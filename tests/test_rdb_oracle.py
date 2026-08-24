import sqlite3

import numpy as np

from scripts.analyze_ecgaim_rdb_oracle import bootstrap
from scripts.evaluate_ecgaim_rdb_oracle_daemon import connect_results, summarize_wave
from scripts.rdb_oracle import interval_metric, lead_role, next_t_onset, round_half_up, st_features, wave_features


def test_half_up_resolution_is_deterministic():
    assert [round_half_up(x) for x in (0.0, 1.49, 1.5, 1.51)] == [0, 1, 2, 2]


def test_lead_roles_match_ecgaim_claim_scope():
    assert [lead_role(i) for i in (0, 1, 7)] == ["observed_control"] * 3
    assert [lead_role(i) for i in (2, 3, 4, 5)] == ["derived_limb_control"] * 4
    assert [lead_role(i) for i in (6, 8, 9, 10, 11)] == ["primary_missing_precordial"] * 5


def test_wave_features_have_no_peak_and_are_dc_invariant():
    signal = np.array([10.0, 11.0, 14.0, 11.0, 10.0])
    assert wave_features(signal, 0, 4) == wave_features(signal + 100, 0, 4)
    assert set(wave_features(signal, 0, 4)) == {"signed_area_mv_ms", "absolute_area_mv_ms"}


def test_interval_metric_uses_exact_inclusive_boundaries():
    reference = np.arange(8, dtype=float)
    predicted = reference.copy(); predicted[2:6] += 1
    metric = interval_metric(reference, predicted, 2, 5)
    assert metric["onset_error_mv"] == 1
    assert metric["offset_error_mv"] == 1
    assert metric["window_rmse_mv"] == 1


def test_j_point_is_first_sample_after_inclusive_qrs_end():
    signal = np.arange(100, dtype=float)
    result = st_features(signal, 10, 70)
    assert result["qrs_offset_mv"] == 10
    assert result["j_mv"] == 11
    assert result["j20_mv"] == 21
    assert result["j60_mv"] == 41


def test_next_t_requires_nonempty_st_interval():
    intervals = np.array([[1, 10, 20], [2, 21, 40], [2, 22, 41], [2, 30, 50]])
    assert next_t_onset(intervals, 20) == 22


def test_compact_schema_has_no_per_interval_predictions_or_dice(tmp_path):
    connection = connect_results(tmp_path / "rdb.sqlite")
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"record_role_signal_metrics", "record_role_wave_metrics", "record_role_st_metrics", "lead_wave_summaries"} <= tables
        assert not ({"event_metrics", "wave_metrics", "oracle_events", "oracle_waves"} & tables)
        columns = [row[1].lower() for table in tables for row in connection.execute(f"PRAGMA table_info({table})")]
        assert not any("dice" in column or column == "f1" or "timing_error" in column for column in columns)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_wave_summary_reports_tail_metrics():
    rows = []
    for error in range(1, 6):
        rows.append({
            "onset_error_mv": error, "offset_error_mv": -error,
            "window_pearson": 1-error/10, "window_rmse_mv": error,
            "window_mae_mv": error, "window_max_abs_error_mv": error,
            "signed_area_error_mv_ms": error, "absolute_area_error_mv_ms": -error,
        })
    result = summarize_wave(rows)
    assert result["intervals"] == 5
    assert result["onset_abs_error_p95_mv"] > result["onset_abs_error_median_mv"]


def test_paired_bootstrap_is_vectorized_and_improvement_oriented(tmp_path):
    connection = connect_results(tmp_path / "bootstrap.sqlite")
    try:
        common = ("sha", 1, "protocol", "dataset", "complete", 1, "now", "now", 1.0, 20, None)
        for evaluation_id, (mask, model) in enumerate((("1000000", "baseline"), ("1100000", "better")), 1):
            connection.execute(
                """INSERT INTO evaluations(evaluation_id,model_id,factorial_mask,checkpoint_sha256,
                checkpoint_size_bytes,protocol_sha256,dataset_sha256,status,attempts,started_at,
                completed_at,duration_seconds,n_records,error,primary_summary_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (evaluation_id, model, mask, *common, "{}"),
            )
            scale = 1.0 if evaluation_id == 1 else 0.5
            for index in range(20):
                record_id = f"R{index:02d}"
                connection.execute(
                    "INSERT INTO record_role_signal_metrics VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (evaluation_id,record_id,"SR","primary_missing_precordial",5,
                     0.5+(1-scale)*0.2,0.4,scale,scale,scale),
                )
                for wave in ("QRS", "T"):
                    values=(evaluation_id,record_id,"SR","primary_missing_precordial",wave,2,
                            0.0,scale,scale,scale,0.0,scale,scale,scale,0.5,0.5,
                            scale,scale,scale,scale,scale,scale,scale,scale)
                    connection.execute("INSERT INTO record_role_wave_metrics VALUES("+",".join("?"*24)+")",values)
                connection.execute(
                    "INSERT INTO record_role_st_metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (evaluation_id,record_id,"SR","primary_missing_precordial",2,scale,scale,scale,
                     scale,scale,scale,scale,0.5,scale),
                )
        connection.commit()
        models=[{"evaluation_id":1,"factorial_mask":"1000000"},{"evaluation_id":2,"factorial_mask":"1100000"}]
        rows=bootstrap(connection,models,n=50,seed=7)
        assert len(rows)==8
        assert all(row["records"]==20 for row in rows)
        assert all(row["improvement_estimate"]>0 for row in rows)
    finally:
        connection.close()
