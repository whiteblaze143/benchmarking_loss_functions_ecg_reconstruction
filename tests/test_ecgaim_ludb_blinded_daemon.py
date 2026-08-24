import sqlite3

import numpy as np

from scripts.evaluate_ecgaim_ludb_blinded_daemon import (
    BOUNDARIES,
    LANDMARKS,
    DelineationResult,
    connect_results,
    delineation_row,
    interval_rows_for_pair,
    monotonic_match_indices,
    primary_summary,
    reference_result,
    summarize_delineation,
)


def result_from_beats(beats):
    return DelineationResult(
        landmarks={
            landmark: np.asarray(
                [beat[landmark] for beat in beats if np.isfinite(beat[landmark])],
                dtype=np.int64,
            )
            for landmark in LANDMARKS
        },
        beats=beats,
        status="ok",
        r_peaks=np.asarray([beat["QRS_peak"] for beat in beats], dtype=np.int64),
    )


def complete_beat(shift=0):
    return {
        "P_onset": 100 + shift,
        "P_peak": 120 + shift,
        "P_offset": 140 + shift,
        "QRS_onset": 180 + shift,
        "QRS_peak": 200 + shift,
        "QRS_offset": 220 + shift,
        "T_onset": 250 + shift,
        "T_peak": 300 + shift,
        "T_offset": 350 + shift,
    }


def test_match_is_one_to_one_and_never_reuses_a_prediction():
    pairs = monotonic_match_indices([100, 110], [105], tolerance_samples=10)
    assert len(pairs) == 1
    assert len({right for _, right in pairs}) == 1


def test_unmatched_reference_is_retained_as_false_negative():
    reference = result_from_beats([complete_beat(0), complete_beat(500)])
    predicted = result_from_beats([complete_beat(0)])
    row = delineation_row(
        "m", "reconstruction", "dwt", "blind", "2", "III", "missing",
        "QRS_onset", reference, predicted,
    )
    assert row["reference_events"] == 2
    assert row["predicted_events"] == 1
    assert row["tp_20"] == 1
    assert row["f1_20"] == 2 / 3


def test_reference_result_exposes_all_nine_landmarks():
    annotations = {
        "P": [{"onset": 100, "peak": 120, "offset": 140}],
        "QRS": [{"onset": 180, "peak": 200, "offset": 220}],
        "T": [{"onset": 250, "peak": 300, "offset": 350}],
    }
    result = reference_result(annotations)
    assert set(result.landmarks) == set(LANDMARKS)
    assert all(len(result.landmarks[name]) == 1 for name in LANDMARKS)
    assert len(result.beats) == 1


def test_intervals_use_matched_beats_and_preserve_signed_error():
    reference = result_from_beats([complete_beat(0), complete_beat(500)])
    predicted_beats = [complete_beat(2), complete_beat(502)]
    # Increase reconstructed QT by 10 samples without changing its QRS anchor.
    for beat in predicted_beats:
        beat["T_offset"] += 10
    predicted = result_from_beats(predicted_beats)
    rows = interval_rows_for_pair(
        "m", "reconstruction", "dwt", "blind", "2", "III", "missing",
        reference, predicted,
    )
    qt = next(row for row in rows if row["interval_name"] == "QT_interval")
    assert qt["reference_intervals"] == 2
    assert qt["matched_150"] == 2
    assert qt["timing_bias_ms"] == 20.0
    assert qt["timing_mae_ms"] == 20.0


def test_rr_uses_consecutive_matched_beats_not_sorted_durations():
    reference = result_from_beats([
        complete_beat(0), complete_beat(500), complete_beat(1000),
    ])
    predicted_beats = [complete_beat(2), complete_beat(522), complete_beat(1002)]
    predicted = result_from_beats(predicted_beats)
    rows = interval_rows_for_pair(
        "m", "reconstruction", "dwt", "blind", "2", "III", "missing",
        reference, predicted,
    )
    rr = next(row for row in rows if row["interval_name"] == "RR_interval")
    assert rr["matched_150"] == 2
    assert rr["timing_bias_ms"] == 0.0
    assert rr["timing_mae_ms"] == 40.0


def test_primary_summary_uses_only_prespecified_six_boundaries():
    rows = []
    for landmark in LANDMARKS:
        rows.append(
            {
                "source": "reconstruction", "method": "dwt", "mode": "blind",
                "subgroup": "all", "lead_role": "missing", "landmark": landmark,
                "micro_f1_20": 0.8 if landmark in BOUNDARIES else 0.0,
            }
        )
    result = primary_summary(rows)
    assert result["boundaries_present"] == 6
    assert np.isclose(result["boundary_macro_f1_20"], 0.8)


def test_database_enforces_unique_record_lead_landmark(tmp_path):
    connection = connect_results(tmp_path / "test.sqlite")
    try:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "delineation_metrics", "interval_metrics", "delineation_summaries",
            "interval_summaries", "signal_summaries",
        } <= tables
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        without_rowid = {
            row[1] for row in connection.execute("PRAGMA table_list") if row[4] == 1
        }
        assert {"delineation_metrics", "interval_metrics", "delineation_summaries"} <= without_rowid
    finally:
        connection.close()


def test_materialized_subgroups_exclude_exact_diagnoses_but_keep_categories():
    metric = {
        "record_id": "2", "method": "dwt", "mode": "blind",
        "lead_role": "missing", "landmark": "P_onset", "detector_status": "ok",
        "reference_events": 1, "predicted_events": 1, "matched_150": 1,
        "timing_bias_ms": 0.0, "timing_mae_ms": 0.0, "timing_p95_abs_ms": 0.0,
        **{f"tp_{value}": 1 for value in (10, 20, 40, 80, 150)},
        **{f"f1_{value}": 1.0 for value in (10, 20, 40, 80, 150)},
    }
    records = [{
        "record_id": "2",
        "metadata": {"subgroups": {
            "all": "derived", "category:ischemia_or_scar": "derived",
            "diagnosis_exact:rare_label": "header",
        }},
    }]
    summaries = summarize_delineation("m", "reconstruction", [metric], records)
    names = {row["subgroup"] for row in summaries}
    assert "all" in names
    assert "category:ischemia_or_scar" in names
    assert "diagnosis_exact:rare_label" not in names


def test_disk_reserve_gate(tmp_path):
    from scripts.evaluate_ecgaim_ludb_blinded_daemon import sufficient_disk_space

    assert sufficient_disk_space(tmp_path, 0)[0]
    assert not sufficient_disk_space(tmp_path, 10**9)[0]
