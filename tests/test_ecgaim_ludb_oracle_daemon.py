import sqlite3

import numpy as np

from scripts.evaluate_ecgaim_ludb_oracle_daemon import (
    connect_results,
    lead_role,
    parse_symbol_stream,
    st_features,
    wave_features,
)


def test_lead_roles_keep_primary_claim_separate():
    assert [lead_role(i) for i in (0, 1, 7)] == ["observed_control"] * 3
    assert [lead_role(i) for i in (2, 3, 4, 5)] == ["derived_limb_control"] * 4
    assert [lead_role(i) for i in (6, 8, 9, 10, 11)] == ["primary_missing_precordial"] * 5


def test_symbol_parser_never_guesses_orphan_boundaries():
    symbols = ["N", ")", "(", "t", ")", "(", "p", ")", "N", "(", ")"]
    samples = [10, 14, 20, 25, 30, 40, 45, 50, 60, 61, 66]
    events = parse_symbol_stream(symbols, samples)
    assert events == [
        {"wave": "QRS", "event_index": 0, "onset": None, "peak": 10, "offset": 14},
        {"wave": "T", "event_index": 0, "onset": 20, "peak": 25, "offset": 30},
        {"wave": "P", "event_index": 0, "onset": 40, "peak": 45, "offset": 50},
        {"wave": "QRS", "event_index": 1, "onset": None, "peak": 60, "offset": None},
    ]


def test_wave_features_use_fixed_interval_and_are_dc_invariant():
    signal = np.array([10.0, 11.0, 14.0, 11.0, 10.0])
    shifted = signal + 123.0
    first = wave_features(signal, 0, 2, 4)
    second = wave_features(shifted, 0, 2, 4)
    assert first == second
    assert first["peak_amplitude_mv"] == 4.0
    assert first["signed_area_mv_ms"] == 12.0
    assert first["absolute_area_mv_ms"] == 12.0


def test_st_features_never_move_the_j_sample():
    signal = np.arange(100, dtype=float) / 10
    result = st_features(signal, 10, 60)
    assert result["j_mv"] == signal[10]
    assert result["j20_mv"] == signal[20]
    assert result["j40_mv"] == signal[30]
    assert result["j60_mv"] == signal[40]
    assert result["j80_mv"] == signal[50]


def test_oracle_schema_has_no_predicted_timing_f1_or_dice(tmp_path):
    connection = connect_results(tmp_path / "oracle.sqlite")
    try:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"oracle_events", "oracle_waves", "oracle_st_segments"} <= tables
        assert {"event_metrics", "wave_metrics", "st_metrics", "record_role_metrics"} <= tables
        columns = []
        for table in tables:
            columns.extend(row[1].lower() for row in connection.execute(f"PRAGMA table_info({table})"))
        assert not any("timing" in column for column in columns)
        assert not any("dice" in column for column in columns)
        assert not any(column == "f1" or column.startswith("f1_") for column in columns)
        without_rowid = {
            row[1] for row in connection.execute("PRAGMA table_list") if row[4] == 1
        }
        assert {"event_metrics", "wave_metrics", "st_metrics", "signal_lead_metrics"} <= without_rowid
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_schema_integrity(tmp_path):
    connection = connect_results(tmp_path / "oracle.sqlite")
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()
