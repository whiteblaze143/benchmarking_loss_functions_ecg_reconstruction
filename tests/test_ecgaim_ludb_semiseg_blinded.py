from pathlib import Path

import numpy as np

from scripts.evaluate_ecgaim_ludb_semiseg_blinded import (
    DEFAULT_SELECTION,
    DEFAULT_TEST_INDEX,
    ROOT,
    class_boundaries,
    connect,
    initialize,
    official_test_records,
    selected_checkpoint,
)


def test_validation_selected_checkpoint_is_content_bound():
    chosen = selected_checkpoint(
        ROOT / "results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42/best-MeanIoU.pth",
        DEFAULT_SELECTION,
        "model_ema",
    )
    assert len(chosen["sha256"]) == 64
    assert chosen["state"] == "model_ema"


def test_official_test_split_is_frozen_and_disjoint_count():
    records, digest, split = official_test_records(ROOT / "data/ludb", DEFAULT_TEST_INDEX, 0)
    assert len(records) == 40
    assert len(split["all_test_subject_ids"]) == 40
    assert len(set(split["all_test_subject_ids"])) == 40
    assert len(digest) == 64


def test_dense_prediction_boundaries_are_inclusive():
    mask = np.zeros(5000, dtype=np.int8)
    mask[10:20], mask[30:45], mask[50:80] = 1, 2, 3
    result = class_boundaries(mask)
    assert result["P_onset"].tolist() == [10]
    assert result["P_offset"].tolist() == [19]
    assert result["QRS_onset"].tolist() == [30]
    assert result["QRS_offset"].tolist() == [44]
    assert result["T_onset"].tolist() == [50]
    assert result["T_offset"].tolist() == [79]


def test_compact_database_has_no_raw_metric_tables(tmp_path: Path):
    path = tmp_path / "compact.sqlite"
    connection = connect(path)
    initialize(connection, {"test": True}, "d" * 64)
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert tables == {"metadata", "evaluations", "boundary_summaries"}
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
