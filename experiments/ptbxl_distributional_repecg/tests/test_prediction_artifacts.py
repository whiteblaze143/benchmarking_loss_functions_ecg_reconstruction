from pathlib import Path

import numpy as np
import pytest

from repecg.evaluation.prediction_artifacts import (
    SCHEMA_VERSION,
    load_prediction_bundle,
    require_paired_bundles,
)


def _write_bundle(path: Path, *, model: str, record_ids: np.ndarray, y_true: np.ndarray) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION), dataset=np.asarray("RDB"),
        model=np.asarray(model), task_name=np.asarray("rhythm"),
        configurations=np.asarray(["Q8_indep", "S1_lead_II"]), record_ids=record_ids,
        patient_ids=np.asarray(["p1", "p2"]), y_true=y_true,
        probabilities=np.asarray([[[0.1], [0.9]], [[0.2], [0.8]]], dtype=np.float32),
    )


def test_paired_prediction_artifacts_require_exact_alignment(tmp_path: Path) -> None:
    labels = np.asarray([[0.0], [1.0]], dtype=np.float32)
    left_path, right_path = tmp_path / "left.npz", tmp_path / "right.npz"
    _write_bundle(left_path, model="left", record_ids=np.asarray(["r1", "r2"]), y_true=labels)
    _write_bundle(right_path, model="right", record_ids=np.asarray(["r1", "r2"]), y_true=labels)
    require_paired_bundles(load_prediction_bundle(left_path), load_prediction_bundle(right_path))


def test_paired_prediction_artifacts_reject_record_reordering(tmp_path: Path) -> None:
    labels = np.asarray([[0.0], [1.0]], dtype=np.float32)
    left_path, right_path = tmp_path / "left.npz", tmp_path / "right.npz"
    _write_bundle(left_path, model="left", record_ids=np.asarray(["r1", "r2"]), y_true=labels)
    _write_bundle(right_path, model="right", record_ids=np.asarray(["r2", "r1"]), y_true=labels)
    with pytest.raises(ValueError, match="record ordering"):
        require_paired_bundles(load_prediction_bundle(left_path), load_prediction_bundle(right_path))
