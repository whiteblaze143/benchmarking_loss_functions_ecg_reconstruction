from __future__ import annotations

import numpy as np

from scripts.task_native.run_paired_task_native_statistics import run


def _write_bundle(root, model: str, probabilities: np.ndarray, patient_ids: np.ndarray | None = None) -> None:
    payload = {
        "schema_version": np.asarray("task_native_aligned_predictions_v1"),
        "dataset": np.asarray("TEST"),
        "model": np.asarray(model),
        "task_name": np.asarray("binary task"),
        "configurations": np.asarray(["Q8_indep"]),
        "record_ids": np.asarray(["a", "b", "c", "d", "e", "f"]),
        "y_true": np.asarray([[0], [1], [0], [1], [0], [1]], dtype=np.float32),
        "probabilities": probabilities[None, :, None].astype(np.float32),
    }
    if patient_ids is not None:
        payload["patient_ids"] = patient_ids
    np.savez_compressed(root / f"test__{model}.npz", **payload)


def test_statistics_are_paired_and_record_bootstrapped_without_patient_ids(tmp_path) -> None:
    _write_bundle(tmp_path, "model_a", np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]))
    _write_bundle(tmp_path, "model_b", np.asarray([0.8, 0.2, 0.7, 0.3, 0.6, 0.4]))
    manifest, rows = run(tmp_path, models=None, configurations=None, replicates=2000, seed=11)
    assert manifest["rows"] == 1
    assert rows[0]["status"] == "complete"
    assert rows[0]["bootstrap_status"] == "complete"
    assert rows[0]["bootstrap_unit"] == "record"
    assert rows[0]["bootstrap_replicates"] == 2000


def test_repeated_patient_rows_are_not_silently_record_bootstrapped(tmp_path) -> None:
    ids = np.asarray(["p1", "p1", "p2", "p2", "p3", "p3"])
    _write_bundle(tmp_path, "model_a", np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]), ids)
    _write_bundle(tmp_path, "model_b", np.asarray([0.8, 0.2, 0.7, 0.3, 0.6, 0.4]), ids)
    _, rows = run(tmp_path, models=None, configurations=None, replicates=2000, seed=11)
    assert rows[0]["bootstrap_status"] == "ineligible"
    assert "aggregation" in rows[0]["bootstrap_reason"]
