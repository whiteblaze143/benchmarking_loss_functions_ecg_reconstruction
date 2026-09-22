"""Validation of immutable, aligned task-native prediction bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "task_native_aligned_predictions_v1"


@dataclass(frozen=True)
class PredictionBundle:
    path: Path
    dataset: str
    model: str
    task_name: str
    configurations: tuple[str, ...]
    record_ids: np.ndarray
    patient_ids: np.ndarray | None
    y_true: np.ndarray
    probabilities: np.ndarray


def _scalar_text(values: np.ndarray, name: str) -> str:
    if values.shape != ():
        raise ValueError(f"prediction artifact field {name!r} must be scalar")
    return str(values.item())


def load_prediction_bundle(path: Path) -> PredictionBundle:
    """Load one task-native prediction artifact and enforce its shape contract."""
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "schema_version", "dataset", "model", "task_name", "configurations",
            "record_ids", "y_true", "probabilities",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"prediction artifact missing fields: {sorted(missing)}")
        if _scalar_text(payload["schema_version"], "schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported prediction artifact schema: {path}")
        configurations = tuple(map(str, payload["configurations"].tolist()))
        record_ids = np.asarray(payload["record_ids"], dtype=str)
        y_true = np.asarray(payload["y_true"], dtype=np.float32)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        patient_ids = np.asarray(payload["patient_ids"], dtype=str) if "patient_ids" in payload else None
        result = PredictionBundle(
            path=path,
            dataset=_scalar_text(payload["dataset"], "dataset"),
            model=_scalar_text(payload["model"], "model"),
            task_name=_scalar_text(payload["task_name"], "task_name"),
            configurations=configurations,
            record_ids=record_ids,
            patient_ids=patient_ids,
            y_true=y_true,
            probabilities=probabilities,
        )
    if not result.configurations or len(set(result.configurations)) != len(result.configurations):
        raise ValueError("prediction artifact configurations must be nonempty and unique")
    if result.record_ids.ndim != 1 or not len(result.record_ids) or len(set(result.record_ids)) != len(result.record_ids):
        raise ValueError("prediction artifact record IDs must be nonempty and unique")
    if result.y_true.ndim != 2 or result.y_true.shape[0] != len(result.record_ids):
        raise ValueError("prediction artifact y_true must be [record,class]")
    if result.probabilities.shape != (len(result.configurations), *result.y_true.shape):
        raise ValueError("prediction artifact probabilities must be [configuration,record,class]")
    if not (np.isfinite(result.y_true).all() and np.isfinite(result.probabilities).all()):
        raise ValueError("prediction artifact contains non-finite values")
    if not np.isin(result.y_true, (0.0, 1.0)).all():
        raise ValueError("prediction artifact y_true must be binary")
    if np.any(result.probabilities < 0.0) or np.any(result.probabilities > 1.0):
        raise ValueError("prediction artifact probabilities must lie in [0,1]")
    if result.patient_ids is not None:
        if result.patient_ids.ndim != 1 or len(result.patient_ids) != len(result.record_ids):
            raise ValueError("prediction artifact patient IDs must align to record IDs")
        if np.any(np.char.strip(result.patient_ids) == ""):
            raise ValueError("prediction artifact patient IDs must be nonempty")
    return result


def require_paired_bundles(left: PredictionBundle, right: PredictionBundle) -> None:
    """Reject any mismatch that would invalidate paired statistical inference."""
    for field in ("dataset", "task_name", "configurations"):
        if getattr(left, field) != getattr(right, field):
            raise ValueError(f"paired prediction artifacts differ in {field}")
    if not np.array_equal(left.record_ids, right.record_ids):
        raise ValueError("paired prediction artifacts have different record ordering")
    if not np.array_equal(left.y_true, right.y_true):
        raise ValueError("paired prediction artifacts have different labels")
    if (left.patient_ids is None) != (right.patient_ids is None):
        raise ValueError("paired prediction artifacts disagree on patient-ID availability")
    if left.patient_ids is not None and not np.array_equal(left.patient_ids, right.patient_ids):
        raise ValueError("paired prediction artifacts have different patient IDs")
