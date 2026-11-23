"""Provenance Guard Contract for T-CAIREM V4.1.

Strict provenance enforcement hooks that prevent unverified, parametric,
or synthetic metrics from being ingested into benchmark tables, reports, or figures.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np


class ProvenanceViolationError(ValueError):
    """Raised when an empirical claim lacks authentic computational lineage."""
    pass


PROVENANCE_CLASSES = {
    "MEASURED",              # Derived from actual raw waveform / model inference on verified data
    "ANALYTICAL",            # Exact mathematical derivation (e.g., exact Charitos integral)
    "CONTROLLED_SIMULATION", # Exact simulation with verified ground truth (e.g., finite-grid permutations)
    "PARAMETRIC_FORMULA",    # UNACCEPTABLE for empirical claims - formulaic scaling
    "HARDCODED_LITERAL",     # UNACCEPTABLE for empirical claims - hand-entered numbers
}


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute hash for non-existent file: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_array_sha256(arr: np.ndarray) -> str:
    """Computes SHA-256 hash of a numpy array buffer."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_empirical_record(
    metric_name: str,
    value: float,
    provenance_class: str,
    source_script: Path | str,
    raw_predictions: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    checkpoint_path: Optional[Path | str] = None,
    input_data_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Validates that an empirical metric record complies with V4.1 integrity gates."""
    if provenance_class not in PROVENANCE_CLASSES:
        raise ProvenanceViolationError(
            f"Invalid provenance class '{provenance_class}'. Must be one of: {PROVENANCE_CLASSES}"
        )
    
    if provenance_class in ("PARAMETRIC_FORMULA", "HARDCODED_LITERAL"):
        raise ProvenanceViolationError(
            f"Integrity rejection: '{metric_name}' is tagged with forbidden provenance class '{provenance_class}'."
        )

    record = {
        "metric_name": metric_name,
        "value": float(value),
        "provenance_class": provenance_class,
        "source_script": str(source_script),
        "timestamp": None,
    }

    if provenance_class == "MEASURED":
        if raw_predictions is None or ground_truth is None:
            raise ProvenanceViolationError(
                f"MEASURED metric '{metric_name}' must provide both raw_predictions and ground_truth arrays."
            )
        if len(raw_predictions) != len(ground_truth):
            raise ProvenanceViolationError(
                f"Prediction count ({len(raw_predictions)}) does not match ground truth count ({len(ground_truth)})."
            )
        if len(raw_predictions) == 0:
            raise ProvenanceViolationError("Predictions array cannot be empty.")
        
        record["prediction_hash"] = compute_array_sha256(raw_predictions)
        record["ground_truth_hash"] = compute_array_sha256(ground_truth)
        record["n_samples"] = int(len(raw_predictions))

        if checkpoint_path is not None:
            p_ckpt = Path(checkpoint_path)
            if not p_ckpt.is_file():
                raise ProvenanceViolationError(f"Checkpoint not found: {p_ckpt}")
            record["checkpoint_hash"] = compute_file_sha256(p_ckpt)
            record["checkpoint_path"] = str(p_ckpt)

        if input_data_path is not None:
            p_data = Path(input_data_path)
            if not p_data.exists():
                raise ProvenanceViolationError(f"Input data not found: {p_data}")
            if p_data.is_file():
                record["input_data_hash"] = compute_file_sha256(p_data)
            record["input_data_path"] = str(p_data)

    return record
