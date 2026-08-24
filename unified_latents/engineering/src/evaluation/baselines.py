"""Baseline models for ECG reconstruction evaluation."""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

LOGGER = logging.getLogger(__name__)


def _collect_arrays(loader) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for batch in loader:
        inp = batch["input"].detach().cpu().numpy()
        tgt = batch["target"].detach().cpu().numpy()
        xs.append(inp.reshape(inp.shape[0], -1))
        ys.append(tgt.reshape(tgt.shape[0], -1))
    if not xs:
        raise ValueError("No data available in loader")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_linear_baseline(train_loader) -> LinearRegression:
    """Train a simple linear regression baseline mapping input leads to outputs."""
    X, Y = _collect_arrays(train_loader)
    model = LinearRegression()
    model.fit(X, Y)
    return model


def evaluate_linear_baseline(model: LinearRegression, loader) -> np.ndarray:
    """Return predictions shaped (batch, leads, time)."""
    xs = []
    preds = []
    for batch in loader:
        inp = batch["input"].detach().cpu().numpy()
        xs.append(inp)
        pred = model.predict(inp.reshape(inp.shape[0], -1))
        pred = pred.reshape(inp.shape[0], batch["target"].shape[1], batch["target"].shape[2])
        preds.append(pred)
    if not preds:
        raise ValueError("Loader produced no batches")
    return np.concatenate(preds, axis=0)


def evaluate_mason_unconditional(
    sunnybrook_root: Path,
    state_dict_dir: Path,
    split: str,
    output_dir: Path,
    device: str = "cuda:0",
) -> Optional[Path]:
    """Evaluate Mason baseline using the third-party reconstruction pipeline if available."""
    sunnybrook_root = sunnybrook_root.expanduser().resolve()
    state_dict_dir = state_dict_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    third_party = Path(__file__).resolve().parents[2] / "third_party" / "ecg_reconstruction-main"
    try:
        third_party = Path(__file__).resolve().parents[2] / "third_party" / "ecg_reconstruction-main"
        sys.path.insert(0, str(third_party))
        from analysis.sunnybrook_eval import evaluate_sunnybrook
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to import Mason evaluation pipeline: %s", exc)
        return None

    manager_args = {
        "device": device,
        "input_leads": "limb+v2",
        "output_leads": "full",
        "dataset": "infarct+noninfarct",
        "data_size": "max",
        "input_channel": None,
        "middle_channel": None,
        "output_channel": None,
        "input_depth": None,
        "middle_depth": None,
        "output_depth": None,
        "input_kernel": None,
        "middle_kernel": None,
        "output_kernel": None,
        "use_residual": None,
        "epoch_num": 200,
        "batch_size": 32,
        "prioritize_percent": None,
        "prioritize_size": None,
        "optimizer": None,
        "learning_rate": None,
        "weight_decay": None,
        "momentum": None,
        "nesterov": None,
        "heteroscedastic": False,
        "stage1_epochs": None,
    }

    cwd = Path.cwd()
    calibration_override = third_party / "output" / "eval_metrics" / "per_lead_gain_identity.json"
    if calibration_override.exists():
        os.environ.setdefault("MASON_CALIBRATION_JSON", str(calibration_override))
    try:
        os.chdir(third_party)
        os.environ["MASON_CALIBRATION_JSON"] = str(calibration_override)
        LOGGER.info("Mason baseline using calibration file: %s", os.environ["MASON_CALIBRATION_JSON"])
        evaluate_sunnybrook(
            manager_args=manager_args,
            state_dict_path=Path(state_dict_dir).resolve(),
            sunnybrook_root=Path(sunnybrook_root).resolve(),
            split=split,
            output_dir=output_dir / split,
            glob_pattern=None,
            limit=None,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Mason baseline evaluation failed: %s", exc)
        fallback = third_party / "output" / "sunnybrook_calibrated" / split / "metrics_summary.csv"
        if fallback.exists():
            LOGGER.info("Falling back to precomputed Mason metrics at %s", fallback)
            return fallback
        return None
    finally:
        os.chdir(cwd)
    return output_dir / split / "metrics_summary.csv"

