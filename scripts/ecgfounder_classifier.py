"""Frozen ECGFounder classifier loading and PTB-XL label integration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_task_names(tasks_path: Path) -> list[str]:
    return [line.strip() for line in tasks_path.read_text().splitlines() if line.strip()]


def load_ptbxl_labels(labels_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(labels_path)
    if "filename_hr" not in frame or "label" not in frame:
        raise ValueError(f"Invalid ECGFounder label file: {labels_path}")
    frame = frame[["filename_hr", "label"]].copy()
    frame["ecgfounder_labels"] = frame["label"].map(
        lambda value: np.asarray(json.loads(value), dtype=np.float32)
    )
    return frame.drop(columns="label").drop_duplicates("filename_hr")


def attach_ptbxl_labels(metadata: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = load_ptbxl_labels(labels_path).set_index("filename_hr")
    output = metadata.copy()
    output["ecgfounder_labels"] = output["filename_hr"].map(labels["ecgfounder_labels"])
    missing = output["ecgfounder_labels"].isna()
    if missing.any():
        examples = output.loc[missing, "filename_hr"].head().tolist()
        raise ValueError(
            f"ECGFounder labels missing for {int(missing.sum())} PTB-XL rows; examples={examples}"
        )
    return output


def preprocess_ecgfounder(signals: torch.Tensor) -> torch.Tensor:
    """Match the official PTB-XL validation's per-record global z-score."""
    signals = signals.float()
    mean = signals.mean(dim=(1, 2), keepdim=True)
    std = signals.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-8)
    return (signals - mean) / std


def load_ecgfounder(
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    n_classes: int,
) -> torch.nn.Module:
    net_path = repo_path / "net1d.py"
    module_spec = importlib.util.spec_from_file_location("comprehensive_ecgfounder_net1d", net_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to load ECGFounder architecture: {net_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
