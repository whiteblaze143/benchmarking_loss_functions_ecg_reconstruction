"""Load the leakage-free frozen ECG-FM PTB-XL signal-quality head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from scripts.bootstrap_paths import ECG_FM_ROOT


def preprocess(signals: torch.Tensor) -> torch.Tensor:
    signals = signals.float()
    return (signals - signals.mean(dim=-1, keepdim=True)) / signals.std(
        dim=-1, keepdim=True, unbiased=False
    ).clamp_min(1e-8)


def load_classifier(
    backbone_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    sys.path.insert(0, str(ECG_FM_ROOT))
    from src.ecg_fm_classifier import ECGFMClassifier

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_used_for_training_or_selection") is not False:
        raise RuntimeError("Signal-quality checkpoint does not prove test isolation")
    classes = list(checkpoint["classes"])
    model = ECGFMClassifier(
        str(backbone_path), num_classes=len(classes), dropout=0.3, freeze_backbone=True
    )
    model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    return model.to(device).eval(), classes
