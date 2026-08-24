"""Feature ablation utilities for conditional ECG reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import pandas as pd

from learn_functions.builder import build_model_from_config, load_config as load_model_config
def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _evaluate_loss(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    masked_features: Optional[Sequence[str]] = None,
) -> float:
    loss = 0.0
    count = 0
    mse = torch.nn.MSELoss(reduction="sum")
    masked_features = masked_features or []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            metadata = {k: v.clone().to(device) for k, v in batch["metadata"].items()}
            for feature in masked_features:
                if feature in metadata:
                    metadata[feature] = torch.zeros_like(metadata[feature])
            outputs = model(inputs, metadata)
            
            if isinstance(outputs, tuple):
                mu, _ = outputs
                loss += mse(mu, targets).item()
            else:
                loss += mse(outputs, targets).item()
                
            count += targets.numel()
    return loss / max(count, 1)


def ablation_study(
    config_path: Path,
    checkpoint_path: Path,
    val_loader,
    features: Sequence[str],
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Compute validation loss when masking each conditioning feature."""
    device_t = torch.device(device)
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()

    model = _load_model(config_path, checkpoint_path, device_t)
    baseline_loss = _evaluate_loss(model, val_loader, device_t, masked_features=None)

    rows: List[Dict[str, float]] = []
    for feature in features:
        masked_loss = _evaluate_loss(model, val_loader, device_t, masked_features=[feature])
        change = (masked_loss - baseline_loss) / baseline_loss * 100 if baseline_loss > 0 else float("nan")
        rows.append(
            {
                "feature": feature,
                "loss_without_feature": masked_loss,
                "loss_baseline": baseline_loss,
                "percent_change": change,
            }
        )
    return pd.DataFrame(rows)

