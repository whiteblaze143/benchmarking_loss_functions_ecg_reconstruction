"""Factory helpers for constructing ConditionalResCNN from config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .conditional_rescnn import ConditionalResCNN
from .mason_reconstructor import MasonReconstructor
from .utils import set_global_seed


def load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return data


def build_model_from_config(config_path: Path) -> ConditionalResCNN:
    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_global_seed(seed)

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "conditional_rescnn")
    if model_type == "conditional_rescnn":
        model = ConditionalResCNN(
            in_channels=model_cfg.get("in_channels", 3),
            out_channels=model_cfg.get("out_channels", 9),
            num_layers=len(model_cfg.get("channels", [])) or model_cfg.get("num_layers", 4),
            conditioning_dim=model_cfg.get("conditioning_dim", 64),
            channels=model_cfg.get("channels"),
            kernel_size=model_cfg.get("kernel_size", 7),
            scaling_params_path=model_cfg.get("scaling_params_path"),
            predict_uncertainty=model_cfg.get("predict_uncertainty", False),
        )
    elif model_type == "mason":
        model = MasonReconstructor(
            input_lead_num=model_cfg.get("input_lead_num", 3),
            output_lead_num=model_cfg.get("output_lead_num", 8),
            input_channel_per_lead=model_cfg.get("input_channels", 32),
            middle_channel_per_lead=model_cfg.get("middle_channels", 32),
            output_channel_per_lead=model_cfg.get("output_channels", 32),
            input_depth=model_cfg.get("input_depth", 3),
            middle_depth=model_cfg.get("middle_depth", 2),
            output_depth=model_cfg.get("output_depth", 3),
            kernel_size=model_cfg.get("kernel_size", 17),
            use_residual=model_cfg.get("use_residual", True),
            heteroscedastic=model_cfg.get("heteroscedastic", False),
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
    return model


def summarize_model(model: torch.nn.Module) -> Dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": params, "trainable": trainable}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Instantiate ConditionalResCNN from config.")
    parser.add_argument("--config", required=True, help="Path to conditional_rescnn_config.json")
    args = parser.parse_args()

    model = build_model_from_config(Path(args.config))
    summary = summarize_model(model)
    print(f"Model built with {summary['parameters']:,} parameters ({summary['trainable']:,} trainable).")

