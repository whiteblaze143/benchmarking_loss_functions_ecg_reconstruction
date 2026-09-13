"""Strict loading helpers for the unmodified author Nef-Net source copy."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def instantiate_author_model(
    source_root: Path,
    device: torch.device,
    super_mode: str = "optimization",
    lead_num: int = 3,
):
    """Instantiate the author model without executing its eager package imports.

    The release's ``network/__init__.py`` eagerly imports unrelated models and
    undeclared dependencies. A namespace package exposes only the requested
    model module while leaving every author file unchanged.
    """
    codes = Path(source_root) / "codes"
    network_dir = codes / "network"
    if not (network_dir / "nefnet_plus.py").is_file():
        raise FileNotFoundError(f"author Nef-Net v2 model not found under {network_dir}")
    package = types.ModuleType("network")
    package.__path__ = [str(network_dir)]
    sys.modules["network"] = package
    importlib.invalidate_caches()
    config_module = types.ModuleType("config")
    config_module.cfg = None
    sys.modules["config"] = config_module
    # The release imports these timm symbols but never uses them in `layer`.
    # Sentinels keep the author file untouched and fail if that assumption changes.
    def _unused_timm_symbol(*args, **kwargs):
        raise RuntimeError("unexpected use of undeclared timm dependency")

    timm_module = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.DropPath = _unused_timm_symbol
    timm_layers.trunc_normal_ = _unused_timm_symbol
    sys.modules.setdefault("timm", timm_module)
    sys.modules.setdefault("timm.models", timm_models)
    sys.modules.setdefault("timm.models.layers", timm_layers)
    model_class = importlib.import_module("network.nefnet_plus").layer
    config = SimpleNamespace(
        MODEL=SimpleNamespace(layers=4, theta_L=1),
        DATA=SimpleNamespace(super_mode=super_mode, lead_num=lead_num),
    )
    return model_class(config).to(device)



def load_author_model(source_root: Path, checkpoint: Path, device: torch.device):
    """Instantiate the author model and load an exact raw state_dict."""
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"trained checkpoint not found: {checkpoint}")
    model = instantiate_author_model(source_root, device)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must be a non-empty raw PyTorch state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("checkpoint is not a raw state_dict; unwrap it explicitly before use")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
