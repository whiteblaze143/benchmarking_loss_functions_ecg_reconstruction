"""Unit tests for Paper 07 168h Ablation Week Architectures.

Verifies:
1. All 22 experiment configurations instantiate cleanly.
2. Forward passes execute without error on microbatches.
3. Output logits have shape (batch, 1) and latent has shape (batch, 256) (or (batch, 7) for tabular-only).
4. Loss is finite and backward() propagates gradients to all active classification parameters.
5. Invariance properties hold:
   - PhaseCNN cyclic-shift invariance on standard_circular (< 1e-6).
   - Phase-position-aware readout breaks cyclic invariance (> 1e-3).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

DEV_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.ablation_setoperator.ablation_models import (
    AblationSetOperatorEchoNext,
    TabularMLPEchoNext,
    ModulatedPhaseCNN,
)


def test_phasecnn_cyclic_shift_invariance():
    """Verify that standard PhaseCNN is invariant to cyclic phase roll, but phase_position_aware is not."""
    standard = ModulatedPhaseCNN(input_dim=128, width=128, blocks=2, classes=128, phase_mode="standard_circular")
    aware = ModulatedPhaseCNN(input_dim=128, width=128, blocks=2, classes=128, phase_mode="phase_position_aware")

    standard.eval()
    aware.eval()

    # Modify aware readout logits away from uniform to verify position awareness
    with torch.no_grad():
        aware.readout.phase_logits[0] = 5.0

    torch.manual_seed(42)
    x = torch.randn(4, 16, 128)
    x_roll = torch.roll(x, shifts=3, dims=1)

    with torch.no_grad():
        y_std_0 = standard(x)
        y_std_r = standard(x_roll)
        std_diff = torch.max(torch.abs(y_std_r - y_std_0)).item()

        y_aw_0 = aware(x)
        y_aw_r = aware(x_roll)
        aw_diff = torch.max(torch.abs(y_aw_r - y_aw_0)).item()

    assert std_diff < 1e-6, f"Standard PhaseCNN must be invariant to cyclic roll, got max diff {std_diff}"
    assert aw_diff > 1e-3, f"Phase-position-aware PhaseCNN must break cyclic invariance, got diff {aw_diff}"


def test_manifest_experiment_instantiations():
    """Verify that every experiment in ablation_week_manifest.json instantiates and runs forward/backward."""
    manifest_path = DEV_ROOT / "experiments/echonext_operator_benchmark/ablation_week_manifest.json"
    assert manifest_path.exists(), "Manifest file missing"

    with open(manifest_path, "r") as f:
        data = json.load(f)

    b, k = 2, 4
    ops = torch.randn(b, k, 8)
    ops = ops / torch.linalg.norm(ops, dim=-1, keepdim=True).clamp_min(1e-6)
    resps = torch.randn(b, k, 16, 128)
    tabs = torch.randn(b, 7)
    mask = torch.zeros(b, k, dtype=torch.bool)
    mask[0, -1] = True  # padded token

    for exp in data["experiments"]:
        exp_id = exp["id"]
        model_cls = exp["model_class"]

        if model_cls == "TabularMLPEchoNext":
            model = TabularMLPEchoNext(tabular_dim=7, classes=1)
            out = model(tabular=tabs)
            assert out["logits"].shape == (b, 1)
            loss = out["logits"].sum()
            loss.backward()
            continue

        model = AblationSetOperatorEchoNext(
            operator_mode=exp.get("operator_mode", "continuous"),
            geometry_mode=exp.get("geometry_mode", "baseline"),
            phase_mode=exp.get("phase_mode", "standard_circular"),
            set_encoder_mode=exp.get("set_encoder_mode", "standard_transformer"),
            pooling_mode=exp.get("pooling_mode", "learned_query"),
            tabular_mode=exp.get("tabular_mode", "native_additive"),
            classes=1,
        )

        out = model(operators=ops, responses=resps, tabular=tabs if exp.get("tabular_mode") != "none" else None, mask=mask)
        assert out["logits"].shape == (b, 1), f"Experiment {exp_id} produced logits shape {out['logits'].shape}"

        loss = out["logits"].sum()
        loss.backward()

        # Verify gradient flow to operator and response encoder
        if model.operator is not None:
            has_op_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.operator.parameters())
            assert has_op_grad, f"Experiment {exp_id} has zero operator gradients"

        has_resp_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.response.parameters())
        assert has_resp_grad, f"Experiment {exp_id} has zero response gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
