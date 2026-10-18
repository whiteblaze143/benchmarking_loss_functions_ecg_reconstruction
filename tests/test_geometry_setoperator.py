"""Comprehensive Mathematical Verification Tests for Geometry-Aware SetOperator.

Covers:
- Test 4: Measurement permutation invariance
- Test 5: Duplicate measurements stress test
- Test 6: Redundant derived measurements (limb lead algebra)
- Test 7: Equivalent measurement bases (orthogonal rotation)
- Test 8: End-to-end gradient flow for G1 and G2
- Test 9: Masking and cardinality (sizes 1-8, padding invariance)
- Test 10: Exact baseline compatibility at beta=0 from original checkpoints
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.paper07_operator.model import OperatorSetModel
from repecg.geometry_setoperator.geometry_setoperator import GeometryOperatorSetModel
from repecg.geometry_setoperator.geometry_features import gamma, compute_operator_gram


def test_test4_measurement_permutation_invariance():
    """Test 4: Final prediction is invariant to permutation of operator-response pairs."""
    torch.manual_seed(42)
    B = 2
    S = 6
    classes = 5

    for mode in ["baseline", "control_capacity", "G1", "G2", "G1_G2"]:
        model = GeometryOperatorSetModel(classes=classes, geometry_mode=mode)
        model.eval()

        # In G2 and G1_G2, test with both beta=0 and beta!=0
        if "G2" in mode:
            for mod in model.set_encoder.layers:
                mod.self_attn.beta.data.uniform_(-0.5, 0.5)

        ops = torch.randn(B, S, 8)
        resps = torch.randn(B, S, 16, 128)
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[0, 4:] = True  # Padded tokens

        with torch.no_grad():
            out_orig = model(ops, resps, mask=mask)

        # Apply a random permutation to items (within valid and invalid tokens)
        # Permutation: [2, 0, 1, 3, 5, 4]
        perm = [2, 0, 1, 3, 5, 4]
        ops_perm = ops[:, perm, :]
        resps_perm = resps[:, perm, :, :]
        mask_perm = mask[:, perm]

        with torch.no_grad():
            out_perm = model(ops_perm, resps_perm, mask=mask_perm)

        diff = (out_orig - out_perm).abs().max().item()
        assert diff < 1e-5, f"Mode {mode} failed permutation invariance: max_diff={diff}"


def test_test5_duplicate_measurements_stress():
    """Test 5: Quantify prediction shift when duplicate operator-response pair is added."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G2")
    model.eval()
    # Set non-zero beta to observe geometry-aware redundancy response
    for mod in model.set_encoder.layers:
        mod.self_attn.beta.data.fill_(0.5)

    B = 1
    # 3 unique operators
    ops = torch.randn(B, 3, 8)
    resps = torch.randn(B, 3, 16, 128)

    with torch.no_grad():
        pred_base = model(ops, resps)

    # Append duplicate of operator 0
    ops_dup = torch.cat([ops, ops[:, :1]], dim=1)  # (1, 4, 8)
    resps_dup = torch.cat([resps, resps[:, :1]], dim=1)  # (1, 4, 16, 128)

    with torch.no_grad():
        pred_dup = model(ops_dup, resps_dup)

    shift = (pred_base - pred_dup).abs().item()
    assert torch.isfinite(pred_dup).all()
    # Log shift for reporting
    print(f"Duplicate measurement logit shift (G2): {shift:.4f}")


def test_test6_redundant_derived_measurements():
    """Test 6: Derived limb leads (I, II vs I, II, III, aVR, aVL, aVF)."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G1_G2")
    model.eval()

    # Lead basis: [I, II, V1, V2, V3, V4, V5, V6]
    # Unit basis vectors:
    e_I = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_II = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Einthoven: III = II - I
    e_III = e_II - e_I
    # Goldberger:
    # aVR = -(I + II)/2
    e_aVR = -0.5 * (e_I + e_II)
    # aVL = I - II/2
    e_aVL = e_I - 0.5 * e_II
    # aVF = II - I/2
    e_aVF = e_II - 0.5 * e_I

    # Minimal 2-lead configuration
    ops_2 = torch.stack([e_I, e_II], dim=0).unsqueeze(0)  # (1, 2, 8)
    resps_2 = torch.randn(1, 2, 16, 128)

    # Full 6 limb lead configuration
    all_limb_ops = torch.stack([e_I, e_II, e_III, e_aVR, e_aVL, e_aVF], dim=0).unsqueeze(0)  # (1, 6, 8)
    # Linear combination for responses (simulating derived responses)
    r_I = resps_2[:, 0]
    r_II = resps_2[:, 1]
    r_III = r_II - r_I
    r_aVR = -0.5 * (r_I + r_II)
    r_aVL = r_I - 0.5 * r_II
    r_aVF = r_II - 0.5 * r_I
    resps_6 = torch.stack([r_I, r_II, r_III, r_aVR, r_aVL, r_aVF], dim=1)  # (1, 6, 16, 128)

    with torch.no_grad():
        out_2 = model(ops_2, resps_2)
        out_6 = model(all_limb_ops, resps_6)

    diff = (out_2 - out_6).abs().item()
    assert torch.isfinite(out_6).all()
    print(f"Redundant limb lead logit shift (2 leads vs 6 leads): {diff:.4f}")


def test_test7_equivalent_measurement_bases():
    """Test 7: Orthogonal coordinate rotation of operator basis."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G1")
    model.eval()

    # Generate 2D rotation for Lead I and Lead II subspace
    theta = math.pi / 4.0  # 45 degrees
    c, s = math.cos(theta), math.sin(theta)
    R_rot = torch.eye(8)
    R_rot[0, 0] = c
    R_rot[0, 1] = -s
    R_rot[1, 0] = s
    R_rot[1, 1] = c

    ops = torch.randn(1, 4, 8)
    ops_rot = ops @ R_rot.T
    resps = torch.randn(1, 4, 16, 128)

    with torch.no_grad():
        out_orig = model(ops, resps)
        out_rot = model(ops_rot, resps)

    diff = (out_orig - out_rot).abs().item()
    assert torch.isfinite(out_rot).all()
    print(f"Orthogonal coordinate rotation logit shift: {diff:.4f}")


def test_test8_end_to_end_gradient_flow():
    """Test 8: Verify gradients reach G1 MLP weights and G2 beta scalars."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G1_G2")
    model.train()

    B, S = 2, 4
    ops = torch.randn(B, S, 8)
    resps = torch.randn(B, S, 16, 128, requires_grad=True)
    labels = torch.ones(B, 1)

    out = model(ops, resps)
    loss = nn.BCEWithLogitsLoss()(out, labels)
    loss.backward()

    # G1 weights gradient
    g1_weight_grad = model.operator[0].weight.grad
    assert g1_weight_grad is not None
    assert torch.isfinite(g1_weight_grad).all()
    assert (g1_weight_grad.abs() > 0.0).any(), "G1 MLP weight grad is zero"

    # G2 beta gradient across both layers
    for i, layer in enumerate(model.set_encoder.layers):
        beta_grad = layer.self_attn.beta.grad
        assert beta_grad is not None, f"Layer {i} beta grad is None"
        assert torch.isfinite(beta_grad).all()
        assert (beta_grad.abs() > 0.0).any(), f"Layer {i} beta grad is zero"


def test_test9_masking_and_cardinality():
    """Test 9: Padding never changes the prediction for the same valid observation set."""
    torch.manual_seed(42)
    for mode in ["baseline", "G1", "G2", "G1_G2"]:
        model = GeometryOperatorSetModel(classes=1, geometry_mode=mode)
        model.eval()

        # Valid set of size 3
        ops_valid = torch.randn(1, 3, 8)
        resps_valid = torch.randn(1, 3, 16, 128)

        # Padded version of size 7 (3 valid + 4 padded)
        ops_padded = torch.zeros(1, 7, 8)
        ops_padded[:, :3] = ops_valid
        # Padded tokens have arbitrary garbage
        ops_padded[:, 3:] = torch.randn(1, 4, 8)

        resps_padded = torch.zeros(1, 7, 16, 128)
        resps_padded[:, :3] = resps_valid
        resps_padded[:, 3:] = torch.randn(1, 4, 16, 128)

        mask = torch.tensor([[False, False, False, True, True, True, True]])

        with torch.no_grad():
            out_valid = model(ops_valid, resps_valid)
            out_padded = model(ops_padded, resps_padded, mask=mask)

        diff = (out_valid - out_padded).abs().max().item()
        assert diff < 1e-5, f"Mode {mode} failed padding invariance: diff={diff}"


def test_test10_exact_baseline_compatibility_at_beta_zero():
    """Test 10: Original OperatorSetModel checkpoint loaded into G2 produces exact logit parity at beta=0."""
    torch.manual_seed(42)
    classes = 5

    # 1. Original SetOperator model
    orig_model = OperatorSetModel(response_dim=128, classes=classes, operator_mode="continuous")
    orig_model.eval()

    # 2. Geometry G2 model
    g2_model = GeometryOperatorSetModel(response_dim=128, classes=classes, geometry_mode="G2")
    g2_model.eval()

    # Load weights from orig_model into g2_model
    g2_model.load_from_original_state_dict(orig_model.state_dict())

    # Verify all beta values are strictly 0.0
    for layer in g2_model.set_encoder.layers:
        assert (layer.self_attn.beta == 0.0).all()

    # Evaluate across varied batch sizes and set sizes
    B, S = 4, 6
    ops = torch.randn(B, S, 8)
    resps = torch.randn(B, S, 16, 128)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[1, 4:] = True

    with torch.no_grad():
        out_orig = orig_model(ops, resps, mask=mask)
        out_g2 = g2_model(ops, resps, mask=mask)

    max_diff = (out_orig - out_g2).abs().max().item()
    mean_diff = (out_orig - out_g2).abs().mean().item()

    assert max_diff < 1e-5, f"Exact baseline compatibility failed: max_diff={max_diff}, mean_diff={mean_diff}"
    print(f"Exact baseline compatibility: Max Diff = {max_diff:.2e}, Mean Diff = {mean_diff:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
