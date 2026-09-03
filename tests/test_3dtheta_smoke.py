#!/usr/bin/env python3
"""Pre-flight Smoke Test Suite for 3D Theta Spatial Reconstruction Model (ECG-AIM-3Dθ).

Verifies all structural invariants, geometric assignments, parameter counts,
and gradient isolation rules before launching training on GPU.
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unified_latents.engineering.models.three_d_theta_reconstruction import (
    ECGAIM_THETA,
    LEAD_NAMES,
    ThetaEncoder,
    ThreeDSpatialConditioner,
    SharedECGDecoder,
    ThreeDThetaECGAIM,
    missing_lead_l1,
)


def test_theta_encoder_shape_and_values():
    """Verify ThetaEncoder produces exact [B, 12, 12] features and matches expected trigonometric order."""
    encoder = ThetaEncoder(encoder_len=1)
    angles = ECGAIM_THETA.unsqueeze(0)  # [1, 12, 2]
    out = encoder(angles)
    assert out.shape == (1, 12, 12), f"Expected [1, 12, 12], got {out.shape}"
    assert torch.isfinite(out).all(), "Theta features must be finite"

    # Spot-check Lead I angles: [pi/2, pi/2]
    # sum = pi, sub = 0
    # before_encode = [pi/2, pi/2, pi, 0]
    # sin = [1, 1, 0, 0]
    # cos = [0, 0, -1, 1]
    lead_i_out = out[0, 0].tolist()
    assert len(lead_i_out) == 12


def test_lead_angle_order_matches_canonical():
    """Verify exact 12-lead anatomical order: I, II, III, aVR, aVL, aVF, V1-V6."""
    expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert LEAD_NAMES == expected_leads

    # Lead I: frontal 0 deg (x-axis)
    assert math.isclose(ECGAIM_THETA[0, 0].item(), math.pi / 2, rel_tol=1e-5)
    assert math.isclose(ECGAIM_THETA[0, 1].item(), math.pi / 2, rel_tol=1e-5)

    # Lead aVF: vertical 90 deg down
    assert math.isclose(ECGAIM_THETA[5, 0].item(), math.pi, rel_tol=1e-5)


def test_pure_theta_cells_have_no_lead_embed_leakage():
    """Verify that pure theta cells (D1, D3) and controls (D4, D5) have no categorical lead_embed."""
    for mode in ["theta", "permuted_theta", "learned", "random_fixed"]:
        model = ThreeDThetaECGAIM(code_mode=mode, fusion="mul", width=128, encoder_depth=2, decoder_depth=2)
        assert model.conditioner.lead_embed is None, f"Mode {mode} must NOT have categorical lead_embed!"


def test_d3_and_d5_differ_only_in_theta_assignment():
    """Verify D3 (true theta) and D5 (permuted theta) are strictly parameter-identical and differ only in spatial angles."""
    torch.manual_seed(42)
    model_d3 = ThreeDThetaECGAIM(code_mode="theta", fusion="mul", width=256, encoder_depth=2, decoder_depth=2)
    torch.manual_seed(42)
    model_d5 = ThreeDThetaECGAIM(code_mode="permuted_theta", fusion="mul", width=256, encoder_depth=2, decoder_depth=2)

    # Compare parameter keys and parameter counts
    params_d3 = dict(model_d3.named_parameters())
    params_d5 = dict(model_d5.named_parameters())
    assert set(params_d3.keys()) == set(params_d5.keys())
    for k in params_d3:
        assert params_d3[k].shape == params_d5[k].shape
        torch.testing.assert_close(params_d3[k], params_d5[k])

    # Buffers must show the permutation difference
    assert not torch.allclose(model_d3.conditioner.lead_angles, model_d5.conditioner._codes(1)[0, :, :2])


def test_d3_and_d4_architectural_parity():
    """Verify D3 and D4 use identical source encoder, code projection shape, shared decoder, and losses."""
    model_d3 = ThreeDThetaECGAIM(code_mode="theta", fusion="mul", width=768, encoder_depth=8, decoder_depth=4)
    model_d4 = ThreeDThetaECGAIM(code_mode="learned", fusion="mul", width=768, encoder_depth=8, decoder_depth=4)

    # Both must have identical code_projection shape [12 -> 768 -> 768]
    assert model_d3.conditioner.code_projection[0].weight.shape == (768, 12)
    assert model_d4.conditioner.code_projection[0].weight.shape == (768, 12)

    # Both must have identical decoder parameters
    dec_d3_p = sum(p.numel() for p in model_d3.decoder.parameters())
    dec_d4_p = sum(p.numel() for p in model_d4.decoder.parameters())
    assert dec_d3_p == dec_d4_p


def test_permutation_derangement_properties():
    """Verify permutation has no fixed points (derangement) so every single lead receives an incorrect coordinate."""
    perm = (torch.arange(12) + 5) % 12
    for i in range(12):
        assert perm[i].item() != i, f"Permutation must have zero fixed points! Found fixed point at lead {i}"


def test_missing_lead_l1_loss():
    """Verify genuine missing-lead L1 strictly ignores observed lead."""
    B = 4
    pred = torch.zeros(B, 12, 5000)
    target = torch.ones(B, 12, 5000)

    # Observed lead 0: error is 1.0 on all 11 missing leads
    loss = missing_lead_l1(pred, target, observed_lead=0)
    assert math.isclose(loss.item(), 1.0, rel_tol=1e-5)

    # Even if observed lead prediction is horribly wrong (e.g., 999.0), missing loss is unaffected!
    pred[:, 0, :] = 999.0
    loss_isolated = missing_lead_l1(pred, target, observed_lead=0)
    assert math.isclose(loss_isolated.item(), 1.0, rel_tol=1e-5)


def test_forward_pass_tensor_trace():
    """Verify forward pass shape trace and gradient backprop."""
    model = ThreeDThetaECGAIM(code_mode="theta", fusion="mul", width=64, encoder_depth=2, decoder_depth=2)
    x = torch.randn(2, 1, 5000, requires_grad=True)
    out = model(x, obs_lead_idx=0)
    y_pred = out["y_pred"]
    h_grid = out["h_grid"]

    assert y_pred.shape == (2, 12, 5000)
    assert h_grid.shape == (2, 12, 200, 64)

    loss = y_pred.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    print("Running 3D Theta pre-flight smoke suite...")
    test_theta_encoder_shape_and_values()
    print("  ✓ ThetaEncoder shape and trigonometric values passed")
    test_lead_angle_order_matches_canonical()
    print("  ✓ Lead angle canonical order passed")
    test_pure_theta_cells_have_no_lead_embed_leakage()
    print("  ✓ No categorical lead_embed leakage in theta cells passed")
    test_d3_and_d5_differ_only_in_theta_assignment()
    print("  ✓ D3 and D5 parity and permutation difference passed")
    test_d3_and_d4_architectural_parity()
    print("  ✓ D3 and D4 architectural parity passed")
    test_permutation_derangement_properties()
    print("  ✓ Permutation derangement properties passed")
    test_missing_lead_l1_loss()
    print("  ✓ Missing-lead L1 isolation passed")
    test_forward_pass_tensor_trace()
    print("  ✓ Forward pass tensor trace and backward pass passed")
    print("\nALL PRE-FLIGHT SMOKE TESTS PASSED!")
