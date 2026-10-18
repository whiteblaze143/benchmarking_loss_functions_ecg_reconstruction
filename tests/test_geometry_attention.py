"""Mathematical Verification Tests for Geometry-Aware Attention.

Covers:
- Exact numerical baseline compatibility at beta = 0.0 with PyTorch nn.TransformerEncoderLayer
- Gradient flow to beta parameters (Test 8)
- Variable set size and key padding mask behavior
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.geometry_setoperator.geometry_attention import (
    GeometryMultiheadAttention,
    GeometryTransformerEncoderLayer,
    GeometryTransformerEncoder,
)
from repecg.geometry_setoperator.geometry_features import compute_operator_gram


def test_geometry_attention_baseline_numerical_parity_beta_zero():
    """Verify that GeometryTransformerEncoderLayer matches standard PyTorch layer exactly at beta=0."""
    torch.manual_seed(42)
    d_model = 192
    nhead = 4
    dim_ff = 384

    # 1. Standard PyTorch layer
    std_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, activation="gelu", batch_first=True
    )
    std_layer.eval()

    # 2. Geometry layer
    geom_layer = GeometryTransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, activation="gelu"
    )
    geom_layer.eval()

    # Copy weights from std_layer into geom_layer
    geom_layer.self_attn.in_proj_weight.data.copy_(std_layer.self_attn.in_proj_weight.data)
    geom_layer.self_attn.in_proj_bias.data.copy_(std_layer.self_attn.in_proj_bias.data)
    geom_layer.self_attn.out_proj.weight.data.copy_(std_layer.self_attn.out_proj.weight.data)
    geom_layer.self_attn.out_proj.bias.data.copy_(std_layer.self_attn.out_proj.bias.data)

    geom_layer.linear1.weight.data.copy_(std_layer.linear1.weight.data)
    geom_layer.linear1.bias.data.copy_(std_layer.linear1.bias.data)
    geom_layer.linear2.weight.data.copy_(std_layer.linear2.weight.data)
    geom_layer.linear2.bias.data.copy_(std_layer.linear2.bias.data)

    geom_layer.norm1.weight.data.copy_(std_layer.norm1.weight.data)
    geom_layer.norm1.bias.data.copy_(std_layer.norm1.bias.data)
    geom_layer.norm2.weight.data.copy_(std_layer.norm2.weight.data)
    geom_layer.norm2.bias.data.copy_(std_layer.norm2.bias.data)

    # beta is strictly 0.0
    assert (geom_layer.self_attn.beta == 0.0).all()

    # Test forward with random inputs and masks
    B, S = 4, 7
    x = torch.randn(B, S, d_model)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[0, 5:] = True
    mask[2, 4:] = True

    # Compute arbitrary non-zero R matrix
    Q = torch.randn(B, S, 8)
    _, R = compute_operator_gram(Q, mask=mask)

    with torch.no_grad():
        out_std = std_layer(x, src_key_padding_mask=mask)
        out_geom = geom_layer(x, src_key_padding_mask=mask, R=R)

    max_diff = (out_std - out_geom).abs().max().item()
    mean_diff = (out_std - out_geom).abs().mean().item()

    assert max_diff < 1e-6, f"Parity failure at beta=0: max_diff={max_diff}, mean_diff={mean_diff}"


def test_test8_gradient_flow_to_beta():
    """Test 8: Gradient flow to beta parameters initialized to 0.0."""
    torch.manual_seed(42)
    d_model = 192
    nhead = 4

    geom_layer = GeometryTransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=384, activation="gelu"
    )
    geom_layer.train()

    B, S = 2, 5
    x = torch.randn(B, S, d_model, requires_grad=True)
    Q = torch.randn(B, S, 8)
    _, R = compute_operator_gram(Q)

    # Forward
    out = geom_layer(x, R=R)
    loss = out.sum()
    loss.backward()

    # Check beta gradients
    beta_grad = geom_layer.self_attn.beta.grad
    assert beta_grad is not None, "beta gradient is None"
    assert torch.isfinite(beta_grad).all(), "beta gradient contains NaN/inf"
    assert (beta_grad.abs() > 0.0).any(), "beta gradient is all zeros"


def test_beta_sensitivity_when_nonzero():
    """Verify that when beta != 0, attention actively incorporates R_ij."""
    torch.manual_seed(42)
    d_model = 192
    nhead = 4

    geom_layer = GeometryTransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=384, activation="gelu"
    )
    geom_layer.eval()

    B, S = 2, 4
    x = torch.randn(B, S, d_model)
    Q = torch.randn(B, S, 8)
    _, R = compute_operator_gram(Q)

    with torch.no_grad():
        out_zero = geom_layer(x, R=R)

        # Set beta to non-zero values
        geom_layer.self_attn.beta.data.fill_(1.5)
        out_nonzero = geom_layer(x, R=R)

    diff = (out_nonzero - out_zero).abs().max().item()
    assert diff > 1e-3, f"Non-zero beta should change output: diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
