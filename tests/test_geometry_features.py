"""Mathematical Verification Tests for Geometry Features.

Covers:
- Test 1: Operator normalization and rejection of zero, NaN, and inf
- Test 2: Projective identity and sign invariance of R_ij
- Test 3: Operator Gram symmetry, unit diagonal, and [0, 1] bounds
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.geometry_setoperator.geometry_features import (
    vech_sqrt2,
    gamma,
    compute_operator_gram,
)


def test_vech_sqrt2_frobenius_isometry():
    """Verify ||vech_sqrt2(M)||_2 == ||M||_F for symmetric matrices."""
    torch.manual_seed(42)
    # Generate random symmetric 8x8 matrices
    A = torch.randn(10, 8, 8)
    M = A + A.transpose(-2, -1)  # Symmetric

    frob_norm = torch.linalg.norm(M, ord="fro", dim=(-2, -1))
    vech_M = vech_sqrt2(M)
    l2_norm = torch.linalg.norm(vech_M, ord=2, dim=-1)

    diff = (frob_norm - l2_norm).abs().max().item()
    assert diff < 1e-6, f"vech_sqrt2 failed Frobenius isometry: diff={diff}"
    assert vech_M.shape == (10, 36)


def test_test1_operator_normalization_and_rejection():
    """Test 1: Operator normalization and strict rejection of zero, NaN, Inf."""
    torch.manual_seed(42)

    # Valid arbitrary nonzero operators
    q = torch.randn(5, 8) * 3.5
    feat = gamma(q)
    assert feat.shape == (5, 44)

    # First 8 features must be strictly unit norm
    q_norm = torch.norm(feat[:, :8], p=2, dim=-1)
    assert torch.allclose(q_norm, torch.ones_like(q_norm), atol=1e-6)

    # Reject zero vector
    q_zero = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="zero or near-zero norm"):
        gamma(q_zero)

    # Reject NaN
    q_nan = torch.randn(1, 8)
    q_nan[0, 2] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        gamma(q_nan)

    # Reject Inf
    q_inf = torch.randn(1, 8)
    q_inf[0, 5] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinite"):
        gamma(q_inf)


def test_test2_projective_identity_and_sign_flips():
    """Test 2: Projective identity and sign invariance of R_ij."""
    torch.manual_seed(42)
    q = torch.randn(8)
    q_neg = -q

    feat_pos = gamma(q)
    feat_neg = gamma(q_neg)

    # Raw-q part (first 8 dims) must be exactly opposite
    raw_diff = (feat_pos[:8] + feat_neg[:8]).abs().max().item()
    assert raw_diff < 1e-6, f"Raw q part should flip sign: {raw_diff}"

    # Projective part (last 36 dims) must be strictly identical
    proj_diff = (feat_pos[8:] - feat_neg[8:]).abs().max().item()
    assert proj_diff < 1e-6, f"Projective part must be invariant to sign flip: {proj_diff}"

    # Relational alignment R_ij must be invariant to sign flips
    Q = torch.randn(2, 6, 8)
    _, R_orig = compute_operator_gram(Q)

    # Flip sign of lead 2 in batch 0
    Q_flipped = Q.clone()
    Q_flipped[0, 2] = -Q_flipped[0, 2]
    # Flip sign of lead 4 in batch 1
    Q_flipped[1, 4] = -Q_flipped[1, 4]

    _, R_flipped = compute_operator_gram(Q_flipped)

    r_diff = (R_orig - R_flipped).abs().max().item()
    assert r_diff < 1e-6, f"R matrix must be strictly invariant to operator sign flips: {r_diff}"


def test_test3_operator_gram_symmetry_and_bounds():
    """Test 3: Operator Gram symmetry, unit diagonal, and [0, 1] bounds."""
    torch.manual_seed(42)
    batch_size = 4
    set_size = 8
    Q = torch.randn(batch_size, set_size, 8)

    G, R = compute_operator_gram(Q)

    # G must be symmetric
    g_sym_diff = (G - G.transpose(1, 2)).abs().max().item()
    assert g_sym_diff < 1e-6, f"Gram matrix G is not symmetric: diff={g_sym_diff}"

    # R must be symmetric
    r_sym_diff = (R - R.transpose(1, 2)).abs().max().item()
    assert r_sym_diff < 1e-6, f"Relational matrix R is not symmetric: diff={r_sym_diff}"

    # Diagonal entries must be 1.0
    for b in range(batch_size):
        for i in range(set_size):
            assert abs(R[b, i, i].item() - 1.0) < 1e-6, f"Diagonal R[{b},{i},{i}] != 1.0"

    # Bounds: R_ij in [0, 1]
    assert (R >= 0.0).all(), "R has negative values"
    assert (R <= 1.0 + 1e-6).all(), "R has values > 1.0"

    # Orthogonal operators produce R_ij == 0
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Q_orth = torch.stack([e1, e2], dim=0).unsqueeze(0)  # (1, 2, 8)
    _, R_orth = compute_operator_gram(Q_orth)
    assert abs(R_orth[0, 0, 1].item()) < 1e-6, "Orthogonal operators should have R_12 == 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
