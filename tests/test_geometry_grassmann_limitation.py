"""Formal Mathematical Tests of the Orthonormal Grassmann Limitation of Relational Attention (G2).

Propositions tested:
1. Orthonormal Grassmann Collapse:
   For any orthonormal operator set Q in R^{k x 8} (Q Q^T = I_k), the relational redundancy
   matrix R = (Q Q^T) * (Q Q^T) is identically the identity matrix I_k, regardless of the subspace.
2. Subspace Blindness:
   For two completely distinct Grassmann subspaces S_1, S_2 in Gr(k, 8) represented by orthonormal
   frames Q_1, Q_2, R_1 = R_2 = I_k. The relational bias cannot distinguish orthonormal subspaces.
3. Rank-1 Inefficacy of G2:
   For k=1, R = [1.0], and adding beta to the scalar attention logit has zero effect on the softmax:
   softmax([s_11 + beta]) = [1.0] for all beta in R. G2 is mathematically identical to baseline at rank 1.
4. Functional Equivalence to Diagonal Self-Attention Bias:
   Under orthonormal frames, G2 is functionally and numerically identical to adding a learned
   scalar beta to self-attention logits relative to cross-attention logits (control_diagonal_bias).
5. Non-orthonormal Discrepancy:
   When operators are non-orthogonal (e.g. redundant limb leads), G2's off-diagonal relational terms
   R_ij = (q_i^T q_j)^2 != 0 become active, producing a distinct output from the diagonal control.
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

from repecg.geometry_setoperator.geometry_features import compute_operator_gram
from repecg.geometry_setoperator.geometry_setoperator import GeometryOperatorSetModel


def test_proposition1_orthonormal_grassmann_collapse():
    """Proposition 1: Q Q^T = I_k implies R = I_k for all ranks k in 1..8."""
    torch.manual_seed(42)
    for k in range(1, 9):
        # Generate random orthonormal frame in R^{k x 8} via QR decomposition
        A = torch.randn(8, k)
        Q_k, _ = torch.linalg.qr(A)  # (8, k)
        Q = Q_k.T.unsqueeze(0)  # (1, k, 8)

        # Q is orthonormal: Q Q^T = I_k
        gram = Q @ Q.transpose(-2, -1)
        I_k = torch.eye(k).unsqueeze(0)
        assert torch.allclose(gram, I_k, atol=1e-6), f"QR frame not orthonormal at rank {k}"

        # Relational alignment: R = Gram * Gram
        _, R = compute_operator_gram(Q)
        diff = (R - I_k).abs().max().item()
        assert diff < 1e-6, f"Proposition 1 failed at rank {k}: max diff={diff}"
        assert torch.allclose(R, I_k, atol=1e-6)


def test_proposition2_subspace_blindness_on_grassmannian():
    """Proposition 2: Different orthonormal subspaces produce identical R = I_k."""
    torch.manual_seed(123)
    k = 4
    # Subspace 1: canonical first 4 leads
    Q1 = torch.eye(8)[:k].unsqueeze(0)  # (1, 4, 8)

    # Subspace 2: random 4-D subspace
    A2 = torch.randn(8, k)
    Q2_k, _ = torch.linalg.qr(A2)
    Q2 = Q2_k.T.unsqueeze(0)  # (1, 4, 8)

    # Subspace 3: another distinct 4-D subspace
    A3 = torch.randn(8, k)
    Q3_k, _ = torch.linalg.qr(A3)
    Q3 = Q3_k.T.unsqueeze(0)  # (1, 4, 8)

    _, R1 = compute_operator_gram(Q1)
    _, R2 = compute_operator_gram(Q2)
    _, R3 = compute_operator_gram(Q3)

    assert torch.allclose(R1, R2, atol=1e-6)
    assert torch.allclose(R2, R3, atol=1e-6)
    assert torch.allclose(R1, torch.eye(k).unsqueeze(0), atol=1e-6)


def test_proposition3_rank1_inefficacy_of_g2():
    """Proposition 3: At rank 1, G2 predictions are completely invariant to beta."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G2")
    model.eval()

    q_unit = torch.randn(1, 1, 8)
    q_unit = q_unit / torch.norm(q_unit, dim=-1, keepdim=True)
    resp = torch.randn(1, 1, 16, 128)

    # Record prediction with beta = 0.0
    for layer in model.set_encoder.layers:
        layer.self_attn.beta.data.zero_()
    with torch.no_grad():
        out_beta_0 = model(q_unit, resp)

    # Evaluate across a wide sweep of beta values
    for test_beta in [-10.0, -2.5, -1.0, 0.5, 1.0, 5.0, 10.0, 100.0]:
        for layer in model.set_encoder.layers:
            layer.self_attn.beta.data.fill_(test_beta)
        with torch.no_grad():
            out_beta_k = model(q_unit, resp)

        diff = (out_beta_0 - out_beta_k).abs().max().item()
        assert diff < 1e-6, f"Rank-1 prediction changed with beta={test_beta}: diff={diff}"


def test_proposition4_equivalence_to_diagonal_bias_control_under_orthonormal():
    """Proposition 4: Under orthonormal frames, G2 == control_diagonal_bias for any beta."""
    torch.manual_seed(999)
    classes = 5

    # Instantiate G2 and control_diagonal_bias
    model_g2 = GeometryOperatorSetModel(classes=classes, geometry_mode="G2")
    model_diag = GeometryOperatorSetModel(classes=classes, geometry_mode="control_diagonal_bias")
    model_g2.eval()
    model_diag.eval()

    # Share all weights between the two models
    model_diag.load_state_dict(model_g2.state_dict())

    # Set non-zero beta in both models
    for layer_g2, layer_diag in zip(model_g2.set_encoder.layers, model_diag.set_encoder.layers):
        test_beta = torch.tensor([0.25, -0.4, 0.8, -0.1])
        layer_g2.self_attn.beta.data.copy_(test_beta)
        layer_diag.self_attn.beta.data.copy_(test_beta)

    # Evaluate on random orthonormal frames across ranks 2..6
    for k in [2, 3, 5, 6]:
        A = torch.randn(8, k)
        Q_k, _ = torch.linalg.qr(A)
        Q = Q_k.T.unsqueeze(0)  # (1, k, 8)
        resps = torch.randn(1, k, 16, 128)

        with torch.no_grad():
            out_g2 = model_g2(Q, resps)
            out_diag = model_diag(Q, resps)

        diff = (out_g2 - out_diag).abs().max().item()
        assert diff < 1e-6, f"G2 and control_diagonal_bias diverged under orthonormal frame (k={k}): diff={diff}"


def test_proposition5_g2_distinct_on_nonorthogonal_sets():
    """Proposition 5: G2 differs from diagonal control when operators are non-orthogonal."""
    torch.manual_seed(999)
    classes = 1

    model_g2 = GeometryOperatorSetModel(classes=classes, geometry_mode="G2")
    model_diag = GeometryOperatorSetModel(classes=classes, geometry_mode="control_diagonal_bias")
    model_g2.eval()
    model_diag.eval()
    model_diag.load_state_dict(model_g2.state_dict())

    # Set non-zero beta
    for layer_g2, layer_diag in zip(model_g2.set_encoder.layers, model_diag.set_encoder.layers):
        layer_g2.self_attn.beta.data.fill_(1.0)
        layer_diag.self_attn.beta.data.fill_(1.0)

    # Non-orthogonal operator set (e.g. Lead I, Lead II, Lead III = II - I)
    e_I = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_II = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_III = e_II - e_I
    Q_limb = torch.stack([e_I, e_II, e_III], dim=0).unsqueeze(0)  # (1, 3, 8)
    resps = torch.randn(1, 3, 16, 128)

    with torch.no_grad():
        out_g2 = model_g2(Q_limb, resps)
        out_diag = model_diag(Q_limb, resps)

    diff = (out_g2 - out_diag).abs().item()
    # On non-orthogonal sets, G2 has non-zero off-diagonal R_ij terms, so out_g2 != out_diag
    assert diff > 1e-4, f"G2 and diagonal control did not diverge on non-orthogonal set: diff={diff}"
    print(f"G2 vs Diagonal Control divergence on non-orthogonal limb leads: {diff:.4f}")


def test_proposition6_higher_order_redundancy_limitation():
    """Proposition 6: Squared Gram features cannot distinguish higher-order redundancy or rank.
    
    Construct two valid 3-operator Gram matrices:
    G_A with pairwise correlations -1/2 (coplanar, 120-degree unit vectors, rank 2, linearly dependent)
    G_B with pairwise correlations +1/2 (tetrahedral/spherical triangle unit vectors, rank 3, linearly independent)
    
    Both satisfy:
    G_A * G_A = G_B * G_B = [[1, 1/4, 1/4], [1/4, 1, 1/4], [1/4, 1/4, 1]]
    
    Thus, R_A == R_B despite rank(G_A) = 2 and rank(G_B) = 3.
    """
    G_A = torch.tensor([
        [1.0, -0.5, -0.5],
        [-0.5, 1.0, -0.5],
        [-0.5, -0.5, 1.0]
    ])
    G_B = torch.tensor([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])

    # 1. Verify PSD
    eig_A = torch.linalg.eigvalsh(G_A)
    eig_B = torch.linalg.eigvalsh(G_B)
    assert (eig_A >= -1e-6).all(), "G_A is not positive semi-definite"
    assert (eig_B >= -1e-6).all(), "G_B is not positive semi-definite"

    # 2. Verify rank difference: rank(G_A) = 2, rank(G_B) = 3
    rank_A = torch.linalg.matrix_rank(G_A).item()
    rank_B = torch.linalg.matrix_rank(G_B).item()
    assert rank_A == 2, f"Expected rank(G_A)=2, got {rank_A}"
    assert rank_B == 3, f"Expected rank(G_B)=3, got {rank_B}"

    # 3. Compute relational redundancy matrices R = G * G
    R_A = G_A.square()
    R_B = G_B.square()
    assert torch.allclose(R_A, R_B, atol=1e-7)

    expected_R = torch.tensor([
        [1.0, 0.25, 0.25],
        [0.25, 1.0, 0.25],
        [0.25, 0.25, 1.0]
    ])
    assert torch.allclose(R_A, expected_R, atol=1e-7)

    # 4. Construct unit vectors in R^8 whose Gram matrices are G_A and G_B
    # For G_A: 3 coplanar vectors at 120 degrees in the (e1, e2) plane
    qA_1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qA_2 = torch.tensor([-0.5, math.sqrt(3)/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qA_3 = torch.tensor([-0.5, -math.sqrt(3)/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Q_A = torch.stack([qA_1, qA_2, qA_3], dim=0)
    assert torch.allclose(Q_A @ Q_A.T, G_A, atol=1e-5)

    # For G_B: 3 vectors in (e1, e2, e3) with mutual dot products +0.5
    # qB_1 = [1, 0, 0]
    # qB_2 = [1/2, sqrt(3)/2, 0]
    # qB_3 = [1/2, 1/(2*sqrt(3)), sqrt(2/3)]
    qB_1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qB_2 = torch.tensor([0.5, math.sqrt(3)/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qB_3 = torch.tensor([0.5, 1.0/(2.0*math.sqrt(3)), math.sqrt(2.0/3.0), 0.0, 0.0, 0.0, 0.0, 0.0])
    Q_B = torch.stack([qB_1, qB_2, qB_3], dim=0)
    assert torch.allclose(Q_B @ Q_B.T, G_B, atol=1e-5)

    # 5. Verify that compute_operator_gram yields bitwise identical R
    _, R_model_A = compute_operator_gram(Q_A.unsqueeze(0))
    _, R_model_B = compute_operator_gram(Q_B.unsqueeze(0))
    diff_R = (R_model_A - R_model_B).abs().max().item()
    assert diff_R < 1e-6, f"R matrices differed in model extractor: diff={diff_R}"
    print(f"Proposition 6 verified: G2 receives identical R matrix despite rank 2 vs 3: max diff={diff_R:.2e}")

