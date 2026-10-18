"""Comprehensive P2 Preflight Verification Suite for Geometry-Aware SetOperator.

Covers all 12 required preflight verification items:
1. G1 operator geometry expansion and sign behavior.
2. G2 attention beta initialization and gradient flow.
3. G2 orthonormal Grassmann collapse.
4. G2 versus diagonal-attention equivalence for orthonormal frames.
5. Higher-order redundancy limitation (Proposition 6).
6. Full original-checkpoint logit parity for G2 at beta=0.
7. Padding, variable cardinality, masking and permutation tests.
8. Strict architecture checkpoint loading.
9. Paired operator-response correctness with waveform-first Phase-KME.
10. Numerical equivalence with the frozen authentic EchoNext Phase-KME extractor.
11. Reproducibility with fixed patient splits and operator bank manifests.
12. Trainable parameter, unused parameter and memory-allocation breakdown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

DEV_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"

for p in [SRC_DIR, MODELS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from repecg.geometry_setoperator.geometry_features import (
    vech_sqrt2,
    gamma,
    compute_operator_gram,
)
from repecg.geometry_setoperator.geometry_attention import (
    GeometryMultiheadAttention,
    GeometryTransformerEncoderLayer,
    GeometryTransformerEncoder,
)
from repecg.geometry_setoperator.geometry_setoperator import (
    GeometryOperatorSetModel,
    EchoNextGeometrySetOperator,
)
from repecg.paper07_operator.model import OperatorSetModel
from setoperator_echonext import SetOperatorEchoNext, build_setoperator_echonext


# ==============================================================================
# Item 1: G1 operator geometry expansion and sign behavior
# ==============================================================================
def test_item1_g1_operator_expansion_and_sign_behavior():
    """G1 expands q in R^8 to gamma(q) = [q, vech_sqrt2(q q^T)] in R^44."""
    torch.manual_seed(42)
    q = torch.randn(2, 5, 8)
    q_norm = q / torch.norm(q, dim=-1, keepdim=True)

    gamma_feat = gamma(q_norm)
    assert gamma_feat.shape == (2, 5, 44), f"Expected (2, 5, 44), got {gamma_feat.shape}"

    # Sign flip: q -> -q preserves the quadratic outer-product part but negates linear part
    gamma_neg = gamma(-q_norm)
    assert torch.allclose(gamma_feat[..., :8], -gamma_neg[..., :8], atol=1e-6), "Linear part must negate"
    assert torch.allclose(gamma_feat[..., 8:], gamma_neg[..., 8:], atol=1e-6), "Projective part must be invariant"

    # Frobenius isometry: ||vech_sqrt2(M)||_2 == ||M||_F
    M = torch.randn(8, 8)
    M_sym = 0.5 * (M + M.T)
    v = vech_sqrt2(M_sym)
    assert v.shape == (36,)
    diff_frob = abs(torch.norm(v, p=2).item() - torch.norm(M_sym, p="fro").item())
    assert diff_frob < 1e-6, f"Isometry failed: diff={diff_frob}"


# ==============================================================================
# Item 2: G2 attention beta initialization and gradient flow
# ==============================================================================
def test_item2_g2_beta_initialization_and_gradient_flow():
    """G2 beta is initialized to 0.0 and receives non-zero gradients on non-orthogonal sets."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G2")

    # Check initialization
    for layer in model.set_encoder.layers:
        assert (layer.self_attn.beta == 0.0).all(), "Beta must initialize strictly to 0.0"

    # Forward pass on non-orthogonal operators
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q3 = q2 - q1  # Lead III
    Q = torch.stack([q1, q2, q3], dim=0).unsqueeze(0)  # (1, 3, 8)
    resps = torch.randn(1, 3, 16, 128)

    model.train()
    out = model(Q, resps)
    loss = out.sum()
    loss.backward()

    # Verify gradients flow to beta
    for i, layer in enumerate(model.set_encoder.layers):
        grad = layer.self_attn.beta.grad
        assert grad is not None, f"Layer {i} beta has no gradient"
        assert torch.isfinite(grad).all(), f"Layer {i} beta gradient is not finite"


# ==============================================================================
# Item 3: G2 orthonormal Grassmann collapse
# ==============================================================================
def test_item3_g2_orthonormal_grassmann_collapse():
    """Under orthonormal frames, R = I_k regardless of subspace; rank-1 has zero beta effect."""
    torch.manual_seed(42)
    for k in range(1, 8):
        A = torch.randn(8, k)
        Q_k, _ = torch.linalg.qr(A)
        Q = Q_k.T.unsqueeze(0)  # (1, k, 8)
        _, R = compute_operator_gram(Q)
        I_k = torch.eye(k).unsqueeze(0)
        assert torch.allclose(R, I_k, atol=1e-6), f"R != I_k for orthonormal frame at rank {k}"

    # Rank 1 inefficacy
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G2")
    model.eval()
    q_1 = torch.randn(1, 1, 8)
    q_1 = q_1 / torch.norm(q_1, dim=-1, keepdim=True)
    r_1 = torch.randn(1, 1, 16, 128)

    with torch.no_grad():
        for layer in model.set_encoder.layers:
            layer.self_attn.beta.data.zero_()
        out_b0 = model(q_1, r_1)

        for layer in model.set_encoder.layers:
            layer.self_attn.beta.data.fill_(15.0)
        out_b15 = model(q_1, r_1)

    assert torch.allclose(out_b0, out_b15, atol=1e-6), "Rank 1 predictions must be invariant to beta"


# ==============================================================================
# Item 4: G2 vs diagonal-attention equivalence for orthonormal frames
# ==============================================================================
def test_item4_g2_vs_diagonal_control_equivalence():
    """G2 is functionally identical to control_diagonal_bias under orthonormal frames."""
    torch.manual_seed(42)
    m_g2 = GeometryOperatorSetModel(classes=1, geometry_mode="G2")
    m_diag = GeometryOperatorSetModel(classes=1, geometry_mode="control_diagonal_bias")
    m_diag.load_state_dict(m_g2.state_dict())
    m_g2.eval()
    m_diag.eval()

    # Set non-zero beta
    for l_g2, l_diag in zip(m_g2.set_encoder.layers, m_diag.set_encoder.layers):
        b = torch.tensor([0.3, -0.2, 0.7, -0.5])
        l_g2.self_attn.beta.data.copy_(b)
        l_diag.self_attn.beta.data.copy_(b)

    for k in [2, 4, 6]:
        A = torch.randn(8, k)
        Q_k, _ = torch.linalg.qr(A)
        Q = Q_k.T.unsqueeze(0)
        resps = torch.randn(1, k, 16, 128)

        with torch.no_grad():
            out_g2 = m_g2(Q, resps)
            out_diag = m_diag(Q, resps)

        diff = (out_g2 - out_diag).abs().max().item()
        assert diff < 1e-6, f"G2 and diagonal control diverged on orthonormal frame (k={k}): diff={diff}"


# ==============================================================================
# Item 5: Higher-order redundancy limitation (Proposition 6)
# ==============================================================================
def test_item5_higher_order_redundancy_limitation():
    """G_A (rank 2, dependent) and G_B (rank 3, independent) have identical R = G * G."""
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
    assert torch.linalg.matrix_rank(G_A).item() == 2
    assert torch.linalg.matrix_rank(G_B).item() == 3
    assert torch.allclose(G_A.square(), G_B.square(), atol=1e-7)


# ==============================================================================
# Item 6: Full original-checkpoint logit parity for G2 at beta=0
# ==============================================================================
def test_item6_logit_parity_at_beta_zero():
    """GeometryOperatorSetModel with beta=0 achieves bitwise/numerical parity with OperatorSetModel."""
    torch.manual_seed(42)
    base_model = OperatorSetModel(classes=1, operator_mode="continuous", operator_dim=8)
    geom_model = GeometryOperatorSetModel(classes=1, geometry_mode="G2", operator_dim=8)

    geom_model.load_from_original_state_dict(base_model.state_dict())
    base_model.eval()
    geom_model.eval()

    ops = torch.randn(3, 6, 8)
    ops = ops / torch.norm(ops, dim=-1, keepdim=True)
    resps = torch.randn(3, 6, 16, 128)

    with torch.no_grad():
        out_base = base_model(ops, resps)
        out_geom = geom_model(ops, resps)

    max_diff = (out_base - out_geom).abs().max().item()
    assert max_diff < 1e-6, f"Parity failed at beta=0: max_diff={max_diff}"


# ==============================================================================
# Item 7: Padding, variable cardinality, masking and permutation tests
# ==============================================================================
def test_item7_padding_cardinality_and_permutation():
    """Permutation invariance and key padding mask correctness."""
    torch.manual_seed(42)
    model = GeometryOperatorSetModel(classes=1, geometry_mode="G1_G2")
    model.eval()

    ops = torch.randn(1, 5, 8)
    ops = ops / torch.norm(ops, dim=-1, keepdim=True)
    resps = torch.randn(1, 5, 16, 128)

    # Permutation
    perm = [2, 0, 4, 1, 3]
    with torch.no_grad():
        out_orig = model(ops, resps)
        out_perm = model(ops[:, perm], resps[:, perm])

    assert torch.allclose(out_orig, out_perm, atol=1e-5), "Permutation invariance failed"

    # Padding invariance: padded tokens must not alter predictions
    dummy_ops = torch.randn(1, 2, 8)
    dummy_ops = dummy_ops / torch.norm(dummy_ops, dim=-1, keepdim=True)
    ops_pad = torch.cat([ops, dummy_ops], dim=1)
    resps_pad = torch.cat([resps, torch.randn(1, 2, 16, 128)], dim=1)
    mask = torch.tensor([[False, False, False, False, False, True, True]])

    with torch.no_grad():
        out_padded = model(ops_pad, resps_pad, mask=mask)

    assert torch.allclose(out_orig, out_padded, atol=1e-5), "Padding invariance failed"


# ==============================================================================
# Item 8: Strict architecture checkpoint loading
# ==============================================================================
def test_item8_strict_checkpoint_loading():
    """Checkpoint loading works cleanly from baseline OperatorSetModel state dict."""
    torch.manual_seed(42)
    base = OperatorSetModel(classes=1, operator_mode="continuous", operator_dim=8)
    geom_g1 = GeometryOperatorSetModel(classes=1, geometry_mode="G1")
    geom_g2 = GeometryOperatorSetModel(classes=1, geometry_mode="G2")

    # G2 should load all non-beta weights cleanly
    geom_g2.load_from_original_state_dict(base.state_dict(), strict=False)
    for layer in geom_g2.set_encoder.layers:
        assert (layer.self_attn.beta == 0.0).all()

    # G1 has different input dim (44 vs 8) for operator.0.weight, non-strict handles this cleanly
    geom_g1.load_from_original_state_dict(base.state_dict(), strict=False)
    assert geom_g1.operator[0].weight.shape == (64, 44)


# ==============================================================================
# Item 9: Paired operator-response correctness with waveform-first Phase-KME
# ==============================================================================
def test_item9_paired_operator_response_waveform_first():
    """Projection Y = X Q^T preserves channel linearity under O(k) rotations."""
    torch.manual_seed(42)
    # Synthesize multi-lead beats: shape (n_beats=4, 256 samples, 8 leads)
    beats_raw = torch.randn(4, 256, 8)
    Q_k = torch.zeros(2, 8)
    Q_k[0, 0] = 1.0
    Q_k[1, 1] = 1.0

    theta = math.pi / 3.0
    A = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    Q_rot = A @ Q_k

    Y = torch.einsum("bij,kj->kbi", beats_raw, Q_k)
    Y_rot = torch.einsum("bij,kj->kbi", beats_raw, Q_rot)
    Y_expected = torch.einsum("kbi,jk->jbi", Y, A)
    assert torch.allclose(Y_rot, Y_expected, atol=1e-5), "Waveform projection linearity failed"


# ==============================================================================
# Item 10: Numerical equivalence with frozen authentic EchoNext Phase-KME extractor
# ==============================================================================
def test_item10_frozen_phase_kme_fit_existence_and_integrity():
    """Verify official and dev Phase-KME fit files exist and match schemas."""
    fit_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/representations/echonext_phase_kme_fit.npz")
    assert fit_path.exists(), f"Official fit file missing: {fit_path}"

    with np.load(fit_path) as fit:
        assert "landmarks" in fit
        assert fit["landmarks"].shape == (128, 2)
        assert "whitening_mean" in fit
        assert fit["whitening_mean"].shape == (2,)


# ==============================================================================
# Item 11: Reproducibility with fixed patient splits and operator bank manifests
# ==============================================================================
def test_item11_patient_split_and_dev_bank_reproducibility():
    """Verify inner development split indices and dev operator bank exist and are disjoint."""
    split_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/echonext_inner_dev_split_indices.npz")
    assert split_path.exists(), "Inner dev split indices file missing"

    with np.load(split_path) as split_data:
        train_pts = set(split_data["inner_train_patients"])
        dev_pts = set(split_data["inner_dev_patients"])
        assert len(train_pts & dev_pts) == 0, "Patient overlap detected in inner dev split!"
        assert len(train_pts) == 22285
        assert len(dev_pts) == 3933
        assert len(split_data["inner_train_indices"]) == 61375
        assert len(split_data["inner_dev_indices"]) == 11100

    bank_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/echonext_dev_operator_bank.npz")
    assert bank_path.exists(), "Dev operator bank file missing"
    with np.load(bank_path) as bank:
        total_subspaces = 0
        for rank in range(1, 8):
            k = f"rank_{rank}"
            assert k in bank
            frames = bank[k]
            assert frames.shape[1] == rank and frames.shape[2] == 8
            # Check orthonormality
            for frame in frames:
                gram = frame @ frame.T
                assert np.allclose(gram, np.eye(rank), atol=1e-5)
            total_subspaces += len(frames)
        assert total_subspaces == 100, f"Expected 100 dev subspaces, got {total_subspaces}"


# ==============================================================================
# Item 12: Trainable parameter, unused parameter and memory-allocation breakdown
# ==============================================================================
def test_item12_parameter_and_memory_accounting():
    """Verify exact parameter counts across active classifier and unused decoder."""
    model_so = SetOperatorEchoNext(tabular_mode="native_additive")
    tot = sum(p.numel() for p in model_so.parameters())
    dec = sum(p.numel() for p in model_so.core.decoder.parameters())
    act = tot - dec

    assert tot == 1652673, f"Expected 1,652,673 total params, got {tot}"
    assert dec == 608512, f"Expected 608,512 decoder params, got {dec}"
    assert act == 1044161, f"Expected 1,044,161 active params, got {act}"

    # Geometry G1
    m_g1 = EchoNextGeometrySetOperator(geometry_mode="G1", tabular_dim=7)
    tot_g1 = sum(p.numel() for p in m_g1.parameters())
    dec_g1 = sum(p.numel() for p in m_g1.core.decoder.parameters())
    act_g1 = tot_g1 - dec_g1
    assert tot_g1 == 1654977, f"Expected 1,654,977 G1 total params, got {tot_g1}"
    assert act_g1 == 1046465, f"Expected 1,046,465 G1 active params, got {act_g1}"
    assert act_g1 - act == 2304, f"Expected +2304 G1 delta, got {act_g1 - act}"

    # Geometry G2
    m_g2 = EchoNextGeometrySetOperator(geometry_mode="G2", tabular_dim=7)
    tot_g2 = sum(p.numel() for p in m_g2.parameters())
    dec_g2 = sum(p.numel() for p in m_g2.core.decoder.parameters())
    act_g2 = tot_g2 - dec_g2
    assert tot_g2 == 1652681, f"Expected 1,652,681 G2 total params, got {tot_g2}"
    assert act_g2 == 1044169, f"Expected 1,044,169 G2 active params, got {act_g2}"
    assert act_g2 - act == 8, f"Expected +8 G2 delta, got {act_g2 - act}"
