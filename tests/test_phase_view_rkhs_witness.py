#!/usr/bin/env python3
"""Exhaustive mathematical verification tests for Phase–View RKHS Witness Network.

Verifies:
1. Exact parameter count: Exactly 18,432 trainable witness coefficients (16 x 9 x 128).
2. Exact polarity reversal invariance: E(-q, p) == E(q, p) under simultaneous lead negation.
3. Subspace basis rotation invariance: Invariant under orthogonal rotation R in O(m).
4. Algebraic redundancy invariance: Invariant to linearly dependent measurement rows.
5. Observability boundary: Refuses to synthesize unobservable outside-span directions.
6. Continuous witness grid evaluation: Evaluates w_{q, p}(v, s) on physical (v, s) grid.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from experiments.echonext_operator_benchmark.phase_view_rkhs_witness import (
    PhaseViewRKHSWitness,
    ObservableSubspaceEvaluator,
)


@pytest.fixture
def mock_rkhs_embeddings():
    """Generates synthetic Phase-KME embeddings (B=2, m=4, 16, 128) and operators."""
    rng = np.random.default_rng(42)
    B, m, d = 2, 4, 8
    num_phases, nystrom_dim = 16, 128

    # Random operators
    Q_np = rng.normal(size=(B, m, d)).astype(np.float32)
    Q_np /= np.linalg.norm(Q_np, axis=-1, keepdims=True)

    # Random embeddings mu_+ and mu_-
    # When y -> -y, physical atoms negate. IMQ kernel is even in distance to landmarks,
    # but whitening mean induces an asymmetry between mu_+ and mu_-.
    mu_pos_np = rng.normal(size=(B, m, num_phases, nystrom_dim)).astype(np.float32)
    mu_neg_np = rng.normal(size=(B, m, num_phases, nystrom_dim)).astype(np.float32)

    Q = torch.from_numpy(Q_np)
    mu_pos = torch.from_numpy(mu_pos_np)
    mu_neg = torch.from_numpy(mu_neg_np)
    tabular = torch.randn(B, 7)

    return {
        "Q": Q,
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "tabular": tabular,
        "B": B,
        "m": m,
        "d": d,
        "num_phases": num_phases,
        "nystrom_dim": nystrom_dim,
    }


def test_exact_witness_parameter_count():
    """Verifies that the witness contains exactly 16 x 9 x 128 = 18,432 trainable coefficients."""
    model = PhaseViewRKHSWitness(num_phases=16, canonical_dim=8, nystrom_dim=128, num_tabular=7)
    
    expected_iso = 16 * 128            # 2,048
    expected_dir = 16 * 8 * 128        # 16,384
    expected_witness = 16 * 9 * 128    # 18,432

    assert model.w_iso.numel() == expected_iso, f"w_iso elements: {model.w_iso.numel()}"
    assert model.w_dir.numel() == expected_dir, f"w_dir elements: {model.w_dir.numel()}"
    assert model.num_witness_parameters == expected_witness, f"Witness params: {model.num_witness_parameters}"
    
    # Total model parameters (witness + bias + 7 tabular)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable == expected_witness + 1 + 7, f"Total trainable: {total_trainable}"


def test_exact_polarity_reversal_invariance(mock_rkhs_embeddings):
    """Verifies that simultaneous reversal of operator and waveform preserves evidence identically.

    Under (q, y) -> (-q, -y):
        mu_pos' = mu_neg
        mu_neg' = mu_pos
        Q' = -Q
    Model evidence E(q, p) MUST be numerically identical!
    """
    model = PhaseViewRKHSWitness(num_phases=16, canonical_dim=8, nystrom_dim=128, num_tabular=7)
    model.eval()

    Q = mock_rkhs_embeddings["Q"]
    mu_pos = mock_rkhs_embeddings["mu_pos"]
    mu_neg = mock_rkhs_embeddings["mu_neg"]
    tabular = mock_rkhs_embeddings["tabular"]

    # Inverted inputs
    Q_flipped = -Q
    mu_pos_flipped = mu_neg  # When waveform is inverted, pos becomes neg
    mu_neg_flipped = mu_pos  # and neg becomes pos

    with torch.no_grad():
        out_orig = model.forward_from_embeddings(mu_pos, mu_neg, Q, tabular=tabular)
        out_flipped = model.forward_from_embeddings(mu_pos_flipped, mu_neg_flipped, Q_flipped, tabular=tabular)

    # Logits and probs must be IDENTICAL within machine precision
    torch.testing.assert_close(out_flipped["logits"], out_orig["logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_flipped["probs"], out_orig["probs"], atol=1e-6, rtol=1e-6)
    
    # Evidence field E(q, p) must be IDENTICAL
    torch.testing.assert_close(out_flipped["evidence_field"], out_orig["evidence_field"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_flipped["phase_evidence"], out_orig["phase_evidence"], atol=1e-6, rtol=1e-6)


def test_subspace_basis_rotation_invariance():
    """Verifies that rotating measurement basis R in O(m) preserves row space and projector."""
    rng = np.random.default_rng(123)
    m, d = 4, 8
    Q = rng.normal(size=(m, d))
    
    evaluator_orig = ObservableSubspaceEvaluator(Q)
    
    # Random orthogonal rotation R in O(m)
    R, _ = np.linalg.qr(rng.normal(size=(m, m)))
    Q_rot = R @ Q
    evaluator_rot = ObservableSubspaceEvaluator(Q_rot)

    # Projectors P_Q must be identical
    np.testing.assert_allclose(evaluator_rot.P_Q, evaluator_orig.P_Q, atol=1e-12)
    # Ranks must be identical
    assert evaluator_rot.rank == evaluator_orig.rank

    # Canonical orthonormal basis must span identical subspace
    U_orig = evaluator_orig.canonical_orthonormal_basis()
    U_rot = evaluator_rot.canonical_orthonormal_basis()
    P_U_orig = U_orig.T @ U_orig
    P_U_rot = U_rot.T @ U_rot
    np.testing.assert_allclose(P_U_rot, P_U_orig, atol=1e-12)


def test_algebraic_redundancy_invariance():
    """Verifies that adding linearly dependent rows leaves the observable subspace identical."""
    rng = np.random.default_rng(456)
    m, d = 3, 8
    Q = rng.normal(size=(m, d))
    evaluator_orig = ObservableSubspaceEvaluator(Q)

    # Add 2 redundant rows
    q_red1 = 0.5 * Q[0] + 0.5 * Q[1]
    q_red2 = 1.3 * Q[1] - 0.8 * Q[2]
    Q_aug = np.vstack([Q, q_red1[None, :], q_red2[None, :]])
    evaluator_aug = ObservableSubspaceEvaluator(Q_aug)

    np.testing.assert_allclose(evaluator_aug.P_Q, evaluator_orig.P_Q, atol=1e-10)
    assert evaluator_aug.rank == evaluator_orig.rank


def test_observability_boundary_refuses_outside_span():
    """Verifies that evaluator refuses to synthesize waveforms for outside-span query directions."""
    rng = np.random.default_rng(789)
    # Measure only Leads I and II (canonical coordinates 0 and 1)
    Q = np.zeros((2, 8))
    Q[0, 0] = 1.0  # Lead I
    Q[1, 1] = 1.0  # Lead II
    evaluator = ObservableSubspaceEvaluator(Q)

    # 1. In-span query: 0.6 * Lead I + 0.8 * Lead II
    q_in = np.zeros(8)
    q_in[0] = 0.6
    q_in[1] = 0.8
    assert evaluator.is_observable(q_in) is True

    # 2. Outside-span query: Lead V2 (coordinate 3)
    q_out = np.zeros(8)
    q_out[3] = 1.0
    assert evaluator.is_observable(q_out) is False

    # Synthesizing outside-span waveform must strictly raise ValueError
    Y = np.zeros((2, 2500))
    with pytest.raises(ValueError, match="Query operator is unobservable"):
        evaluator.recover_waveform(Y, q_out)


def test_witness_grid_evaluation():
    """Verifies continuous witness evaluation w_{q, p}(v, s) on 2D physical grid."""
    model = PhaseViewRKHSWitness(num_phases=16, canonical_dim=8, nystrom_dim=128, num_tabular=7)
    
    # Synthetic fit data
    rng = np.random.default_rng(999)
    fit_data = {
        "whitening_mean": np.array([0.0, 0.0], dtype=np.float64),
        "whitening_components": np.eye(2, dtype=np.float64),
        "whitening_scales": np.array([1.0, 1.0], dtype=np.float64),
        "landmarks": rng.normal(size=(128, 2)).astype(np.float64),
        "inverse_root": np.eye(128, dtype=np.float64),
        "c2": 0.5,
    }

    q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Lead I
    v_grid = np.linspace(-1.5, 1.5, 20)
    s_grid = np.linspace(-30.0, 30.0, 20)

    score_grid = model.evaluate_witness_grid(q, p=4, v_grid=v_grid, s_grid=s_grid, fit_data=fit_data)
    
    assert score_grid.shape == (20, 20)
    assert np.all(np.isfinite(score_grid))
    # Non-trivial spatial variation
    assert np.std(score_grid) > 0.0
