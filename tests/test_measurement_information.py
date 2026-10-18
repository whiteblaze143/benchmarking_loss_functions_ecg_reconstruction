#!/usr/bin/env python3
"""Exhaustive mathematical unit tests for Module A: Measurement Information Layer.

Verifies:
1. Permutation Invariance: Permuting measurement rows leaves X_obs and P_Q unchanged.
2. Invertible Basis Invariance: Any invertible M in GL(m) leaves X_obs and P_Q unchanged.
3. Orthogonal Rotation Invariance: R in O(m) leaves X_obs and P_Q unchanged.
4. Polarity Reversal Invariance: q_i -> -q_i, y_i -> -y_i leaves representation unchanged.
5. Algebraic Redundancy Invariance: Adding dependent measurement rows does not alter X_obs or P_Q.
6. Independent Measurement Monotonicity: Adding independent row strictly expands P_Q (PSD difference).
7. Consistent Unit Scaling: q_i -> c*q_i, y_i -> c*y_i leaves representation unchanged.
8. Gaussian Likelihood Invariance: (G, h) invariant under Sigma' = M Sigma M^T.
9. Observability and Nullspace Testing: Exact classification of observable vs unsupported directions.
10. PyTorch Module Batch Consistency: Batch processing matches individual unbatched calls.
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

from experiments.echonext_operator_benchmark.measurement_information import (
    compute_noiseless_observable_np,
    compute_noiseless_observable_torch,
    compute_gaussian_information_np,
    compute_gaussian_information_torch,
    evaluate_concept_observability,
    MeasurementInformationLayer,
)


@pytest.fixture
def canonical_ecg_data():
    """Generates synthetic 8-lead ECG data (d=8, T=2500) and random operator (m=4, d=8)."""
    rng = np.random.default_rng(42)
    d, T, m = 8, 2500, 4
    # Random multi-lead signal
    X = rng.normal(scale=1.0, size=(d, T)).astype(np.float64)
    # Random full-rank measurement operator
    Q = rng.normal(scale=1.0, size=(m, d)).astype(np.float64)
    # Orthonormalize rows of Q for clean conditioning
    Q, _ = np.linalg.qr(Q.T)
    Q = Q.T[:m]
    # Noiseless acquisition
    Y = Q @ X
    return {"X": X, "Q": Q, "Y": Y, "d": d, "T": T, "m": m}


def test_noiseless_projection_idempotent_and_symmetric(canonical_ecg_data):
    """Verifies that P_Q is symmetric and idempotent (P_Q^2 == P_Q)."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    X_obs, P_Q, P_perp = compute_noiseless_observable_np(Q, Y)
    
    # Symmetry
    np.testing.assert_allclose(P_Q, P_Q.T, atol=1e-12)
    # Idempotence: P_Q @ P_Q == P_Q
    np.testing.assert_allclose(P_Q @ P_Q, P_Q, atol=1e-12)
    # Orthogonality: P_Q @ P_perp == 0
    np.testing.assert_allclose(P_Q @ P_perp, np.zeros_like(P_Q), atol=1e-12)
    # Trace equals rank
    assert np.isclose(np.trace(P_Q), canonical_ecg_data["m"], atol=1e-10)


def test_permutation_invariance(canonical_ecg_data):
    """Verifies that permuting the order of measurement rows leaves X_obs and P_Q identical."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    m = canonical_ecg_data["m"]
    
    X_obs_orig, P_Q_orig, _ = compute_noiseless_observable_np(Q, Y)
    
    perm = np.array([2, 0, 3, 1])
    Q_perm = Q[perm]
    Y_perm = Y[perm]
    
    X_obs_perm, P_Q_perm, _ = compute_noiseless_observable_np(Q_perm, Y_perm)
    
    np.testing.assert_allclose(P_Q_perm, P_Q_orig, atol=1e-12)
    np.testing.assert_allclose(X_obs_perm, X_obs_orig, atol=1e-12)


def test_orthogonal_basis_invariance(canonical_ecg_data):
    """Verifies that rotating measurement coordinates (R in O(m)) leaves canonical representation unchanged."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    m = canonical_ecg_data["m"]
    rng = np.random.default_rng(123)
    
    # Generate random orthogonal matrix R in O(m)
    R, _ = np.linalg.qr(rng.normal(size=(m, m)))
    
    Q_rot = R @ Q
    Y_rot = R @ Y
    
    X_obs_orig, P_Q_orig, _ = compute_noiseless_observable_np(Q, Y)
    X_obs_rot, P_Q_rot, _ = compute_noiseless_observable_np(Q_rot, Y_rot)
    
    np.testing.assert_allclose(P_Q_rot, P_Q_orig, atol=1e-12)
    np.testing.assert_allclose(X_obs_rot, X_obs_orig, atol=1e-12)


def test_general_invertible_basis_invariance(canonical_ecg_data):
    """Verifies that ANY invertible linear combination M in GL(m) leaves X_obs and P_Q unchanged."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    m = canonical_ecg_data["m"]
    rng = np.random.default_rng(456)
    
    # Well-conditioned invertible matrix M
    M = rng.normal(size=(m, m))
    U, s, Vt = np.linalg.svd(M)
    s_cond = np.linspace(2.0, 0.5, m)
    M = U @ np.diag(s_cond) @ Vt
    
    Q_new = M @ Q
    Y_new = M @ Y
    
    X_obs_orig, P_Q_orig, _ = compute_noiseless_observable_np(Q, Y)
    X_obs_new, P_Q_new, _ = compute_noiseless_observable_np(Q_new, Y_new)
    
    np.testing.assert_allclose(P_Q_new, P_Q_orig, atol=1e-10)
    np.testing.assert_allclose(X_obs_new, X_obs_orig, atol=1e-10)


def test_polarity_reversal_invariance(canonical_ecg_data):
    """Verifies that flipping lead polarities (q_i -> -q_i, y_i -> -y_i) preserves representation."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    
    X_obs_orig, P_Q_orig, _ = compute_noiseless_observable_np(Q, Y)
    
    # Invert rows 0 and 2
    Q_flip = Q.copy()
    Y_flip = Y.copy()
    Q_flip[[0, 2]] *= -1.0
    Y_flip[[0, 2]] *= -1.0
    
    X_obs_flip, P_Q_flip, _ = compute_noiseless_observable_np(Q_flip, Y_flip)
    
    np.testing.assert_allclose(P_Q_flip, P_Q_orig, atol=1e-12)
    np.testing.assert_allclose(X_obs_flip, X_obs_orig, atol=1e-12)


def test_algebraic_redundancy_invariance(canonical_ecg_data):
    """Verifies that adding linearly dependent measurement rows does NOT change X_obs or P_Q."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    
    X_obs_orig, P_Q_orig, _ = compute_noiseless_observable_np(Q, Y)
    
    # Create 2 redundant rows as linear combinations of existing rows
    q_red1 = 0.5 * Q[0] + 0.5 * Q[1]
    y_red1 = 0.5 * Y[0] + 0.5 * Y[1]
    q_red2 = 1.2 * Q[2] - 0.7 * Q[3]
    y_red2 = 1.2 * Y[2] - 0.7 * Y[3]
    
    Q_augmented = np.vstack([Q, q_red1[None, :], q_red2[None, :]])
    Y_augmented = np.vstack([Y, y_red1[None, :], y_red2[None, :]])
    
    X_obs_aug, P_Q_aug, _ = compute_noiseless_observable_np(Q_augmented, Y_augmented)
    
    np.testing.assert_allclose(P_Q_aug, P_Q_orig, atol=1e-10)
    np.testing.assert_allclose(X_obs_aug, X_obs_orig, atol=1e-10)


def test_independent_measurement_monotonicity(canonical_ecg_data):
    """Verifies that adding a genuinely independent measurement strictly expands P_Q."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    X = canonical_ecg_data["X"]
    
    X_obs_orig, P_Q_orig, P_perp_orig = compute_noiseless_observable_np(Q, Y)
    
    # Pick a vector strictly in the nullspace of Q
    rng = np.random.default_rng(789)
    v_null = P_perp_orig @ rng.normal(size=canonical_ecg_data["d"])
    v_null /= np.linalg.norm(v_null)
    
    # Add new measurement
    q_new = v_null
    y_new = q_new @ X
    
    Q_new = np.vstack([Q, q_new[None, :]])
    Y_new = np.vstack([Y, y_new[None, :]])
    
    X_obs_new, P_Q_new, _ = compute_noiseless_observable_np(Q_new, Y_new)
    
    # Difference matrix Delta_P = P_Q_new - P_Q_orig
    Delta_P = P_Q_new - P_Q_orig
    # Must be positive semi-definite (eigenvalues >= -1e-12)
    eigvals = np.linalg.eigvalsh(Delta_P)
    assert np.all(eigvals >= -1e-12), f"Min eigenvalue negative: {np.min(eigvals)}"
    # Rank must increase exactly by 1
    assert np.isclose(np.trace(P_Q_new), np.trace(P_Q_orig) + 1.0, atol=1e-10)


def test_unit_scaling_invariance(canonical_ecg_data):
    """Verifies that scaling measurement units consistently leaves representations invariant."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    
    X_obs_orig, P_Q_orig, _ = compute_noiseless_observable_np(Q, Y)
    
    # Scale individual rows by arbitrary factors
    scales = np.array([1000.0, 0.001, 42.0, 0.5])[:, None]
    Q_scaled = Q * scales
    Y_scaled = Y * scales
    
    X_obs_scaled, P_Q_scaled, _ = compute_noiseless_observable_np(Q_scaled, Y_scaled)
    
    np.testing.assert_allclose(P_Q_scaled, P_Q_orig, atol=1e-10)
    np.testing.assert_allclose(X_obs_scaled, X_obs_orig, atol=1e-10)


def test_gaussian_information_invariance(canonical_ecg_data):
    """Verifies that Gaussian information terms (G, h) are strictly invariant under Sigma' = M Sigma M^T."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    m = canonical_ecg_data["m"]
    rng = np.random.default_rng(999)
    
    # Non-trivial noise covariance Sigma_eps
    A = rng.normal(size=(m, m))
    Sigma_eps = A @ A.T + 0.1 * np.eye(m)
    
    G_orig, h_orig = compute_gaussian_information_np(Q, Y, Sigma_eps)
    
    # Invertible coordinate transformation M
    M = rng.normal(size=(m, m))
    Q_trans = M @ Q
    Y_trans = M @ Y
    Sigma_trans = M @ Sigma_eps @ M.T
    
    G_trans, h_trans = compute_gaussian_information_np(Q_trans, Y_trans, Sigma_trans)
    
    np.testing.assert_allclose(G_trans, G_orig, atol=1e-10)
    np.testing.assert_allclose(h_trans, h_orig, atol=1e-10)


def test_concept_observability_classification(canonical_ecg_data):
    """Verifies observability classification: observable vs unobservable."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    _, P_Q, P_perp = compute_noiseless_observable_np(Q, Y)
    
    # 1. Concept direction lying strictly in row space of Q (observable)
    v_obs = Q[0]  # direction of first measurement
    res_obs = evaluate_concept_observability(P_Q, v_obs)
    assert res_obs["status"] == "OBSERVABLE"
    assert np.isclose(res_obs["observable_fraction"], 1.0, atol=1e-5)
    assert np.isclose(res_obs["nullspace_residue"], 0.0, atol=1e-5)
    
    # 2. Concept direction lying strictly in nullspace of Q (unobservable)
    rng = np.random.default_rng(101)
    v_unobs = P_perp @ rng.normal(size=canonical_ecg_data["d"])
    v_unobs /= np.linalg.norm(v_unobs)
    res_unobs = evaluate_concept_observability(P_Q, v_unobs)
    assert res_unobs["status"] == "UNOBSERVABLE"
    assert np.isclose(res_unobs["observable_fraction"], 0.0, atol=1e-5)
    assert np.isclose(res_unobs["nullspace_residue"], 1.0, atol=1e-5)
    
    # 3. Partial concept direction (50% in, 50% out)
    v_part = v_obs / np.linalg.norm(v_obs) + v_unobs
    res_part = evaluate_concept_observability(P_Q, v_part)
    assert res_part["status"] == "PARTIALLY_OBSERVABLE"
    assert 0.10 < res_part["observable_fraction"] < 0.90


def test_pytorch_module_batching(canonical_ecg_data):
    """Verifies that PyTorch MeasurementInformationLayer yields exact batched results."""
    Q, Y = canonical_ecg_data["Q"], canonical_ecg_data["Y"]
    
    Q_t = torch.from_numpy(Q).float()
    Y_t = torch.from_numpy(Y).float()
    
    layer = MeasurementInformationLayer(canonical_dim=8)
    
    # Unbatched forward
    X_obs_single, P_Q_single, _ = layer(Q_t, Y_t)
    
    # Batched forward with duplicate
    Q_batch = torch.stack([Q_t, Q_t], dim=0)
    Y_batch = torch.stack([Y_t, Y_t], dim=0)
    X_obs_batch, P_Q_batch, _ = layer(Q_batch, Y_batch)
    
    torch.testing.assert_close(X_obs_batch[0], X_obs_single, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(P_Q_batch[0], P_Q_single, atol=1e-5, rtol=1e-5)
