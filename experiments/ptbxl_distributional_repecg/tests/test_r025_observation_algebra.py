from __future__ import annotations

import pytest
import torch
import numpy as np

from repecg.paper07_operator.observable_subspace import (
    ObservableState,
    observable_state,
    compute_projector,
)


def test_shapes_and_finite_outputs():
    torch.manual_seed(42)
    B, T, m, d = 4, 100, 5, 8
    q = torch.randn(B, m, d)
    y = torch.randn(B, T, m)

    state = observable_state(q, y, rcond=1e-5)
    assert state.projected.shape == (B, T, d)
    assert state.projector.shape == (B, d, d)
    assert state.rank.shape == (B,)
    assert state.singular_values.shape == (B, d)
    assert torch.isfinite(state.projected).all()
    assert torch.isfinite(state.projector).all()
    assert torch.isfinite(state.singular_values).all()


def test_projector_algebraic_identities():
    """Verify P^2 = P and P^T = P."""
    torch.manual_seed(42)
    q = torch.randn(6, 4, 8, dtype=torch.float64)
    y = torch.randn(6, 200, 4, dtype=torch.float64)

    state = observable_state(q, y, rcond=1e-5)
    P = state.projector

    # P is symmetric: P^T = P
    sym_diff = torch.max(torch.abs(P - P.transpose(-2, -1))).item()
    assert sym_diff < 1e-12, f"Projector symmetry violated: max diff = {sym_diff}"

    # P is idempotent: P @ P = P
    idemp_diff = torch.max(torch.abs(P @ P - P)).item()
    assert idemp_diff < 1e-12, f"Projector idempotence violated: max diff = {idemp_diff}"


def test_noiseless_observable_projection():
    """Verify X_obs = X @ P and X_obs @ P = X_obs in noiseless exact model."""
    torch.manual_seed(42)
    B, T, m, d = 4, 500, 5, 8
    X = torch.randn(B, T, d, dtype=torch.float64)
    Q = torch.randn(B, m, d, dtype=torch.float64)
    Y = X @ Q.transpose(-2, -1)  # Exact noiseless measurements

    state = observable_state(Q, Y, rcond=1e-5)
    X_obs = state.projected
    P = state.projector

    # X_obs = X @ P
    diff_xp = torch.max(torch.abs(X_obs - (X @ P))).item()
    assert diff_xp < 1e-12, f"X_obs != X @ P: max diff = {diff_xp}"

    # X_obs @ P = X_obs
    diff_proj = torch.max(torch.abs((X_obs @ P) - X_obs)).item()
    assert diff_proj < 1e-12, f"X_obs @ P != X_obs: max diff = {diff_proj}"


def test_channel_reorder_invariance():
    """Permuting the measured channels must yield identical P and X_obs."""
    torch.manual_seed(123)
    T, m, d = 250, 6, 8
    X = torch.randn(T, d, dtype=torch.float64)
    Q = torch.randn(m, d, dtype=torch.float64)
    Y = X @ Q.T

    state_orig = observable_state(Q, Y, rcond=1e-5)

    # Permute channels
    perm = torch.randperm(m)
    Q_perm = Q[perm]
    Y_perm = Y[:, perm]

    state_perm = observable_state(Q_perm, Y_perm, rcond=1e-5)

    diff_P = torch.max(torch.abs(state_orig.projector - state_perm.projector)).item()
    diff_X = torch.max(torch.abs(state_orig.projected - state_perm.projected)).item()
    assert diff_P < 1e-12, f"Reorder invariance for P violated: diff = {diff_P}"
    assert diff_X < 1e-12, f"Reorder invariance for X_obs violated: diff = {diff_X}"


def test_channel_sign_reversal_invariance():
    """Negating an operator row with matching waveform negation must yield identical P and X_obs."""
    torch.manual_seed(456)
    T, m, d = 200, 4, 8
    X = torch.randn(T, d, dtype=torch.float64)
    Q = torch.randn(m, d, dtype=torch.float64)
    Y = X @ Q.T

    state_orig = observable_state(Q, Y, rcond=1e-5)

    # Invert signs of channels 0 and 2
    signs = torch.tensor([[-1.0, 1.0, -1.0, 1.0]], dtype=torch.float64)
    Q_neg = Q * signs.T
    Y_neg = Y * signs

    state_neg = observable_state(Q_neg, Y_neg, rcond=1e-5)

    diff_P = torch.max(torch.abs(state_orig.projector - state_neg.projector)).item()
    diff_X = torch.max(torch.abs(state_orig.projected - state_neg.projected)).item()
    assert diff_P < 1e-12, f"Sign reversal invariance for P violated: diff = {diff_P}"
    assert diff_X < 1e-12, f"Sign reversal invariance for X_obs violated: diff = {diff_X}"


def test_invertible_row_remixing_invariance():
    """Any well-conditioned invertible R in R^(m x m) must preserve P and X_obs."""
    torch.manual_seed(789)
    T, m, d = 300, 5, 8
    X = torch.randn(T, d, dtype=torch.float64)
    Q = torch.randn(m, d, dtype=torch.float64)
    Y = X @ Q.T

    state_orig = observable_state(Q, Y, rcond=1e-5)

    # Generate well-conditioned invertible matrix R
    R = torch.randn(m, m, dtype=torch.float64)
    while torch.linalg.cond(R) > 50.0:
        R = torch.randn(m, m, dtype=torch.float64)

    Q_remix = R @ Q
    Y_remix = Y @ R.T

    state_remix = observable_state(Q_remix, Y_remix, rcond=1e-5)

    diff_P = torch.max(torch.abs(state_orig.projector - state_remix.projector)).item()
    diff_X = torch.max(torch.abs(state_orig.projected - state_remix.projected)).item()
    assert diff_P < 1e-11, f"Invertible remixing invariance for P violated: diff = {diff_P}"
    assert diff_X < 1e-11, f"Invertible remixing invariance for X_obs violated: diff = {diff_X}"


def test_dependent_measurement_consistency():
    """Adding a linearly dependent row (e.g. III = II - I) or duplicating a row must not change P or X_obs."""
    torch.manual_seed(999)
    T, d = 250, 8
    X = torch.randn(T, d, dtype=torch.float64)
    # Basis: Lead I, Lead II, Lead V1
    e = torch.eye(d, dtype=torch.float64)
    Q = torch.stack([e[0], e[1], e[2]], dim=0)  # m = 3
    Y = X @ Q.T

    state_orig = observable_state(Q, Y, rcond=1e-5)
    assert state_orig.rank.item() == 3

    # Add dependent row: III = II - I
    Q_dep = torch.cat([Q, (e[1] - e[0]).unsqueeze(0)], dim=0)
    Y_dep = torch.cat([Y, (Y[:, 1:2] - Y[:, 0:1])], dim=1)

    state_dep = observable_state(Q_dep, Y_dep, rcond=1e-5)
    assert state_dep.rank.item() == 3, f"Dependent row increased rank: {state_dep.rank.item()}"

    diff_P = torch.max(torch.abs(state_orig.projector - state_dep.projector)).item()
    diff_X = torch.max(torch.abs(state_orig.projected - state_dep.projected)).item()
    assert diff_P < 1e-12, f"Dependent lead consistency for P violated: diff = {diff_P}"
    assert diff_X < 1e-12, f"Dependent lead consistency for X_obs violated: diff = {diff_X}"

    # Also test duplicating row 2
    Q_dup = torch.cat([Q, Q[2:3]], dim=0)
    Y_dup = torch.cat([Y, Y[:, 2:3]], dim=1)
    state_dup = observable_state(Q_dup, Y_dup, rcond=1e-5)
    assert state_dup.rank.item() == 3

    diff_P_dup = torch.max(torch.abs(state_orig.projector - state_dup.projector)).item()
    diff_X_dup = torch.max(torch.abs(state_orig.projected - state_dup.projected)).item()
    assert diff_P_dup < 1e-12
    assert diff_X_dup < 1e-12


def test_monotonic_observability():
    """If row(Q1) subset of row(Q2), then P2 - P1 is positive semi-definite and rank(P2) >= rank(P1)."""
    torch.manual_seed(777)
    d = 8
    e = torch.eye(d, dtype=torch.float64)
    # Q1 spans 3 leads
    Q1 = torch.stack([e[0], e[2], e[4]], dim=0)
    # Q2 adds 2 more independent leads
    Q2 = torch.stack([e[0], e[2], e[4], e[1], e[6]], dim=0)

    P1, r1 = compute_projector(Q1, rcond=1e-5)
    P2, r2 = compute_projector(Q2, rcond=1e-5)

    assert r2.item() >= r1.item()
    diff_P = (P2 - P1).squeeze(0)

    # Eigenvalues of P2 - P1 must be >= -1e-12
    eigvals = torch.linalg.eigvalsh(diff_P)
    assert torch.all(eigvals >= -1e-12), f"Monotonic observability violated: min eigval = {eigvals.min().item()}"


def test_degeneracy_and_near_collinearity():
    """Test m=1, zero measurements, near-collinear rows, and NaN guards."""
    torch.manual_seed(888)
    d = 8
    T = 100

    # m = 1
    Q1 = torch.randn(1, d, dtype=torch.float64)
    Y1 = torch.randn(T, 1, dtype=torch.float64)
    state1 = observable_state(Q1, Y1, rcond=1e-5)
    assert state1.rank.item() == 1
    assert torch.isfinite(state1.projector).all()
    assert torch.isfinite(state1.projected).all()

    # m = 0 (empty measurements)
    Q0 = torch.zeros(0, d, dtype=torch.float64)
    Y0 = torch.zeros(T, 0, dtype=torch.float64)
    state0 = observable_state(Q0, Y0, rcond=1e-5)
    assert state0.rank.item() == 0
    assert torch.all(state0.projector == 0)
    assert torch.all(state0.projected == 0)

    # Near-collinear rows: row 1 + 1e-8 * noise
    Q_near = torch.stack([Q1[0], Q1[0] + 1e-8 * torch.randn(d, dtype=torch.float64)], dim=0)
    Y_near = torch.randn(T, 2, dtype=torch.float64)
    # With rcond=1e-5, the second singular value will be discarded, rank should be 1
    state_near = observable_state(Q_near, Y_near, rcond=1e-5)
    assert state_near.rank.item() == 1
    assert torch.isfinite(state_near.projector).all()
    assert torch.isfinite(state_near.projected).all()


def test_float32_tolerances():
    """Verify float32 inputs meet production tolerances."""
    torch.manual_seed(333)
    B, T, m, d = 8, 250, 4, 8
    X = torch.randn(B, T, d, dtype=torch.float32)
    Q = torch.randn(B, m, d, dtype=torch.float32)
    Y = X @ Q.transpose(-2, -1)

    state = observable_state(Q, Y, rcond=1e-5)
    P = state.projector
    X_obs = state.projected

    # Float32 tolerances
    sym_diff = torch.max(torch.abs(P - P.transpose(-2, -1))).item()
    idemp_diff = torch.max(torch.abs(P @ P - P)).item()
    proj_diff = torch.max(torch.abs(X_obs - (X @ P))).item()

    assert sym_diff < 1e-5, f"Float32 symmetry diff: {sym_diff}"
    assert idemp_diff < 1e-4, f"Float32 idempotence diff: {idemp_diff}"
    assert proj_diff < 1e-4, f"Float32 projection diff: {proj_diff}"
