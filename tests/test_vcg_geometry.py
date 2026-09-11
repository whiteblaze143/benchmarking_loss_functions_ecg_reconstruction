"""Tests for VCG Geometry and Pseudoinverse Lift Contracts (PRD Section 65)."""
import numpy as np
import pytest
import torch

from rep_stat_ecg.src.vcg.geometry import (
    CANONICAL_12_LEADS,
    TABLE_7_LEAD_DIRECTIONS,
    get_lead_direction_matrix,
    compute_tikhonov_svd_pinv,
)
from rep_stat_ecg.src.vcg.lift import VCGLift


def test_lvcg_direction_matrix_exact():
    """Verify Table 7 values match exact specification."""
    A = get_lead_direction_matrix()
    assert A.shape == (12, 3)

    # Check Lead I
    assert np.allclose(A[0].numpy(), [1.0, 0.0, 0.0], atol=1e-5)
    # Check Lead II
    assert np.allclose(A[1].numpy(), [0.5, 0.86603, 0.0], atol=1e-5)
    # Check Lead III
    assert np.allclose(A[2].numpy(), [-0.5, 0.86603, 0.0], atol=1e-5)
    # Check V1
    assert np.allclose(A[6].numpy(), [-0.33682, 0.17365, 0.92542], atol=1e-5)
    # Check V6
    assert np.allclose(A[11].numpy(), [0.63302, 0.17365, -0.75441], atol=1e-5)


def test_direction_matrix_rank3():
    """Verify standard 12-lead direction matrix has full rank 3."""
    A = get_lead_direction_matrix()
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    assert S.shape == (3,)
    assert S[2].item() > 0.5, f"Smallest singular value too small: {S[2].item()}"
    rank = torch.linalg.matrix_rank(A).item()
    assert rank == 3


def test_svd_pseudoinverse_matches_lstsq():
    """Verify SVD Tikhonov pseudoinverse matches regularized least squares."""
    A = get_lead_direction_matrix()
    lam = 1e-6
    A_pinv = compute_tikhonov_svd_pinv(A, lam=lam)
    assert A_pinv.shape == (3, 12)

    # Analytic normal equation check: (A^T A + lam I)^-1 A^T
    AtA = A.T @ A
    reg = lam * torch.eye(3)
    analytic_pinv = torch.linalg.inv(AtA + reg) @ A.T

    diff = torch.max(torch.abs(A_pinv - analytic_pinv)).item()
    assert diff < 1e-5, f"SVD pseudoinverse deviates from normal equation: {diff}"


def test_project_lift_consistency():
    """Verify exact dipole ECG is recovered faithfully: A (A^+ E) ~= E."""
    lift_mod = VCGLift(lam=1e-6)

    # Generate synthetic exact dipole trajectory v(t) [B, 3, T]
    B, T = 4, 500
    torch.manual_seed(42)
    v_true = torch.randn(B, 3, T)

    # Project to 12 leads
    E_true = lift_mod.project(v_true)  # [B, 12, T]
    assert E_true.shape == (B, 12, T)

    # Lift back to VCG
    v_rec = lift_mod.lift(E_true)  # [B, 3, T]
    assert v_rec.shape == (B, 3, T)

    max_err = torch.max(torch.abs(v_true - v_rec)).item()
    assert max_err < 1e-4, f"Dipole recovery error too high: {max_err}"

    # Residual should be zero for exact dipoles
    _, E_proj, r_norm = lift_mod.compute_residual(E_true)
    assert torch.max(r_norm).item() < 1e-4


def test_derived_limb_geometry():
    """Verify Einthoven's law (II - I = III) holds on projections."""
    lift_mod = VCGLift(lam=1e-6)
    v = torch.randn(2, 3, 100)
    E = lift_mod.project(v)  # [2, 12, 100]

    lead_i = E[:, 0, :]
    lead_ii = E[:, 1, :]
    lead_iii = E[:, 2, :]

    # In Einthoven geometry: Lead II - Lead I = Lead III
    diff = torch.max(torch.abs((lead_ii - lead_i) - lead_iii)).item()
    assert diff < 1e-5, f"Einthoven law violated: {diff}"
