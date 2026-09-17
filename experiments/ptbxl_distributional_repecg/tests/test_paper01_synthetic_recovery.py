import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.paper01.build_paper01_representations import _allowed_mask, fit_tau, recurrence


def compute_rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Unbiased MMD^2 estimator with RBF kernel between sample sets x and y."""
    m, n = len(x), len(y)
    gamma = 1.0 / (2.0 * sigma**2)
    
    # K_xx
    diff_xx = x[:, None] - x[None, :]
    dist_xx = np.square(diff_xx).sum(axis=-1)
    k_xx = np.exp(-gamma * dist_xx)
    np.fill_diagonal(k_xx, 0.0)
    mmd_xx = k_xx.sum() / (m * (m - 1))
    
    # K_yy
    diff_yy = y[:, None] - y[None, :]
    dist_yy = np.square(diff_yy).sum(axis=-1)
    k_yy = np.exp(-gamma * dist_yy)
    np.fill_diagonal(k_yy, 0.0)
    mmd_yy = k_yy.sum() / (n * (n - 1))
    
    # K_xy
    diff_xy = x[:, None] - y[None, :]
    dist_xy = np.square(diff_xy).sum(axis=-1)
    k_xy = np.exp(-gamma * dist_xy)
    mmd_xy = k_xy.mean()
    
    return float(mmd_xx + mmd_yy - 2.0 * mmd_xy)


def test_synthetic_a_same_means_distinct_distributions() -> None:
    """Synthetic A: Verify that MMD separates distributions with identical means where Euclidean distance fails."""
    rng = np.random.default_rng(42)
    n_samples = 2000
    
    # P1: Unimodal standard normal N(0, 1)
    p1_samples = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 1))
    
    # P2: Bimodal Gaussian mixture 0.5*N(-2, 0.2) + 0.5*N(2, 0.2)
    component = rng.choice([0, 1], size=n_samples)
    p2_samples = np.where(
        component[:, None] == 0,
        rng.normal(loc=-2.0, scale=0.2, size=(n_samples, 1)),
        rng.normal(loc=2.0, scale=0.2, size=(n_samples, 1)),
    )
    
    # Check that empirical means are both close to 0
    mean1 = p1_samples.mean(axis=0)
    mean2 = p2_samples.mean(axis=0)
    euclidean_mean_diff = float(np.linalg.norm(mean1 - mean2))
    assert euclidean_mean_diff < 0.1, f"Means not matched: {mean1} vs {mean2}"
    
    # Mean-distance recurrence sees virtually zero distance
    euclidean_dist_sq = euclidean_mean_diff**2
    assert euclidean_dist_sq < 0.01
    
    # MMD recurrence with sigma=1.0 easily separates them
    mmd2 = compute_rbf_mmd2(p1_samples, p2_samples, sigma=1.0)
    assert mmd2 > 0.35, f"MMD failed to separate distributions: {mmd2}"


def test_synthetic_b_finite_sample_convergence() -> None:
    """Synthetic B: Verify finite-sample MMD convergence to zero for identical distributions."""
    rng = np.random.default_rng(123)
    dim = 2
    
    sample_sizes = [20, 100, 500, 2000]
    mmd_estimates = []
    
    for n in sample_sizes:
        x = rng.normal(size=(n, dim))
        y = rng.normal(size=(n, dim))
        mmd_val = max(0.0, compute_rbf_mmd2(x, y, sigma=1.0))
        mmd_estimates.append(mmd_val)
        
    # Large sample size should have MMD close to 0
    assert mmd_estimates[-1] < 0.02, f"MMD did not converge: {mmd_estimates}"
    assert mmd_estimates[-1] < mmd_estimates[0], "MMD should be lower at n=2000 than n=20"


def test_synthetic_c_cyclic_shift_equivariance() -> None:
    """Synthetic C: Verify cyclic shift equivariance of recurrence operators."""
    rng = np.random.default_rng(42)
    phases = 16
    features = 10
    
    x = rng.normal(size=(1, phases, features))
    tau = fit_tau(x)
    r = recurrence(x, tau)[0]  # shape (16, 16)
    
    shift = 3
    x_shifted = np.roll(x, shift, axis=1)
    r_shifted = recurrence(x_shifted, tau)[0]
    
    # The recurrence of the shifted trajectory must equal the rolled recurrence operator
    # P_k R P_k^T is rolling both axes
    r_expected = np.roll(np.roll(r, shift, axis=0), shift, axis=1)
    
    max_diff = np.abs(r_shifted - r_expected).max()
    assert max_diff < 1e-5, f"Cyclic shift equivariance violated: {max_diff}"


def test_synthetic_d_scale_disentanglement() -> None:
    """Synthetic D: Verify that scale normalization produces scale-invariant recurrence."""
    rng = np.random.default_rng(999)
    phases = 16
    features = 8
    
    x = rng.normal(size=(1, phases, features))
    scale_factor = 3.5
    x_scaled = x * scale_factor
    
    # Extract scale
    scale_x = np.sqrt(np.mean(np.square(x)))
    scale_scaled = np.sqrt(np.mean(np.square(x_scaled)))
    
    assert np.isclose(scale_scaled, scale_x * scale_factor, rtol=1e-5)
    
    # Normalized shape representations
    x_shape = x / scale_x
    x_scaled_shape = x_scaled / scale_scaled
    
    assert np.allclose(x_shape, x_scaled_shape, atol=1e-6)
    
    # Recurrence on shape representations is identical
    tau = fit_tau(x_shape)
    r_orig = recurrence(x_shape, tau)[0]
    r_from_scaled = recurrence(x_scaled_shape, tau)[0]
    
    assert np.allclose(r_orig, r_from_scaled, atol=1e-6)
