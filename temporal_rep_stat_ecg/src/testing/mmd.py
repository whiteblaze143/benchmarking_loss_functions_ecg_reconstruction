"""Maximum Mean Discrepancy (MMD^2) computation with characteristic kernels.

Implements Step 3(a) of repSpat adapted to ECG beat feature distributions:
- Inverse Multiquadratic (IMQ) Kernel: k(x, y) = (||x - y||^2 + c^2)^{-1/2}
- Gaussian / RBF Kernel: k(x, y) = exp(-||x - y||^2 / sigma^2)
- Empirical MMD^2 statistic computation
"""
from __future__ import annotations

import numpy as np
import scipy.spatial.distance as sp_dist


def imq_kernel(
    dists_sq: np.ndarray,
    c_param: float = 1.0,
) -> np.ndarray:
    """Inverse Multiquadratic (IMQ) kernel: k(x, y) = (||x - y||^2 + c^2)^{-1/2}.

    Parameters
    ----------
    dists_sq : np.ndarray
        Matrix of pairwise squared Euclidean distances.
    c_param : float
        Kernel shape parameter (default: 1.0, c >= 1).

    Returns
    -------
    K : np.ndarray
        Kernel Gram matrix.
    """
    return 1.0 / np.sqrt(dists_sq + (c_param ** 2))


def gaussian_kernel(
    dists_sq: np.ndarray,
    sigma_param: float = 1.0,
) -> np.ndarray:
    """Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||^2 / sigma^2).

    Parameters
    ----------
    dists_sq : np.ndarray
        Matrix of pairwise squared Euclidean distances.
    sigma_param : float
        Bandwidth parameter.

    Returns
    -------
    K : np.ndarray
        Kernel Gram matrix.
    """
    gamma = 1.0 / max(sigma_param, 1e-9)
    return np.exp(-dists_sq * gamma)


def compute_mmd_sq_from_features(
    X_g: np.ndarray,
    X_h: np.ndarray,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
) -> float:
    """Computes empirical MMD^2 between two samples from feature matrices.

    Parameters
    ----------
    X_g : np.ndarray
        Feature matrix for episode g of shape [n_g, p].
    X_h : np.ndarray
        Feature matrix for episode h of shape [n_h, p].
    kernel : str
        'IMQ' (default) or 'Gaussian'.
    kernel_param : float
        Shape parameter c for IMQ or bandwidth for Gaussian.

    Returns
    -------
    mmd_sq : float
        Empirical MMD^2 statistic.
    """
    n_g = len(X_g)
    n_h = len(X_h)
    if n_g == 0 or n_h == 0:
        return 0.0

    D_gg = sp_dist.cdist(X_g, X_g, metric="sqeuclidean")
    D_hh = sp_dist.cdist(X_h, X_h, metric="sqeuclidean")
    D_gh = sp_dist.cdist(X_g, X_h, metric="sqeuclidean")

    if kernel.upper() == "IMQ":
        K_gg = imq_kernel(D_gg, c_param=kernel_param)
        K_hh = imq_kernel(D_hh, c_param=kernel_param)
        K_gh = imq_kernel(D_gh, c_param=kernel_param)
    elif kernel.lower() == "gaussian":
        K_gg = gaussian_kernel(D_gg, sigma_param=kernel_param)
        K_hh = gaussian_kernel(D_hh, sigma_param=kernel_param)
        K_gh = gaussian_kernel(D_gh, sigma_param=kernel_param)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Supported: 'IMQ', 'Gaussian'")

    term1 = K_gg.sum() / (n_g * n_g)
    term2 = K_hh.sum() / (n_h * n_h)
    term3 = -2.0 * K_gh.sum() / (n_g * n_h)

    mmd_sq = float(term1 + term2 + term3)
    return max(0.0, mmd_sq)


def compute_mmd_sq(
    idx_g: np.ndarray | list[int],
    idx_h: np.ndarray | list[int],
    dist_matrix: np.ndarray,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
) -> float:
    """Computes empirical MMD^2 using precomputed distance matrix.

    Parameters
    ----------
    idx_g : array-like
        Indices belonging to episode g.
    idx_h : array-like
        Indices belonging to episode h.
    dist_matrix : np.ndarray
        Full pairwise Euclidean distance matrix D [n, n].
    kernel : str
        'IMQ' (default) or 'Gaussian'.
    kernel_param : float
        Kernel parameter.

    Returns
    -------
    mmd_sq : float
        Empirical MMD^2 statistic.
    """
    idx_g = np.asarray(idx_g, dtype=int)
    idx_h = np.asarray(idx_h, dtype=int)
    n_g = len(idx_g)
    n_h = len(idx_h)
    if n_g == 0 or n_h == 0:
        return 0.0

    D_gg_sq = dist_matrix[np.ix_(idx_g, idx_g)] ** 2
    D_hh_sq = dist_matrix[np.ix_(idx_h, idx_h)] ** 2
    D_gh_sq = dist_matrix[np.ix_(idx_g, idx_h)] ** 2

    if kernel.upper() == "IMQ":
        K_gg = imq_kernel(D_gg_sq, c_param=kernel_param)
        K_hh = imq_kernel(D_hh_sq, c_param=kernel_param)
        K_gh = imq_kernel(D_gh_sq, c_param=kernel_param)
    elif kernel.lower() == "gaussian":
        K_gg = gaussian_kernel(D_gg_sq, sigma_param=kernel_param)
        K_hh = gaussian_kernel(D_hh_sq, sigma_param=kernel_param)
        K_gh = gaussian_kernel(D_gh_sq, sigma_param=kernel_param)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Supported: 'IMQ', 'Gaussian'")

    term1 = K_gg.sum() / (n_g * n_g)
    term2 = K_hh.sum() / (n_h * n_h)
    term3 = -2.0 * K_gh.sum() / (n_g * n_h)

    mmd_sq = float(term1 + term2 + term3)
    return max(0.0, mmd_sq)
