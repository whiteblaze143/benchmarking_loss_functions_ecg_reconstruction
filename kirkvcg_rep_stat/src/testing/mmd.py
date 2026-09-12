"""Empirical Maximum Mean Discrepancy squared (MMD^2) and characteristic kernels.

Implements Step 3(a) of repSpat:
- Empirical MMD^2 two-sample test statistic between feature distributions P(g) and P(h).
- Inverse Multiquadratic (IMQ) Kernel: k(x, y) = (||x-y||^2 + c^2)^(-1/2).
- Characteristic Gaussian RBF kernel fallback.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def imq_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    c: float = 1.0,
) -> np.ndarray:
    """Computes the Inverse Multiquadratic (IMQ) kernel matrix: k(x, y) = 1 / sqrt(||x-y||^2 + c^2).

    Parameters
    ----------
    X : np.ndarray
        Sample matrix [n_x, p].
    Y : np.ndarray, optional
        Sample matrix [n_y, p]. If None, Y = X.
    c : float
        Kernel shape scale parameter (default: 1.0).

    Returns
    -------
    K : np.ndarray
        Kernel Gram matrix [n_x, n_y].
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float64)

    # Pairwise squared Euclidean distances
    dist_sq = cdist(X_arr, Y_arr, metric="sqeuclidean")
    return 1.0 / np.sqrt(dist_sq + c**2)


def rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """Computes Gaussian RBF kernel matrix: k(x, y) = exp(-gamma * ||x-y||^2).

    Parameters
    ----------
    X : np.ndarray
        Sample matrix [n_x, p].
    Y : np.ndarray, optional
        Sample matrix [n_y, p]. If None, Y = X.
    gamma : float, optional
        RBF scale (default: 1 / p).

    Returns
    -------
    K : np.ndarray
        Kernel Gram matrix [n_x, n_y].
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float64)
    p = X_arr.shape[1]
    g = 1.0 / float(p) if gamma is None else float(gamma)

    dist_sq = cdist(X_arr, Y_arr, metric="sqeuclidean")
    return np.exp(-g * dist_sq)


def compute_mmd2(
    X_g: np.ndarray,
    X_h: np.ndarray,
    kernel: str = "imq",
    kernel_param: float = 1.0,
    unbiased: bool = False,
) -> float:
    """Computes the empirical Maximum Mean Discrepancy squared statistic MMD^2(X_g, X_h).

    MMD^2 = 1/n_g^2 sum k(x_i^g, x_j^g) + 1/n_h^2 sum k(x_i^h, x_j^h) - 2/(n_g n_h) sum k(x_i^g, x_j^h)

    Parameters
    ----------
    X_g : np.ndarray
        Feature vectors of cluster g of shape [n_g, p].
    X_h : np.ndarray
        Feature vectors of cluster h of shape [n_h, p].
    kernel : str
        'imq' (default) or 'rbf'.
    kernel_param : float
        c parameter for IMQ or gamma for RBF.
    unbiased : bool
        Whether to use unbiased MMD^2 (excludes self-similarities). Default: False (per repSpat).

    Returns
    -------
    mmd2_stat : float
        Non-negative empirical MMD^2 statistic.
    """
    X_g = np.asarray(X_g, dtype=np.float64)
    X_h = np.asarray(X_h, dtype=np.float64)

    n_g = len(X_g)
    n_h = len(X_h)

    if n_g == 0 or n_h == 0:
        return 0.0

    k_fn = (
        (lambda x, y: imq_kernel(x, y, c=kernel_param))
        if kernel.lower() == "imq"
        else (lambda x, y: rbf_kernel(x, y, gamma=kernel_param))
    )

    K_gg = k_fn(X_g, X_g)
    K_hh = k_fn(X_h, X_h)
    K_gh = k_fn(X_g, X_h)

    if unbiased and n_g > 1 and n_h > 1:
        term_gg = (K_gg.sum() - np.trace(K_gg)) / (n_g * (n_g - 1))
        term_hh = (K_hh.sum() - np.trace(K_hh)) / (n_h * (n_h - 1))
        term_gh = 2.0 * K_gh.mean()
        stat = term_gg + term_hh - term_gh
    else:
        term_gg = K_gg.mean()
        term_hh = K_hh.mean()
        term_gh = 2.0 * K_gh.mean()
        stat = term_gg + term_hh - term_gh

    # Prevent minor floating point negative values
    return float(max(0.0, stat))
