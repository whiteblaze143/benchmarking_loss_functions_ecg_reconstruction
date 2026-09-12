"""Attribute dissimilarity matrix computation for VCG repSpat.

Implements Step 1 of repSpat adapted to electrophysiological features:
- Continuous attributes: Euclidean distance d_ij = ||X(t_i) - X(t_j)||_2 on standardized features
- Binary markers: Jaccard distance bounded in [0, 1] on thresholded morphological markers.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_attribute_distances(
    X: np.ndarray,
    metric: str = "euclidean",
    standardize: bool = True,
) -> np.ndarray:
    """Computes the pairwise attribute dissimilarity matrix D of shape [n, n].

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape [n, p].
    metric : str
        'euclidean' for continuous attributes or 'jaccard' for binary markers.
    standardize : bool
        Whether to standardize continuous features before distance calculation (default: True).

    Returns
    -------
    D : np.ndarray
        Symmetric dissimilarity matrix [n, n] with zero diagonal.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D array [n, p], got {X_arr.shape}")

    metric = metric.lower()
    if metric not in ("euclidean", "jaccard"):
        raise ValueError(f"Metric must be 'euclidean' or 'jaccard', got '{metric}'")

    if metric == "jaccard":
        # Ensure binary {0, 1}
        unique_vals = np.unique(X_arr)
        if not set(unique_vals).issubset({0.0, 1.0}):
            # Nonparametric median thresholding
            medians = np.median(X_arr, axis=0, keepdims=True)
            X_bin = (X_arr > medians).astype(bool)
        else:
            X_bin = X_arr.astype(bool)
        condensed = pdist(X_bin, metric="jaccard")
        condensed = np.nan_to_num(condensed, nan=0.0)
    else:
        if standardize:
            mean = np.mean(X_arr, axis=0, keepdims=True)
            std = np.std(X_arr, axis=0, keepdims=True) + 1e-8
            X_proc = (X_arr - mean) / std
        else:
            X_proc = X_arr
        condensed = pdist(X_proc, metric="euclidean")

    D = squareform(condensed)
    np.fill_diagonal(D, 0.0)
    return D
