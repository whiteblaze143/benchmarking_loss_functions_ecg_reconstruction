"""Attribute dissimilarity representation for temporal ECG features.

Implements Step 1 of repSpat adapted to 1D temporal beat features:
- Continuous attributes: Pairwise Euclidean distance
- Binary / Thresholded markers: Pairwise Jaccard distance
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_attribute_distances(
    X: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Computes pairwise attribute dissimilarity matrix D [n, n].

    Parameters
    ----------
    X : np.ndarray
        Array of shape [n, p] containing p-dimensional attribute vectors
        observed across n beats.
    metric : str
        Dissimilarity metric: 'euclidean' for continuous attributes or
        'jaccard' for binary/thresholded markers.

    Returns
    -------
    D : np.ndarray
        Symmetric dissimilarity matrix of shape [n, n] with zeros on the diagonal.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape [n, p], got {X.shape}")

    metric = metric.lower()
    if metric not in ("euclidean", "jaccard"):
        raise ValueError(f"Metric must be 'euclidean' or 'jaccard', got '{metric}'")

    if metric == "jaccard":
        # Ensure binary values {0, 1} for jaccard
        unique_vals = np.unique(X)
        if not set(unique_vals).issubset({0.0, 1.0}):
            # Threshold at median or non-zero if continuous values passed
            X_bin = (X > 0.0).astype(bool)
        else:
            X_bin = X.astype(bool)
        condensed_dist = pdist(X_bin, metric="jaccard")
        # Handle zero-division (when both vectors are all-zeros, pdist can yield NaN)
        condensed_dist = np.nan_to_num(condensed_dist, nan=0.0)
    else:
        condensed_dist = pdist(X, metric="euclidean")

    D = squareform(condensed_dist)
    np.fill_diagonal(D, 0.0)
    return D
