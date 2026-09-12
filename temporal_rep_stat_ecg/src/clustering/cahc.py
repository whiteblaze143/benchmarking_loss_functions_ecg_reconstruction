"""Temporal adjacency representation and Constrained Agglomerative Hierarchical Clustering (CAHC).

Implements Step 2 of the repSpat pipeline translated to 1D temporal domain:
- 1D Temporal Adjacency Matrix L based on m-nearest temporal neighbors.
- Constrained Agglomerative Hierarchical Clustering (CAHC) with temporal contiguity.
- Lance-Williams Ward.D2 distance updates.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


def construct_temporal_adjacency(
    timestamps: np.ndarray,
    m_neighbors: int = 4,
    symmetric: bool = True,
) -> sp.csr_matrix:
    """Constructs 1D temporal adjacency matrix L [n, n].

    Parameters
    ----------
    timestamps : np.ndarray
        1D array of length n containing timestamps (e.g. R-peak times in seconds or indices).
    m_neighbors : int
        Number of nearest temporal neighbors (m >= 2).
    symmetric : bool
        Whether to enforce undirected/symmetric adjacency: L = max(L, L^T).

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Binary sparse adjacency matrix of shape [n, n] with zero diagonal.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1, 1)
    n = len(timestamps)

    if n <= 1:
        return sp.csr_matrix((n, n), dtype=int)

    # Ensure m is within [1, n-1]
    k = min(max(1, m_neighbors), n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(timestamps)
    # Exclude self (index 0)
    indices = nn.kneighbors(timestamps, return_distance=False)[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = np.ones(len(rows), dtype=int)

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if symmetric:
        L = L.maximum(L.T)

    L.setdiag(0)
    L.eliminate_zeros()
    return L


def temporal_constrained_hac(
    X: np.ndarray,
    timestamps: np.ndarray,
    n_clusters: int = 4,
    m_neighbors: int = 4,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[np.ndarray, AgglomerativeClustering, sp.csr_matrix]:
    """Performs Constrained Agglomerative Hierarchical Clustering with temporal contiguity.

    Parameters
    ----------
    X : np.ndarray
        Attribute matrix [n, p].
    timestamps : np.ndarray
        1D timestamps [n].
    n_clusters : int
        Number of clusters / temporal episodes G (G >= 2).
    m_neighbors : int
        Temporal neighborhood size m.
    linkage : str
        HAC linkage criterion ('ward', 'complete', 'average', 'single').
    metric : str
        Metric ('euclidean' for continuous).

    Returns
    -------
    labels : np.ndarray
        1-indexed cluster labels of shape [n] (values in 1, ..., G).
    model : AgglomerativeClustering
        Fitted clustering model.
    connectivity : scipy.sparse.csr_matrix
        Temporal adjacency matrix used as constraint.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    n_clusters = min(max(1, n_clusters), n)

    connectivity = construct_temporal_adjacency(timestamps, m_neighbors=m_neighbors)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric if linkage != "ward" else "euclidean",
        linkage=linkage,
        connectivity=connectivity,
    )

    labels_0indexed = model.fit_predict(X)
    # Convert to 1-indexed labels {1, ..., G} per repSpat convention
    labels = labels_0indexed + 1
    return labels, model, connectivity
