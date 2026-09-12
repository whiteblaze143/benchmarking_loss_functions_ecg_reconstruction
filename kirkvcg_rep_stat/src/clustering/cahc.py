"""Constrained Agglomerative Hierarchical Clustering (CAHC) in 3D VCG Space.

Implements Step 2 of the repSpat pipeline mapped to 3D VCG trajectories:
- Contiguity constraint restricts merges strictly to VCG-adjacent cluster pairs.
- Lance-Williams Ward.D2 distance updates minimize within-cluster feature variance.
- Outputs 1-indexed initial cluster labels {1, ..., G}.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from .vcg_adjacency import construct_vcg_adjacency, construct_hybrid_adjacency


def vcg_constrained_hac(
    X: np.ndarray,
    vcg: np.ndarray,
    n_clusters: int = 4,
    m_neighbors: int = 4,
    use_hybrid: bool = False,
    temporal_window: int = 1,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[np.ndarray, AgglomerativeClustering, sp.csr_matrix]:
    """Performs Constrained Agglomerative Hierarchical Clustering under VCG spatial adjacency.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix [n, p].
    vcg : np.ndarray
        3D VCG coordinates [n, 3].
    n_clusters : int
        Number of clusters G (G >= 2).
    m_neighbors : int
        Number of nearest neighbors in VCG space R^3.
    use_hybrid : bool
        Whether to combine VCG adjacency with temporal contiguity.
    temporal_window : int
        Lag window for temporal contiguity if use_hybrid is True.
    linkage : str
        HAC linkage criterion ('ward', 'complete', 'average', 'single').
    metric : str
        Dissimilarity metric ('euclidean').

    Returns
    -------
    labels : np.ndarray
        1-indexed cluster labels of shape [n] (values in 1, ..., G).
    model : AgglomerativeClustering
        Fitted clustering model.
    connectivity : scipy.sparse.csr_matrix
        Adjacency matrix used as spatial constraint.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n = len(X_arr)
    n_clusters = min(max(1, n_clusters), n)

    if use_hybrid:
        connectivity = construct_hybrid_adjacency(
            vcg, m_neighbors=m_neighbors, temporal_window=temporal_window
        )
    else:
        connectivity = construct_vcg_adjacency(vcg, m_neighbors=m_neighbors)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric if linkage != "ward" else "euclidean",
        linkage=linkage,
        connectivity=connectivity,
    )

    labels_0 = model.fit_predict(X_arr)
    # Convert to 1-indexed cluster labels {1, ..., G}
    labels = labels_0 + 1

    return labels, model, connectivity
