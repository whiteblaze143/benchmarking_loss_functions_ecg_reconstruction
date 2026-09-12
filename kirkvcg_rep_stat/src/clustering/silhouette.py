"""Spatially-Informed (Modified) Silhouette Score and Hyperparameter Grid Search.

Implements Step 2.3 of repSpat mapped to 3D VCG trajectories:
- Within-cluster cohesion a(i) computed across cluster members.
- Modified between-cluster separation b(i) strictly restricted to VCG-adjacent
  clusters sharing at least one link in the 3D VCG adjacency matrix L.
- Modified silhouette width: s_h(i) = (b(i) - a(i)) / max(a(i), b(i)).
- Grid optimization over m (VCG spatial neighbors) and G (initial clusters).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from .cahc import vcg_constrained_hac
from .distance import compute_attribute_distances
from .vcg_adjacency import construct_vcg_adjacency, construct_hybrid_adjacency


def compute_vcg_modified_silhouette(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    adjacency: np.ndarray | sp.csr_matrix,
) -> tuple[float, np.ndarray]:
    """Computes the spatially-informed modified silhouette score in VCG space.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise attribute dissimilarity matrix D [n, n].
    labels : np.ndarray
        1-indexed or 0-indexed cluster labels [n].
    adjacency : np.ndarray or scipy.sparse.csr_matrix
        VCG spatial adjacency matrix L [n, n].

    Returns
    -------
    avg_score : float
        Mean modified silhouette score across all observations.
    silhouettes : np.ndarray
        Per-sample silhouette score array of shape [n].
    """
    if sp.issparse(adjacency):
        adj_dense = adjacency.toarray().astype(bool)
    else:
        adj_dense = np.asarray(adjacency, dtype=bool)

    labels = np.asarray(labels)
    n = len(labels)
    silhouettes = np.zeros(n, dtype=float)

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0, silhouettes

    label_indices = {
        lab: np.flatnonzero(labels == lab)
        for lab in unique_labels
    }

    # Find VCG-adjacent cluster neighbors for each cluster
    neighbor_labels: dict[int, list[int]] = {}
    for lab, indices in label_indices.items():
        connected = adj_dense[indices].any(axis=0)
        neighbors = [
            other for other in np.unique(labels[connected])
            if other != lab
        ]
        neighbor_labels[lab] = neighbors

    for lab, members in label_indices.items():
        if len(members) > 1:
            intra = dist_matrix[np.ix_(members, members)]
            a = (intra.sum(axis=1) - np.diag(intra)) / (len(members) - 1)
        else:
            a = np.zeros(len(members), dtype=float)

        neighbors = neighbor_labels.get(lab, [])
        if not neighbors:
            continue

        b_candidates = [
            dist_matrix[np.ix_(members, label_indices[neighbor])].mean(axis=1)
            for neighbor in neighbors
        ]
        b = np.minimum.reduce(b_candidates)
        denom = np.maximum(a, b)
        silhouettes[members] = np.divide(
            b - a,
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0,
        )

    avg_score = float(np.mean(silhouettes))
    return avg_score, silhouettes


def optimize_vcg_hyperparameters(
    X: np.ndarray,
    vcg: np.ndarray,
    m_neighbors_list: list[int] | range = (3, 5, 8),
    n_clusters_range: list[int] | range = range(4, 9),
    metric: str = "euclidean",
    linkage: str = "ward",
    use_hybrid: bool = False,
) -> tuple[int, int, pd.DataFrame]:
    """Grid search over (m, G) maximizing the VCG-informed modified silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix [n, p].
    vcg : np.ndarray
        3D VCG coordinates [n, 3].
    m_neighbors_list : iterable of int
        Candidate spatial neighborhood sizes m.
    n_clusters_range : iterable of int
        Candidate cluster counts G.
    metric : str
        Attribute metric ('euclidean' or 'jaccard').
    linkage : str
        HAC linkage criterion ('ward').
    use_hybrid : bool
        Whether to use hybrid adjacency.

    Returns
    -------
    best_m : int
        Optimal m nearest neighbors.
    best_G : int
        Optimal initial cluster count G.
    results_df : pd.DataFrame
        Table containing ('m', 'G', 'mean_silhouette', 'status').
    """
    X_arr = np.asarray(X, dtype=np.float64)
    vcg_arr = np.asarray(vcg, dtype=np.float64)
    D = compute_attribute_distances(X_arr, metric=metric)

    records = []
    best_score = -float("inf")
    best_m = list(m_neighbors_list)[0]
    best_G = list(n_clusters_range)[0]

    for m in m_neighbors_list:
        if use_hybrid:
            adj = construct_hybrid_adjacency(vcg_arr, m_neighbors=m)
        else:
            adj = construct_vcg_adjacency(vcg_arr, m_neighbors=m)

        for G in n_clusters_range:
            try:
                labels, _, _ = vcg_constrained_hac(
                    X_arr,
                    vcg_arr,
                    n_clusters=G,
                    m_neighbors=m,
                    use_hybrid=use_hybrid,
                    linkage=linkage,
                    metric=metric,
                )
                score, _ = compute_vcg_modified_silhouette(D, labels, adj)
                records.append({
                    "m": m,
                    "G": G,
                    "mean_silhouette": score,
                    "status": "success",
                })
                if score > best_score:
                    best_score = score
                    best_m = m
                    best_G = G
            except Exception as exc:
                records.append({
                    "m": m,
                    "G": G,
                    "mean_silhouette": -1.0,
                    "status": f"failed: {exc}",
                })

    results_df = pd.DataFrame(records)
    return best_m, best_G, results_df
