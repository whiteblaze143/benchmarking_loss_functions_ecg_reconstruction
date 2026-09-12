"""Temporally-Informed (Modified) Silhouette Score and Hyperparameter Grid Search.

Implements Step 2.3 of repSpat adapted to 1D temporal domain:
- Evaluates within-cluster cohesion a(i)
- Evaluates modified between-cluster separation b(i) strictly restricted to
  temporally adjacent episodes (sharing >= 1 temporal link in L)
- Computes modified silhouette width s_h(i) = (b(i) - a(i)) / max(a(i), b(i))
- Optimizes grid over m (temporal neighbors) and G (number of episodes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from .cahc import construct_temporal_adjacency, temporal_constrained_hac
from .distance import compute_attribute_distances


def compute_modified_silhouette(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    adjacency: np.ndarray | sp.csr_matrix,
) -> tuple[float, np.ndarray]:
    """Computes the temporally-informed modified silhouette score.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise attribute dissimilarity matrix D [n, n].
    labels : np.ndarray
        Cluster labels [n].
    adjacency : np.ndarray or scipy.sparse.csr_matrix
        Temporal adjacency matrix L [n, n].

    Returns
    -------
    avg_silhouette : float
        Mean modified silhouette score over all observations.
    silhouettes : np.ndarray
        Per-sample silhouette score [n].
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
        label: np.flatnonzero(labels == label)
        for label in unique_labels
    }

    # Find temporally adjacent cluster neighbors for each cluster
    neighbor_labels: dict[int, list[int]] = {}
    for label, indices in label_indices.items():
        connected = adj_dense[indices].any(axis=0)
        neighbors = [
            other for other in np.unique(labels[connected])
            if other != label
        ]
        neighbor_labels[label] = neighbors

    for label, members in label_indices.items():
        if len(members) > 1:
            intra = dist_matrix[np.ix_(members, members)]
            a = (intra.sum(axis=1) - np.diag(intra)) / (len(members) - 1)
        else:
            a = np.zeros(len(members), dtype=float)

        neighbors = neighbor_labels.get(label, [])
        if not neighbors:
            # Fallback to all other clusters if no adjacent neighbor found
            neighbors = [o for o in unique_labels if o != label]
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

    avg_silhouette = float(np.mean(silhouettes))
    return avg_silhouette, silhouettes


def temporal_silhouette_analysis(
    X: np.ndarray,
    timestamps: np.ndarray,
    dist_matrix: np.ndarray | None = None,
    m_neighbors_list: list[int] | None = None,
    n_clusters_range: list[int] | range | None = None,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[pd.DataFrame, tuple[int, int], float]:
    """Evaluates grid of (m, G) to optimize the temporally-informed silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Attribute matrix [n, p].
    timestamps : np.ndarray
        1D temporal coordinates [n].
    dist_matrix : np.ndarray, optional
        Precomputed dissimilarity matrix D [n, n].
    m_neighbors_list : list of int, optional
        Grid of temporal neighbors m (default: [2, 4, 6]).
    n_clusters_range : iterable of int, optional
        Grid of cluster counts G (default: range(3, 8)).
    linkage : str
        Linkage criterion ('ward').
    metric : str
        Metric for attributes ('euclidean' or 'jaccard').

    Returns
    -------
    results_df : pd.DataFrame
        Table with columns: ['m_neighbors', 'n_clusters', 'avg_silhouette'].
    best_params : tuple of (int, int)
        (best_m, best_G) maximizing average modified silhouette.
    best_score : float
        The maximum average modified silhouette score.
    """
    if m_neighbors_list is None:
        m_neighbors_list = [2, 4, 6]
    if n_clusters_range is None:
        n_clusters_range = range(3, 8)

    if dist_matrix is None:
        dist_matrix = compute_attribute_distances(X, metric=metric)

    results = []

    for m in m_neighbors_list:
        L = construct_temporal_adjacency(timestamps, m_neighbors=m)
        adj_dense = L.toarray().astype(bool)

        for G in n_clusters_range:
            if G >= len(X):
                continue

            labels, _, _ = temporal_constrained_hac(
                X=X,
                timestamps=timestamps,
                n_clusters=G,
                m_neighbors=m,
                linkage=linkage,
                metric=metric,
            )

            avg_sil, _ = compute_modified_silhouette(dist_matrix, labels, adj_dense)

            results.append({
                "m_neighbors": int(m),
                "n_clusters": int(G),
                "avg_silhouette": float(avg_sil),
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df, (m_neighbors_list[0], list(n_clusters_range)[0]), 0.0

    best_row = results_df.loc[results_df["avg_silhouette"].idxmax()]
    best_params = (int(best_row["m_neighbors"]), int(best_row["n_clusters"]))
    best_score = float(best_row["avg_silhouette"])

    return results_df, best_params, best_score
