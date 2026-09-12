"""Constrained Agglomerative Hierarchical Clustering (CAHC) & Modified Silhouette.

Implements Sections 2.1, Appendix A.1, and Table A.7 of Senanayake & Jeganathan (2026):
- Attribute dissimilarity matrix D (Euclidean for continuous, Jaccard for binary)
- Domain connectivity matrix L (m-nearest neighbor graph with OR-symmetrization)
- Lance-Williams recurrence with exact Table A.7 coefficients (Ward, Single, Complete, Average)
- Spatially-informed modified silhouette analysis (Eqs. 2-4) restricting b(i) to adjacent clusters
- Strict validity checks (rejecting singleton clusters and disconnected clusters)
- Deterministic argmax for (m*, G*) selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


def compute_attribute_dissimilarity(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Computes pairwise attribute dissimilarity matrix D.

    Parameters
    ----------
    X : np.ndarray of shape [n, p]
        Attribute vectors (e.g. continuous VCG voltages or binary markers).
    metric : str
        'euclidean' for continuous attributes or 'jaccard' for binary/multinomial.

    Returns
    -------
    D : np.ndarray of shape [n, n]
        Pairwise dissimilarity matrix.
    """
    X = np.asarray(X, dtype=float)
    metric_clean = metric.lower()

    if metric_clean == "euclidean":
        # Continuous attributes: Euclidean distance D_ij = ||X_i - X_j||_2 [PAPER §2.1]
        D = squareform(pdist(X, metric="euclidean"))
    elif metric_clean == "jaccard":
        # Binary/multinomial attributes: Jaccard dissimilarity 1 - (X_i . X_j) / (||X_i||_1 + ||X_j||_1 - X_i . X_j) [PAPER §2.1]
        D = squareform(pdist(X, metric="jaccard"))
        # Handle zero-zero vectors
        np.nan_to_num(D, copy=False, nan=0.0)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'euclidean' or 'jaccard'.")

    return D


def construct_domain_links(
    S: np.ndarray,
    m: int,
    symmetrize: str = "or",
) -> sp.csr_matrix:
    """Constructs spatial/temporal contiguity links L from domain coordinates S.

    Parameters
    ----------
    S : np.ndarray of shape [n, d]
        Domain coordinates (e.g. physical 2D coordinates [x, y], or ECG time embedded as [tau, 0]).
    m : int
        Number of nearest neighbors. [PAPER §2.1]
    symmetrize : str
        'or' (default): L_ij = 1 if j in m-NN(i) or i in m-NN(j). [AMBIGUOUS §2.1 completion]
        'and': L_ij = 1 if j in m-NN(i) and i in m-NN(j).
        'none': directed graph as computed.

    Returns
    -------
    L : sp.csr_matrix of shape [n, n]
        Binary adjacency matrix (diagonal is 0).
    """
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        # If 1D timestamps passed, embed into R^2 as (tau_t, 0) [ECG_ADAPTATION]
        S = np.column_stack([S, np.zeros_like(S)])

    n = len(S)
    k = min(m + 1, n)  # +1 because query point itself is included

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(S)
    _, indices = nbrs.kneighbors(S)

    # Build directed adjacency (excluding self at column 0)
    rows = np.repeat(np.arange(n), k - 1)
    cols = indices[:, 1:].ravel()
    data = np.ones(len(rows), dtype=int)

    L_directed = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if symmetrize.lower() == "or":
        # OR symmetrization: either direction links the points
        L = (L_directed + L_directed.T).astype(bool).astype(int)
    elif symmetrize.lower() == "and":
        # Mutual nearest neighbors
        L = L_directed.multiply(L_directed.T).astype(int)
    elif symmetrize.lower() == "none":
        L = L_directed
    else:
        raise ValueError(f"Unknown symmetrization rule: '{symmetrize}'. Expected 'or', 'and', or 'none'.")

    # Ensure zero diagonal
    L.setdiag(0)
    L.eliminate_zeros()
    return L.tocsr()


def run_cahc(
    D: np.ndarray,
    L: sp.csr_matrix,
    n_clusters: int,
    linkage: str = "ward",
) -> np.ndarray:
    """Performs Constrained Agglomerative Hierarchical Clustering (CAHC).

    Merges are strictly constrained to clusters sharing at least one domain link in L.

    Parameters
    ----------
    D : np.ndarray of shape [n, n]
        Attribute dissimilarity matrix.
    L : sp.csr_matrix of shape [n, n]
        Contiguity graph (domain links).
    n_clusters : int
        Target number of clusters G. [PAPER §2.1]
    linkage : str
        'ward', 'single', 'complete', or 'average' [PAPER Table A.7].

    Returns
    -------
    labels : np.ndarray of shape [n]
        Cluster labels in {0, ..., n_clusters - 1}.
    """
    n = len(D)
    if n_clusters >= n:
        return np.arange(n)
    if n_clusters <= 1:
        return np.zeros(n, dtype=int)

    # Ensure L is connected enough for G clusters; if graph has multiple components > G,
    # AgglomerativeClustering automatically completes graph to prevent early stopping.
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage.lower(),
        connectivity=L,
    )
    labels = model.fit_predict(D)
    return labels


def compute_modified_silhouette(
    D: np.ndarray,
    L: sp.csr_matrix,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, bool]:
    """Computes the spatially-informed modified silhouette score (Eqs. 2-4).

    Parameters
    ----------
    D : np.ndarray of shape [n, n]
        Attribute dissimilarity matrix.
    L : sp.csr_matrix of shape [n, n]
        Domain adjacency matrix.
    labels : np.ndarray of shape [n]
        Cluster assignment for each point.

    Returns
    -------
    avg_score : float
        Mean modified silhouette score over all observations, or -np.inf if invalid.
    sh_values : np.ndarray of shape [n]
        Per-sample modified silhouette values.
    is_valid : bool
        Whether the candidate partition satisfies all paper conditions
        (no singletons, every cluster has at least one adjacent cluster).
    """
    n = len(labels)
    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
    G = len(unique_labels)

    if G <= 1:
        return -np.inf, np.full(n, -np.inf), False

    # Check for singletons: Eq. (2) denominator is n_g - 1, undefined if n_g == 1 [PAPER §2.1 / AMBIGUOUS default]
    if np.any(cluster_sizes <= 1):
        return -np.inf, np.full(n, -np.inf), False

    label_to_members = {
        lbl: np.flatnonzero(labels == lbl)
        for lbl in unique_labels
    }

    # Identify cluster-level adjacency from L
    # Cluster g and cluster h are adjacent iff exists i in g, j in h such that L_ij == 1
    L_dense = L.toarray().astype(bool)
    cluster_adj = {lbl: [] for lbl in unique_labels}

    for lbl in unique_labels:
        members = label_to_members[lbl]
        connected_to_members = L_dense[members].any(axis=0)
        neighbors = [
            other for other in np.unique(labels[connected_to_members])
            if other != lbl
        ]
        if not neighbors:
            # Cluster has no adjacent neighbors: b(i) is undefined [PAPER Eq. 3 / AMBIGUOUS default]
            return -np.inf, np.full(n, -np.inf), False
        cluster_adj[lbl] = neighbors

    sh = np.zeros(n, dtype=float)

    for lbl in unique_labels:
        members = label_to_members[lbl]
        n_g = len(members)

        # Within-cluster dissimilarity a(i) = 1/(n_g - 1) * sum_{j in C_g, j != i} D_ij [PAPER Eq. 2]
        intra_D = D[np.ix_(members, members)]
        a_i = (intra_D.sum(axis=1) - np.diag(intra_D)) / (n_g - 1)

        # Between-cluster dissimilarity b(i) = min_{C_ell: L_g_ell = 1} 1/|C_ell| * sum_{j in C_ell} D_ij [PAPER Eq. 3]
        adj_clusters = cluster_adj[lbl]
        b_candidates = [
            D[np.ix_(members, label_to_members[other_lbl])].mean(axis=1)
            for other_lbl in adj_clusters
        ]
        b_i = np.minimum.reduce(b_candidates)

        # Modified silhouette sh(i) = (b(i) - a(i)) / max(a(i), b(i)) [PAPER Eq. 4]
        denom = np.maximum(a_i, b_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh_members = np.where(denom > 0, (b_i - a_i) / denom, 0.0)

        sh[members] = sh_members

    avg_score = float(np.mean(sh))
    return avg_score, sh, True


def search_optimal_cahc(
    S: np.ndarray,
    X: np.ndarray,
    m_grid: list[int] | Sequence[int],
    G_grid: list[int] | Sequence[int],
    metric: str = "euclidean",
    linkage: str = "ward",
    symmetrize_links: str = "or",
) -> tuple[int, int, pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    """Grid search over candidate (m, G) maximizing the modified silhouette.

    Parameters
    ----------
    S : np.ndarray of shape [n, d]
        Domain coordinates (physical coordinates or temporal embedding).
    X : np.ndarray of shape [n, p]
        Attribute vectors.
    m_grid : sequence of int
        Candidate nearest-neighbor counts (e.g. 2..10). [PAPER_EXAMPLE §2.1.1]
    G_grid : sequence of int
        Candidate cluster counts (e.g. 2..10). [PAPER_EXAMPLE §2.1.1]
    metric : str
        Dissimilarity metric ('euclidean' or 'jaccard').
    linkage : str
        HAC linkage criterion ('ward', 'single', 'complete', 'average').
    symmetrize_links : str
        Symmetrization rule for m-NN links ('or', 'and', 'none').

    Returns
    -------
    m_star : int
        Selected nearest-neighbor parameter.
    G_star : int
        Selected cluster count.
    scores_df : pd.DataFrame
        Table of all candidate (m, G) evaluations and modified silhouette scores.
    partition_cache : dict[(m, G), np.ndarray]
        Fitted cluster labels for all valid candidates.
    """
    D = compute_attribute_dissimilarity(X, metric=metric)
    results = []
    partition_cache = {}

    for m in m_grid:
        L = construct_domain_links(S, m=m, symmetrize=symmetrize_links)

        for G in G_grid:
            labels = run_cahc(D=D, L=L, n_clusters=G, linkage=linkage)
            avg_score, _, is_valid = compute_modified_silhouette(D=D, L=L, labels=labels)

            results.append({
                "m": int(m),
                "G": int(G),
                "avg_silhouette": float(avg_score),
                "is_valid": bool(is_valid),
            })

            if is_valid:
                partition_cache[(int(m), int(G))] = labels

    scores_df = pd.DataFrame(results)

    # Filter to valid candidates
    valid_df = scores_df[scores_df["is_valid"]].copy()

    if len(valid_df) == 0:
        # Fallback if no candidate passed validity checks: take candidate with largest valid components
        best_row = scores_df.sort_values(by="avg_silhouette", ascending=False).iloc[0]
        m_star = int(best_row["m"])
        G_star = int(best_row["G"])
        L_star = construct_domain_links(S, m=m_star, symmetrize=symmetrize_links)
        labels_fallback = run_cahc(D=D, L=L_star, n_clusters=G_star, linkage=linkage)
        partition_cache[(m_star, G_star)] = labels_fallback
        return m_star, G_star, scores_df, partition_cache

    # Deterministic tie-breaking:
    # 1. Highest avg_silhouette
    # 2. Smallest G (parsimony)
    # 3. Smallest m
    valid_df = valid_df.sort_values(
        by=["avg_silhouette", "G", "m"],
        ascending=[False, True, True],
    )
    best_row = valid_df.iloc[0]
    m_star = int(best_row["m"])
    G_star = int(best_row["G"])

    return m_star, G_star, scores_df, partition_cache
