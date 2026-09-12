"""Pairwise Biased MMD^2, Attribute k-Means Block Permutation, and Multiple Testing.

Implements Sections 2.2, 2.3, and Appendix A.2 of Senanayake & Jeganathan (2026):
- Biased empirical V-statistic MMD^2 (Eq. 6) including diagonal terms
- IMQ kernel k(x, y) = (||x-y||^2 + c^2)^{-1/2} [PAPER Table 1]
- Attribute k-means blocking within each CAHC cluster (Eq. A.1)
- Strict '>' whole-block permutation resampling without replacement (Appendix A.2)
- Upper-tail Monte-Carlo p-value with (R+1)/(B+1) correction (Phipson & Smyth, 2010)
- Benjamini-Hochberg FDR correction across all G*(G-1)/2 pairwise tests.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests


def apply_kernel(
    dist_sq: np.ndarray,
    kernel: str = "imq",
    kernel_param: float = 1.0,
) -> np.ndarray:
    """Computes kernel Gram matrix from squared Euclidean distance matrix.

    Parameters
    ----------
    dist_sq : np.ndarray
        Squared Euclidean distance matrix ||x_i - x_j||_2^2.
    kernel : str
        'imq' for Inverse Multiquadric, or 'gaussian' for RBF [PAPER Table 1].
    kernel_param : float
        Parameter c for IMQ (c^2 in denominator), or sigma for Gaussian.

    Returns
    -------
    K : np.ndarray
        Gram matrix.
    """
    kernel_clean = kernel.lower()
    if kernel_clean == "imq":
        # IMQ: k(x,y) = 1 / sqrt(||x-y||^2 + c^2) [PAPER Table 1]
        c_sq = float(kernel_param) ** 2
        return 1.0 / np.sqrt(dist_sq + c_sq)
    elif kernel_clean in ("gaussian", "rbf"):
        # Gaussian: k(x,y) = exp(-||x-y||^2 / (2 * sigma^2)) or exp(-||x-y||^2 / sigma^2)
        # Paper Table 1 writes exp(-||x-y||_2^2 / sigma^2)
        scale = float(kernel_param) ** 2
        return np.exp(-dist_sq / scale)
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Expected 'imq' or 'gaussian'.")


def compute_biased_mmd2_from_dist(
    idx1: np.ndarray | list[int],
    idx2: np.ndarray | list[int],
    dist_matrix: np.ndarray,
    kernel: str = "imq",
    kernel_param: float = 1.0,
) -> float:
    """Computes biased empirical MMD^2 (V-statistic) from precomputed distance matrix.

    Follows Eq. (6) exactly, including diagonal (i=j) terms.

    Parameters
    ----------
    idx1 : array-like
        Indices of first sample.
    idx2 : array-like
        Indices of second sample.
    dist_matrix : np.ndarray
        Global Euclidean distance matrix.
    kernel : str
        'imq' or 'gaussian'.
    kernel_param : float
        Kernel parameter c or sigma.

    Returns
    -------
    mmd2 : float
        Biased empirical MMD^2 V-statistic.
    """
    idx1 = np.asarray(idx1, dtype=int)
    idx2 = np.asarray(idx2, dtype=int)
    N = len(idx1)
    M = len(idx2)

    if N == 0 or M == 0:
        return np.nan

    D_xx_sq = dist_matrix[np.ix_(idx1, idx1)] ** 2
    D_xy_sq = dist_matrix[np.ix_(idx1, idx2)] ** 2
    D_yy_sq = dist_matrix[np.ix_(idx2, idx2)] ** 2

    K_xx = apply_kernel(D_xx_sq, kernel=kernel, kernel_param=kernel_param)
    K_xy = apply_kernel(D_xy_sq, kernel=kernel, kernel_param=kernel_param)
    K_yy = apply_kernel(D_yy_sq, kernel=kernel, kernel_param=kernel_param)

    # Eq. (6): 1/N^2 sum(K_xx) - 2/(N*M) sum(K_xy) + 1/M^2 sum(K_yy) [PAPER Eq. 6]
    term1 = K_xx.sum() / (N * N)
    term2 = -2.0 * K_xy.sum() / (N * M)
    term3 = K_yy.sum() / (M * M)

    mmd2 = float(term1 + term2 + term3)
    return mmd2


def create_attribute_blocks(
    X: np.ndarray,
    labels: np.ndarray,
    m_star: int,
    rounding_rule: str = "nearest",
    n_init: int = 50,
    random_state: int = 42,
) -> dict[int, list[np.ndarray]]:
    """Partitions each CAHC cluster separately into attribute k-means blocks.

    Parameters
    ----------
    X : np.ndarray of shape [n, p]
        Attribute vectors. [PAPER §2.3]
    labels : np.ndarray of shape [n]
        Initial CAHC cluster labels.
    m_star : int
        Selected nearest-neighbor count.
    rounding_rule : str
        'nearest': b_g = max(1, floor(n_g / m_star + 0.5)) [AMBIGUOUS completion]
        'floor': b_g = max(1, n_g // m_star)
        'ceil': b_g = max(1, int(np.ceil(n_g / m_star)))
    n_init : int
        Number of k-means initializations (retaining min SSE).
    random_state : int
        Fixed random seed.

    Returns
    -------
    cluster_blocks : dict[int, list[np.ndarray]]
        Map from cluster label g to list of block member index arrays.
    """
    unique_labels = np.unique(labels)
    cluster_blocks = {}

    for g in unique_labels:
        members = np.flatnonzero(labels == g)
        n_g = len(members)

        # Paper Eq. (A.1) & Appendix A.2: If m > n_g, use one block.
        if m_star > n_g:
            num_blocks = 1
        else:
            if rounding_rule == "nearest":
                num_blocks = max(1, int(np.floor(n_g / m_star + 0.5)))
            elif rounding_rule == "floor":
                num_blocks = max(1, n_g // m_star)
            elif rounding_rule == "ceil":
                num_blocks = max(1, int(np.ceil(n_g / m_star)))
            else:
                num_blocks = max(1, int(np.round(n_g / m_star)))

        # Do not request more clusters than distinct attribute rows
        distinct_rows = len(np.unique(X[members], axis=0))
        num_blocks = min(num_blocks, max(1, distinct_rows))

        if num_blocks <= 1:
            cluster_blocks[g] = [members]
        else:
            km = KMeans(
                n_clusters=num_blocks,
                n_init=n_init,
                random_state=random_state + int(g),
            )
            km_labels = km.fit_predict(X[members])
            blocks_g = [
                members[km_labels == b]
                for b in range(num_blocks)
                if np.any(km_labels == b)
            ]
            cluster_blocks[g] = blocks_g

    return cluster_blocks


def run_block_permutation_test(
    g: int,
    h: int,
    blocks_g: list[np.ndarray],
    blocks_h: list[np.ndarray],
    dist_matrix: np.ndarray,
    kernel: str = "imq",
    kernel_param: float = 1.0,
    n_permutations: int = 9999,
    strict_exceed: bool = True,
    p_value_correction: bool = True,
    random_state: int = 42,
) -> dict:
    """Performs whole-block permutation test between cluster g and cluster h.

    Parameters
    ----------
    g, h : int
        Cluster identifiers.
    blocks_g : list of np.ndarray
        Blocks of cluster g.
    blocks_h : list of np.ndarray
        Blocks of cluster h.
    dist_matrix : np.ndarray
        Global attribute distance matrix.
    kernel : str
        'imq' or 'gaussian'.
    kernel_param : float
        Kernel parameter.
    n_permutations : int
        Number of permutations B (e.g. 9999).
    strict_exceed : bool
        If True, stops selection strictly when selected count > n_min (Appendix A.2).
    p_value_correction : bool
        If True, uses (R + 1) / (B + 1) formula.
    random_state : int
        Seed for deterministic resampling.

    Returns
    -------
    result : dict
        Observed MMD^2, empirical p-value, and permutation distribution summary.
    """
    idx_g = np.concatenate(blocks_g)
    idx_h = np.concatenate(blocks_h)
    n_g, n_h = len(idx_g), len(idx_h)

    # Observed biased MMD^2 [PAPER Eq. 6]
    T_obs = compute_biased_mmd2_from_dist(
        idx_g, idx_h, dist_matrix, kernel=kernel, kernel_param=kernel_param
    )

    # Pool blocks from both clusters: B_g U B_h [PAPER Appendix A.2]
    all_blocks = blocks_g + blocks_h
    n_blocks = len(all_blocks)
    block_sizes = np.array([len(b) for b in all_blocks], dtype=int)
    total_obs = len(idx_g) + len(idx_h)

    # Smaller original cluster size [PAPER Appendix A.2]
    # If equal size, cluster with lower ID is defined as smaller [AMBIGUOUS completion]
    n_min = min(n_g, n_h)

    rng = np.random.RandomState(random_state + (g * 1000 + h))
    null_dist = np.zeros(n_permutations, dtype=float)

    valid_perms = 0
    max_attempts = n_permutations * 3
    attempt = 0

    while valid_perms < n_permutations and attempt < max_attempts:
        attempt += 1
        perm_order = rng.permutation(n_blocks)

        selected_block_indices = []
        selected_n = 0

        # Appendix A.2: Continue selecting until observations in selected blocks EXCEED n_min.
        for b_idx in perm_order:
            selected_block_indices.append(b_idx)
            selected_n += block_sizes[b_idx]
            if strict_exceed:
                if selected_n > n_min:  # Strict > [PAPER Appendix A.2]
                    break
            else:
                if selected_n >= n_min:
                    break

        # Selected blocks -> pseudo-small cluster; remaining blocks -> other pseudo-group
        remaining_block_indices = [
            b_idx for b_idx in range(n_blocks) if b_idx not in selected_block_indices
        ]

        if not remaining_block_indices or not selected_block_indices:
            # Pathological case where all blocks were selected; redraw [AMBIGUOUS completion]
            continue

        pseudo_small = np.concatenate([all_blocks[b] for b in selected_block_indices])
        pseudo_large = np.concatenate([all_blocks[b] for b in remaining_block_indices])

        if len(pseudo_small) == 0 or len(pseudo_large) == 0:
            continue

        T_perm = compute_biased_mmd2_from_dist(
            pseudo_small, pseudo_large, dist_matrix, kernel=kernel, kernel_param=kernel_param
        )
        null_dist[valid_perms] = T_perm
        valid_perms += 1

    if valid_perms < n_permutations:
        null_dist = null_dist[:valid_perms]

    # Upper-tail count: R = sum 1(T_perm >= T_obs) [PAPER §2.3 / AMBIGUOUS formula]
    R = int(np.sum(null_dist >= T_obs))
    B = len(null_dist)

    if p_value_correction:
        # Phipson & Smyth (2010): p = (R + 1) / (B + 1)
        p_val = float(R + 1) / float(B + 1)
    else:
        p_val = float(R) / float(B) if B > 0 else 1.0

    return {
        "cluster_1": int(g),
        "cluster_2": int(h),
        "obs_mmd2": float(T_obs),
        "p_value": float(p_val),
        "perm_count": int(B),
        "r_count": int(R),
        "null_mean": float(np.mean(null_dist)) if len(null_dist) > 0 else np.nan,
        "null_std": float(np.std(null_dist)) if len(null_dist) > 0 else np.nan,
    }


def run_all_pairwise_mmd_tests(
    X: np.ndarray,
    labels: np.ndarray,
    dist_matrix: np.ndarray,
    m_star: int,
    kernel: str = "imq",
    kernel_param: float = 1.0,
    n_permutations: int = 9999,
    fdr_alpha: float = 0.05,
    rounding_rule: str = "nearest",
    strict_exceed: bool = True,
    kmeans_n_init: int = 50,
    p_value_correction: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Runs pairwise MMD^2 block permutation tests across all unique cluster pairs.

    Applies Benjamini-Hochberg FDR correction across all G*(G-1)/2 tests.

    Parameters
    ----------
    X : np.ndarray
        Attributes [n, p].
    labels : np.ndarray
        Initial CAHC labels.
    dist_matrix : np.ndarray
        Pairwise attribute dissimilarity matrix D.
    m_star : int
        Selected m nearest neighbors.
    kernel : str
        'imq' or 'gaussian'.
    kernel_param : float
        Kernel parameter c or sigma.
    n_permutations : int
        Number of block permutations B.
    fdr_alpha : float
        FDR threshold (0.05).
    rounding_rule : str
        Rounding rule for block counts.
    strict_exceed : bool
        Strict '>' stopping criterion in permutation.
    kmeans_n_init : int
        Number of initializations for k-means.
    p_value_correction : bool
        (R+1)/(B+1) Monte-Carlo p-value formula.
    random_state : int
        Random seed.

    Returns
    -------
    results_df : pd.DataFrame
        Table of all tested pairs with observed MMD^2, raw p-value, and BH-adjusted q-value.
    """
    unique_clusters = sorted(np.unique(labels))
    G = len(unique_clusters)

    # 1. Precompute k-means blocks for each CAHC cluster once [Appendix A.2]
    cluster_blocks = create_attribute_blocks(
        X=X,
        labels=labels,
        m_star=m_star,
        rounding_rule=rounding_rule,
        n_init=kmeans_n_init,
        random_state=random_state,
    )

    pair_records = []
    # Loop over all distinct unordered pairs: G*(G-1)/2 tests [PAPER §2.2]
    for g, h in combinations(unique_clusters, 2):
        res = run_block_permutation_test(
            g=g,
            h=h,
            blocks_g=cluster_blocks[g],
            blocks_h=cluster_blocks[h],
            dist_matrix=dist_matrix,
            kernel=kernel,
            kernel_param=kernel_param,
            n_permutations=n_permutations,
            strict_exceed=strict_exceed,
            p_value_correction=p_value_correction,
            random_state=random_state,
        )
        pair_records.append(res)

    results_df = pd.DataFrame(pair_records)

    if len(results_df) > 0:
        # Benjamini-Hochberg FDR correction across ALL distinct pairs [PAPER §2.3]
        reject, q_values, _, _ = multipletests(
            results_df["p_value"], alpha=fdr_alpha, method="fdr_bh"
        )
        results_df["q_value"] = q_values
        results_df["rejected"] = reject
        # Retain edge in similarity graph if NOT rejected (q > alpha) [PAPER §2.4]
        results_df["is_similar_edge"] = ~reject
    else:
        results_df["q_value"] = []
        results_df["rejected"] = []
        results_df["is_similar_edge"] = []

    return results_df
