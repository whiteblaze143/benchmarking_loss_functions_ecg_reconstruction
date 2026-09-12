"""Attribute-based block permutation testing for empirical MMD^2 null distribution.

Implements Step 3(b) of repSpat:
- Partitions observations within each cluster into feature-space blocks via k-means:
  b(g) = max(1, floor(n_g / m))
- Block permutation preserves local autocorrelation structure under H_0: P(g) = P(h).
- Shuffles blocks without replacement B times to construct the empirical null distribution.
- Computes empirical two-sample p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .mmd import compute_mmd2


def partition_into_blocks(
    X_cluster: np.ndarray,
    m_neighbors: int = 4,
    random_state: int = 42,
) -> list[np.ndarray]:
    """Partitions observations within a cluster into feature-space blocks via k-means.

    b = max(1, floor(n / m))

    Parameters
    ----------
    X_cluster : np.ndarray
        Feature matrix for one cluster [n, p].
    m_neighbors : int
        Spatial neighborhood size m.
    random_state : int
        Random seed for k-means reproducibility.

    Returns
    -------
    blocks : list of np.ndarray
        List of feature subsets representing each block.
    """
    n = len(X_cluster)
    n_blocks = max(1, int(np.floor(n / float(max(1, m_neighbors)))))
    n_blocks = min(n_blocks, n)

    if n_blocks <= 1:
        return [X_cluster]

    km = KMeans(n_clusters=n_blocks, random_state=random_state, n_init=1)
    block_labels = km.fit_predict(X_cluster)

    blocks = [X_cluster[block_labels == k] for k in range(n_blocks)]
    # Filter any empty blocks
    return [b for b in blocks if len(b) > 0]


def block_permutation_test(
    X_g: np.ndarray,
    X_h: np.ndarray,
    m_neighbors: int = 4,
    kernel: str = "imq",
    kernel_param: float = 1.0,
    n_permutations: int = 200,
    random_state: int = 42,
) -> tuple[float, float, list[float]]:
    """Performs attribute-based block permutation two-sample test comparing P(g) vs P(h).

    Parameters
    ----------
    X_g : np.ndarray
        Features of cluster g [n_g, p].
    X_h : np.ndarray
        Features of cluster h [n_h, p].
    m_neighbors : int
        Spatial neighborhood size m.
    kernel : str
        'imq' or 'rbf'.
    kernel_param : float
        c or gamma.
    n_permutations : int
        Number of permutations B (default: 200).
    random_state : int
        Random seed.

    Returns
    -------
    obs_stat : float
        Observed MMD^2 statistic.
    p_val : float
        Empirical p-value under the permutation null.
    null_dist : list of float
        Permuted MMD^2 values [B].
    """
    X_g = np.asarray(X_g, dtype=np.float64)
    X_h = np.asarray(X_h, dtype=np.float64)

    obs_stat = compute_mmd2(X_g, X_h, kernel=kernel, kernel_param=kernel_param)

    n_g, n_h = len(X_g), len(X_h)
    if n_g == 0 or n_h == 0:
        return obs_stat, 1.0, []

    # 1. Feature-space k-means blocking
    blocks_g = partition_into_blocks(X_g, m_neighbors=m_neighbors, random_state=random_state)
    blocks_h = partition_into_blocks(X_h, m_neighbors=m_neighbors, random_state=random_state + 1)
    all_blocks = blocks_g + blocks_h
    n_blocks = len(all_blocks)

    if n_blocks <= 1:
        return obs_stat, 1.0, [obs_stat]

    n_target = min(n_g, n_h)
    rng = np.random.default_rng(random_state)
    null_dist: list[float] = []

    for b in range(n_permutations):
        # Permute block indices
        perm_indices = rng.permutation(n_blocks)

        group1_blocks = []
        count = 0
        split_idx = 0
        for idx in perm_indices:
            blk = all_blocks[idx]
            group1_blocks.append(blk)
            count += len(blk)
            split_idx += 1
            if count >= n_target:
                break

        group2_blocks = [all_blocks[idx] for idx in perm_indices[split_idx:]]

        if not group2_blocks:
            # Fallback if all blocks taken into group 1
            perm_stat = 0.0
        else:
            X_perm1 = np.vstack(group1_blocks)
            X_perm2 = np.vstack(group2_blocks)
            perm_stat = compute_mmd2(X_perm1, X_perm2, kernel=kernel, kernel_param=kernel_param)

        null_dist.append(perm_stat)

    # Empirical p-value
    null_arr = np.array(null_dist)
    p_val = float(np.mean(null_arr >= obs_stat))

    return obs_stat, p_val, null_dist


def pairwise_cluster_testing(
    X: np.ndarray,
    labels: np.ndarray,
    m_neighbors: int = 4,
    kernel: str = "imq",
    kernel_param: float = 1.0,
    n_permutations: int = 200,
    random_state: int = 42,
    K_gram: np.ndarray | None = None,
) -> pd.DataFrame:
    """Executes pairwise MMD^2 block permutation tests across all cluster pairs.

    Accelerated via precomputed kernel Gram matrix slicing and single-pass
    feature-space k-means block partitioning (faithful to repSpat Steps 3a & 3b).

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix [n, p].
    labels : np.ndarray
        Cluster labels [n].
    m_neighbors : int
        Spatial neighborhood size.
    kernel : str
        'imq' or 'rbf'.
    kernel_param : float
        c or gamma.
    n_permutations : int
        Permutations per pair.
    random_state : int
        Random seed.
    K_gram : np.ndarray, optional
        Precomputed kernel Gram matrix [n, n]. If None, computed once on X.

    Returns
    -------
    results_df : pd.DataFrame
        Table with columns: ['cluster_1', 'cluster_2', 'n_1', 'n_2', 'obs_mmd_sq', 'p_val'].
    """
    X_arr = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_clusters = sorted(np.unique(labels))
    K = len(unique_clusters)

    # 1. Precompute full kernel Gram matrix ONCE (matching author repspat design)
    if K_gram is None:
        if kernel == "imq":
            from .mmd import imq_kernel
            K_gram = imq_kernel(X_arr, c=kernel_param)
        else:
            from .mmd import rbf_kernel
            K_gram = rbf_kernel(X_arr, gamma=kernel_param)

    # 2. Partition each unique cluster into feature-space blocks ONCE
    cluster_indices = {}
    cluster_blocks = {}
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        cluster_indices[c] = idx
        n_c = len(idx)
        n_blocks = max(1, int(np.floor(n_c / float(max(1, m_neighbors)))))
        n_blocks = min(n_blocks, n_c)
        if n_blocks <= 1:
            cluster_blocks[c] = [idx]
        else:
            km = KMeans(n_clusters=n_blocks, random_state=random_state, n_init=1)
            blk_lbls = km.fit_predict(X_arr[idx])
            cluster_blocks[c] = [idx[blk_lbls == b] for b in range(n_blocks) if (blk_lbls == b).any()]

    # 3. Pairwise testing using precomputed Gram matrix submatrices
    records = []
    pair_count = 0
    for i in range(K):
        for j in range(i + 1, K):
            c1, c2 = unique_clusters[i], unique_clusters[j]
            idx1, idx2 = cluster_indices[c1], cluster_indices[c2]
            n1, n2 = len(idx1), len(idx2)

            if n1 == 0 or n2 == 0:
                records.append({
                    "cluster_1": c1,
                    "cluster_2": c2,
                    "n_1": n1,
                    "n_2": n2,
                    "obs_mmd_sq": 0.0,
                    "p_val": 1.0,
                })
                pair_count += 1
                continue

            # Observed MMD^2 via Gram submatrices
            term1 = K_gram[np.ix_(idx1, idx1)].sum() / (n1 * n1)
            term2 = -2.0 * K_gram[np.ix_(idx1, idx2)].sum() / (n1 * n2)
            term3 = K_gram[np.ix_(idx2, idx2)].sum() / (n2 * n2)
            obs_stat = float(max(0.0, term1 + term2 + term3))

            # Block permutation testing
            blocks_all = cluster_blocks[c1] + cluster_blocks[c2]
            n_blocks = len(blocks_all)
            if n_blocks <= 1:
                records.append({
                    "cluster_1": c1,
                    "cluster_2": c2,
                    "n_1": n1,
                    "n_2": n2,
                    "obs_mmd_sq": obs_stat,
                    "p_val": 1.0,
                })
                pair_count += 1
                continue

            target_n = min(n1, n2)
            rng = np.random.default_rng(random_state + pair_count)
            null_dist: list[float] = []

            for b in range(n_permutations):
                perm = rng.permutation(n_blocks)
                g1_idx: list[np.ndarray] = []
                count = 0
                split_idx = 0
                for k in perm:
                    blk = blocks_all[k]
                    g1_idx.append(blk)
                    count += len(blk)
                    split_idx += 1
                    if count >= target_n:
                        break

                g2_blocks = [blocks_all[k] for k in perm[split_idx:]]
                if not g2_blocks:
                    null_dist.append(0.0)
                else:
                    p1 = np.concatenate(g1_idx)
                    p2 = np.concatenate(g2_blocks)
                    np1, np2 = len(p1), len(p2)
                    p_term1 = K_gram[np.ix_(p1, p1)].sum() / (np1 * np1)
                    p_term2 = -2.0 * K_gram[np.ix_(p1, p2)].sum() / (np1 * np2)
                    p_term3 = K_gram[np.ix_(p2, p2)].sum() / (np2 * np2)
                    null_dist.append(float(max(0.0, p_term1 + p_term2 + p_term3)))

            null_arr = np.array(null_dist)
            p_val = float(np.mean(null_arr >= obs_stat))

            records.append({
                "cluster_1": c1,
                "cluster_2": c2,
                "n_1": n1,
                "n_2": n2,
                "obs_mmd_sq": obs_stat,
                "p_val": p_val,
            })
            pair_count += 1

    return pd.DataFrame(records)

