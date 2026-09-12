"""Attribute-Based Block Permutation for Dependence-Preserving Two-Sample MMD Testing.

Implements Step 3(b) of repSpat adapted to ECG temporal autocorrelation:
- Within-episode k-means blocking: b(g) = max(1, floor(n_g / m))
- Block-level resampling preserving local temporal dependence
- Empirical null distribution generation and p-value computation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .mmd import compute_mmd_sq


def create_feature_blocks(
    X: np.ndarray,
    labels: np.ndarray,
    m_neighbors: int = 4,
    random_state: int = 42,
) -> np.ndarray:
    """Partitions observations within each episode into feature-space blocks.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix [n, p].
    labels : np.ndarray
        Episode labels [n].
    m_neighbors : int
        Temporal neighborhood parameter m.
    random_state : int
        Random seed for k-means.

    Returns
    -------
    block_ids : np.ndarray
        Unique block identifier for each observation [n].
    """
    n = len(X)
    block_ids = np.zeros(n, dtype=int)
    current_block_id = 1

    for label in np.unique(labels):
        mask = labels == label
        indices = np.flatnonzero(mask)
        n_g = len(indices)
        if n_g == 0:
            continue

        num_blocks = max(1, n_g // max(1, m_neighbors))
        # Ensure num_blocks does not exceed unique samples
        unique_samples = len(np.unique(X[indices], axis=0))
        num_blocks = min(num_blocks, unique_samples)

        if num_blocks <= 1:
            block_ids[indices] = current_block_id
            current_block_id += 1
        else:
            km = KMeans(n_clusters=num_blocks, n_init=10, random_state=random_state)
            cluster_assignments = km.fit_predict(X[indices])
            block_ids[indices] = current_block_id + cluster_assignments
            current_block_id += num_blocks

    return block_ids


def permute_blocks_once(
    idx_g: np.ndarray,
    idx_h: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates one block-permuted partition of observations.

    Parameters
    ----------
    idx_g : np.ndarray
        Indices belonging to episode g.
    idx_h : np.ndarray
        Indices belonging to episode h.
    block_ids : np.ndarray
        Global block IDs for each observation.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    perm_idx_1 : np.ndarray
        Sampled indices for first permuted group.
    perm_idx_2 : np.ndarray
        Remaining indices for second permuted group.
    """
    # Order so group 1 is smaller
    if len(idx_g) <= len(idx_h):
        smaller_idx, larger_idx = idx_g, idx_h
    else:
        smaller_idx, larger_idx = idx_h, idx_g

    n1_target = len(smaller_idx)
    all_idx = np.concatenate([smaller_idx, larger_idx])
    sub_block_ids = block_ids[all_idx]

    unique_blocks = np.unique(sub_block_ids)
    rng.shuffle(unique_blocks)

    sampled_blocks = []
    n_accum = 0

    for blk in unique_blocks:
        sampled_blocks.append(blk)
        n_accum += int(np.sum(sub_block_ids == blk))
        if n_accum >= n1_target:
            break

    sampled_blocks_set = set(sampled_blocks)
    is_in_sampled = np.isin(sub_block_ids, list(sampled_blocks_set))

    perm_idx_1 = all_idx[is_in_sampled]
    perm_idx_2 = all_idx[~is_in_sampled]

    return perm_idx_1, perm_idx_2


def two_sample_block_permutation_test(
    idx_g: np.ndarray | list[int],
    idx_h: np.ndarray | list[int],
    dist_matrix: np.ndarray,
    block_ids: np.ndarray,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
    n_permutations: int = 200,
    random_state: int = 42,
) -> dict:
    """Performs block-permutation two-sample MMD^2 test between two temporal episodes.

    Parameters
    ----------
    idx_g : array-like
        Indices of episode g.
    idx_h : array-like
        Indices of episode h.
    dist_matrix : np.ndarray
        Precomputed pairwise attribute distance matrix D [n, n].
    block_ids : np.ndarray
        Array [n] of block assignments.
    kernel : str
        Kernel function ('IMQ' or 'Gaussian').
    kernel_param : float
        Kernel parameter c or sigma.
    n_permutations : int
        Number of permutations B (default: 200).
    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        {
            'obs_mmd_sq': float,
            'p_value': float,
            'null_distribution': np.ndarray,
        }
    """
    idx_g = np.asarray(idx_g, dtype=int)
    idx_h = np.asarray(idx_h, dtype=int)

    obs_mmd_sq = compute_mmd_sq(
        idx_g, idx_h, dist_matrix, kernel=kernel, kernel_param=kernel_param
    )

    rng = np.random.default_rng(random_state)
    null_dist = np.zeros(n_permutations, dtype=float)

    for b in range(n_permutations):
        perm_1, perm_2 = permute_blocks_once(idx_g, idx_h, block_ids, rng=rng)
        if len(perm_1) == 0 or len(perm_2) == 0:
            null_dist[b] = 0.0
            continue
        null_dist[b] = compute_mmd_sq(
            perm_1, perm_2, dist_matrix, kernel=kernel, kernel_param=kernel_param
        )

    # Empirical p-value
    p_val = float(np.mean(null_dist >= obs_mmd_sq))

    return {
        "obs_mmd_sq": obs_mmd_sq,
        "p_value": p_val,
        "null_distribution": null_dist,
    }
