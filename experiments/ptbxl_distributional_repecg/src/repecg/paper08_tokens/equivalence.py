from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def simultaneous_upper_bounds(point: np.ndarray, bootstrap: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    estimate = np.asarray(point, dtype=np.float64)
    draws = np.asarray(bootstrap, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != len(estimate):
        raise ValueError("bootstrap must have shape (replicate,pair)")
    maximum_error = np.max(estimate[None, :] - draws, axis=1)
    critical = np.quantile(maximum_error, 1.0 - alpha)
    return estimate + critical


def complete_linkage_merge(upper: np.ndarray, delta: float) -> list[tuple[int, ...]]:
    clusters, _ = complete_linkage_merge_with_trace(upper, delta)
    return clusters


def complete_linkage_merge_with_trace(
    upper: np.ndarray, delta: float
) -> tuple[list[tuple[int, ...]], list[dict[str, object]]]:
    matrix = np.asarray(upper, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper bounds must be square")
    clusters = [tuple([i]) for i in range(len(matrix))]
    trace: list[dict[str, object]] = []
    while True:
        candidates = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cross = matrix[np.ix_(clusters[i], clusters[j])]
                maximum = float(cross.max())
                if maximum < delta:
                    candidates.append((maximum, clusters[i], clusters[j], i, j))
        if not candidates:
            break
        _, left, right, i, j = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        merged = tuple(sorted(left + right))
        trace.append({
            "left": list(left), "right": list(right), "merged": list(merged),
            "maximum_upper_bound": float(matrix[np.ix_(left, right)].max()),
        })
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
        clusters.sort()
    return clusters, trace


def odd_even_agreement(odd_counts: np.ndarray, even_counts: np.ndarray) -> float:
    odd = np.asarray(odd_counts, dtype=np.float64)
    even = np.asarray(even_counts, dtype=np.float64)
    if odd.shape != even.shape or odd.ndim != 1:
        raise ValueError("token counts must be matching vectors")
    if odd.sum() <= 0 or even.sum() <= 0:
        raise ValueError("token counts must be nonempty")
    odd /= odd.sum()
    even /= even.sum()
    return float(1.0 - jensenshannon(odd, even, base=np.e) ** 2 / np.log(2.0))


def pairwise_phase_distances(representations: np.ndarray) -> np.ndarray:
    """
    Computes average pairwise Euclidean distances between phase cells across records.
    Input shape: (N, num_phases, feature_dim).
    Output shape: (num_phases, num_phases).
    """
    data = np.asarray(representations, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("representations must have shape (N, num_phases, feature_dim)")
    N, P, D = data.shape
    dist_matrix = np.zeros((P, P), dtype=np.float64)
    for i in range(P):
        for j in range(i + 1, P):
            diff = data[:, i, :] - data[:, j, :]
            dist = np.linalg.norm(diff, axis=-1).mean()
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def bootstrap_pairwise_phase_distances(
    representations: np.ndarray,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes sample pairwise distances and bootstrap replicates for simultaneous bounds.
    Returns:
        point_pairs: (P*(P-1)//2,) vector of upper-triangular pairwise distances.
        bootstrap_pairs: (n_bootstraps, P*(P-1)//2) matrix of bootstrap draws.
    """
    data = np.asarray(representations, dtype=np.float64)
    N, P, D = data.shape
    rng = np.random.default_rng(seed)
    
    triu_idx = np.triu_indices(P, k=1)
    
    # Point estimate
    sample_mat = pairwise_phase_distances(data)
    point_pairs = sample_mat[triu_idx]
    
    # Bootstrap replicates
    bootstrap_pairs = np.zeros((n_bootstraps, len(point_pairs)), dtype=np.float64)
    for b in range(n_bootstraps):
        idx = rng.integers(0, N, size=N)
        boot_sample = data[idx]
        boot_mat = pairwise_phase_distances(boot_sample)
        bootstrap_pairs[b] = boot_mat[triu_idx]
        
    return point_pairs, bootstrap_pairs
