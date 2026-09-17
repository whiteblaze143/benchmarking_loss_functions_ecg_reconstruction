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
    matrix = np.asarray(upper, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper bounds must be square")
    clusters = [tuple([i]) for i in range(len(matrix))]
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
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
        clusters.sort()
    return clusters


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
