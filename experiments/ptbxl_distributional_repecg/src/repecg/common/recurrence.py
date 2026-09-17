from __future__ import annotations

import numpy as np


def recurrence_operator(
    means: np.ndarray,
    tau: float,
    *,
    local_exclusion: int = 1,
) -> np.ndarray:
    values = np.asarray(means, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("means must have shape (phase,feature)")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    groups = len(values)
    d2 = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=-1)
    affinity = np.exp(-d2 / tau)
    index = np.arange(groups)
    distance = np.abs(index[:, None] - index[None, :])
    cyclic = np.minimum(distance, groups - distance)
    affinity[cyclic <= local_exclusion] = 0.0
    degree = np.maximum(affinity.sum(axis=1), 1e-8)
    inverse = degree ** -0.5
    return inverse[:, None] * affinity * inverse[None, :]


def upper_triangle(operator: np.ndarray) -> np.ndarray:
    values = np.asarray(operator)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("operator must be square")
    return values[np.triu_indices(values.shape[0], k=1)]
