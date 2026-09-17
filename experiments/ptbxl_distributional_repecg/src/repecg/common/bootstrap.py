from __future__ import annotations

from collections.abc import Callable

import numpy as np


def clustered_paired_bootstrap(
    patient_ids: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    replicates: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    patients = np.unique(np.asarray(patient_ids, dtype=np.int64))
    rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        index = np.concatenate([rows[patient] for patient in sampled])
        draws.append(np.atleast_1d(statistic(index)))
    return np.stack(draws)


def interval(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(draws, dtype=np.float64)
    low, high = np.quantile(values, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(low), float(high)
