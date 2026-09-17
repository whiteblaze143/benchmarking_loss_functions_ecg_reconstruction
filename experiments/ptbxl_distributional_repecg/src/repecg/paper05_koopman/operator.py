from __future__ import annotations

import numpy as np


def soft_observables(means: np.ndarray, anchors: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    d2 = ((means[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=-1)
    logits = -d2 / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def median_anchor_bandwidth(means: np.ndarray, anchors: np.ndarray) -> float:
    """Training-only median squared distance to the nearest fitted anchor."""
    values = np.asarray(means, dtype=np.float64)
    centers = np.asarray(anchors, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("means and anchors must be compatible matrices")
    nearest_squared = ((values[:, None] - centers[None]) ** 2).sum(axis=-1).min(axis=1)
    tau = float(np.median(nearest_squared))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("training-median anchor bandwidth is not positive and finite")
    return tau


def koopman_operator(observables: np.ndarray, *, ridge: float = 1e-2) -> np.ndarray:
    values = np.asarray(observables, dtype=np.float64)
    if values.ndim != 2 or len(values) - 1 < values.shape[1]:
        raise ValueError("ineligible: fewer transitions than observable dimensions")
    z_minus = values[:-1].T
    z_plus = values[1:].T
    gram = z_minus @ z_minus.T + ridge * np.eye(values.shape[1])
    return np.linalg.solve(gram.T, (z_plus @ z_minus.T).T).T


def operator_descriptor(operator: np.ndarray, observables: np.ndarray) -> np.ndarray:
    values = np.asarray(observables, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(operator)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    one = np.linalg.norm(values[1:].T - operator @ values[:-1].T) / max(np.linalg.norm(values[1:]), 1e-12)
    if len(values) > 4:
        four = np.linalg.norm(values[4:].T - np.linalg.matrix_power(operator, 4) @ values[:-4].T) / max(np.linalg.norm(values[4:]), 1e-12)
    else:
        four = np.nan
    nonnormal = np.linalg.norm(operator.T @ operator - operator @ operator.T)
    return np.concatenate(
        (np.abs(eigenvalues), np.angle(eigenvalues), [np.max(np.abs(eigenvalues)), one, four, nonnormal])
    )


def chronological_permutation(length: int, record_id: int, seed: int) -> np.ndarray:
    if length < 2:
        raise ValueError("chronology permutation requires at least two states")
    sequence = np.random.SeedSequence([seed, int(record_id)])
    return np.random.default_rng(sequence).permutation(length)
