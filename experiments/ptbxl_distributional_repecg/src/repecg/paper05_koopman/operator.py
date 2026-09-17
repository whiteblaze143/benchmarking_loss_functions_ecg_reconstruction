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


def koopman_operator(observables: np.ndarray, *, ridge: float = 1e-2) -> np.ndarray:
    values = np.asarray(observables, dtype=np.float64)
    if values.ndim != 2 or len(values) - 1 < values.shape[1]:
        raise ValueError("ineligible: fewer transitions than observable dimensions")
    z_minus = values[:-1].T
    z_plus = values[1:].T
    gram = z_minus @ z_minus.T + ridge * np.eye(values.shape[1])
    return z_plus @ z_minus.T @ np.linalg.inv(gram)


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
        (operator.reshape(-1), np.abs(eigenvalues), np.angle(eigenvalues), [np.max(np.abs(eigenvalues)), one, four, nonnormal])
    )
