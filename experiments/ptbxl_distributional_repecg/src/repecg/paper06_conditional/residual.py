from __future__ import annotations

import numpy as np


def decompose_macro_residual(
    x: np.ndarray,
    train_mean: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(x, dtype=np.float64) - train_mean
    if basis.shape != (centered.shape[-1], 3):
        raise ValueError("macro basis must have shape (lead,3)")
    macro = centered @ basis
    residual = centered - macro @ basis.T
    return macro, residual


def soft_membership(macro: np.ndarray, anchors: np.ndarray, sigma2: float) -> np.ndarray:
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    d2 = ((macro[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=-1)
    logits = -d2 / sigma2
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def conditional_distance(
    feature_cells: list[np.ndarray],
    membership_cells: list[np.ndarray],
    *,
    min_effective_mass: float = 2.0,
) -> np.ndarray:
    if len(feature_cells) != len(membership_cells):
        raise ValueError("feature and membership cell counts differ")
    groups = len(feature_cells)
    states = membership_cells[0].shape[1]
    means = []
    prevalence = []
    effective = []
    for features, membership in zip(feature_cells, membership_cells, strict=True):
        if len(features) != len(membership):
            raise ValueError("cell feature and membership rows differ")
        mass = membership.sum(axis=0)
        means.append((membership.T @ features) / (mass[:, None] + 1e-8))
        prevalence.append(mass / len(membership))
        effective.append(mass**2 / (np.square(membership).sum(axis=0) + 1e-12))
    means = np.stack(means)
    prevalence = np.stack(prevalence)
    effective = np.stack(effective)
    distance = np.zeros((groups, groups), dtype=np.float64)
    for g in range(groups):
        for h in range(g + 1, groups):
            active = (effective[g] >= min_effective_mass) & (effective[h] >= min_effective_mass)
            raw = np.where(active, np.sqrt(prevalence[g] * prevalence[h]), 0.0)
            denominator = raw.sum()
            if denominator < 1e-8:
                raise ValueError(f"ineligible: no shared effective macrostate for cells {g},{h}")
            per_state = np.square(means[g] - means[h]).sum(axis=1)
            distance[g, h] = distance[h, g] = float(raw @ per_state / denominator)
    return distance
