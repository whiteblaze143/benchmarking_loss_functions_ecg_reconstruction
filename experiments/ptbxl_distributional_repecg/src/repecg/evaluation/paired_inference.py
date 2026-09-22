"""Paired inference for saved, aligned binary endpoint predictions.

This module deliberately accepts only one binary endpoint at a time.  Macro
aggregation, patient-level cohort construction, and multiplicity correction
belong to the caller because each needs an explicit, preregistered contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm


@dataclass(frozen=True)
class DeLongResult:
    """Two-sided paired DeLong comparison of model A minus model B AUROC."""

    auc_a: float
    auc_b: float
    delta_auc: float
    variance: float
    z_score: float
    p_value: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Patient-equal paired bootstrap for an already patient-collapsed endpoint."""

    observed_delta: float
    ci_lower: float
    ci_upper: float
    replicates: int
    valid_replicates: int


def _midranks(values: np.ndarray) -> np.ndarray:
    """Return one-indexed midranks, including exact-tie handling."""
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    result = np.empty(len(values), dtype=np.float64)
    result[order] = ranks
    return result


def _fast_delong(scores_positive_first: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return AUROCs and their covariance for paired score rows."""
    if scores_positive_first.ndim != 2:
        raise ValueError("scores must be [model, observation]")
    model_count, observation_count = scores_positive_first.shape
    negative_count = observation_count - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong variance requires at least two positive and two negative observations")

    positive = scores_positive_first[:, :positive_count]
    negative = scores_positive_first[:, positive_count:]
    tx = np.empty((model_count, positive_count), dtype=np.float64)
    ty = np.empty((model_count, negative_count), dtype=np.float64)
    tz = np.empty((model_count, observation_count), dtype=np.float64)
    for row in range(model_count):
        tx[row] = _midranks(positive[row])
        ty[row] = _midranks(negative[row])
        tz[row] = _midranks(scores_positive_first[row])

    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count - (positive_count + 1) / (2.0 * negative_count)
    v_positive = (tz[:, :positive_count] - tx) / negative_count
    v_negative = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = np.atleast_2d(np.cov(v_positive)) / positive_count + np.atleast_2d(np.cov(v_negative)) / negative_count
    return aucs, covariance


def paired_delong_auc(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> DeLongResult:
    """Compare two aligned score vectors with a two-sided paired DeLong test.

    Inputs must be finite one-dimensional arrays of equal length.  `y_true`
    must be binary 0/1 and both classes must occur.  There is intentionally no
    implicit row filtering, thresholding, resampling, or unpaired fallback.
    """
    labels = np.asarray(y_true)
    left = np.asarray(scores_a, dtype=np.float64)
    right = np.asarray(scores_b, dtype=np.float64)
    if labels.ndim != 1 or left.ndim != 1 or right.ndim != 1:
        raise ValueError("y_true and both score vectors must be one-dimensional")
    if not (len(labels) == len(left) == len(right)):
        raise ValueError("y_true and both score vectors must have identical length")
    if len(labels) < 2:
        raise ValueError("DeLong requires at least two observations")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain binary 0/1 labels only")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("score vectors must be finite")
    positive = labels == 1
    positive_count = int(positive.sum())
    if positive_count == 0 or positive_count == len(labels):
        raise ValueError("DeLong requires both outcome classes")

    ordering = np.concatenate((np.flatnonzero(positive), np.flatnonzero(~positive)))
    aucs, covariance = _fast_delong(np.stack((left[ordering], right[ordering])), positive_count)
    contrast = np.array((1.0, -1.0))
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[0] - aucs[1])
    if variance < -1e-14:
        raise RuntimeError(f"DeLong produced invalid negative variance {variance}")
    variance = max(variance, 0.0)
    if variance == 0.0:
        z_score = 0.0 if delta == 0.0 else float(np.copysign(np.inf, delta))
        p_value = 1.0 if delta == 0.0 else 0.0
    else:
        z_score = delta / float(np.sqrt(variance))
        p_value = float(2.0 * norm.sf(abs(z_score)))
    return DeLongResult(
        auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta_auc=delta,
        variance=variance, z_score=z_score, p_value=p_value,
    )


def patient_equal_paired_bootstrap(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    replicates: int = 2000,
    seed: int = 2026,
) -> PairedBootstrapResult:
    """Percentile CI for paired metric(A)-metric(B), sampling patients equally.

    The caller must provide a pre-specified patient-level endpoint table: each
    patient ID exactly once. This avoids silently weighting patients by their
    number of ECGs. Degenerate resamples are excluded rather than replaced by
    a fabricated metric value.
    """
    labels = np.asarray(y_true)
    left = np.asarray(scores_a, dtype=np.float64)
    right = np.asarray(scores_b, dtype=np.float64)
    ids = np.asarray(patient_ids)
    if labels.ndim != 1 or left.ndim != 1 or right.ndim != 1 or ids.ndim != 1:
        raise ValueError("labels, scores, and patient_ids must be one-dimensional")
    if not (len(labels) == len(left) == len(right) == len(ids)):
        raise ValueError("labels, scores, and patient_ids must have identical length")
    if len(labels) < 2:
        raise ValueError("bootstrap requires at least two patient-level observations")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("patient_equal_paired_bootstrap requires exactly one row per patient")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("score vectors must be finite")
    try:
        observed = float(metric(labels, left) - metric(labels, right))
    except ValueError as error:
        raise ValueError("observed patient-level endpoint is not metric-eligible") from error
    if not np.isfinite(observed):
        raise ValueError("observed patient-level endpoint is not metric-eligible")

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    count = len(labels)
    for _ in range(replicates):
        indices = rng.integers(0, count, size=count)
        if np.unique(labels[indices]).size < 2:
            continue
        try:
            delta = float(metric(labels[indices], left[indices]) - metric(labels[indices], right[indices]))
        except ValueError:
            continue
        if np.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        raise ValueError("all bootstrap resamples were metric-ineligible")
    distribution = np.asarray(deltas, dtype=np.float64)
    return PairedBootstrapResult(
        observed_delta=observed,
        ci_lower=float(np.quantile(distribution, 0.025)),
        ci_upper=float(np.quantile(distribution, 0.975)),
        replicates=replicates,
        valid_replicates=len(distribution),
    )
