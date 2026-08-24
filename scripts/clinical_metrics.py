"""Leakage-safe per-task metrics for binary and multilabel ECG classifiers.

All thresholded metrics use an externally specified threshold (0.5 by
default). This module deliberately does not optimize thresholds on test data.
Undefined threshold-free metrics are represented as ``None`` rather than NaN
so downstream JSON remains standards compliant.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _finite_float(value: float | np.floating[Any]) -> float | None:
    output = float(value)
    return output if np.isfinite(output) else None


def _safe_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def binary_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | int | None]:
    """Return Brier/log-loss and equal-width ECE/MCE for one binary task."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    valid = np.isfinite(probabilities) & np.isfinite(labels)
    probabilities = np.clip(probabilities[valid], 1e-7, 1.0 - 1e-7)
    labels = labels[valid].astype(np.int64)
    if probabilities.size == 0:
        return {"n": 0, "brier": None, "log_loss": None, "ece": None, "mce": None}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    mce = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        gap = abs(float(probabilities[mask].mean() - labels[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return {
        "n": int(probabilities.size),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": float(ece),
        "mce": float(mce),
    }


def binary_task_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
    threshold_source: str = "fixed_predefined",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute extensive metrics for a single real-ground-truth binary task."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probabilities.shape != labels.shape:
        raise ValueError("Binary probability and label arrays must have identical shapes")
    valid = np.isfinite(probabilities) & np.isfinite(labels)
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    labels = labels[valid]
    if labels.size and not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Binary labels must contain only 0/1 values")
    labels = labels.astype(np.int64)

    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = (
        confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        if labels.size
        else (0, 0, 0, 0)
    )
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    both_classes = positives > 0 and negatives > 0
    calibration = binary_calibration_metrics(probabilities, labels, n_bins=n_bins)
    precision = (
        float(precision_score(labels, predictions, zero_division=0)) if labels.size else None
    )
    recall = float(recall_score(labels, predictions, zero_division=0)) if labels.size else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    npv = float(tn / (tn + fn)) if tn + fn else None

    return {
        "status": "ok" if both_classes else ("single_class" if labels.size else "empty"),
        "n": int(labels.size),
        "support_positive": positives,
        "support_negative": negatives,
        "prevalence": float(positives / labels.size) if labels.size else None,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, probabilities)) if both_classes else None
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)) if labels.size else None,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "accuracy": float(accuracy_score(labels, predictions)) if labels.size else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions)) if both_classes else None
        ),
        "brier": calibration["brier"],
        "log_loss": calibration["log_loss"],
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def multilabel_classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    thresholds: float | Sequence[float] = 0.5,
    threshold_source: str = "fixed_predefined",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute macro, micro, and fully enumerated per-task metrics."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != labels.shape:
        raise ValueError("Multilabel probabilities and labels must be equal [N,T] arrays")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Task-name count does not match probability columns")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Task names must be unique")
    if np.isscalar(thresholds):
        threshold_values = np.full(probabilities.shape[1], float(thresholds))
    else:
        threshold_values = np.asarray(thresholds, dtype=np.float64)
        if threshold_values.shape != (probabilities.shape[1],):
            raise ValueError("Thresholds must be scalar or one value per task")

    per_task: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        task = binary_task_metrics(
            probabilities[:, index],
            labels[:, index],
            threshold=float(threshold_values[index]),
            threshold_source=threshold_source,
            n_bins=n_bins,
        )
        task["task_index"] = index
        per_task[str(name)] = task

    macro_fields = (
        "auroc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "negative_predictive_value",
        "accuracy",
        "balanced_accuracy",
        "brier",
        "log_loss",
        "ece",
        "mce",
    )
    macro = {
        field: _safe_mean([task[field] for task in per_task.values()])
        for field in macro_fields
    }
    valid = np.isfinite(probabilities) & np.isfinite(labels)
    flat_probabilities = probabilities[valid]
    flat_labels = labels[valid]
    # Micro metrics are explicitly labeled as such; they do not replace
    # per-task or macro reporting.
    micro = binary_task_metrics(
        flat_probabilities,
        flat_labels,
        threshold=float(threshold_values[0])
        if np.allclose(threshold_values, threshold_values[0])
        else 0.5,
        threshold_source=threshold_source,
        n_bins=n_bins,
    )
    return {
        "schema_version": 2,
        "n_records": int(probabilities.shape[0]),
        "n_tasks": int(probabilities.shape[1]),
        "n_tasks_with_both_classes": int(
            sum(task["status"] == "ok" for task in per_task.values())
        ),
        "macro": macro,
        "micro": micro,
        # Backward-compatible names used by the existing analysis.
        "macro_auroc": macro["auroc"],
        "macro_average_precision": macro["average_precision"],
        "macro_f1": macro["f1"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_sensitivity": macro["sensitivity"],
        "macro_specificity": macro["specificity"],
        "calibration": {
            "brier": macro["brier"],
            "ece": macro["ece"],
            "mce": macro["mce"],
            "log_loss": macro["log_loss"],
        },
        "per_class": per_task,
        "per_task": per_task,
    }


def probability_fidelity_metrics(
    reconstructed_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Clinical-output fidelity proxy when diagnostic ground truth is absent.

    These are not classification-performance metrics. They quantify agreement
    with the same frozen classifier applied to the paired reference ECG.
    """
    reconstructed = np.asarray(reconstructed_probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    if reconstructed.ndim != 2 or reconstructed.shape != reference.shape:
        raise ValueError("Reconstructed/reference probabilities must be equal [N,T] arrays")
    if reconstructed.shape[1] != len(class_names):
        raise ValueError("Task-name count does not match probability columns")
    epsilon = 1e-7
    reconstructed = np.clip(reconstructed, epsilon, 1.0 - epsilon)
    reference = np.clip(reference, epsilon, 1.0 - epsilon)
    per_task: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        pred = reconstructed[:, index]
        target = reference[:, index]
        midpoint = 0.5 * (pred + target)
        kl_target_pred = target * np.log(target / pred) + (1.0 - target) * np.log(
            (1.0 - target) / (1.0 - pred)
        )
        kl_target_mid = target * np.log(target / midpoint) + (1.0 - target) * np.log(
            (1.0 - target) / (1.0 - midpoint)
        )
        kl_pred_mid = pred * np.log(pred / midpoint) + (1.0 - pred) * np.log(
            (1.0 - pred) / (1.0 - midpoint)
        )
        correlation = None
        if pred.size > 1 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
            correlation = _finite_float(np.corrcoef(pred, target)[0, 1])
        per_task[str(name)] = {
            "task_index": index,
            "n": int(pred.size),
            "evaluation_type": "frozen_classifier_probability_fidelity_proxy",
            "reference_mean_probability": float(target.mean()),
            "reconstructed_mean_probability": float(pred.mean()),
            "probability_mae": float(np.mean(np.abs(pred - target))),
            "probability_rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
            "probability_pearson": correlation,
            "bernoulli_kl_reference_to_reconstruction": float(kl_target_pred.mean()),
            "jensen_shannon_divergence": float(
                0.5 * np.mean(kl_target_mid + kl_pred_mid)
            ),
            "threshold": float(threshold),
            "threshold_agreement": float(
                np.mean((pred >= threshold) == (target >= threshold))
            ),
            "reference_positive_rate": float(np.mean(target >= threshold)),
            "reconstructed_positive_rate": float(np.mean(pred >= threshold)),
        }
    fidelity_fields = (
        "probability_mae",
        "probability_rmse",
        "probability_pearson",
        "bernoulli_kl_reference_to_reconstruction",
        "jensen_shannon_divergence",
        "threshold_agreement",
    )
    return {
        "schema_version": 1,
        "evaluation_type": "frozen_classifier_probability_fidelity_proxy",
        "ground_truth_status": "unavailable",
        "n_records": int(reconstructed.shape[0]),
        "n_tasks": int(reconstructed.shape[1]),
        "macro": {
            field: _safe_mean([task[field] for task in per_task.values()])
            for field in fidelity_fields
        },
        "per_task": per_task,
    }
