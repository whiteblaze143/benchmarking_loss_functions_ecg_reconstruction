from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Macro classwise equal-width expected calibration error."""
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("target and probability must be matching two-dimensional arrays")
    if bins <= 0:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    classwise = []
    for column in range(y.shape[1]):
        assigned = np.minimum(np.searchsorted(edges, p[:, column], side="right") - 1, bins - 1)
        value = 0.0
        for index in range(bins):
            mask = assigned == index
            if mask.any():
                value += float(mask.mean()) * abs(float(y[mask, column].mean() - p[mask, column].mean()))
        classwise.append(value)
    return float(np.mean(classwise))


def select_f1_thresholds(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Select one validation threshold per class by maximum observed F1."""
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("target and probability must be matching two-dimensional arrays")
    selected = np.empty(y.shape[1], dtype=np.float64)
    for column in range(y.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y[:, column], p[:, column])
        if len(thresholds) == 0:
            selected[column] = 0.5
            continue
        denominator = precision[:-1] + recall[:-1]
        scores = np.divide(
            2.0 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        selected[column] = thresholds[int(np.argmax(scores))]
    return selected


def macro_f1(target: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> float:
    prediction = np.asarray(probability) >= np.asarray(thresholds)[None, :]
    return float(f1_score(np.asarray(target), prediction, average="macro", zero_division=0))


def multilabel_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | list[float]]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("target and probability must be matching two-dimensional arrays")
    auroc = [float(roc_auc_score(y[:, index], p[:, index])) for index in range(y.shape[1])]
    auprc = [float(average_precision_score(y[:, index], p[:, index])) for index in range(y.shape[1])]
    thresholds = select_f1_thresholds(y, p)
    return {
        "macro_auroc": float(np.mean(auroc)),
        "macro_auprc": float(np.mean(auprc)),
        "micro_auroc": float(roc_auc_score(y.ravel(), p.ravel())),
        "brier": float(np.square(y - p).mean()),
        "ece": expected_calibration_error(y, p),
        "validation_threshold_macro_f1": macro_f1(y, p, thresholds),
        "validation_f1_thresholds": thresholds.tolist(),
        "classwise_auroc": auroc,
        "classwise_auprc": auprc,
    }
