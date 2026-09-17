from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def multilabel_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | list[float]]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("target and probability must be matching two-dimensional arrays")
    auroc = [float(roc_auc_score(y[:, index], p[:, index])) for index in range(y.shape[1])]
    auprc = [float(average_precision_score(y[:, index], p[:, index])) for index in range(y.shape[1])]
    return {
        "macro_auroc": float(np.mean(auroc)),
        "macro_auprc": float(np.mean(auprc)),
        "micro_auroc": float(roc_auc_score(y.ravel(), p.ravel())),
        "brier": float(np.square(y - p).mean()),
        "classwise_auroc": auroc,
        "classwise_auprc": auprc,
    }
