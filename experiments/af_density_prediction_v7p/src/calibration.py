"""Comprehensive classification, calibration, and discrimination metrics."""
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression


def compute_calibration_intercept_and_slope(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Computes calibration intercept and slope via logistic regression of true labels on log-odds."""
    eps = 1e-6
    probs_clipped = np.clip(y_prob, eps, 1.0 - eps)
    log_odds = np.log(probs_clipped / (1.0 - probs_clipped)).reshape(-1, 1)

    try:
        lr = LogisticRegression(C=1e9, solver="lbfgs")
        lr.fit(log_odds, y_true)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0][0])
        return intercept, slope
    except Exception:
        return np.nan, np.nan


def compute_ece_10_bins(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Expected Calibration Error over 10 uniform probability bins."""
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(y_prob)
    if n == 0:
        return 0.0
    for i in range(10):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < 9 else y_prob <= high)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    operating_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes full suite of classification and calibration metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Filter nonfinite
    valid = np.isfinite(y_prob)
    if not np.all(valid):
        y_true = y_true[valid]
        y_prob = y_prob[valid]

    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return {
            "n_samples": n,
            "prevalence": float(np.mean(y_true)) if n > 0 else 0.0,
            "auroc": np.nan,
            "auprc": np.nan,
            "brier_score": np.nan,
            "brier_skill_score": np.nan,
            "log_loss": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "ece_10_bins": np.nan,
            "balanced_accuracy": np.nan,
            "f1_score": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
        }

    prev = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    brier_ref = prev * (1.0 - prev)
    brier_skill = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = prev

    ll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6)))
    calib_int, calib_slope = compute_calibration_intercept_and_slope(y_true, y_prob)
    ece = compute_ece_10_bins(y_true, y_prob)

    y_pred = (y_prob >= operating_threshold).astype(int)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "n_samples": n,
        "prevalence": prev,
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "brier_skill_score": brier_skill,
        "log_loss": ll,
        "calibration_intercept": calib_int,
        "calibration_slope": calib_slope,
        "ece_10_bins": ece,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": prec,
        "npv": npv,
    }
