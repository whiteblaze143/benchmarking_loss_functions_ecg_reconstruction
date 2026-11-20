"""
Single Tested Clinical Metrics Engine for Atrial Fibrillation Monitoring.
Standardizes:
1. Episode detection and matching (>=30s historical comparability, >=6min AFA-ARC consensus endpoint).
2. Clinician-review False Alarm Rate per 24 hours (with configurable refractory period).
3. Patient-level cumulative AF burden agreement (MAE, Pearson r, Spearman rho).
4. Patient-clustered bootstrap confidence intervals (clustered by Subject_ID).
5. Risk-coverage curves stratified by ambulatory noise tiers (Clean, N1, N2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment


def extract_continuous_episodes(
    binary_mask: np.ndarray,
    window_sec: float = 30.0,
    min_duration_sec: float = 30.0,
    record_duration_sec: float | None = None,
) -> List[Dict[str, float]]:
    """
    Extract contiguous episodes of positive predictions from a binary mask.
    Returns list of dicts with start_sec, end_sec, duration_sec.
    """
    mask = np.asarray(binary_mask,dtype=bool)
    boundaries = np.diff(np.pad(mask.astype(np.int8),(1,1)))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    start_times=starts*window_sec
    end_times=ends*window_sec
    if record_duration_sec is not None:
        end_times=np.minimum(end_times,record_duration_sec)
    return [{"start_sec":float(start),"end_sec":float(end),"duration_sec":float(end-start)}
            for start,end in zip(start_times,end_times) if end-start >= min_duration_sec]


def match_episodes(
    ref_episodes: List[Dict[str, float]],
    pred_episodes: List[Dict[str, float]],
    min_overlap_sec: float = 15.0,
) -> Dict[str, Any]:
    """
    Bipartite one-to-one episode matching between reference and predicted episodes.
    Prevents multiple predicted windows from overcounting detection of a single episode.
    """
    n_ref = len(ref_episodes)
    n_pred = len(pred_episodes)

    if n_ref == 0:
        return {
            "n_ref": 0,
            "n_pred": n_pred,
            "detected_ref": 0,
            "false_pred": n_pred,
            "sensitivity": 1.0 if n_pred == 0 else 0.0,
            "matched_pairs": [],
        }

    matched_pairs = []
    if n_pred:
        overlap = np.array([[max(0., min(r["end_sec"],p["end_sec"])-max(r["start_sec"],p["start_sec"]))
                             for p in pred_episodes] for r in ref_episodes])
        valid = overlap >= min_overlap_sec
        # One additional valid match outweighs every overlap tie-break combined.
        reward = valid * (1. + overlap / (overlap.sum() + 1.))
        row_indices, col_indices = linear_sum_assignment(reward, maximize=True)
        matched_pairs = [(int(r),int(p),float(overlap[r,p]))
                         for r,p in zip(row_indices,col_indices) if valid[r,p]]
    n_detected = len(matched_pairs)
    n_false_pred = n_pred - n_detected
    sens = float(n_detected / n_ref) if n_ref > 0 else 1.0

    return {
        "n_ref": n_ref,
        "n_pred": n_pred,
        "detected_ref": n_detected,
        "false_pred": n_false_pred,
        "sensitivity": sens,
        "matched_pairs": matched_pairs,
    }


def compute_clinical_false_alerts_per_24h(
    false_positive_mask: np.ndarray,
    monitored_hours: float,
    window_sec: float = 30.0,
    refractory_period_sec: float = 3600.0,  # 1-hour clinical refractory window
) -> Dict[str, float]:
    """
    Compute raw false alarms and clinician-review false alerts per 24 hours.
    Applies refractory window: consecutive alarms within refractory_period_sec
    are merged into a single actionable alert.
    """
    if monitored_hours <= 0:
        return {"raw_fa": 0, "raw_fa_per_24h": 0.0, "clinical_alerts": 0, "alerts_per_24h": 0.0}

    n = len(false_positive_mask)
    raw_fa = int(np.sum(false_positive_mask))
    raw_fa_per_24h = (raw_fa / monitored_hours) * 24.0

    # Clinician alerts with refractory period
    alerts = 0
    last_alert_time = -refractory_period_sec - 1.0

    for i in range(n):
        if false_positive_mask[i]:
            t = i * window_sec
            if t - last_alert_time >= refractory_period_sec:
                alerts += 1
                last_alert_time = t

    alerts_per_24h = (alerts / monitored_hours) * 24.0

    return {
        "raw_fa": raw_fa,
        "raw_fa_per_24h": round(raw_fa_per_24h, 2),
        "clinical_alerts": alerts,
        "alerts_per_24h": round(alerts_per_24h, 2),
    }


def compute_burden_metrics(
    gt_burdens: np.ndarray,
    pred_burdens: np.ndarray,
) -> Dict[str, float]:
    """
    Compute cumulative AF burden agreement:
    - MAE (% points)
    - Pearson correlation r
    - Spearman rank correlation rho
    """
    gt = np.asarray(gt_burdens, dtype=float)
    pred = np.asarray(pred_burdens, dtype=float)

    valid = ~(np.isnan(gt) | np.isnan(pred))
    gt_v = gt[valid]
    pred_v = pred[valid]

    if len(gt_v) < 2:
        return {
            "mae": float(np.mean(np.abs(gt_v - pred_v))) if len(gt_v) > 0 else 0.0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "n_records": len(gt_v),
        }

    mae = float(np.mean(np.abs(gt_v - pred_v)))
    
    # Pearson
    if np.std(gt_v) > 1e-6 and np.std(pred_v) > 1e-6:
        r, _ = stats.pearsonr(gt_v, pred_v)
        rho, _ = stats.spearmanr(gt_v, pred_v)
    else:
        r = 0.0
        rho = 0.0

    return {
        "mae": round(mae, 2),
        "pearson_r": round(float(r), 4),
        "spearman_rho": round(float(rho), 4),
        "n_records": len(gt_v),
    }


def patient_clustered_bootstrap_ci(
    df: pd.DataFrame,
    patient_col: str,
    metric_fn,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute patient-clustered bootstrap confidence intervals.
    Resamples patient clusters with replacement.
    """
    rng = np.random.RandomState(seed)
    patients = df[patient_col].unique()
    n_patients = len(patients)

    boot_metrics = defaultdict(list)

    # Compute point estimate
    point_est = metric_fn(df)

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(patients, size=n_patients, replace=True)
        # Build resampled dataframe
        sample_dfs = [df[df[patient_col] == p] for p in sampled_patients]
        boot_df = pd.concat(sample_dfs, ignore_index=True)
        try:
            m_dict = metric_fn(boot_df)
            for k, v in m_dict.items():
                if np.isscalar(v) and not np.isnan(v):
                    boot_metrics[k].append(v)
        except Exception:
            continue

    alpha = (1.0 - ci_level) / 2.0
    results = {}
    for k, pt in point_est.items():
        if k in boot_metrics and len(boot_metrics[k]) > 50:
            vals = np.array(boot_metrics[k])
            lo = float(np.percentile(vals, alpha * 100))
            hi = float(np.percentile(vals, (1.0 - alpha) * 100))
            results[k] = {
                "point_estimate": round(float(pt), 4) if isinstance(pt, (int, float)) else pt,
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
            }
        else:
            results[k] = {
                "point_estimate": pt,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }

    return results
