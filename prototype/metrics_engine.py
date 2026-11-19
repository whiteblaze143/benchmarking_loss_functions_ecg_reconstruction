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


def extract_continuous_episodes(
    binary_mask: np.ndarray,
    window_sec: float = 30.0,
    min_duration_sec: float = 30.0,
) -> List[Dict[str, float]]:
    """
    Extract contiguous episodes of positive predictions from a binary mask.
    Returns list of dicts with start_sec, end_sec, duration_sec.
    """
    episodes = []
    n = len(binary_mask)
    in_ep = False
    start_idx = 0

    for i in range(n):
        if binary_mask[i] and not in_ep:
            in_ep = True
            start_idx = i
        elif not binary_mask[i] and in_ep:
            in_ep = False
            dur = (i - start_idx) * window_sec
            if dur >= min_duration_sec:
                episodes.append({
                    "start_sec": start_idx * window_sec,
                    "end_sec": i * window_sec,
                    "duration_sec": dur,
                })

    if in_ep:
        dur = (n - start_idx) * window_sec
        if dur >= min_duration_sec:
            episodes.append({
                "start_sec": start_idx * window_sec,
                "end_sec": n * window_sec,
                "duration_sec": dur,
            })

    return episodes


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

    detected_ref_indices = set()
    used_pred_indices = set()
    matched_pairs = []

    for r_idx, r_ep in enumerate(ref_episodes):
        r_start, r_end = r_ep["start_sec"], r_ep["end_sec"]
        best_p_idx = -1
        max_overlap = 0.0

        for p_idx, p_ep in enumerate(pred_episodes):
            if p_idx in used_pred_indices:
                continue
            p_start, p_end = p_ep["start_sec"], p_ep["end_sec"]
            # Compute overlap
            overlap = max(0.0, min(r_end, p_end) - max(r_start, p_start))
            if overlap >= min_overlap_sec and overlap > max_overlap:
                max_overlap = overlap
                best_p_idx = p_idx

        if best_p_idx >= 0:
            detected_ref_indices.add(r_idx)
            used_pred_indices.add(best_p_idx)
            matched_pairs.append((r_idx, best_p_idx, max_overlap))

    n_detected = len(detected_ref_indices)
    n_false_pred = n_pred - len(used_pred_indices)
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
