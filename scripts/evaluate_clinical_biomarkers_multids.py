#!/usr/bin/env python3
"""
Comprehensive Multi-Dataset Clinical Biomarker Evaluator.
Evaluates baselines, single-loss term models, and top multi-loss Pareto models across
PTB-XL, EchoNext, LUDB, ISP, Zhejiang, and Sunnybrook datasets.
Enforces CPU execution capped at <= 2 cores.
"""

import os
# Two single-threaded workers below form the hard two-core ceiling.  Leaving
# these at two would allow each process to fan out and oversubscribe the host.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import ast
import gc
import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import concurrent.futures
import statsmodels.api as sm

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from torch.utils.data import Dataset, DataLoader
from scripts.evaluate_comprehensive_registry import load_adapter
from scripts.ecgfounder_classifier import (
    load_ecgfounder,
    load_ptbxl_labels,
    load_task_names,
    preprocess_ecgfounder,
)
from scripts.evaluate_external_delineation_watch import load_ludb, load_isp, load_zhejiang
from scripts.evaluate_sunnybrook_registry import load_record as load_sunnybrook_record
from scripts.echonext_classifier import (
    EchoNextMiniModel,
    SHD_TASKS,
    load_echonext_test_metadata,
)
from scripts.evaluate_echonext import load_and_validate as load_echonext_waveforms

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fmt(val, fmt_spec=".4f"):
    """Format float values safely. Return empty string for NaNs or None for clean CSVs."""
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return ""
    return f"{val:{fmt_spec}}"

# Statistical Functions
def compute_bland_altman_and_regression(y_true, y_pred):
    """Compute Bland-Altman agreement (bias, LoA), R2, Pearson r, and MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid_idx], y_pred[valid_idx]
    
    if len(y_true) < 2:
        return {"mae": np.nan, "pearson_r": np.nan, "r2": np.nan, "bias": np.nan, "sd_diff": np.nan, "loa_low": np.nan, "loa_high": np.nan}
        
    diff = y_pred - y_true
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff
    
    mae = float(np.mean(np.abs(diff)))
    r_val, _ = stats.pearsonr(y_true, y_pred)
    slope, intercept, r_value, _, _ = stats.linregress(y_true, y_pred)
    r2 = float(r_value ** 2)
    
    return {
        "mae": mae,
        "pearson_r": float(r_val),
        "r2": r2,
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_low": loa_low,
        "loa_high": loa_high,
    }

def compute_classification_metrics(y_true, y_pred, y_score, n_bootstraps=1000):
    """Compute AUROC, AUPRC, F1, Sensitivity, Specificity, PPV, NPV with 1000-sample bootstrapped CIs."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    
    def _metrics(idx):
        t = y_true[idx]
        p = y_pred[idx]
        s = y_score[idx]
        if np.sum(t) == 0 or np.sum(t) == len(t):
            return [np.nan] * 7
        auroc = roc_auc_score(t, s)
        auprc = average_precision_score(t, s)
        f1 = f1_score(t, p)
        tp = np.sum((t == 1) & (p == 1))
        fn = np.sum((t == 1) & (p == 0))
        tn = np.sum((t == 0) & (p == 0))
        fp = np.sum((t == 0) & (p == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        return [auroc, auprc, f1, sens, spec, ppv, npv]
        
    base_metrics = _metrics(np.arange(len(y_true)))
    
    np.random.seed(42)
    boot_res = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        res = _metrics(idx)
        if not np.isnan(res[0]):
            boot_res.append(res)
            
    if not boot_res:
        return base_metrics, [(np.nan, np.nan)] * 7
        
    boot_res = np.array(boot_res)
    cis = [(float(np.percentile(boot_res[:, i], 2.5)), float(np.percentile(boot_res[:, i], 97.5))) for i in range(7)]
    return base_metrics, cis


def _finite_paired_mask(series_a, series_b):
    """Return a boolean mask where both series are finite (non-NaN, non-inf)."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    return np.isfinite(a) & np.isfinite(b)


def _classification_with_nan_filter(y_true_raw, y_pred_raw, y_score_raw, n_bootstraps=1000):
    """Drop non-finite score rows then delegate to compute_classification_metrics.

    Returns all-NaN metrics/CIs when fewer than 2 finite rows remain, rather
    than letting roc_auc_score raise ValueError on NaN inputs.
    """
    y_score_arr = np.asarray(y_score_raw, dtype=float)
    finite_mask = np.isfinite(y_score_arr)
    n_finite = int(np.sum(finite_mask))
    if n_finite < 2:
        nan7 = [np.nan] * 7
        nan_cis = [(np.nan, np.nan)] * 7
        logging.warning(
            "_classification_with_nan_filter: only %d finite score rows; "
            "returning NaN metrics.",
            n_finite,
        )
        return nan7, nan_cis
    y_true_f = np.asarray(y_true_raw, dtype=int)[finite_mask]
    y_pred_f = np.asarray(y_pred_raw, dtype=int)[finite_mask]
    y_score_f = y_score_arr[finite_mask]
    return compute_classification_metrics(y_true_f, y_pred_f, y_score_f, n_bootstraps=n_bootstraps)

def fit_logistic_regression(df, target_col, predictor_col):
    """Fit multivariable logistic regression controlling for age & sex and return Adjusted OR & p-val."""
    if "age" not in df.columns or "sex" not in df.columns:
        return np.nan, (np.nan, np.nan), np.nan
    clean_df = df[[target_col, predictor_col, "age", "sex"]].dropna()
    if len(clean_df) < 10 or clean_df[target_col].nunique() < 2:
        return np.nan, (np.nan, np.nan), np.nan
        
    y = clean_df[target_col].astype(float)
    X = clean_df[[predictor_col, "age", "sex"]].astype(float)
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        odds_ratio = np.exp(model.params[predictor_col])
        conf = model.conf_int()
        ci_low = np.exp(conf.loc[predictor_col, 0])
        ci_high = np.exp(conf.loc[predictor_col, 1])
        pval = model.pvalues[predictor_col]
        return float(odds_ratio), (float(ci_low), float(ci_high)), float(pval)
    except Exception:
        return np.nan, (np.nan, np.nan), np.nan

def compute_fisher_exact(y_true, y_pred):
    """Compute Fisher exact test p-value for 2x2 contingency matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    table = [[tp, fp], [fn, tn]]
    odds_ratio, pval = stats.fisher_exact(table)
    return float(odds_ratio), float(pval)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
OBSERVED_LEAD_INDICES = (0, 1, 7)  # I, II, V2 are copied from the acquisition.
MISSING_LEAD_INDICES = tuple(
    index for index in range(len(LEAD_NAMES)) if index not in OBSERVED_LEAD_INDICES
)
MISSING_LEAD_NAMES = tuple(LEAD_NAMES[index] for index in MISSING_LEAD_INDICES)

# Increment this whenever metric semantics change. Resume checks are scoped to
# this value so legacy rows cannot make a corrected evaluator skip a model.
EVALUATION_VERSION = "missing_leads_v2"
PATIENT_BOOTSTRAPS = 500
QRS_DURATION_RANGE_MS = (30.0, 240.0)
DELINEATION_MODE = "independent_missing_leads"


def _expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        return np.nan
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        included = (
            (probabilities >= lower) & (probabilities < upper)
            if index < bins - 1
            else (probabilities >= lower) & (probabilities <= upper)
        )
        if np.any(included):
            value += (
                np.mean(included)
                * abs(np.mean(labels[included]) - np.mean(probabilities[included]))
            )
    return float(value)


def _binary_probability_metric(labels, probabilities, metric):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim == 2:
        if probabilities.shape != labels.shape:
            raise ValueError("Multilabel probabilities must match label shape")
        values = [
            _binary_probability_metric(labels[:, index], probabilities[:, index], metric)
            for index in range(labels.shape[1])
        ]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else np.nan
    if metric == "auroc":
        return float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    if metric == "auprc":
        return float(average_precision_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    if metric == "brier":
        return float(np.mean((probabilities - labels) ** 2))
    if metric == "ece":
        return _expected_calibration_error(labels, probabilities)
    raise ValueError(f"Unsupported paired metric: {metric}")


def patient_cluster_bootstrap_delta(
    labels,
    reference_probabilities,
    reconstructed_probabilities,
    patient_ids,
    metric,
    *,
    n_bootstraps=PATIENT_BOOTSTRAPS,
    seed=42,
):
    """Paired original-vs-reconstruction delta with patient-cluster bootstrap."""
    labels = np.asarray(labels)
    reference = np.asarray(reference_probabilities, dtype=float)
    reconstructed = np.asarray(reconstructed_probabilities, dtype=float)
    patient_ids = np.asarray(patient_ids)
    if not (len(labels) == len(reference) == len(reconstructed) == len(patient_ids)):
        raise ValueError("Paired bootstrap inputs must have equal record counts")

    reference_value = _binary_probability_metric(labels, reference, metric)
    reconstruction_value = _binary_probability_metric(labels, reconstructed, metric)
    point_delta = reconstruction_value - reference_value

    unique_patients = np.unique(patient_ids)
    patient_rows = {patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients}
    rng = np.random.default_rng(seed)
    bootstrap_deltas = []
    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rows = np.concatenate([patient_rows[patient] for patient in sampled_patients])
        reference_boot = _binary_probability_metric(labels[rows], reference[rows], metric)
        reconstruction_boot = _binary_probability_metric(
            labels[rows], reconstructed[rows], metric
        )
        delta = reconstruction_boot - reference_boot
        if np.isfinite(delta):
            bootstrap_deltas.append(delta)

    if not bootstrap_deltas:
        return {
            "reference": reference_value,
            "reconstruction": reconstruction_value,
            "delta": point_delta,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n_records": len(labels),
            "n_patients": len(unique_patients),
            "n_bootstraps": 0,
        }

    bootstrap_deltas = np.asarray(bootstrap_deltas)
    # Two-sided paired bootstrap test with a finite-sample correction.
    non_positive = (np.sum(bootstrap_deltas <= 0) + 1) / (len(bootstrap_deltas) + 1)
    non_negative = (np.sum(bootstrap_deltas >= 0) + 1) / (len(bootstrap_deltas) + 1)
    return {
        "reference": reference_value,
        "reconstruction": reconstruction_value,
        "delta": point_delta,
        "ci_low": float(np.percentile(bootstrap_deltas, 2.5)),
        "ci_high": float(np.percentile(bootstrap_deltas, 97.5)),
        "p_value": float(min(1.0, 2.0 * min(non_positive, non_negative))),
        "n_records": len(labels),
        "n_patients": len(unique_patients),
        "n_bootstraps": len(bootstrap_deltas),
    }


def patient_cluster_bootstrap_agreement(
    target,
    reconstruction,
    patient_ids,
    metric,
    *,
    n_bootstraps=PATIENT_BOOTSTRAPS,
    seed=42,
):
    """Patient-cluster CI for a paired continuous agreement endpoint."""
    target = np.asarray(target, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    patient_ids = np.asarray(patient_ids)
    finite = np.isfinite(target) & np.isfinite(reconstruction)
    target, reconstruction, patient_ids = (
        target[finite], reconstruction[finite], patient_ids[finite]
    )
    if not len(target):
        raise ValueError("No finite paired observations")

    def statistic(rows):
        differences = reconstruction[rows] - target[rows]
        if metric == "mae":
            return float(np.mean(np.abs(differences)))
        if metric == "bias":
            return float(np.mean(differences))
        raise ValueError(f"Unsupported agreement metric: {metric}")

    unique_patients = np.unique(patient_ids)
    patient_rows = {patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstraps):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rows = np.concatenate([patient_rows[patient] for patient in sampled])
        estimates.append(statistic(rows))

    patient_differences = np.asarray([
        np.mean(reconstruction[rows] - target[rows]) for rows in patient_rows.values()
    ])
    try:
        paired_p = float(stats.wilcoxon(patient_differences).pvalue)
    except ValueError:
        paired_p = np.nan
    estimate = statistic(np.arange(len(target)))
    return {
        "reference": 0.0,
        "reconstruction": estimate,
        "delta": estimate,
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
        "p_value": paired_p if metric == "bias" else np.nan,
        "n_records": len(target),
        "n_patients": len(unique_patients),
        "n_bootstraps": len(estimates),
    }

# Helper function for Fast DTW distance calculation
def compute_dtw_distance(s1, s2, max_points=500):
    """Compute Fast DTW distance between 1D signals s1 and s2."""
    try:
        if len(s1) > max_points:
            idx = np.linspace(0, len(s1) - 1, max_points).astype(int)
            s1 = s1[idx]
            s2 = s2[idx]
        diff_mat = np.abs(s1[:, None] - s2[None, :])
        return float(np.mean(np.min(diff_mat, axis=1)))
    except Exception:
        return float(np.mean(np.abs(s1 - s2)))

def _wave_mask(onsets, offsets, n_samples):
    """Build a one-dimensional wave mask from valid paired boundaries."""
    mask = np.zeros(n_samples, dtype=bool)
    for onset, offset in zip(onsets, offsets):
        if (
            np.isfinite(onset)
            and np.isfinite(offset)
            and 0 <= int(onset) <= int(offset) < n_samples
        ):
            mask[int(onset):int(offset) + 1] = True
    return mask


def _extract_lead_biomarkers(lead, fs=500):
    """Delineate one lead and return timing, ST, and morphology measurements."""
    signals, info = nk.ecg_process(lead, sampling_rate=fs)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(rpeaks) < 2:
        return None

    _, waves = nk.ecg_delineate(
        signals["ECG_Clean"].values,
        rpeaks,
        sampling_rate=fs,
        method="dwt",
    )

    def wave_array(name):
        return np.asarray(waves.get(name, []), dtype=float)

    r_onsets = wave_array("ECG_R_Onsets")
    r_offsets = wave_array("ECG_R_Offsets")
    p_onsets = wave_array("ECG_P_Onsets")
    p_offsets = wave_array("ECG_P_Offsets")
    t_onsets = wave_array("ECG_T_Onsets")
    t_offsets = wave_array("ECG_T_Offsets")

    qrs_durations = []
    valid_r_offsets = []
    for peak in rpeaks:
        onset_candidates = r_onsets[np.isfinite(r_onsets) & (r_onsets < peak)]
        offset_candidates = r_offsets[np.isfinite(r_offsets) & (r_offsets > peak)]
        if len(onset_candidates) == 0 or len(offset_candidates) == 0:
            continue
        onset = onset_candidates[-1]
        offset = offset_candidates[0]
        if offset > onset:
            duration_ms = (offset - onset) / fs * 1000.0
            if QRS_DURATION_RANGE_MS[0] <= duration_ms <= QRS_DURATION_RANGE_MS[1]:
                qrs_durations.append(duration_ms)
                valid_r_offsets.append(offset)
    if not qrs_durations:
        return None

    st_amplitudes = []
    for offset in valid_r_offsets:
        j60 = int(offset + 0.06 * fs)
        if j60 < len(lead):
            st_amplitudes.append(lead[j60])

    n_samples = len(lead)
    return {
        "qrs_ms": float(np.mean(qrs_durations)),
        "st_mv": float(np.mean(st_amplitudes)) if st_amplitudes else np.nan,
        "rpeaks": rpeaks,
        "p_onsets": p_onsets,
        "p_offsets": p_offsets,
        "r_onsets": r_onsets,
        "r_offsets": r_offsets,
        "t_onsets": t_onsets,
        "t_offsets": t_offsets,
        "p_mask": _wave_mask(p_onsets, p_offsets, n_samples),
        "qrs_mask": _wave_mask(r_onsets, r_offsets, n_samples),
        "t_mask": _wave_mask(t_onsets, t_offsets, n_samples),
    }


def _extract_missing_lead_biomarkers(signal, fs=500):
    """Delineate all nine missing leads independently and aggregate them."""
    lead_results = {}
    for lead_index in MISSING_LEAD_INDICES:
        lead_name = LEAD_NAMES[lead_index]
        try:
            result = _extract_lead_biomarkers(signal[lead_index], fs=fs)
        except Exception:
            result = None
        if result is not None:
            lead_results[lead_name] = result
    if not lead_results:
        return None

    # Sokolow-Lyon uses reconstructed V1 and V5, with timing detected on the
    # same missing lead. Copied V2 is absent from timing and amplitude.
    lvh_voltage = np.nan
    if "V1" in lead_results and "V5" in lead_results:
        v1_result = lead_results["V1"]
        v5_result = lead_results["V5"]
        v1 = signal[LEAD_NAMES.index("V1")]
        v5 = signal[LEAD_NAMES.index("V5")]
        s_amplitudes = []
        for peak in v1_result["rpeaks"]:
            offsets = v1_result["r_offsets"][
                np.isfinite(v1_result["r_offsets"])
                & (v1_result["r_offsets"] > peak)
            ]
            if len(offsets) and int(offsets[0]) < len(v1):
                s_amplitudes.append(np.min(v1[int(peak):int(offsets[0]) + 1]))
        r_amplitudes = [
            v5[peak] for peak in v5_result["rpeaks"] if peak < len(v5)
        ]
        if s_amplitudes and r_amplitudes:
            mean_s_v1 = float(np.mean(s_amplitudes))
            mean_r_v5 = float(np.mean(r_amplitudes))
            lvh_voltage = abs(mean_s_v1) + max(0.0, mean_r_v5)

    return {
        "qrs_ms": float(np.mean([
            result["qrs_ms"] for result in lead_results.values()
        ])),
        "lvh_mv": lvh_voltage,
        "st_lead_mv": {
            lead_name: result["st_mv"] for lead_name, result in lead_results.items()
        },
        "lead_results": lead_results,
        "delineated_missing_leads": len(lead_results),
        "missing_lead_coverage": len(lead_results) / len(MISSING_LEAD_INDICES),
        "delineation_mode": DELINEATION_MODE,
    }


def _timing_mae_ms(arr_true, arr_recon, fs):
    valid_true = np.asarray(arr_true)[np.isfinite(arr_true)]
    valid_recon = np.asarray(arr_recon)[np.isfinite(arr_recon)]
    if len(valid_true) == 0 or len(valid_recon) == 0:
        return np.nan
    paired = min(len(valid_true), len(valid_recon))
    return float(np.mean(np.abs(valid_recon[:paired] - valid_true[:paired])) / fs * 1000.0)


def _dice(mask_true, mask_recon):
    intersection = np.sum(mask_true & mask_recon)
    denominator = np.sum(mask_true) + np.sum(mask_recon)
    return float(2.0 * intersection / denominator) if denominator > 0 else np.nan


# Biomarker & 12-Lead Extraction Kernel
def extract_biomarkers_kernel(item_tuple, fs=500):
    """
    Extract QRS duration, LVH voltage criteria, ST deviation, per-lead metrics,
    missing-lead metrics (Pearson r, MSE, SNR dB, DTW), boundary timing errors
    (P_Onset, P_Offset, R_Onset, R_Offset, T_Onset, T_Offset), and morphological
    Dice wave overlap scores across the nine reconstructed (missing) leads.
    """
    target, recon = item_tuple
    
    # 1. 12-Lead Signal Metrics (MAE, Pearson r, SNR dB, MSE, DTW per lead)
    per_lead_metrics = {}
    missing_pearsons, missing_mses, missing_snrs, missing_dtws = [], [], [], []
    
    for ch_idx, lead_name in enumerate(LEAD_NAMES):
        t_lead = target[ch_idx]
        r_lead = recon[ch_idx]
        diff = r_lead - t_lead
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        bias = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        
        # Pearson r
        if np.std(t_lead) > 1e-6 and np.std(r_lead) > 1e-6:
            r_val, _ = stats.pearsonr(t_lead, r_lead)
            r_val = float(r_val)
        else:
            r_val = 0.0
            
        # R2
        var_t = float(np.var(t_lead))
        var_err = float(np.var(diff))
        r2 = float(1.0 - (var_err / (var_t + 1e-8))) if var_t > 1e-6 else 0.0
            
        # SNR dB
        snr_db = float(10.0 * np.log10((var_t + 1e-8) / (var_err + 1e-8)))
        
        # Fast DTW distance
        dtw_val = compute_dtw_distance(t_lead, r_lead)
        
        per_lead_metrics[lead_name] = {
            "mae": mae,
            "mse": mse,
            "pearson_r": r_val,
            "r2": r2,
            "bias": bias,
            "sd_diff": sd_diff,
            "snr_db": snr_db,
            "dtw": dtw_val,
        }
        
        # Exactly nine missing leads; observed V2 must never enter this summary.
        if ch_idx in MISSING_LEAD_INDICES:
            missing_pearsons.append(r_val)
            missing_mses.append(mse)
            missing_snrs.append(snr_db)
            missing_dtws.append(dtw_val)
            
    val_missing_pearson = float(np.mean(missing_pearsons)) if missing_pearsons else 0.0
    val_missing_mse = float(np.mean(missing_mses)) if missing_mses else 0.0
    val_missing_snr_db = float(np.mean(missing_snrs)) if missing_snrs else 0.0
    val_missing_dtw = float(np.mean(missing_dtws)) if missing_dtws else 0.0

    t_res = _extract_missing_lead_biomarkers(target, fs=fs)
    r_res = _extract_missing_lead_biomarkers(recon, fs=fs)
    if t_res is not None and r_res is not None:
        paired_leads = sorted(
            set(t_res["lead_results"]) & set(r_res["lead_results"])
        )
        if not paired_leads:
            return None

        boundary_pairs = {
            "p_onset_mae_ms": ("p_onsets", "p_onsets"),
            "p_offset_mae_ms": ("p_offsets", "p_offsets"),
            "r_onset_mae_ms": ("r_onsets", "r_onsets"),
            "r_offset_mae_ms": ("r_offsets", "r_offsets"),
            "t_onset_mae_ms": ("t_onsets", "t_onsets"),
            "t_offset_mae_ms": ("t_offsets", "t_offsets"),
        }
        boundary_errors = {}
        for output_name, (target_name, recon_name) in boundary_pairs.items():
            values = [
                _timing_mae_ms(
                    t_res["lead_results"][lead_name][target_name],
                    r_res["lead_results"][lead_name][recon_name],
                    fs,
                )
                for lead_name in paired_leads
            ]
            finite = [value for value in values if np.isfinite(value)]
            boundary_errors[output_name] = float(np.mean(finite)) if finite else np.nan

        dice_pairs = {
            "p_wave_dice": "p_mask",
            "qrs_wave_dice": "qrs_mask",
            "t_wave_dice": "t_mask",
        }
        dice_scores = {}
        for output_name, mask_name in dice_pairs.items():
            values = [
                _dice(
                    t_res["lead_results"][lead_name][mask_name],
                    r_res["lead_results"][lead_name][mask_name],
                )
                for lead_name in paired_leads
            ]
            finite = [value for value in values if np.isfinite(value)]
            dice_scores[output_name] = float(np.mean(finite)) if finite else np.nan

        boundary_errors["paired_missing_lead_coverage"] = (
            len(paired_leads) / len(MISSING_LEAD_INDICES)
        )
        
        missing_summary = {
            "val_missing_pearson": val_missing_pearson,
            "val_missing_mse": val_missing_mse,
            "val_missing_snr_db": val_missing_snr_db,
            "val_missing_dtw": val_missing_dtw,
        }
        
        def extract_peak(sig):
            return {
                "V1": float(np.percentile(sig[LEAD_NAMES.index("V1")], 99.0)),
                "V3": float(np.percentile(sig[LEAD_NAMES.index("V3")], 99.0)),
                "V6": float(np.percentile(sig[LEAD_NAMES.index("V6")], 99.0)),
                "I": float(np.percentile(sig[LEAD_NAMES.index("I")], 99.0)),
            }
            
        t_peaks = extract_peak(target)
        r_peaks = extract_peak(recon)
        
        return t_res, r_res, per_lead_metrics, missing_summary, boundary_errors, dice_scores, t_peaks, r_peaks
        
    return None

class SimplePTBDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.pt"), key=lambda p: int(p.stem))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        sig = torch.load(path, weights_only=True).float()
        return sig, int(path.stem)

def main():
    device = torch.device("cpu") # Execute strictly on CPU capped at 2 cores
    project_root = _ROOT
    
    registry_file = project_root / "results/clinical_biomarkers_model_registry.json"
    if not registry_file.exists():
        logging.error(f"Registry file not found: {registry_file}")
        sys.exit(1)
        
    with open(registry_file) as f:
        registry = json.load(f)
        
    models = registry["models"]
    logging.info(f"Loaded {len(models)} models for multi-dataset clinical biomarker evaluation.")
    
    out_dir = project_root / "results/clinical_biomarkers_multids"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_per_architecture = 160
    architecture_aliases = {
        "unet": "unet",
        "msvae": "msvae",
        "multiscale_vae": "msvae",
        "ecg_aim": "ecg_aim",
        "alitok": "ecg_aim",  # registry stores ECG-AIM checkpoints as kind='alitok'
    }
    architecture_counts = {"unet": 0, "msvae": 0, "ecg_aim": 0}
    for spec in models:
        architecture = architecture_aliases.get(spec.get("kind"), spec.get("kind"))
        if architecture in architecture_counts:
            architecture_counts[architecture] += 1
    architecture_gate = {
        "evaluation_version": EVALUATION_VERSION,
        "expected_models_per_architecture": expected_per_architecture,
        "registered_models": architecture_counts,
        "complete": {
            architecture: count == expected_per_architecture
            for architecture, count in architecture_counts.items()
        },
    }
    architecture_gate["architecture_claims_allowed"] = all(
        architecture_gate["complete"].values()
    )
    architecture_gate["reason"] = (
        "All three architecture registries are complete."
        if architecture_gate["architecture_claims_allowed"]
        else "Architecture comparisons are blocked until all 160 masks are registered "
             "for U-Net, MSVAE, and ECG-AIM."
    )
    gate_path = out_dir / "architecture_completeness.json"
    staging_gate_path = gate_path.with_suffix(".json.tmp")
    staging_gate_path.write_text(json.dumps(architecture_gate, indent=2))
    os.replace(staging_gate_path, gate_path)
    if not architecture_gate["architecture_claims_allowed"]:
        logging.warning(architecture_gate["reason"])
    # Keep corrected rows out of the append-only legacy CSV whose metric
    # semantics included copied V2. SQLite remains the authoritative store.
    summary_file = out_dir / f"clinical_metrics_summary_{EVALUATION_VERSION}.csv"
    
    headers = [
        "dataset", "model_id", "target", "mae", "pearson_r", "r2",
        "bland_bias", "loa_low", "loa_high",
        "auroc", "auroc_ci_low", "auroc_ci_high",
        "auprc", "auprc_ci_low", "auprc_ci_high",
        "f1", "sens", "spec", "ppv", "npv",
        "adj_or", "adj_or_ci_low", "adj_or_ci_high", "pval_logistic", "fisher_pval",
        "evaluation_version"
    ]
    
    evaluated_ptbxl = set()
    evaluated_echonext = set()
    evaluated_sunnybrook = set()
    evaluated_ludb = set()
    evaluated_isp = set()
    evaluated_zhejiang = set()
    
    import sqlite3
    db_path = out_dir / "clinical_metrics.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinical_metrics (
        dataset TEXT,
        model_id TEXT,
        target TEXT,
        mae REAL,
        pearson_r REAL,
        r2 REAL,
        bland_bias REAL,
        loa_low REAL,
        loa_high REAL,
        auroc REAL,
        auroc_ci_low REAL,
        auroc_ci_high REAL,
        auprc REAL,
        auprc_ci_low REAL,
        auprc_ci_high REAL,
        f1 REAL,
        sens REAL,
        spec REAL,
        ppv REAL,
        npv REAL,
        adj_or REAL,
        adj_or_ci_low REAL,
        adj_or_ci_high REAL,
        pval_logistic REAL,
        fisher_pval REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (dataset, model_id, target)
    )
    """)
    metric_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(clinical_metrics)").fetchall()
    }
    if "evaluation_version" not in metric_columns:
        cursor.execute(
            "ALTER TABLE clinical_metrics ADD COLUMN evaluation_version TEXT "
            "NOT NULL DEFAULT 'legacy_v1'"
        )
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paired_inference (
        dataset TEXT NOT NULL,
        model_id TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        metric TEXT NOT NULL,
        reference_value REAL,
        reconstruction_value REAL,
        delta REAL,
        ci_low REAL,
        ci_high REAL,
        p_value REAL,
        n_records INTEGER NOT NULL,
        n_patients INTEGER NOT NULL,
        n_bootstraps INTEGER NOT NULL,
        cluster_source TEXT NOT NULL,
        evaluation_version TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (dataset, model_id, endpoint, metric)
    )
    """)
    conn.commit()
    
    # A model/dataset is complete only when its final deterministic metric was
    # committed.  Using any row as a completion marker caused disk-full or
    # interrupted partial writes to be skipped forever on resume.
    cursor.execute("""
        SELECT DISTINCT dataset, model_id
        FROM clinical_metrics
        WHERE evaluation_version = ?
          AND ((dataset = 'ptb_xl' AND target = 'ST_Lead_V6')
            OR (dataset <> 'ptb_xl' AND target = 'Signal_Lead_V6'))
    """, (EVALUATION_VERSION,))
    for ds, mid in cursor.fetchall():
        if ds == "ptb_xl": evaluated_ptbxl.add(mid)
        elif ds == "echonext": evaluated_echonext.add(mid)
        elif ds == "sunnybrook": evaluated_sunnybrook.add(mid)
        elif ds == "ludb": evaluated_ludb.add(mid)
        elif ds == "isp": evaluated_isp.add(mid)
        elif ds == "zhejiang": evaluated_zhejiang.add(mid)
    conn.close()
    
    def export_csv_from_sqlite():
        with sqlite3.connect(str(db_path), timeout=60) as conn_rebuild:
            cursor_rebuild = conn_rebuild.cursor()
            cursor_rebuild.execute(
                "SELECT dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high, "
                "auroc, auroc_ci_low, auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high, "
                "f1, sens, spec, ppv, npv, adj_or, adj_or_ci_low, adj_or_ci_high, pval_logistic, "
                "fisher_pval, evaluation_version FROM clinical_metrics ORDER BY dataset, model_id, target"
            )
            existing_rows = cursor_rebuild.fetchall()
        with open(summary_file, "w") as f:
            f.write(",".join(headers) + "\n")
            for row in existing_rows:
                f.write(",".join(
                    "" if v is None else str(v)
                    for v in row
                ) + "\n")

    # Rebuild the CSV from the deduplicated SQLite at startup.
    export_csv_from_sqlite()
    logging.info(
        "CSV synchronized with SQLite at startup: exported to %s.",
        summary_file.name,
    )
            
    def write_metric(ds, mid, target, v_dict):
        mae = v_dict.get("mae")
        pearson_r = v_dict.get("pearson_r")
        r2 = v_dict.get("r2")
        bland_bias = v_dict.get("bland_bias")
        loa_low = v_dict.get("loa_low")
        loa_high = v_dict.get("loa_high")
        auroc = v_dict.get("auroc")
        auroc_ci_low = v_dict.get("auroc_ci_low")
        auroc_ci_high = v_dict.get("auroc_ci_high")
        auprc = v_dict.get("auprc")
        auprc_ci_low = v_dict.get("auprc_ci_low")
        auprc_ci_high = v_dict.get("auprc_ci_high")
        f1 = v_dict.get("f1")
        sens = v_dict.get("sens")
        spec = v_dict.get("spec")
        ppv = v_dict.get("ppv")
        npv = v_dict.get("npv")
        adj_or = v_dict.get("adj_or")
        adj_or_ci_low = v_dict.get("adj_or_ci_low")
        adj_or_ci_high = v_dict.get("adj_or_ci_high")
        pval_logistic = v_dict.get("pval_logistic")
        fisher_pval = v_dict.get("fisher_pval")
        
        # Fail closed on a database write error.  The outer continuous watcher
        # retries the pass, and the final-metric resume sentinel above ensures
        # an interrupted model/dataset is recomputed rather than silently lost.
        with sqlite3.connect(str(db_path), timeout=60) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO clinical_metrics (
                    dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
                    auroc, auroc_ci_low, auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high,
                    f1, sens, spec, ppv, npv, adj_or, adj_or_ci_low, adj_or_ci_high, pval_logistic, fisher_pval,
                    evaluation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ds, mid, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
                  auroc, auroc_ci_low, auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high,
                  f1, sens, spec, ppv, npv, adj_or, adj_or_ci_low, adj_or_ci_high, pval_logistic, fisher_pval,
                  EVALUATION_VERSION))
            
        export_csv_from_sqlite()

    def write_paired_inference(ds, mid, endpoint, metric, result, cluster_source):
        with sqlite3.connect(str(db_path), timeout=60) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO paired_inference (
                    dataset, model_id, endpoint, metric, reference_value,
                    reconstruction_value, delta, ci_low, ci_high, p_value,
                    n_records, n_patients, n_bootstraps, cluster_source,
                    evaluation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ds, mid, endpoint, metric, result["reference"],
                result["reconstruction"], result["delta"], result["ci_low"],
                result["ci_high"], result["p_value"], result["n_records"],
                result["n_patients"], result["n_bootstraps"], cluster_source,
                EVALUATION_VERSION,
            ))
        
    # Scaled evaluator throughput to 6 workers
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)
    
    # -------------------------------------------------------------------------
    # ECGFOUNDER INITIALIZATION
    # -------------------------------------------------------------------------
    # Enforce CPU execution to keep GPU 100% free for training queue
    device = torch.device("cpu")
    ecgfounder_repo = project_root / "ecg_fm_integration/ecgfounder_repo"
    ecgfounder_ckpt = ecgfounder_repo / "checkpoint/12_lead_ECGFounder.pth"
    ecgfounder_tasks_file = ecgfounder_repo / "tasks.txt"
    ecgfounder_labels_file = ecgfounder_repo / "csv/ptbxl_label.csv"

    ecgfounder_model = None
    ecgfounder_tasks = []
    labels_map = {}

    if ecgfounder_repo.exists() and ecgfounder_ckpt.exists() and ecgfounder_tasks_file.exists():
        try:
            ecgfounder_tasks = load_task_names(ecgfounder_tasks_file)
            ecgfounder_model = load_ecgfounder(ecgfounder_repo, ecgfounder_ckpt, device, len(ecgfounder_tasks))
            if ecgfounder_labels_file.exists():
                labels_df = load_ptbxl_labels(ecgfounder_labels_file)
                labels_map = dict(zip(labels_df["filename_hr"], labels_df["ecgfounder_labels"]))
            logging.info(f"Successfully loaded ECGFounder Foundation Model ({len(ecgfounder_tasks)} tasks) on {device}")
        except Exception as e:
            logging.warning(f"Could not load ECGFounder model: {e}")

    # -------------------------------------------------------------------------
    # DATASET 1: PTB-XL
    # -------------------------------------------------------------------------
    logging.info("=== Starting Dataset 1: PTB-XL ===")
    ptb_data_dir = project_root / "data/ptb_xl/tensors/test"
    if ptb_data_dir.exists():
        df_ptb = pd.read_csv(project_root / "data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
        df_scp = pd.read_csv(project_root / "data/ptb_xl/scp_statements.csv", index_col=0)
        
        def get_superclasses(scp_str):
            try:
                scps = ast.literal_eval(scp_str)
                res = set()
                for code, prob in scps.items():
                    if prob > 0 and code in df_scp.index:
                        cls = df_scp.loc[code, "diagnostic_class"]
                        if pd.notna(cls): res.add(cls)
                return res
            except: return set()
            
        df_ptb["superclasses"] = df_ptb["scp_codes"].apply(get_superclasses)
        ptb_dataset = SimplePTBDataset(ptb_data_dir)
        ptb_loader = DataLoader(ptb_dataset, batch_size=32, shuffle=False)
        
        total_models = len(models)
        for model_idx, spec in enumerate(models, 1):
            mid = spec["id"]
            if mid in evaluated_ptbxl:
                continue
            logging.info(f"PTB-XL | [{model_idx}/{total_models}] Evaluating {mid}...")
            try:
                adapter = load_adapter(spec, device)
            except Exception as e:
                logging.error(f"Failed to load {mid}: {e}")
                continue
                
            records = []
            ecgfounder_records = []
            with torch.inference_mode():
                for batch in tqdm(ptb_loader, desc=f"PTB-XL {mid}"):
                    target = batch[0].to(device)
                    ecg_ids = batch[1].numpy()
                    try: recon = adapter.reconstruct(target)
                    except: continue
                    
                    if ecgfounder_model is not None:
                        t_norm = preprocess_ecgfounder(target)
                        r_norm = preprocess_ecgfounder(recon)
                        probs_t = torch.sigmoid(ecgfounder_model(t_norm)).cpu().numpy()
                        probs_r = torch.sigmoid(ecgfounder_model(r_norm)).cpu().numpy()
                        
                        for idx, ecg_id in enumerate(ecg_ids):
                            if ecg_id in df_ptb.index:
                                fn_hr = df_ptb.loc[ecg_id, "filename_hr"]
                                if fn_hr in labels_map:
                                    ecgfounder_records.append({
                                        "target_probs": probs_t[idx],
                                        "recon_probs": probs_r[idx],
                                        "gt_labels": labels_map[fn_hr],
                                        "patient_id": df_ptb.loc[ecg_id, "patient_id"],
                                    })

                    target_np, recon_np = target.cpu().numpy(), recon.cpu().numpy()
                    tasks = [(target_np[i], recon_np[i]) for i in range(target_np.shape[0])]
                    batch_res = list(executor.map(extract_biomarkers_kernel, tasks, chunksize=8))
                    
                    for ecg_id, res in zip(ecg_ids, batch_res):
                        if res is not None and ecg_id in df_ptb.index:
                            t_res, r_res, lead_metrics, missing_summary, boundary_errors, dice_scores, t_peaks, r_peaks = res
                            meta = df_ptb.loc[ecg_id]
                            classes = meta["superclasses"]
                            records.append({
                                "ecg_id": ecg_id, "patient_id": meta["patient_id"],
                                "age": meta["age"], "sex": meta["sex"],
                                "CD_label": 1 if "CD" in classes else 0,
                                "HYP_label": 1 if "HYP" in classes else 0,
                                "MI_label": 1 if "MI" in classes else 0,
                                "t_qrs": t_res["qrs_ms"], "r_qrs": r_res["qrs_ms"],
                                "t_lvh": t_res["lvh_mv"], "r_lvh": r_res["lvh_mv"],
                                "lead_metrics": lead_metrics,
                                "st_t_leads": t_res["st_lead_mv"],
                                "st_r_leads": r_res["st_lead_mv"],
                                "paired_missing_lead_coverage": boundary_errors[
                                    "paired_missing_lead_coverage"
                                ],
                                "t_peaks": t_peaks, "r_peaks": r_peaks
                            })
                            
            if records:
                df = pd.DataFrame(records)
                df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
                df["age"] = pd.to_numeric(df["age"], errors="coerce")
                
                # ECGFounder 150-Task Downstream Clinical Diagnostic Classification Metrics
                if ecgfounder_records:
                    y_true = np.array([r["gt_labels"] for r in ecgfounder_records])
                    reference_score = np.array([r["target_probs"] for r in ecgfounder_records])
                    y_score = np.array([r["recon_probs"] for r in ecgfounder_records])
                    patient_ids = np.array([r["patient_id"] for r in ecgfounder_records])
                    
                    macro_auroc, macro_auprc = [], []
                    task_summaries = []
                    
                    for t_idx, task_name in enumerate(ecgfounder_tasks):
                        pos_count = int(np.sum(y_true[:, t_idx] > 0.5))
                        if pos_count > 0 and pos_count < len(y_true):
                            yt = (y_true[:, t_idx] > 0.5).astype(int)
                            ys = y_score[:, t_idx]
                            yp = (ys > 0.5).astype(int)
                            
                            try:
                                auroc_val = roc_auc_score(yt, ys)
                                auprc_val = average_precision_score(yt, ys)
                                macro_auroc.append(auroc_val)
                                macro_auprc.append(auprc_val)
                                
                                clean_name = task_name.strip().replace(" ", "_").replace("'", "").replace("-", "_")
                                m, c = compute_classification_metrics(yt, yp, ys, n_bootstraps=50)
                                _, fish_p = compute_fisher_exact(yt, yp)
                                task_summaries.append((clean_name, m, c, fish_p))
                            except Exception:
                                pass
                                
                    mean_macro_auroc = float(np.mean(macro_auroc)) if macro_auroc else np.nan
                    mean_macro_auprc = float(np.mean(macro_auprc)) if macro_auprc else np.nan
                    
                    write_metric("ptb_xl", mid, "ECGFounder_Macro_150", {"auroc": mean_macro_auroc, "auprc": mean_macro_auprc})
                    for metric_name in ("auroc", "auprc", "brier", "ece"):
                        paired = patient_cluster_bootstrap_delta(
                            y_true,
                            reference_score,
                            y_score,
                            patient_ids,
                            metric_name,
                        )
                        write_paired_inference(
                            "ptb_xl",
                            mid,
                            "ECGFounder_Macro",
                            metric_name,
                            paired,
                            "ptbxl_database.patient_id",
                        )
                    for clean_name, m, c, fish_p in task_summaries:
                        write_metric("ptb_xl", mid, f"ECGFounder_{clean_name}", {
                            "auroc": m[0], "auroc_ci_low": c[0][0], "auroc_ci_high": c[0][1],
                            "auprc": m[1], "auprc_ci_low": c[1][0], "auprc_ci_high": c[1][1],
                            "f1": m[2], "sens": m[3], "spec": m[4], "ppv": m[5], "npv": m[6],
                            "fisher_pval": fish_p
                        })
                
                # QRS Duration (All Leads reference)
                ba_qrs = compute_bland_altman_and_regression(df["t_qrs"], df["r_qrs"])
                y_t_qrs, y_p_qrs = (df["t_qrs"] > 120).astype(int), (df["r_qrs"] > 120).astype(int)
                m_qrs, c_qrs = compute_classification_metrics(y_t_qrs, y_p_qrs, df["r_qrs"])
                df["pred_prolonged_qrs"] = y_p_qrs
                or_qrs, or_ci_qrs, p_qrs = fit_logistic_regression(df, "CD_label", "pred_prolonged_qrs")
                _, fish_qrs = compute_fisher_exact(y_t_qrs, y_p_qrs)
                
                arr_real_v1 = np.array([x["V1"] for x in df["t_peaks"]])
                arr_real_v3 = np.array([x["V3"] for x in df["t_peaks"]])
                arr_real_v6 = np.array([x["V6"] for x in df["t_peaks"]])
                arr_real_i  = np.array([x["I"]  for x in df["t_peaks"]])
                
                arr_rec_v1 = np.array([x["V1"] for x in df["r_peaks"]])
                arr_rec_v3 = np.array([x["V3"] for x in df["r_peaks"]])
                arr_rec_v6 = np.array([x["V6"] for x in df["r_peaks"]])
                arr_rec_i  = np.array([x["I"]  for x in df["r_peaks"]])
                
                var_real_v3 = np.var(arr_real_v3)
                var_real_v6 = np.var(arr_real_v6)
                
                slope_v3, _, r_v3, _, _ = stats.linregress(arr_real_v3, arr_rec_v3)
                _, _, r_v1, _, _ = stats.linregress(arr_real_v1, arr_rec_v1)
                _, _, r_v6, _, _ = stats.linregress(arr_real_v6, arr_rec_v6)
                _, _, r_inter, _, _ = stats.linregress(arr_rec_i, arr_rec_v3)
                
                bland_v1_r2 = r_v1**2
                bland_v3_r2 = r_v3**2
                bland_v6_r2 = r_v6**2
                interlead_r2 = r_inter**2
                bland_v3_slope = slope_v3
                
                var_rec_v3 = np.var(arr_rec_v3)
                var_rec_v6 = np.var(arr_rec_v6)
                bland_v3_var_ret = (var_rec_v3 / var_real_v3) * 100.0 if var_real_v3 > 0 else 0.0
                bland_v6_var_ret = (var_rec_v6 / var_real_v6) * 100.0 if var_real_v6 > 0 else 0.0
                
                write_metric("ptb_xl", mid, "QRS_Overall", {
                    "mae": ba_qrs['mae'], "pearson_r": ba_qrs['pearson_r'], "r2": ba_qrs['r2'],
                    "bland_bias": ba_qrs['bias'], "loa_low": ba_qrs['loa_low'], "loa_high": ba_qrs['loa_high'],
                    "auroc": m_qrs[0], "auroc_ci_low": c_qrs[0][0], "auroc_ci_high": c_qrs[0][1],
                    "auprc": m_qrs[1], "auprc_ci_low": c_qrs[1][0], "auprc_ci_high": c_qrs[1][1],
                    "f1": m_qrs[2], "sens": m_qrs[3], "spec": m_qrs[4], "ppv": m_qrs[5], "npv": m_qrs[6],
                    "adj_or": or_qrs, "adj_or_ci_low": or_ci_qrs[0], "adj_or_ci_high": or_ci_qrs[1],
                    "pval_logistic": p_qrs, "fisher_pval": fish_qrs,
                    "bland_v1_r2": bland_v1_r2, "bland_v3_r2": bland_v3_r2, "bland_v6_r2": bland_v6_r2,
                    "bland_v3_slope": bland_v3_slope, "bland_v3_var_ret": bland_v3_var_ret, 
                    "bland_v6_var_ret": bland_v6_var_ret, "interlead_r2": interlead_r2
                })
                for endpoint, target_values, recon_values in (
                    ("QRS_MissingLeads", df["t_qrs"], df["r_qrs"]),
                    ("LVH_SokolowLyon", df["t_lvh"], df["r_lvh"]),
                ):
                    # Guard: patient_cluster_bootstrap_agreement raises
                    # ValueError if no finite paired observations survive.
                    _pair_finite = _finite_paired_mask(target_values, recon_values)
                    if int(_pair_finite.sum()) < 2:
                        logging.warning(
                            "PTB-XL %s: fewer than 2 finite paired rows; "
                            "skipping paired-inference write.",
                            endpoint,
                        )
                        continue
                    for metric_name in ("mae", "bias"):
                        agreement = patient_cluster_bootstrap_agreement(
                            target_values,
                            recon_values,
                            df["patient_id"],
                            metric_name,
                        )
                        write_paired_inference(
                            "ptb_xl",
                            mid,
                            endpoint,
                            metric_name,
                            agreement,
                            "ptbxl_database.patient_id",
                        )
                    
                # LVH — filter to finite pairs before any AUROC/classification call.
                # df["r_lvh"] is NaN for records where V1/V5 delineation found no
                # valid beats; passing those NaNs to roc_auc_score raises ValueError.
                ba_lvh = compute_bland_altman_and_regression(df["t_lvh"], df["r_lvh"])
                lvh_finite = _finite_paired_mask(df["t_lvh"], df["r_lvh"])
                n_lvh_finite = int(lvh_finite.sum())
                logging.info(
                    "PTB-XL LVH: %d/%d records have finite paired LVH voltage.",
                    n_lvh_finite, len(df),
                )
                df_lvh = df[lvh_finite].copy()
                y_t_lvh = (df_lvh["t_lvh"] > 3.5).astype(int)
                y_p_lvh = (df_lvh["r_lvh"] > 3.5).astype(int)
                m_lvh, c_lvh = _classification_with_nan_filter(
                    y_t_lvh, y_p_lvh, df_lvh["r_lvh"]
                )
                df_lvh["pred_lvh"] = y_p_lvh
                or_lvh, or_ci_lvh, p_lvh = fit_logistic_regression(df_lvh, "HYP_label", "pred_lvh")
                _, fish_lvh = compute_fisher_exact(y_t_lvh, y_p_lvh)

                write_metric("ptb_xl", mid, "LVH_SokolowLyon", {
                    "mae": ba_lvh['mae'], "pearson_r": ba_lvh['pearson_r'], "r2": ba_lvh['r2'],
                    "bland_bias": ba_lvh['bias'], "loa_low": ba_lvh['loa_low'], "loa_high": ba_lvh['loa_high'],
                    "auroc": m_lvh[0], "auroc_ci_low": c_lvh[0][0], "auroc_ci_high": c_lvh[0][1],
                    "auprc": m_lvh[1], "auprc_ci_low": c_lvh[1][0], "auprc_ci_high": c_lvh[1][1],
                    "f1": m_lvh[2], "sens": m_lvh[3], "spec": m_lvh[4], "ppv": m_lvh[5], "npv": m_lvh[6],
                    "adj_or": or_lvh, "adj_or_ci_low": or_ci_lvh[0], "adj_or_ci_high": or_ci_lvh[1],
                    "pval_logistic": p_lvh, "fisher_pval": fish_lvh
                })

                # 12-Lead Signal Metrics Summary Output
                for lead_name in LEAD_NAMES:
                    lead_maes = [rec["lead_metrics"][lead_name]["mae"] for rec in records]
                    lead_pearsons = [rec["lead_metrics"][lead_name]["pearson_r"] for rec in records]
                    lead_r2s = [rec["lead_metrics"][lead_name]["r2"] for rec in records]
                    lead_biases = [rec["lead_metrics"][lead_name]["bias"] for rec in records]
                    lead_sds = [rec["lead_metrics"][lead_name]["sd_diff"] for rec in records]
                    
                    mean_mae = float(np.mean(lead_maes))
                    mean_r = float(np.mean(lead_pearsons))
                    mean_r2 = float(np.mean(lead_r2s))
                    mean_bias = float(np.mean(lead_biases))
                    mean_sd = float(np.mean(lead_sds))
                    loa_low = mean_bias - 1.96 * mean_sd
                    loa_high = mean_bias + 1.96 * mean_sd
                    
                    write_metric("ptb_xl", mid, f"Signal_Lead_{lead_name}", {
                        "mae": mean_mae, "pearson_r": mean_r, "r2": mean_r2,
                        "bland_bias": mean_bias, "loa_low": loa_low, "loa_high": loa_high
                    })

                # Per-Lead ST-segment Elevation/Depression Deviation
                write_metric(
                    "ptb_xl",
                    mid,
                    "Delineation_Missing_Lead_Coverage",
                    {"mae": float(df["paired_missing_lead_coverage"].mean())},
                )
                for lead_name in MISSING_LEAD_NAMES:
                    st_pairs = [
                        (
                            rec.get("st_t_leads", {}).get(lead_name, np.nan),
                            rec.get("st_r_leads", {}).get(lead_name, np.nan),
                        )
                        for rec in records
                    ]
                    st_t_vals = np.asarray([pair[0] for pair in st_pairs], dtype=float)
                    st_r_vals = np.asarray([pair[1] for pair in st_pairs], dtype=float)
                    finite = np.isfinite(st_t_vals) & np.isfinite(st_r_vals)
                    if np.sum(finite) >= 2:
                        ba_st = compute_bland_altman_and_regression(st_t_vals[finite], st_r_vals[finite])
                        write_metric("ptb_xl", mid, f"ST_Lead_{lead_name}", {
                            "mae": ba_st["mae"], "pearson_r": ba_st["pearson_r"], "r2": ba_st["r2"],
                            "bland_bias": ba_st["bias"], "loa_low": ba_st["loa_low"], "loa_high": ba_st["loa_high"]
                        })

    # -------------------------------------------------------------------------
    # DATASET 2: EchoNext
    # -------------------------------------------------------------------------
    logging.info("=== Starting Dataset 2: EchoNext ===")
    echo_file = project_root / "data/echonext/EchoNext_test_waveforms.npy"
    if echo_file.exists():
        echo_data, _ = load_echonext_waveforms(project_root / "data/echonext")
        n_echo = len(echo_data)
        echo_clinical_records = min(1000, n_echo)
        echonext_model_root = (
            project_root
            / "ecg_fm_integration/echonext_minimodel_repo/7-EchoNext Minimodel"
        )
        shd_classifier = EchoNextMiniModel(echonext_model_root, device)
        shd_metadata, shd_tabular, shd_labels = load_echonext_test_metadata(
            project_root / "data/echonext/echonext_metadata_100k.csv",
            shd_classifier.transformer_path,
        )
        if len(shd_metadata) != n_echo:
            raise ValueError(
                f"EchoNext metadata/waveform mismatch: {len(shd_metadata)} vs {n_echo}"
            )
        reference_shd_chunks = []
        for start_idx in range(0, n_echo, 256):
            stop_idx = min(start_idx + 256, n_echo)
            reference_shd_chunks.append(
                shd_classifier.predict_official_waveforms(
                    echo_data.official_normalized_batch(start_idx, stop_idx),
                    shd_tabular[start_idx:stop_idx],
                )
            )
        reference_shd = np.concatenate(reference_shd_chunks)
        reference_macro_auroc = _binary_probability_metric(
            shd_labels, reference_shd, "auroc"
        )
        if not 0.79 <= reference_macro_auroc <= 0.82:
            raise RuntimeError(
                "EchoNext official-waveform classifier sanity gate failed: "
                f"macro AUROC={reference_macro_auroc:.4f}"
            )
        logging.info(
            "EchoNext loaded with %d classifier records and %d delineation records.",
            n_echo,
            echo_clinical_records,
        )
        
        for spec in models:
            mid = spec["id"]
            if mid in evaluated_echonext:
                continue
            logging.info(f"EchoNext | Evaluating {mid}...")
            try: adapter = load_adapter(spec, device)
            except Exception as e: logging.error(f"Failed to load {mid}: {e}"); continue
            
            records = []
            reconstructed_shd_chunks = []
            bs = 32
            with torch.inference_mode():
                for start_idx in range(0, n_echo, bs):
                    stop_idx = min(start_idx + bs, n_echo)
                    target = torch.from_numpy(
                        echo_data.batch(start_idx, stop_idx)
                    ).float().to(device)
                    try: recon = adapter.reconstruct(target)
                    except: continue
                    reconstructed_shd_chunks.append(
                        shd_classifier.predict_reconstruction_500hz(
                            recon,
                            shd_tabular[start_idx:stop_idx],
                        )
                    )

                    if start_idx < echo_clinical_records:
                        take = min(stop_idx, echo_clinical_records) - start_idx
                        target_np = target[:take].cpu().numpy()
                        recon_np = recon[:take].cpu().numpy()
                        tasks = [
                            (target_np[index], recon_np[index])
                            for index in range(target_np.shape[0])
                        ]
                        batch_res = list(executor.map(extract_biomarkers_kernel, tasks))

                        for local_index, res in enumerate(batch_res):
                            if res is not None:
                                t_res, r_res, lead_metrics, missing_summary, boundary_errors, dice_scores, t_peaks, r_peaks = res
                                metadata_index = start_idx + local_index
                                records.append({
                                    "patient_id": shd_metadata.iloc[metadata_index]["patient_key"],
                                    "t_qrs": t_res["qrs_ms"], "r_qrs": r_res["qrs_ms"],
                                    "t_lvh": t_res["lvh_mv"], "r_lvh": r_res["lvh_mv"],
                                    "lead_metrics": lead_metrics,
                                    "paired_missing_lead_coverage": boundary_errors[
                                        "paired_missing_lead_coverage"
                                    ],
                                    "t_peaks": t_peaks, "r_peaks": r_peaks
                                })

            reconstructed_shd = np.concatenate(reconstructed_shd_chunks)
                            
            if records:
                df = pd.DataFrame(records)
                ba_qrs = compute_bland_altman_and_regression(df["t_qrs"], df["r_qrs"])
                y_t_qrs, y_p_qrs = (df["t_qrs"] > 120).astype(int), (df["r_qrs"] > 120).astype(int)
                m_qrs, c_qrs = compute_classification_metrics(y_t_qrs, y_p_qrs, df["r_qrs"])
                _, fish_qrs = compute_fisher_exact(y_t_qrs, y_p_qrs)
                
                write_metric("echonext", mid, "QRS_Overall", {
                    "mae": ba_qrs['mae'], "pearson_r": ba_qrs['pearson_r'], "r2": ba_qrs['r2'],
                    "bland_bias": ba_qrs['bias'], "loa_low": ba_qrs['loa_low'], "loa_high": ba_qrs['loa_high'],
                    "auroc": m_qrs[0], "auroc_ci_low": c_qrs[0][0], "auroc_ci_high": c_qrs[0][1],
                    "auprc": m_qrs[1], "auprc_ci_low": c_qrs[1][0], "auprc_ci_high": c_qrs[1][1],
                    "f1": m_qrs[2], "sens": m_qrs[3], "spec": m_qrs[4], "ppv": m_qrs[5], "npv": m_qrs[6],
                    "fisher_pval": fish_qrs
                })
                for endpoint, target_values, recon_values in (
                    ("QRS_MissingLeads", df["t_qrs"], df["r_qrs"]),
                    ("LVH_SokolowLyon", df["t_lvh"], df["r_lvh"]),
                ):
                    # Guard: same NaN-LVH issue exists for EchoNext.
                    _pair_finite = _finite_paired_mask(target_values, recon_values)
                    if int(_pair_finite.sum()) < 2:
                        logging.warning(
                            "EchoNext %s: fewer than 2 finite paired rows; "
                            "skipping paired-inference write.",
                            endpoint,
                        )
                        continue
                    for metric_name in ("mae", "bias"):
                        agreement = patient_cluster_bootstrap_agreement(
                            target_values,
                            recon_values,
                            df["patient_id"],
                            metric_name,
                        )
                        write_paired_inference(
                            "echonext",
                            mid,
                            endpoint,
                            metric_name,
                            agreement,
                            "echonext_metadata.patient_key",
                        )

                write_metric(
                    "echonext",
                    mid,
                    "Delineation_Missing_Lead_Coverage",
                    {"mae": float(df["paired_missing_lead_coverage"].mean())},
                )

                shd_aurocs, shd_auprcs = [], []
                for task_index, task_name in enumerate(SHD_TASKS):
                    labels = shd_labels[:, task_index].astype(int)
                    probabilities = reconstructed_shd[:, task_index]
                    predictions = (probabilities >= 0.5).astype(int)
                    metrics, intervals = compute_classification_metrics(
                        labels,
                        predictions,
                        probabilities,
                        n_bootstraps=50,
                    )
                    shd_aurocs.append(metrics[0])
                    shd_auprcs.append(metrics[1])
                    write_metric(
                        "echonext",
                        mid,
                        f"EchoNextSHD_{task_name}",
                        {
                            "auroc": metrics[0],
                            "auroc_ci_low": intervals[0][0],
                            "auroc_ci_high": intervals[0][1],
                            "auprc": metrics[1],
                            "auprc_ci_low": intervals[1][0],
                            "auprc_ci_high": intervals[1][1],
                            "f1": metrics[2],
                            "sens": metrics[3],
                            "spec": metrics[4],
                            "ppv": metrics[5],
                            "npv": metrics[6],
                        },
                    )
                write_metric(
                    "echonext",
                    mid,
                    "EchoNextSHD_Macro_12",
                    {
                        "auroc": float(np.nanmean(shd_aurocs)),
                        "auprc": float(np.nanmean(shd_auprcs)),
                    },
                )
                for metric_name in ("auroc", "auprc", "brier", "ece"):
                    paired = patient_cluster_bootstrap_delta(
                        shd_labels,
                        reference_shd,
                        reconstructed_shd,
                        shd_metadata["patient_key"].to_numpy(),
                        metric_name,
                    )
                    write_paired_inference(
                        "echonext",
                        mid,
                        "EchoNextSHD_Macro_12",
                        metric_name,
                        paired,
                        "echonext_metadata.patient_key",
                    )

                # 12-Lead Signal Metrics Summary Output for EchoNext
                for lead_name in LEAD_NAMES:
                    lead_maes = [rec["lead_metrics"][lead_name]["mae"] for rec in records]
                    lead_pearsons = [rec["lead_metrics"][lead_name]["pearson_r"] for rec in records]
                    lead_r2s = [rec["lead_metrics"][lead_name]["r2"] for rec in records]
                    lead_biases = [rec["lead_metrics"][lead_name]["bias"] for rec in records]
                    lead_sds = [rec["lead_metrics"][lead_name]["sd_diff"] for rec in records]
                    
                    mean_mae = float(np.mean(lead_maes))
                    mean_r = float(np.mean(lead_pearsons))
                    mean_r2 = float(np.mean(lead_r2s))
                    mean_bias = float(np.mean(lead_biases))
                    mean_sd = float(np.mean(lead_sds))
                    loa_low = mean_bias - 1.96 * mean_sd
                    loa_high = mean_bias + 1.96 * mean_sd
                    
                    write_metric("echonext", mid, f"Signal_Lead_{lead_name}", {
                        "mae": mean_mae, "pearson_r": mean_r, "r2": mean_r2,
                        "bland_bias": mean_bias, "loa_low": loa_low, "loa_high": loa_high
                    })

    # Global Dataset Signal Cache (load datasets ONCE into RAM across all model evaluations)
    _DATASET_CACHE = {}

    def get_cached_signals(ds_name, loader_fn):
        if ds_name not in _DATASET_CACHE:
            logging.info(f"Loading {ds_name} dataset into cache...")
            _DATASET_CACHE[ds_name] = loader_fn()
            logging.info(f"Cached {len(_DATASET_CACHE[ds_name])} records for {ds_name}.")
        return _DATASET_CACHE[ds_name]

    # -------------------------------------------------------------------------
    # HELPER FOR EXTRA DATASETS (Sunnybrook, LUDB, ISP, Zhejiang) WITH CACHING & OOM GUARDS
    # -------------------------------------------------------------------------
    def evaluate_waveform_signals(ds_name, loader_fn, evaluated_set):
        dataset_records = get_cached_signals(ds_name, loader_fn)
        if not dataset_records:
            return
        signals_list = [record["signal"] for record in dataset_records]
        cluster_ids = np.asarray([record["cluster_id"] for record in dataset_records])
        cluster_source = dataset_records[0]["cluster_source"]
            
        total_models = len(models)
        for model_idx, spec in enumerate(models, 1):
            mid = spec["id"]
            if mid in evaluated_set:
                continue
            logging.info(f"{ds_name} | [{model_idx}/{total_models}] Evaluating {mid}...")
            try: adapter = load_adapter(spec, device)
            except Exception as e: logging.error(f"Failed to load {mid}: {e}"); continue
            
            records = []
            bs = 32
            with torch.inference_mode():
                for start_idx in range(0, len(signals_list), bs):
                    chunk = signals_list[start_idx:start_idx+bs]
                    target = torch.from_numpy(np.stack(chunk)).float().to(device)
                    try: recon = adapter.reconstruct(target)
                    except: continue
                    
                    target_np, recon_np = target.cpu().numpy(), recon.cpu().numpy()
                    tasks = [(target_np[i], recon_np[i]) for i in range(target_np.shape[0])]
                    batch_res = list(executor.map(extract_biomarkers_kernel, tasks, chunksize=8))
                    
                    for local_index, res in enumerate(batch_res):
                        if res is not None:
                            t_res, r_res, lead_metrics, missing_summary, boundary_errors, dice_scores, t_peaks, r_peaks = res
                            records.append({
                                "patient_id": cluster_ids[start_idx + local_index],
                                "t_qrs": t_res["qrs_ms"], "r_qrs": r_res["qrs_ms"],
                                "t_lvh": t_res["lvh_mv"], "r_lvh": r_res["lvh_mv"],
                                "lead_metrics": lead_metrics,
                                "missing": missing_summary,
                                "boundary": boundary_errors,
                                "dice": dice_scores,
                                "paired_missing_lead_coverage": boundary_errors[
                                    "paired_missing_lead_coverage"
                                ],
                            })
                            
                    del target, recon, target_np, recon_np
                    
            if records:
                df = pd.DataFrame(records)
                ba_qrs = compute_bland_altman_and_regression(df["t_qrs"], df["r_qrs"])
                y_t_qrs, y_p_qrs = (df["t_qrs"] > 120).astype(int), (df["r_qrs"] > 120).astype(int)
                m_qrs, c_qrs = compute_classification_metrics(y_t_qrs, y_p_qrs, df["r_qrs"])
                _, fish_qrs = compute_fisher_exact(y_t_qrs, y_p_qrs)
                
                # QRS Duration Biomarker
                write_metric(ds_name, mid, "QRS_Overall", {
                    "mae": ba_qrs['mae'], "pearson_r": ba_qrs['pearson_r'], "r2": ba_qrs['r2'],
                    "bland_bias": ba_qrs['bias'], "loa_low": ba_qrs['loa_low'], "loa_high": ba_qrs['loa_high'],
                    "auroc": m_qrs[0], "auroc_ci_low": c_qrs[0][0], "auroc_ci_high": c_qrs[0][1],
                    "auprc": m_qrs[1], "auprc_ci_low": c_qrs[1][0], "auprc_ci_high": c_qrs[1][1],
                    "f1": m_qrs[2], "sens": m_qrs[3], "spec": m_qrs[4], "ppv": m_qrs[5], "npv": m_qrs[6],
                    "fisher_pval": fish_qrs
                })
                for endpoint, target_values, recon_values in (
                    ("QRS_MissingLeads", df["t_qrs"], df["r_qrs"]),
                    ("LVH_SokolowLyon", df["t_lvh"], df["r_lvh"]),
                ):
                    for metric_name in ("mae", "bias"):
                        agreement = patient_cluster_bootstrap_agreement(
                            target_values,
                            recon_values,
                            df["patient_id"],
                            metric_name,
                        )
                        write_paired_inference(
                            ds_name,
                            mid,
                            endpoint,
                            metric_name,
                            agreement,
                            cluster_source,
                        )

                write_metric(
                    ds_name,
                    mid,
                    "Delineation_Missing_Lead_Coverage",
                    {"mae": float(df["paired_missing_lead_coverage"].mean())},
                )

                # LVH Sokolow-Lyon Biomarker
                ba_lvh = compute_bland_altman_and_regression(df["t_lvh"], df["r_lvh"])
                write_metric(ds_name, mid, "LVH_SokolowLyon", {
                    "mae": ba_lvh['mae'], "pearson_r": ba_lvh['pearson_r'], "r2": ba_lvh['r2'],
                    "bland_bias": ba_lvh['bias'], "loa_low": ba_lvh['loa_low'], "loa_high": ba_lvh['loa_high']
                })

                # Missing Lead Metrics Summary
                for m_key, m_label in [("val_missing_pearson", "Signal_Missing_Leads_Pearson"), ("val_missing_mse", "Signal_Missing_Leads_MSE"), ("val_missing_snr_db", "Signal_Missing_Leads_SNR_dB"), ("val_missing_dtw", "Signal_Missing_Leads_DTW")]:
                    vals = [rec["missing"][m_key] for rec in records if not np.isnan(rec["missing"][m_key])]
                    if vals:
                        m_mean = float(np.mean(vals))
                        write_metric(ds_name, mid, m_label, {"mae": m_mean})

                # Boundary Timing Error Metrics (MAE in ms)
                for b_key, b_label in [("p_onset_mae_ms", "Boundary_P_Onset_MAE_ms"), ("p_offset_mae_ms", "Boundary_P_Offset_MAE_ms"), ("r_onset_mae_ms", "Boundary_R_Onset_MAE_ms"), ("r_offset_mae_ms", "Boundary_R_Offset_MAE_ms"), ("t_onset_mae_ms", "Boundary_T_Onset_MAE_ms"), ("t_offset_mae_ms", "Boundary_T_Offset_MAE_ms")]:
                    b_vals = [rec["boundary"][b_key] for rec in records if not np.isnan(rec["boundary"][b_key])]
                    if b_vals:
                        b_mean = float(np.mean(b_vals))
                        write_metric(ds_name, mid, b_label, {"mae": b_mean})

                # Morphological Wave Dice Overlap Scores
                for d_key, d_label in [("p_wave_dice", "Morphology_P_Wave_Dice"), ("qrs_wave_dice", "Morphology_QRS_Wave_Dice"), ("t_wave_dice", "Morphology_T_Wave_Dice")]:
                    d_vals = [rec["dice"][d_key] for rec in records if not np.isnan(rec["dice"][d_key])]
                    if d_vals:
                        d_mean = float(np.mean(d_vals))
                        write_metric(ds_name, mid, d_label, {"mae": d_mean})

                for lead_name in LEAD_NAMES:
                    lead_maes = [rec["lead_metrics"][lead_name]["mae"] for rec in records]
                    lead_pearsons = [rec["lead_metrics"][lead_name]["pearson_r"] for rec in records]
                    lead_r2s = [rec["lead_metrics"][lead_name]["r2"] for rec in records]
                    lead_biases = [rec["lead_metrics"][lead_name]["bias"] for rec in records]
                    lead_sds = [rec["lead_metrics"][lead_name]["sd_diff"] for rec in records]
                    
                    mean_mae = float(np.mean(lead_maes))
                    mean_r = float(np.mean(lead_pearsons))
                    mean_r2 = float(np.mean(lead_r2s))
                    mean_bias = float(np.mean(lead_biases))
                    mean_sd = float(np.mean(lead_sds))
                    loa_low = mean_bias - 1.96 * mean_sd
                    loa_high = mean_bias + 1.96 * mean_sd
                    
                    write_metric(ds_name, mid, f"Signal_Lead_{lead_name}", {
                        "mae": mean_mae, "pearson_r": mean_r, "r2": mean_r2,
                        "bland_bias": mean_bias, "loa_low": loa_low, "loa_high": loa_high
                    })
                        
            # OOM & RAM Guardrail
            del adapter
            gc.collect()

    # -------------------------------------------------------------------------
    # DATASET 3: Sunnybrook
    # -------------------------------------------------------------------------
    sunny_dir = project_root / "data/sunnybrook_12_lead_ecg_samples"
    if sunny_dir.exists():
        def load_sunny():
            sunny_xmls = sorted(list(sunny_dir.glob("*.xml")))
            signals = []
            for xml in sunny_xmls:
                try:
                    record = load_sunnybrook_record(xml)
                    signals.append({
                        "signal": record["signal"],
                        "cluster_id": record["record_id"],
                        "cluster_source": "sunnybrook.record_id_patient_mapping_unavailable",
                    })
                except Exception as e: logging.warning(f"Skipping Sunnybrook {xml.name}: {e}")
            return signals
        evaluate_waveform_signals("sunnybrook", load_sunny, evaluated_sunnybrook)

    # -------------------------------------------------------------------------
    # DATASET 4: LUDB
    # -------------------------------------------------------------------------
    ludb_dir = project_root / "data/ludb"
    if ludb_dir.exists():
        def load_ludb_data():
            try:
                recs, _ = load_ludb(ludb_dir, max_records=0)
                return [{
                    "signal": r.signal_mv,
                    "cluster_id": r.record_id,
                    "cluster_source": "ludb.record_id",
                } for r in recs]
            except Exception as e:
                logging.warning(f"Could not load LUDB: {e}")
                return []
        evaluate_waveform_signals("ludb", load_ludb_data, evaluated_ludb)

    # -------------------------------------------------------------------------
    # DATASET 5: ISP Delineation
    # -------------------------------------------------------------------------
    isp_dir = project_root / "data/isp_delineation_dataset"
    if isp_dir.exists():
        def load_isp_data():
            try:
                recs, _ = load_isp(isp_dir, max_records=0)
                return [{
                    "signal": r.signal_mv,
                    "cluster_id": r.record_id,
                    "cluster_source": "isp.record_id_patient_mapping_unavailable",
                } for r in recs]
            except Exception as e:
                logging.warning(f"Could not load ISP: {e}")
                return []
        evaluate_waveform_signals("isp", load_isp_data, evaluated_isp)

    # -------------------------------------------------------------------------
    # DATASET 6: Zhejiang
    # -------------------------------------------------------------------------
    zhej_dir = project_root / "data/zhejiang"
    if zhej_dir.exists():
        def load_zhej_data():
            try:
                recs, _ = load_zhejiang(zhej_dir, max_records=0)
                return [{
                    "signal": r.signal_mv,
                    "cluster_id": r.record_id,
                    "cluster_source": "zhejiang.record_id_patient_mapping_unavailable",
                } for r in recs]
            except Exception as e:
                logging.warning(f"Could not load Zhejiang: {e}")
                return []
        evaluate_waveform_signals("zhejiang", load_zhej_data, evaluated_zhejiang)

    logging.info("=== Multi-Dataset Clinical Biomarker Evaluation Completed Successfully ===")

if __name__ == "__main__":
    main()
