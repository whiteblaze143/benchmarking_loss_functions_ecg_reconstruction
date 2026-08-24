"""
Structural and alignment-aware ECG metrics used by Sunnybrook evaluation.

This module centralizes beat-aligned, morphology, and lead-order utilities so the
`eval_clinical_sunnybrook.py` script can remain small and focused.

Functions provided (selected):
- prd: Percentage RMS Difference
- median_beat_correlation: median beat correlation across R‑peaks
- cross_correlation_lag: peak cross‑correlation and lag (samples)
- dtw_distance: wrapper around fastdtw (fallback returns np.nan)
- rpeak_aligned_mse: R‑peak aligned MSE over short beat windows
- detect_and_fix_avr_avl_swap: detect and correct common aVR/aVL lead swap

The implementations favor safe fallbacks when optional deps (neurokit2 / fastdtw)
are not installed.
"""
from typing import Tuple
import numpy as np
import torch

# Prefer importing LeadIndex for clarity; fallback to indices if import not possible
try:
    from scripts.inference_pipeline import LeadIndex
    AVR_IDX = LeadIndex.aVR
    AVL_IDX = LeadIndex.aVL
except Exception:
    AVR_IDX = 3
    AVL_IDX = 4


def prd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage RMS Difference (PRD) between y_true and y_pred.

    Args:
        y_true, y_pred: (C, T) numpy arrays or (T,) arrays
    Returns:
        PRD in percent
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sqrt(np.sum((y_true - y_pred) ** 2))
    denom = np.sqrt(np.sum(y_true ** 2)) + 1e-8
    return float(100.0 * num / denom)


def cross_correlation_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 25) -> Tuple[float, int]:
    """Compute normalized max cross‑correlation and lag (in samples).

    a, b: 1D numpy arrays
    """
    if len(a) != len(b):
        L = min(len(a), len(b))
        a = a[:L]
        b = b[:L]
    a_z = (a - a.mean()) / (a.std() + 1e-8)
    b_z = (b - b.mean()) / (b.std() + 1e-8)
    corrs = [
        np.correlate(a_z[max_lag:-max_lag], np.roll(b_z, lag)[max_lag:-max_lag])[0] / (len(a_z)-2*max_lag)
        for lag in range(-max_lag, max_lag + 1)
    ]
    corrs = np.array(corrs)
    best_idx = int(np.argmax(corrs))
    best_lag = best_idx - max_lag
    return float(corrs[best_idx]), int(best_lag)


def dtw_distance(a: np.ndarray, b: np.ndarray):
    try:
        from scipy.spatial.distance import euclidean
        from fastdtw import fastdtw
        dist, _ = fastdtw(a, b, dist=euclidean)
        return float(dist)
    except Exception:
        return float('nan')


def rpeak_aligned_mse(gt: np.ndarray, recon: np.ndarray, fs: int = 500) -> float:
    """Compute MSE of short beat windows aligned by detected R‑peaks.

    Both inputs are 1D signals (single lead) or 2D arrays (n_beats x beat_len).
    This is a best‑effort implementation that uses neurokit2 if available,
    otherwise returns np.nan.
    """
    try:
        import neurokit2 as nk
        r_gt, _ = nk.ecg_peaks(gt, sampling_rate=fs)
        r_recon, _ = nk.ecg_peaks(recon, sampling_rate=fs)
        r_gt_idx = np.where(r_gt['ECG_R_Peaks'])[0]
        r_recon_idx = np.where(r_recon['ECG_R_Peaks'])[0]
        if len(r_gt_idx) < 2 or len(r_recon_idx) < 2:
            return float('nan')
        min_beats = min(len(r_gt_idx), len(r_recon_idx))
        gt_beats = [gt[max(0, idx-50):idx+50] for idx in r_gt_idx[:min_beats]]
        recon_beats = [recon[max(0, idx-50):idx+50] for idx in r_recon_idx[:min_beats]]
        beat_len = min([len(b) for b in gt_beats + recon_beats])
        gt_beats = np.stack([b[:beat_len] for b in gt_beats], axis=0)
        recon_beats = np.stack([b[:beat_len] for b in recon_beats], axis=0)
        mse = float(((gt_beats - recon_beats) ** 2).mean())
        return mse
    except Exception:
        return float('nan')


def median_beat_correlation(gt: np.ndarray, recon: np.ndarray, fs: int = 500) -> float:
    """Compute median correlation across corresponding beats for Lead II.

    Returns np.nan if beat detection is unavailable.
    """
    try:
        import neurokit2 as nk
        r_gt, _ = nk.ecg_peaks(gt, sampling_rate=fs)
        r_recon, _ = nk.ecg_peaks(recon, sampling_rate=fs)
        r_gt_idx = np.where(r_gt['ECG_R_Peaks'])[0]
        r_recon_idx = np.where(r_recon['ECG_R_Peaks'])[0]
        if len(r_gt_idx) < 2 or len(r_recon_idx) < 2:
            return float('nan')
        min_beats = min(len(r_gt_idx), len(r_recon_idx))
        corrs = []
        for i in range(min_beats):
            g = gt[max(0, r_gt_idx[i]-50):r_gt_idx[i]+50]
            r = recon[max(0, r_recon_idx[i]-50):r_recon_idx[i]+50]
            L = min(len(g), len(r))
            if L < 10:
                continue
            corr = np.corrcoef(g[:L], r[:L])[0, 1]
            corrs.append(corr)
        if len(corrs) == 0:
            return float('nan')
        return float(np.median(corrs))
    except Exception:
        return float('nan')


def detect_and_fix_avr_avl_swap(recon: torch.Tensor, gt: torch.Tensor, threshold: float = 0.02) -> Tuple[torch.Tensor, bool]:
    """Detect and correct a common aVR/aVL swap in predicted leads.

    Strategy: compare sum of per-lead correlations for (aVR→aVR + aVL→aVL)
    against the swapped assignment (aVR→aVL + aVL→aVR). If the swapped
    assignment improves correlation by `threshold`, perform the swap and
    return True.

    Args:
        recon: (B,12,T) or (12,T) torch.Tensor in mV
        gt: (B,12,T) or (12,T) torch.Tensor in mV
        threshold: minimum improvement in summed correlation to accept swap

    Returns: (recon_fixed, did_swap)
    """
    single_sample = False
    if recon.dim() == 2:
        single_sample = True
        recon = recon.unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)

    recon_np = recon.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    B = recon_np.shape[0]
    did_any = False

    for b in range(B):
        a_vr_r = recon_np[b, int(AVR_IDX)].copy()
        a_vl_r = recon_np[b, int(AVL_IDX)].copy()
        a_vr_g = gt_np[b, int(AVR_IDX)].copy()
        a_vl_g = gt_np[b, int(AVL_IDX)].copy()

        # Safe correlation with NaN handling
        def safe_corr(x, y):
            try:
                if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    return 0.0
                return float(np.corrcoef(x, y)[0, 1])
            except Exception:
                return 0.0

        corr_orig = safe_corr(a_vr_r, a_vr_g) + safe_corr(a_vl_r, a_vl_g)
        corr_swapped = safe_corr(a_vr_r, a_vl_g) + safe_corr(a_vl_r, a_vr_g)

        if corr_swapped > corr_orig + threshold:
            # Perform swap in-place on recon
            recon[b, int(AVR_IDX)] = torch.from_numpy(a_vl_r).to(recon.device)
            recon[b, int(AVL_IDX)] = torch.from_numpy(a_vr_r).to(recon.device)
            did_any = True

    if single_sample:
        return recon.squeeze(0), did_any
    return recon, did_any
