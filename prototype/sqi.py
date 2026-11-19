"""
Signal Quality Index (SQI) Library for Wearable & Ambulatory ECG.
Evaluates physiological plausibility, beat detection consensus, spectral purity,
and higher-order statistical moments to support quality-aware abstention.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats
import wfdb.processing as wp


def compute_bsqi(
    sig: np.ndarray,
    fs: float,
    rpeaks_ref: np.ndarray | None = None,
    tolerance_ms: float = 50.0,
) -> float:
    """
    Compute Beat Detection Consensus Index (bSQI).
    Compares two independent QRS peak detectors (e.g. XQRS vs GQRS, or detector vs ref).
    Agreement ratio: 2 * N_matched / (N1 + N2).
    """
    if len(sig) < int(fs * 2.0):
        return 0.5  # Too short to evaluate

    try:
        if rpeaks_ref is not None and len(rpeaks_ref) > 0:
            peaks1 = np.asarray(rpeaks_ref, dtype=int)
        else:
            xqrs = wp.XQRS(sig=sig, fs=fs)
            xqrs.detect(verbose=False)
            peaks1 = np.asarray(xqrs.qrs_inds, dtype=int)

        peaks2 = np.asarray(wp.gqrs_detect(sig=sig, fs=fs), dtype=int)
    except Exception:
        return 0.0

    if len(peaks1) == 0 and len(peaks2) == 0:
        return 0.0  # Asystole or severe dropout
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0

    tol_samples = int((tolerance_ms / 1000.0) * fs)
    matched = 0
    for p in peaks1:
        if np.any(np.abs(peaks2 - p) <= tol_samples):
            matched += 1

    bsqi = float((2.0 * matched) / (len(peaks1) + len(peaks2)))
    return float(np.clip(bsqi, 0.0, 1.0))


def compute_psqi(
    sig: np.ndarray,
    fs: float,
    f_qrs_low: float = 5.0,
    f_qrs_high: float = 15.0,
    f_total_low: float = 1.0,
    f_total_high: float = 40.0,
) -> float:
    """
    Compute Power Spectrum Distribution Index (pSQI).
    Ratio of power in the physiological QRS band (5-15 Hz) to the total band (1-40 Hz).
    High values (>0.35) indicate dominant QRS energy. Low values indicate broadband noise or drift.
    """
    if len(sig) < int(fs):
        return 0.0

    nperseg = min(len(sig), int(fs * 4))
    f, psd = signal.welch(sig, fs=fs, nperseg=nperseg)

    band_qrs = np.logical_and(f >= f_qrs_low, f <= f_qrs_high)
    band_total = np.logical_and(f >= f_total_low, f <= f_total_high)

    power_qrs = float(np.sum(psd[band_qrs]))
    power_total = float(np.sum(psd[band_total])) + 1e-10

    ratio = power_qrs / power_total
    # Normalizing ratio: in clean ECG ratio typically ranges between 0.35 and 0.75
    psqi = float(np.clip((ratio - 0.15) / 0.50, 0.0, 1.0))
    return psqi


def compute_ksqi(sig: np.ndarray) -> float:
    """
    Compute Kurtosis Index (kSQI).
    Physiological ECG typically has leptokurtic distribution (kurtosis > 5) due to sharp QRS spikes.
    Gaussian noise has kurtosis ~ 3. Artifact/clipping has low kurtosis (< 2.5).
    """
    if len(sig) < 100:
        return 0.0

    try:
        k = float(stats.kurtosis(sig, fisher=False))
    except Exception:
        return 0.0

    if np.isnan(k) or np.isinf(k):
        return 0.0

    # Gaussian noise ~3.0; clean ECG typically 5.0 - 25.0
    if k < 3.0:
        score = max(0.0, (k - 1.5) / 1.5)
    elif k <= 20.0:
        score = 0.5 + 0.5 * min(1.0, (k - 3.0) / 10.0)
    else:
        # Extremely huge spikes (impulse noise / electrostatic discharge)
        score = max(0.2, 1.0 - (k - 20.0) / 100.0)

    return float(np.clip(score, 0.0, 1.0))


def compute_ssqi(sig: np.ndarray) -> float:
    """
    Compute Skewness Index (sSQI).
    Evaluates waveform asymmetry and baseline stability.
    """
    if len(sig) < 100:
        return 0.0

    try:
        s = float(stats.skew(sig))
    except Exception:
        return 0.0

    if np.isnan(s) or np.isinf(s):
        return 0.0

    # Clean single-lead ECG often has skew between -3 and +3 depending on polarity
    # Extreme skew indicates huge baseline drift or railing
    score = 1.0 - min(1.0, abs(s) / 5.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_baseline_wander_ratio(sig: np.ndarray, fs: float) -> float:
    """
    Evaluate low-frequency baseline drift (<0.5 Hz) relative to ECG signal power.
    Returns drift penalty score in [0, 1] where 1 is clean (no drift) and 0 is severe wander.
    """
    nperseg = min(len(sig), int(fs * 4))
    f, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
    band_drift = f < 0.5
    band_ecg = np.logical_and(f >= 0.5, f <= 40.0)

    p_drift = float(np.sum(psd[band_drift]))
    p_ecg = float(np.sum(psd[band_ecg])) + 1e-10

    drift_ratio = p_drift / (p_drift + p_ecg)
    # Score is 1 when drift is low, decreases when drift dominates
    bw_score = 1.0 - min(1.0, drift_ratio / 0.7)
    return float(np.clip(bw_score, 0.0, 1.0))


def compute_composite_sqi(
    sig: np.ndarray,
    fs: float = 200.0,
    rpeaks_ref: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute Multi-metric Composite Signal Quality Index (SQI).
    Weights:
      - bSQI (beat detection agreement): 0.40
      - pSQI (QRS spectral ratio): 0.30
      - kSQI (kurtosis leptokurtic score): 0.15
      - bwSQI (baseline wander stability): 0.15
    Returns dict containing individual scores and composite 'Q'.
    """
    # Remove NaN or inf
    clean_sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    # Center signal
    clean_sig = clean_sig - np.median(clean_sig)

    bsqi = compute_bsqi(clean_sig, fs, rpeaks_ref=rpeaks_ref)
    psqi = compute_psqi(clean_sig, fs)
    ksqi = compute_ksqi(clean_sig)
    bw_sqi = compute_baseline_wander_ratio(clean_sig, fs)
    ssqi = compute_ssqi(clean_sig)

    # Composite weighted score Q in [0, 1]
    composite_q = (
        0.40 * bsqi +
        0.30 * psqi +
        0.15 * ksqi +
        0.15 * bw_sqi
    )

    return {
        "bSQI": round(bsqi, 4),
        "pSQI": round(psqi, 4),
        "kSQI": round(ksqi, 4),
        "sSQI": round(ssqi, 4),
        "bwSQI": round(bw_sqi, 4),
        "Q": round(float(np.clip(composite_q, 0.0, 1.0)), 4),
    }
