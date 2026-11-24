"""Adaptive QRST cancellation and Atrial Fibrillatory Rate (AFR) dominant frequency estimation.
Implementation based on Petrutiu et al. (IEEE TBME 2006) and Stridh & Sornmo (IEEE TBME 2001).
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from scipy import signal


def bandpass_filter_ecg(
    ecg: np.ndarray,
    fs: float,
    lowcut: float = 1.0,
    highcut: float = 30.0,
    order: int = 3
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter to isolate fibrillatory and QRS components."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, ecg)


def adaptive_median_qrst_cancellation(
    ecg: np.ndarray,
    qrs_samples: np.ndarray,
    fs: float,
    pre_ms: float = 80.0,
    post_ms: float = 240.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Cancels ventricular QRST complexes via aligned running median template subtraction.
    
    Parameters:
      ecg: 1D ECG voltage array.
      qrs_samples: Array of detected R-peak sample indices.
      fs: Sampling frequency in Hz.
      pre_ms: Milliseconds before R-peak for QRST window (default 80ms).
      post_ms: Milliseconds after R-peak for QRST window (default 240ms, covering QRS + ST-T).
      
    Returns:
      (residual_atrial_signal, cancelled_ventricular_template)
    """
    n_samples = len(ecg)
    pre_samp = int(round(pre_ms * fs / 1000.0))
    post_samp = int(round(post_ms * fs / 1000.0))
    win_len = pre_samp + post_samp

    residual = ecg.copy()
    template_signal = np.zeros(n_samples, dtype=np.float64)

    # Valid beats fully within bounds
    valid_beats = [q for q in qrs_samples if (q - pre_samp >= 0 and q + post_samp <= n_samples)]
    if len(valid_beats) < 5:
        return residual, template_signal

    # Extract all QRST complexes
    complexes = np.array([ecg[q - pre_samp : q + post_samp] for q in valid_beats])

    # Moving median template (5-beat adaptive median)
    k_window = 7
    half_k = k_window // 2

    for i, q in enumerate(valid_beats):
        w_start = max(0, i - half_k)
        w_end = min(len(valid_beats), i + half_k + 1)
        local_median = np.median(complexes[w_start:w_end], axis=0)

        # Scale factor alpha = (complex . template) / (template . template)
        c_i = complexes[i]
        denom = np.dot(local_median, local_median)
        alpha = np.dot(c_i, local_median) / denom if denom > 1e-12 else 1.0
        alpha = np.clip(alpha, 0.5, 1.5)

        sub_complex = alpha * local_median
        template_signal[q - pre_samp : q + post_samp] = sub_complex
        residual[q - pre_samp : q + post_samp] -= sub_complex

    return residual, template_signal


def estimate_dominant_frequency(
    atrial_signal: np.ndarray,
    fs: float,
    f_min: float = 3.0,
    f_max: float = 9.0,
    nperseg_s: float = 2.5
) -> Dict[str, Any]:
    """Estimates the dominant frequency (DF) of atrial fibrillatory waves in [f_min, f_max] Hz.
    Uses Welch's average periodogram with Hann windowing.
    
    Returns:
      Dictionary with DF, peak power, spectral concentration, SNR, and validity flag.
    """
    if len(atrial_signal) < int(fs * nperseg_s):
        return {
            "dominant_frequency_hz": np.nan,
            "peak_power": np.nan,
            "spectral_concentration": np.nan,
            "spectral_snr_db": np.nan,
            "is_valid_df": False,
            "rejection_reason": "Signal too short"
        }

    # Bandpass filter specifically for fibrillatory wave band (2.5 - 12.0 Hz)
    nyq = 0.5 * fs
    b, a = signal.butter(3, [2.5 / nyq, 12.0 / nyq], btype='band')
    filtered_aa = signal.filtfilt(b, a, atrial_signal)

    nperseg = int(round(nperseg_s * fs))
    noverlap = int(round(nperseg * 0.5))

    freqs, psd = signal.welch(filtered_aa, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nperseg * 4)

    # Restrict to search band [f_min, f_max]
    band_mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band_mask):
        return {
            "dominant_frequency_hz": np.nan,
            "is_valid_df": False,
            "rejection_reason": "No frequencies in physiological band"
        }

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    peak_idx = np.argmax(band_psd)
    df_hz = float(band_freqs[peak_idx])
    peak_power = float(band_psd[peak_idx])

    # Spectral concentration: power in [DF - 0.5Hz, DF + 0.5Hz] divided by total band power [3, 9] Hz
    df_band = (freqs >= (df_hz - 0.5)) & (freqs <= (df_hz + 0.5))
    peak_band_power = np.sum(psd[df_band])
    total_band_power = np.sum(band_psd)
    sc = float(peak_band_power / total_band_power) if total_band_power > 0 else 0.0

    # Spectral SNR: Peak power vs median background power in band
    bg_power = float(np.median(band_psd))
    snr_db = float(10.0 * np.log10(peak_power / bg_power)) if bg_power > 0 else 0.0

    # Quality criterion: Petrutiu requires SC >= 0.25 and SNR >= 3.0 dB to avoid ambiguity
    is_valid = (sc >= 0.25) and (snr_db >= 3.0)
    reason = "Valid DF" if is_valid else ("Low spectral concentration" if sc < 0.25 else "Low SNR")

    return {
        "dominant_frequency_hz": df_hz,
        "peak_power": peak_power,
        "spectral_concentration": sc,
        "spectral_snr_db": snr_db,
        "is_valid_df": is_valid,
        "rejection_reason": reason
    }
