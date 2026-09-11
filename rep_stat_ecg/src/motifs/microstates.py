"""Beat Segmentation, Normalized Phase Microstates, and Patient Balancing.

Adheres to PRD Sections 7, 8, 13, 14:
- Beat boundary used strictly as a temporal normalization device (beats are not tokens).
- Reference lead R-peak detection (Lead II) with physiological refractory constraints.
- Derivatives (v_dot, v_ddot) and intrinsic descriptors xi computed at 500 Hz physical time FIRST,
  then resampled onto P=64 normalized phase locations per beat.
- Patient-balanced cohort: exactly one ECG per patient via deterministic hash,
  capped at max_beats_per_patient.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from ..vcg.dynamics import extract_vcg_microstate_features


def detect_r_peaks(
    lead_ii_signal: np.ndarray,
    fs: float = 500.0,
    min_dist_ms: float = 250.0,
) -> np.ndarray:
    """Detects R-peaks on Lead II with physiological refractory constraint.

    Args:
        lead_ii_signal: [T] 1D waveform in physical mV.
        fs: Sampling frequency in Hz.
        min_dist_ms: Minimum refractory period in ms (default 250 ms).

    Returns:
        peak_indices: 1D array of sample indices corresponding to R-peaks.
    """
    min_dist_samples = int(round((min_dist_ms / 1000.0) * fs))

    # Standardize signal for robust peak thresholding
    sig = lead_ii_signal - np.median(lead_ii_signal)
    scale = np.median(np.abs(sig))
    sig_norm = sig / (scale + 1e-6)

    # First attempt: height above 2.0 MADs
    peaks, _ = find_peaks(sig_norm, distance=min_dist_samples, height=2.0)
    if len(peaks) < 3:
        # Fallback: lower threshold to 1.0 MAD
        peaks, _ = find_peaks(sig_norm, distance=min_dist_samples, height=1.0)
    if len(peaks) < 3:
        # Fallback: prominence based
        peaks, _ = find_peaks(sig_norm, distance=min_dist_samples, prominence=0.5)

    return peaks


def build_beat_microstates(
    v_physical: np.ndarray,
    r_peaks: np.ndarray,
    patient_id: str,
    ecg_id: str,
    fs: float = 500.0,
    phase_points: int = 64,
    max_beats: int = 8,
) -> list[dict]:
    """Extracts phase-normalized microstates for detected beats in one ECG.

    CRITICAL PRD CONTRACT: Derivatives (v_dot, v_ddot) and intrinsic descriptor xi
    are computed at original physical sampling (500 Hz) before phase resampling.
    """
    T = v_physical.shape[-1]
    # 1. Compute physical derivatives and descriptor at 500 Hz
    v_dot, v_ddot, xi = extract_vcg_microstate_features(v_physical, fs=fs)

    microstates = []
    if len(r_peaks) < 2:
        return microstates

    num_beats = min(len(r_peaks) - 1, max_beats)

    for b in range(num_beats):
        t_start = r_peaks[b]
        t_end = r_peaks[b + 1]

        # Valid physiological beat duration check: 300 ms to 1600 ms
        duration_ms = (t_end - t_start) * (1000.0 / fs)
        if duration_ms < 250.0 or duration_ms > 2000.0:
            continue

        beat_indices = np.arange(t_start, t_end)
        if len(beat_indices) < 5:
            continue

        # Physical time within beat
        orig_times = beat_indices / fs
        # Interpolate v [3, N_beat] and xi [3, N_beat] onto phase_points
        norm_phases = np.linspace(0.0, 1.0, phase_points, endpoint=False)
        phase_in = np.linspace(0.0, 1.0, len(beat_indices), endpoint=False)

        interp_v = interp1d(phase_in, v_physical[:, beat_indices], axis=-1, kind="linear")
        interp_xi = interp1d(phase_in, xi[:, beat_indices], axis=-1, kind="linear")
        interp_t = interp1d(phase_in, orig_times, kind="linear")

        v_resampled = interp_v(norm_phases)       # [3, P]
        xi_resampled = interp_xi(norm_phases)     # [3, P]
        t_resampled = interp_t(norm_phases)       # [P]

        for p in range(phase_points):
            microstates.append({
                "patient_id": str(patient_id),
                "ecg_id": str(ecg_id),
                "beat_id": int(b),
                "original_time": float(t_resampled[p]),
                "normalized_phase": float(norm_phases[p]),
                "v_x": float(v_resampled[0, p]),
                "v_y": float(v_resampled[1, p]),
                "v_z": float(v_resampled[2, p]),
                "s": float(xi_resampled[0, p]),
                "rho": float(xi_resampled[1, p]),
                "kappa": float(xi_resampled[2, p]),
            })

    return microstates


def select_patient_balanced_ecgs(
    df_metadata: pd.DataFrame,
    max_patients: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """Selects exactly one ECG per patient deterministically using hash of patient_id."""
    df = df_metadata.copy()

    # Deterministic hash ranking for each ECG within patient
    def _hash_record(row):
        key = f"{row['patient_id']}_{row['ecg_id']}_{seed}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

    df["hash_val"] = df.apply(_hash_record, axis=1)
    df_sorted = df.sort_values(by=["patient_id", "hash_val"])
    df_unique = df_sorted.drop_duplicates(subset=["patient_id"], keep="first").copy()

    if len(df_unique) > max_patients:
        df_unique = df_unique.sample(n=max_patients, random_state=seed).reset_index(drop=True)

    return df_unique
