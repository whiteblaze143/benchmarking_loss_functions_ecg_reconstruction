"""Ventricular rate estimation, QRS extraction, and electrophysiological endpoint computation.
Implements nonoverlapping 30s local ventricular-rate estimation within observed AF runs:
  r(t) = (N_beats / Delta_t) * 60 bpm
and calculates primary endpoint:
  Q_120 = R_120 / M
  B_120 = R_120 / T
"""

import os
import math
import glob
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import wfdb
import wfdb.processing
import h5py


def extract_beats_ltafdb(
    record_id: str,
    base_dir: str = "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference",
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reads expert-adjudicated beat annotations from LTAFDB .atr file.
    Returns: (beat_samples, beat_symbols, fs)
    """
    rec_path = os.path.join(base_dir, record_id)
    ann = wfdb.rdann(rec_path, 'atr', sampfrom=start_sample or 0, sampto=end_sample)
    fs = float(ann.fs)

    samples = np.array(ann.sample, dtype=np.int64)
    symbols = np.array(ann.symbol)

    # Exclude non-beat markers: rhythm change '+', comment '~', etc.
    # All true ventricular responses during AF (N, V, A, F, j, etc.) are valid beats
    valid_mask = np.isin(symbols, ['+', '~', '|', 'x', '[', ']', '!', '"'], invert=True)
    return samples[valid_mask], symbols[valid_mask], fs


def extract_beats_shdb(
    record_id: str,
    base_dir: str = "/data/mithunmanivannan/datasets/shdb_af",
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reads expert QRS annotations from SHDB-AF .qrs file.
    Returns: (beat_samples, beat_symbols, fs)
    """
    rec_path = os.path.join(base_dir, record_id)
    try:
        ann = wfdb.rdann(rec_path, 'qrs', sampfrom=start_sample or 0, sampto=end_sample)
    except Exception:
        ann = wfdb.rdann(rec_path, 'atr', sampfrom=start_sample or 0, sampto=end_sample)

    fs = float(ann.fs)
    samples = np.array(ann.sample, dtype=np.int64)
    symbols = np.array(ann.symbol)
    valid_mask = np.isin(symbols, ['+', '~', '|', 'x', '[', ']', '!', '"'], invert=True)
    return samples[valid_mask], symbols[valid_mask], fs


def extract_beats_iridia(
    record_id: str,
    start_s: float,
    end_s: float,
    base_dir: str = "/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1"
) -> Tuple[np.ndarray, float]:
    """Runs validated WFDB XQRS detector on IRIDIA Lead 0 ECG slice.
    Returns: (beat_samples_relative_to_start, fs)
    """
    fs = 200.0
    s_idx = int(round(start_s * fs))
    e_idx = int(round(end_s * fs))

    # Identify which daily file in ecg_hdf5/record_XXX contains [s_idx, e_idx]
    # Each daily file is 86,400s (17,280,000 samples)
    day_idx = int(s_idx // 17280000)
    h5_path = os.path.join(base_dir, "ecg_hdf5", record_id, f"{record_id}_ecg_{day_idx:02d}.h5")

    if not os.path.exists(h5_path):
        # Fallback search
        h5_files = sorted(glob.glob(os.path.join(base_dir, "ecg_hdf5", record_id, "*.h5")))
        if h5_files:
            h5_path = h5_files[min(day_idx, len(h5_files) - 1)]
        else:
            return np.array([], dtype=np.int64), fs

    # Read signal
    day_offset = day_idx * 17280000
    local_s = max(0, s_idx - day_offset)
    local_e = max(local_s + 1, e_idx - day_offset)

    with h5py.File(h5_path, 'r') as f:
        key = "ecg" if "ecg" in f else list(f.keys())[0]
        # In IRIDIA, shape is (samples, channels)
        if f[key].ndim == 2:
            if f[key].shape[1] <= 12:
                sig_slice = f[key][local_s:local_e, 0]
            else:
                sig_slice = f[key][0, local_s:local_e]
        else:
            sig_slice = f[key][local_s:local_e]

    if len(sig_slice) < int(fs * 2.0):
        return np.array([], dtype=np.int64), fs

    # Run xqrs_detect
    qrs_inds = wfdb.processing.xqrs_detect(sig_slice, fs=fs, verbose=False)
    # Global samples
    global_qrs = qrs_inds + s_idx
    return global_qrs, fs


def compute_local_ventricular_rates(
    beat_samples: np.ndarray,
    fs: float,
    af_start_s: float,
    af_end_s: float,
    window_s: float = 30.0
) -> List[Dict[str, Any]]:
    """Calculates nonoverlapping window_s local ventricular-rate estimates entirely within an AF run.
    """
    dur_s = af_end_s - af_start_s
    if dur_s < window_s:
        # Episode too short for full window
        return []

    n_windows = int(dur_s // window_s)
    rate_intervals = []

    for w_i in range(n_windows):
        ws = af_start_s + w_i * window_s
        we = ws + window_s
        ws_samp = int(round(ws * fs))
        we_samp = int(round(we * fs))

        # Beats in window
        in_win = (beat_samples >= ws_samp) & (beat_samples < we_samp)
        n_beats = int(np.sum(in_win))
        win_beats = beat_samples[in_win]

        rate_bpm = (n_beats / window_s) * 60.0

        # RR metrics
        if n_beats >= 3:
            rr_ms = np.diff(win_beats) * (1000.0 / fs)
            mean_rr = float(np.mean(rr_ms))
            sd_rr = float(np.std(rr_ms))
            cv_rr = sd_rr / mean_rr if mean_rr > 0 else 0.0
        else:
            mean_rr = np.nan
            cv_rr = np.nan

        rate_intervals.append({
            "start_time_s": ws,
            "end_time_s": we,
            "duration_s": window_s,
            "n_beats": n_beats,
            "rate_bpm": rate_bpm,
            "is_rapid_120": (rate_bpm > 120.0),
            "is_rapid_110": (rate_bpm > 110.0),
            "is_rapid_130": (rate_bpm > 130.0),
            "mean_rr_ms": mean_rr,
            "cv_rr": cv_rr
        })

    return rate_intervals


def compute_physiological_endpoints(
    rate_intervals: List[Dict[str, Any]],
    total_af_s: float,
    window_duration_s: float
) -> Dict[str, Any]:
    """Computes Q_120, B_120, coverage, and secondary rate metrics for an observation window."""
    if not rate_intervals or total_af_s <= 0:
        return {
            "measurable_af_s": 0.0,
            "rate_coverage_fraction": 0.0,
            "r_120_s": 0.0,
            "q_120": 0.0,
            "q_120_lower_bound": 0.0,
            "q_120_upper_bound": 1.0,
            "b_120": 0.0,
            "mean_af_rate_bpm": np.nan,
            "median_af_rate_bpm": np.nan,
            "p90_af_rate_bpm": np.nan,
            "q_110": 0.0,
            "q_130": 0.0,
            "longest_rapid_run_s": 0.0,
            "mean_cv_rr": np.nan
        }

    df_rates = pd.DataFrame(rate_intervals)
    meas_af_s = df_rates["duration_s"].sum()
    coverage = min(1.0, meas_af_s / total_af_s)
    unknown_af_s = max(0.0, total_af_s - meas_af_s)

    r_120_s = df_rates[df_rates["is_rapid_120"]]["duration_s"].sum()
    r_110_s = df_rates[df_rates["is_rapid_110"]]["duration_s"].sum()
    r_130_s = df_rates[df_rates["is_rapid_130"]]["duration_s"].sum()

    q_120 = r_120_s / total_af_s
    b_120 = r_120_s / window_duration_s

    # Bounds under unknown rate
    q_120_lower = r_120_s / total_af_s
    q_120_upper = min(1.0, (r_120_s + unknown_af_s) / total_af_s)

    # Longest contiguous rapid run
    longest_rapid = 0.0
    curr_rapid = 0.0
    for _, row in df_rates.iterrows():
        if row["is_rapid_120"]:
            curr_rapid += row["duration_s"]
            longest_rapid = max(longest_rapid, curr_rapid)
        else:
            curr_rapid = 0.0

    return {
        "measurable_af_s": meas_af_s,
        "rate_coverage_fraction": coverage,
        "r_120_s": r_120_s,
        "q_120": q_120,
        "q_120_lower_bound": q_120_lower,
        "q_120_upper_bound": q_120_upper,
        "b_120": b_120,
        "mean_af_rate_bpm": float(df_rates["rate_bpm"].mean()),
        "median_af_rate_bpm": float(df_rates["rate_bpm"].median()),
        "p90_af_rate_bpm": float(np.percentile(df_rates["rate_bpm"], 90)),
        "q_110": r_110_s / total_af_s,
        "q_130": r_130_s / total_af_s,
        "longest_rapid_run_s": longest_rapid,
        "mean_cv_rr": float(df_rates["cv_rr"].dropna().mean()) if not df_rates["cv_rr"].dropna().empty else np.nan
    }
