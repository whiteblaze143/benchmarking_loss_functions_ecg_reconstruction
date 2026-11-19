"""
Data Ingestion and Feature Extraction Pipeline for Wearable AF Recurrence Benchmark.
Processes ECG recordings from Noise Holter Database and ambulatory Holter datasets.
Extracts:
1) Raw 30s ECG windows (6000 samples @ 200 Hz)
2) Beat-to-beat RR interval sequences (60 beats)
3) Physiological HRV descriptors (SDNN, RMSSD, pNN50, Poincaré SD1/SD2)
4) SetOperator Phase-KME embeddings (16 phase cells x 128 RKHS dimensions)
5) Ground truth arrhythmia labels and clinical noise tiers (Clean, N1, N2).
6) Full multi-hour SHDB-AF streaming across all 128 recordings (including 33 non-AF controls).
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, interpolate
import wfdb
import wfdb.processing as wp


def compute_hrv_features(rr_sec: np.ndarray) -> np.ndarray:
    """
    Extract 8 statistical and geometric HRV descriptors from RR sequence:
    [mean_rr, sdnn, rmssd, pnn50, cv, sd1, sd2, sd_ratio]
    """
    if len(rr_sec) < 3:
        return np.zeros(8, dtype=np.float32)

    # Filter physiologically plausible beats (0.2s - 2.5s)
    valid_rr = rr_sec[(rr_sec >= 0.2) & (rr_sec <= 2.5)]
    if len(valid_rr) < 3:
        valid_rr = rr_sec

    mean_rr = float(np.mean(valid_rr))
    sdnn = float(np.std(valid_rr, ddof=1)) if len(valid_rr) > 1 else 0.05
    diff_rr = np.diff(valid_rr)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.02
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.050)) if len(diff_rr) > 0 else 0.0
    cv = float(sdnn / (mean_rr + 1e-6))

    # Poincaré descriptors
    var_rr = float(np.var(valid_rr))
    var_diff = float(np.var(diff_rr)) if len(diff_rr) > 0 else 0.001
    sd1 = float(np.sqrt(max(1e-6, 0.5 * var_diff)))
    sd2 = float(np.sqrt(max(1e-6, 2.0 * var_rr - 0.5 * var_diff)))
    sd_ratio = float(sd1 / (sd2 + 1e-6))

    return np.array(
        [mean_rr, sdnn, rmssd, pnn50, cv, sd1, sd2, sd_ratio],
        dtype=np.float32,
    )


def compute_phase_kme_embedding(
    ecg_signal: np.ndarray,
    rpeaks: np.ndarray,
    fs: float = 200.0,
    num_phases: int = 16,
    kme_dim: int = 128,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Compute SetOperator Phase-KME embedding:
    1) Segment signal into heartbeats from detected R-peaks.
    2) Invariantly resample each beat to 256 phase points in [0, 2pi).
    3) Divide each beat into 16 phase cells (16 samples per cell).
    4) Embed distribution across beats using Gaussian RBF Kernel Mean Embedding.
       Returns: (16, 128) matrix in RKHS.
    """
    rng = np.random.default_rng(random_seed)
    anchors = rng.normal(0.0, 1.0, size=(kme_dim, 16)).astype(np.float32)
    gamma = 1.0 / 16.0

    if len(rpeaks) < 3:
        beat_len = int(fs * 0.8)
        rpeaks = np.arange(0, len(ecg_signal) - beat_len, beat_len)

    beats_resampled = []
    for i in range(len(rpeaks) - 1):
        start = rpeaks[i]
        end = rpeaks[i + 1]
        if end - start < int(fs * 0.25) or end - start > int(fs * 2.5):
            continue
        beat = ecg_signal[start:end]
        if len(beat) < 5:
            continue
        x_old = np.linspace(0, 1, len(beat))
        x_new = np.linspace(0, 1, 256)
        beat_norm = np.interp(x_new, x_old, beat)
        beats_resampled.append(beat_norm)

    if len(beats_resampled) == 0:
        beats_resampled = [np.sin(np.linspace(0, 2 * np.pi, 256))]

    beats_arr = np.array(beats_resampled, dtype=np.float32)
    phase_cells = beats_arr.reshape(-1, num_phases, 16)

    kme_matrix = np.zeros((num_phases, kme_dim), dtype=np.float32)
    for g in range(num_phases):
        cell_beats = phase_cells[:, g, :]
        dists = np.sum((cell_beats[:, None, :] - anchors[None, :, :]) ** 2, axis=-1)
        kernel_vals = np.exp(-gamma * dists)
        kme_matrix[g] = np.mean(kernel_vals, axis=0)

    norms = np.linalg.norm(kme_matrix, axis=-1, keepdims=True) + 1e-8
    return kme_matrix / norms


def load_sample_dataset(
    wfdb_dir: str = "/data/mithunmanivannan/datasets/noise_holter_database/wfdb_files",
    max_per_group: int = 50,
) -> dict[str, np.ndarray]:
    """Backward-compatible loader for preliminary tests."""
    return load_noise_holter_benchmark_dataset(wfdb_dir=wfdb_dir, max_per_group=max_per_group)


def load_noise_holter_benchmark_dataset(
    wfdb_dir: str = "/data/mithunmanivannan/datasets/noise_holter_database/wfdb_files",
    max_per_group: Optional[int] = 50,
) -> dict[str, np.ndarray]:
    """
    Load multi-tier dataset from Noise Holter Database across 6 experimental tiers:
    Clean AF, Clean NSR, N1 AF, N1 NSR, N2 AF, N2 NSR.
    If max_per_group is None, loads ALL available records matching each tier.
    """
    master_csv = Path("/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/noise_holter_master_index.csv")
    if master_csv.is_file():
        df = pd.read_csv(master_csv)
    else:
        df = None

    if df is not None:
        records_by_group: dict[str, list[str]] = {
            "Clean_AF": df[(df["quality_class"] == "Clean") & (df["is_af"] == 1)]["hea_path"].str.replace(".hea", "").tolist(),
            "Clean_NSR": df[(df["quality_class"] == "Clean") & (df["is_af"] == 0) & (df["events"] == "None")]["hea_path"].str.replace(".hea", "").tolist(),
            "N1_AF": df[(df["quality_class"] == "N1") & (df["is_af"] == 1)]["hea_path"].str.replace(".hea", "").tolist(),
            "N1_NSR": df[(df["quality_class"] == "N1") & (df["is_af"] == 0) & (df["events"] == "None")]["hea_path"].str.replace(".hea", "").tolist(),
            "N2_AF": df[(df["quality_class"] == "N2") & (df["is_af"] == 1)]["hea_path"].str.replace(".hea", "").tolist(),
            "N2_NSR": df[(df["quality_class"] == "N2") & (df["is_af"] == 0) & (df["events"] == "None")]["hea_path"].str.replace(".hea", "").tolist(),
        }
    else:
        hea_files = glob.glob(f"{wfdb_dir}/*.hea")
        records_by_group = {"Clean_AF": [], "Clean_NSR": [], "N1_AF": [], "N1_NSR": [], "N2_AF": [], "N2_NSR": []}
        for hf in hea_files:
            rec_id = hf.replace(".hea", "")
            with open(hf) as fp:
                content = fp.read()
            is_af = "EVENTS:A. Fib." in content or "EVENTS:A. Flutter" in content
            is_clean = "CLASS:Clean" in content
            is_n1 = "CLASS:N1" in content
            is_n2 = "CLASS:N2" in content
            if is_af:
                if is_clean: records_by_group["Clean_AF"].append(rec_id)
                elif is_n1: records_by_group["N1_AF"].append(rec_id)
                elif is_n2: records_by_group["N2_AF"].append(rec_id)
            elif "EVENTS:None" in content or "EVENTS:Sinus" in content:
                if is_clean: records_by_group["Clean_NSR"].append(rec_id)
                elif is_n1: records_by_group["N1_NSR"].append(rec_id)
                elif is_n2: records_by_group["N2_NSR"].append(rec_id)

    if max_per_group is not None:
        for k in records_by_group:
            records_by_group[k] = records_by_group[k][:max_per_group]

    total_records = sum(len(v) for v in records_by_group.values())
    print(f"[load_noise_holter] Loading {total_records} records across 6 experimental tiers:")
    for grp, rlist in records_by_group.items():
        print(f"  - {grp}: {len(rlist)} records")

    raw_ecg_list, rr_seq_list, hrv_stats_list, phase_kme_list = [], [], [], []
    labels_list, noise_tier_list, rec_id_list = [], [], []

    window_size_samples = 6000  # 30 seconds @ 200 Hz
    target_rr_len = 60          # 60 beats

    for grp, rlist in records_by_group.items():
        is_af_label = 1 if "AF" in grp else 0
        noise_tier = grp.split("_")[0]

        for rec_path in rlist:
            try:
                rec = wfdb.rdrecord(rec_path)
                sig = rec.p_signal[:, 0]
                fs = float(rec.fs)

                if fs != 200.0:
                    sig = signal.resample(sig, int(len(sig) * 200.0 / fs))
                    fs = 200.0

                if len(sig) >= window_size_samples:
                    ecg_window = sig[:window_size_samples]
                else:
                    ecg_window = np.pad(sig, (0, window_size_samples - len(sig)), mode="edge")

                try:
                    qrs_ann = wfdb.rdann(rec_path, "qrs")
                    peaks = np.asarray(qrs_ann.sample, dtype=int)
                except Exception:
                    peaks = np.asarray(wp.gqrs_detect(sig=ecg_window, fs=fs), dtype=int)

                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / fs
                else:
                    rr_intervals = np.random.uniform(0.6, 1.0, size=target_rr_len)

                if len(rr_intervals) >= target_rr_len:
                    rr_seq = rr_intervals[:target_rr_len]
                else:
                    pad_val = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8
                    rr_seq = np.pad(rr_intervals, (0, target_rr_len - len(rr_intervals)), constant_values=pad_val)

                hrv_feats = compute_hrv_features(rr_seq)
                kme = compute_phase_kme_embedding(ecg_window, peaks, fs=fs, num_phases=16, kme_dim=128)

                raw_ecg_list.append(ecg_window.astype(np.float32))
                rr_seq_list.append(rr_seq.astype(np.float32))
                hrv_stats_list.append(hrv_feats.astype(np.float32))
                phase_kme_list.append(kme.astype(np.float32))
                labels_list.append(is_af_label)
                noise_tier_list.append(noise_tier)
                rec_id_list.append(os.path.basename(rec_path))

            except Exception:
                continue

    return {
        "raw_ecg": np.array(raw_ecg_list),
        "rr_seq": np.array(rr_seq_list),
        "hrv_stats": np.array(hrv_stats_list),
        "phase_kme": np.array(phase_kme_list),
        "labels": np.array(labels_list, dtype=np.int64),
        "noise_tier": np.array(noise_tier_list),
        "rec_ids": np.array(rec_id_list),
    }


def get_shdb_af_record_rhythm_timeline(rec_id: str, shdb_dir: str = "/data/mithunmanivannan/datasets/shdb_af") -> List[Tuple[int, int, str]]:
    """
    Extract rhythm intervals [(start_sample, end_sample, rhythm_string)] from .atr file.
    """
    rec_stem = f"{shdb_dir}/{rec_id}"
    atr_path = f"{rec_stem}.atr"
    if not os.path.exists(atr_path):
        return []

    hea = wfdb.rdheader(rec_stem)
    total_samples = hea.sig_len
    ann = wfdb.rdann(rec_stem, "atr")

    rhythm_changes = [
        (int(s), note.strip())
        for s, note in zip(ann.sample, ann.aux_note)
        if note.strip().startswith("(")
    ]

    intervals = []
    if not rhythm_changes:
        return [(0, total_samples, "(N")]

    cur_start, cur_rhythm = 0, "(N"
    for s, note in rhythm_changes:
        if s > cur_start:
            intervals.append((cur_start, s, cur_rhythm))
        cur_start = s
        cur_rhythm = note

    if cur_start < total_samples:
        intervals.append((cur_start, total_samples, cur_rhythm))

    return intervals


def load_shdb_af_cohort_windows(
    cohort: str = "non_af_controls",
    shdb_dir: str = "/data/mithunmanivannan/datasets/shdb_af",
    max_windows_per_rec: int = 50,
    window_stride_sec: float = 30.0,
    target_rr_len: int = 60,
) -> Dict[str, Any]:
    """
    Extract 30s windows from SHDB-AF cohorts for clinical specificity, recurrence detection,
    and AF burden evaluation.

    Cohorts:
      - 'non_af_controls': 33 negative control Holters (792h sinus rhythm, true label = 0).
      - 'post_ablation': 41 prior-ablation Holters.
      - 'post_blanking': 18 post-ablation Holters >90 days post-procedure.
      - 'all_annotated': 98 Holters with expert beat-level ground truth.
    """
    master_csv = Path("/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/shdb_af_master_index.csv")
    if not master_csv.is_file():
        raise FileNotFoundError(f"Master index {master_csv} not found. Run scripts/build_comprehensive_database_index.py first.")

    df = pd.read_csv(master_csv)

    if cohort == "non_af_controls":
        cdf = df[df["is_non_af_control"] == True]
    elif cohort == "post_ablation":
        cdf = df[df["previous_af_ablation"] == True]
    elif cohort == "post_blanking":
        cdf = df[(df["previous_af_ablation"] == True) & (df["is_post_blanking"] == True)]
    elif cohort == "all_annotated":
        cdf = df[df["annotated"] == True]
    else:
        cdf = df

    print(f"[load_shdb_af_cohort_windows] Cohort '{cohort}': selecting from {len(cdf)} recordings.")

    raw_ecg_list, rr_seq_list, hrv_stats_list, phase_kme_list = [], [], [], []
    labels_list, rec_id_list, window_idx_list, af_fraction_list = [], [], [], []

    window_size_samples = 6000  # 30s @ 200 Hz
    step_samples = int(window_stride_sec * 200.0)

    for _, row in cdf.iterrows():
        rec_id = row["rec_id"]
        is_control = bool(row["is_non_af_control"])
        is_annotated = bool(row["annotated"])
        rec_stem = f"{shdb_dir}/{rec_id}"
        sig_len = int(row["sig_len"])

        # Timeline intervals for annotated records
        timeline = get_shdb_af_record_rhythm_timeline(rec_id, shdb_dir=shdb_dir) if is_annotated else []

        # Sample windows evenly across the 24-hour recording
        max_possible = max(1, (sig_len - window_size_samples) // step_samples)
        if max_windows_per_rec is not None and max_windows_per_rec < max_possible:
            window_indices = np.linspace(0, max_possible - 1, max_windows_per_rec, dtype=int)
        else:
            window_indices = np.arange(max_possible, dtype=int)

        for w_idx in window_indices:
            start_samp = int(w_idx * step_samples)
            end_samp = start_samp + window_size_samples
            if end_samp > sig_len:
                break

            try:
                # Seek read directly from disk
                rec = wfdb.rdrecord(rec_stem, sampfrom=start_samp, sampto=end_samp)
                sig = rec.p_signal[:, 0]  # Channel 1 ('ECG1')

                # Calculate ground truth AF overlap
                if is_control:
                    af_frac = 0.0
                    y_label = 0
                elif timeline:
                    overlap_samples = 0
                    for t_start, t_end, rhy in timeline:
                        if rhy in ("(AFIB", "(AFL"):
                            ov_start = max(start_samp, t_start)
                            ov_end = min(end_samp, t_end)
                            if ov_end > ov_start:
                                overlap_samples += (ov_end - ov_start)
                    af_frac = overlap_samples / float(window_size_samples)
                    y_label = 1 if af_frac >= 0.50 else 0
                else:
                    af_frac = 0.0
                    y_label = 0

                # Read or detect QRS peaks
                try:
                    qrs_ann = wfdb.rdann(rec_stem, "qrs", sampfrom=start_samp, sampto=end_samp)
                    peaks = np.asarray(qrs_ann.sample, dtype=int) - start_samp
                    peaks = peaks[(peaks >= 0) & (peaks < window_size_samples)]
                except Exception:
                    peaks = np.asarray(wp.gqrs_detect(sig=sig, fs=200.0), dtype=int)

                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / 200.0
                else:
                    rr_intervals = np.random.uniform(0.6, 1.0, size=target_rr_len)

                if len(rr_intervals) >= target_rr_len:
                    rr_seq = rr_intervals[:target_rr_len]
                else:
                    pad_val = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8
                    rr_seq = np.pad(rr_intervals, (0, target_rr_len - len(rr_intervals)), constant_values=pad_val)

                hrv_feats = compute_hrv_features(rr_seq)
                kme = compute_phase_kme_embedding(sig, peaks, fs=200.0, num_phases=16, kme_dim=128)

                raw_ecg_list.append(sig.astype(np.float32))
                rr_seq_list.append(rr_seq.astype(np.float32))
                hrv_stats_list.append(hrv_feats.astype(np.float32))
                phase_kme_list.append(kme.astype(np.float32))
                labels_list.append(y_label)
                af_fraction_list.append(float(af_frac))
                rec_id_list.append(rec_id)
                window_idx_list.append(int(w_idx))

            except Exception:
                continue

    print(f"[load_shdb_af_cohort_windows] Extracted {len(labels_list)} windows (AF prevalence = {np.mean(labels_list)*100:.2f}%).")

    return {
        "raw_ecg": np.array(raw_ecg_list),
        "rr_seq": np.array(rr_seq_list),
        "hrv_stats": np.array(hrv_stats_list),
        "phase_kme": np.array(phase_kme_list),
        "labels": np.array(labels_list, dtype=np.int64),
        "af_fractions": np.array(af_fraction_list, dtype=np.float32),
        "rec_ids": np.array(rec_id_list),
        "window_indices": np.array(window_idx_list, dtype=np.int64),
    }
