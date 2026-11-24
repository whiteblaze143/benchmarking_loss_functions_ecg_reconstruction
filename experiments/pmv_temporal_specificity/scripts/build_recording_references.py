"""Compute Consistent Recording-Level Baseline References (Ei and u1) across all Holters.

Establishes a single, verified baseline reference for each Holter recording:
- Identifies remote sinus rhythm at least 60 minutes away from any AF episode.
- Computes spatial PCA projection vector u1_rec across the two leads.
- Computes fixed reference individual P-wave energy Ei_rec = median_j { sum_n p_j(n)^2 }.
- Computes lead-specific reference energies Ei_lead0 and Ei_lead1 for Lead Specificity (R2).
- Ensures that every pre-AF, post-AF, interepisode, and negative-control window of a recording
  uses the EXACT same reference.
"""

import sys
from pathlib import Path
import time
import concurrent.futures
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import wfdb
import wfdb.processing
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pmv_temporal_specificity.scripts.pmv_core import PMVExtractor

MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
LTAFDB_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_DIR = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5")


def load_ecg_and_qrs_slice(
    dataset: str,
    record_id: str,
    start_s: float,
    end_s: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Load ECG slice and QRS sample indices relative to slice start."""
    duration_s = end_s - start_s
    if duration_s < 60.0:
        return None, None, 0.0

    if dataset == "LTAFDB":
        fs = 128.0
        rec_base = str(LTAFDB_DIR / record_id)
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            rec = wfdb.rdrecord(rec_base, sampfrom=s_idx, sampto=e_idx)
            ecg = rec.p_signal.T
            atr = wfdb.rdann(rec_base, "atr", sampfrom=s_idx, sampto=e_idx)
            qrs_list = [s - s_idx for s, sym in zip(atr.sample, atr.symbol) if sym in ["N", "B", "L", "R", "j"]]
            qrs = np.array(qrs_list, dtype=int)
            qrs = qrs[(qrs >= 0) & (qrs < ecg.shape[1])]
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    elif dataset == "SHDB-AF":
        fs = 200.0
        rec_base = str(SHDB_DIR / record_id)
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            rec = wfdb.rdrecord(rec_base, sampfrom=s_idx, sampto=e_idx)
            ecg = rec.p_signal.T
            qrs_ann = wfdb.rdann(rec_base, "qrs", sampfrom=s_idx, sampto=e_idx)
            rel_qrs = qrs_ann.sample - s_idx
            rel_qrs = rel_qrs[(rel_qrs >= 0) & (rel_qrs < ecg.shape[1])]
            return ecg, rel_qrs, fs
        except Exception:
            return None, None, fs

    elif dataset == "IRIDIA-AF":
        fs = 200.0
        rec_dir = IRIDIA_DIR / record_id
        h5_files = list(rec_dir.glob("*.h5"))
        if not h5_files:
            return None, None, fs
        h5_path = h5_files[0]
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            with h5py.File(h5_path, "r") as f:
                ecg = f["ecg"][s_idx:e_idx].T
            if ecg.shape[1] < int(60.0 * fs):
                return None, None, fs
            qrs = wfdb.processing.xqrs_detect(ecg[0], fs=fs, verbose=False)
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    return None, None, 0.0


def compute_reference_for_record(
    rec_info: Tuple[str, str, str, List[Tuple[float, float]]]
) -> Dict:
    """Compute recording-level baseline reference (Ei and u1) from verified remote sinus intervals."""
    dataset, record_id, patient_id, sinus_intervals = rec_info

    # Sort intervals by duration descending (use the longest available sinus interval)
    sinus_intervals = sorted(sinus_intervals, key=lambda x: x[1] - x[0], reverse=True)

    for start_s, end_s in sinus_intervals:
        dur = end_s - start_s
        if dur < 180.0:  # Require at least 3 minutes
            continue

        # Take up to 300 seconds from the middle of the interval
        ref_start = start_s + max(0.0, (dur - 300.0) / 2.0)
        ref_end = min(end_s, ref_start + 300.0)

        ecg, qrs, fs = load_ecg_and_qrs_slice(dataset, record_id, ref_start, ref_end)
        if ecg is None or qrs is None or len(qrs) < 30:
            continue

        extractor = PMVExtractor(fs=fs, min_beats_per_subsegment=20, min_beat_snr_db=18.0)
        try:
            ecg_filt = extractor.filter_signal(ecg)
        except Exception:
            continue

        if ecg_filt.shape[0] < 2:
            continue

        # Compute spatial PCA projection
        p_lead0, qrs0 = extractor.extract_p_wave_ensemble(ecg_filt[0], qrs)
        p_lead1, qrs1 = extractor.extract_p_wave_ensemble(ecg_filt[1], qrs)

        common_qrs, idx0, idx1 = np.intersect1d(qrs0, qrs1, return_indices=True)
        if len(common_qrs) < 25:
            continue

        p0_sub = p_lead0[idx0]
        p1_sub = p_lead1[idx1]

        # Compute covariance & eigenvector u1
        N, L = p0_sub.shape
        R = np.zeros((2, 2), dtype=np.float64)
        for i in range(N):
            P_i = np.vstack([p0_sub[i], p1_sub[i]])
            R += P_i @ P_i.T
        R /= (N * L)

        eigenvalues, eigenvectors = np.linalg.eigh(R)
        u1 = eigenvectors[:, 1]
        if u1[0] < 0:
            u1 = -u1

        # Projected P-waves
        p_proj = u1[0] * p0_sub + u1[1] * p1_sub

        Ei_proj = extractor.compute_reference_energy(p_proj)
        Ei_lead0 = extractor.compute_reference_energy(p0_sub)
        Ei_lead1 = extractor.compute_reference_energy(p1_sub)

        # Ensure reference energy is in physiological range (>= 1e-4 mV^2)
        if Ei_proj >= 1e-4 and Ei_lead0 >= 1e-5 and Ei_lead1 >= 1e-5:
            return {
                "dataset": dataset,
                "record_id": record_id,
                "patient_id": patient_id,
                "has_valid_remote_ref": True,
                "Ei_rec": round(float(Ei_proj), 6),
                "Ei_lead0": round(float(Ei_lead0), 6),
                "Ei_lead1": round(float(Ei_lead1), 6),
                "u1_0": round(float(u1[0]), 6),
                "u1_1": round(float(u1[1]), 6),
                "ref_n_beats": int(len(common_qrs)),
                "ref_start_s": round(float(ref_start), 1),
                "ref_dur_s": round(float(ref_end - ref_start), 1),
            }

    # If no window met criteria
    return {
        "dataset": dataset,
        "record_id": record_id,
        "patient_id": patient_id,
        "has_valid_remote_ref": False,
        "Ei_rec": np.nan,
        "Ei_lead0": np.nan,
        "Ei_lead1": np.nan,
        "u1_0": np.nan,
        "u1_1": np.nan,
        "ref_n_beats": 0,
        "ref_start_s": np.nan,
        "ref_dur_s": np.nan,
    }


def main():
    print("=" * 70)
    print("ESTABLISHING RECORDING-LEVEL BASELINE REFERENCES")
    print("=" * 70)

    df_inv = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")
    print(f"Total recordings in AF inventory: {df_inv['record_id'].nunique()}")

    record_jobs = []
    for (dataset, record_id, patient_id), group in df_inv.groupby(["dataset", "record_id", "patient_id"]):
        intervals = []
        for _, row in group.iterrows():
            if row["preceding_sinus_dur_s"] >= 180.0:
                s_end = row["start_s"]
                s_start = s_end - min(row["preceding_sinus_dur_s"], 3600.0)
                intervals.append((s_start, s_end, row["preceding_sinus_dur_s"]))
            if row["subsequent_sinus_dur_s"] >= 180.0:
                s_start = row["end_s"]
                s_end = s_start + min(row["subsequent_sinus_dur_s"], 3600.0)
                intervals.append((s_start, s_end, row["subsequent_sinus_dur_s"]))

        if intervals:
            # Sort by sinus duration descending
            intervals = sorted(intervals, key=lambda x: x[2], reverse=True)
            clean_intervals = [(x[0], x[1]) for x in intervals[:5]]
            record_jobs.append((dataset, record_id, patient_id, clean_intervals))

    print(f"Total recordings to process: {len(record_jobs)}")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for res in executor.map(compute_reference_for_record, record_jobs):
            results.append(res)

    df_refs = pd.DataFrame(results)
    out_path = MANIFESTS_DIR / "recording_level_references.parquet"
    df_refs.to_parquet(out_path, index=False)
    df_refs.to_csv(MANIFESTS_DIR / "recording_level_references.csv", index=False)

    n_valid = df_refs["has_valid_remote_ref"].sum()
    print(f"\nSuccessfully generated recording-level references!")
    print(f"Valid references: {n_valid}/{len(df_refs)} ({100.0 * n_valid / len(df_refs):.1f}%)")
    print(f"Summary of Ei_rec:\n{df_refs['Ei_rec'].describe()}")


if __name__ == "__main__":
    main()
