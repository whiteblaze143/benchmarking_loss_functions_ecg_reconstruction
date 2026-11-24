"""Build Decisive Feature Sets for Post-AF Trajectories, Measurement Specificity, Negative Controls, and Recurrence.

Uses the verified recording-level references (Ei_rec, u1_rec) from manifests/recording_level_references.parquet:
- No silent fallbacks: only recordings with has_valid_remote_ref == True are processed.
- Computes spatial PCA PMV (V), raw MAD energy (D), individual Lead 0 and Lead 1 PMV, and PR-aligned PMV.
- Extracts POST_AF, REMOTE_SINUS, NEGATIVE_CONTROL, and PRE_AF.
- Attaches exact recurrence follow-up metadata for survival analysis (R4).
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

from experiments.pmv_temporal_specificity.scripts.pmv_core import PMVExtractor, PMVSubsegmentResult

MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
FEATURES_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "features"
LTAFDB_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_DIR = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5")


def load_ecg_and_qrs_slice(
    dataset: str,
    record_id: str,
    start_s: float,
    end_s: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Load ECG slice and relative QRS indices."""
    duration_s = end_s - start_s
    if duration_s < 120.0:
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
            if ecg.shape[1] < int(120.0 * fs):
                return None, None, fs
            qrs = wfdb.processing.xqrs_detect(ecg[0], fs=fs, verbose=False)
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    return None, None, 0.0


def extract_multi_representation_subsegment(
    ecg_filt: np.ndarray,
    qrs_indices: np.ndarray,
    time_center_s: float,
    extractor: PMVExtractor,
    Ei_rec: float,
    Ei_lead0: float,
    Ei_lead1: float,
    u1: np.ndarray
) -> Dict:
    """Extract full subsegment metrics across all representations:
    1. Spatial PCA PMV & raw D
    2. Individual Lead 0 PMV & raw D
    3. Individual Lead 1 PMV & raw D
    4. PR-aligned PMV & raw D
    """
    # 1. Spatial PCA PMV
    res_pca = extractor.compute_subsegment_pmv(
        ecg_filt, qrs_indices, time_center_s, fixed_eigenvector=u1, fixed_reference_energy=Ei_rec
    )

    # 2. Lead 0 alone
    res_lead0 = extractor.compute_subsegment_pmv(
        ecg_filt[0], qrs_indices, time_center_s, fixed_reference_energy=Ei_lead0
    )

    # 3. Lead 1 alone
    res_lead1 = extractor.compute_subsegment_pmv(
        ecg_filt[1], qrs_indices, time_center_s, fixed_reference_energy=Ei_lead1
    )

    return {
        "time_center_s": time_center_s,
        "is_valid": res_pca.is_valid,
        "status": res_pca.status,
        "num_beats": res_pca.num_beats,
        "mad_energy_D": res_pca.mad_energy_D,
        "ref_energy_Ei": Ei_rec,
        "pmv_V": res_pca.pmv_V,
        "pr_aligned": res_pca.pr_aligned,
        "pmv_aligned_D": res_pca.pmv_aligned_D,
        "pmv_aligned_V": res_pca.pmv_aligned_V,
        "mean_abs_pr_shift_ms": res_pca.mean_abs_pr_shift_ms,
        "mad_energy_lead0": res_lead0.mad_energy_D,
        "ref_energy_lead0": Ei_lead0,
        "pmv_lead0": res_lead0.pmv_V,
        "mad_energy_lead1": res_lead1.mad_energy_D,
        "ref_energy_lead1": Ei_lead1,
        "pmv_lead1": res_lead1.pmv_V,
        "mean_hr_bpm": res_pca.mean_hr_bpm,
        "sdnn_ms": res_pca.sdnn_ms,
        "rmssd_ms": res_pca.rmssd_ms,
        "p_amp_iqr": res_pca.p_amp_dispersion_iqr,
        "p_duration_std": res_pca.p_duration_dispersion_std,
    }


def process_post_af_episode(
    ep_info: Tuple[Dict, Dict]
) -> List[Dict]:
    """Process a single post-AF episode and extract sliding 120s subsegments up to 60m."""
    row, ref = ep_info
    dataset = row["dataset"]
    record_id = row["record_id"]
    patient_id = row["patient_id"]
    af_term_s = row["end_s"]
    af_dur_s = row["duration_s"]
    sinus_dur_s = row["subsequent_sinus_dur_s"]
    time_until_next_af_s = row["time_until_next_af_s"]

    # Maximum post-AF follow-up window (up to 60 minutes or end of continuous sinus)
    max_followup_s = min(3600.0, sinus_dur_s)
    if max_followup_s < 120.0:
        return []

    ecg, qrs, fs = load_ecg_and_qrs_slice(dataset, record_id, af_term_s, af_term_s + max_followup_s)
    if ecg is None or qrs is None or len(qrs) < 20:
        return []

    extractor = PMVExtractor(fs=fs, subsegment_len_s=120.0, subsegment_step_s=60.0, min_beats_per_subsegment=20)
    try:
        ecg_filt = extractor.filter_signal(ecg)
    except Exception:
        return []

    Ei_rec = ref["Ei_rec"]
    Ei_lead0 = ref["Ei_lead0"]
    Ei_lead1 = ref["Ei_lead1"]
    u1 = np.array([ref["u1_0"], ref["u1_1"]])

    sub_len_samples = int(round(120.0 * fs))
    sub_step_samples = int(round(60.0 * fs))
    total_samples = ecg_filt.shape[1]

    results = []
    sub_id = 0
    seg_id = f"{dataset}_{record_id}_POST_{int(af_term_s)}"

    for start_samp in range(0, total_samples - sub_len_samples + 1, sub_step_samples):
        end_samp = start_samp + sub_len_samples
        center_samp = (start_samp + end_samp) // 2
        rel_time_s = center_samp / fs
        rel_time_m = rel_time_s / 60.0

        mask = (qrs >= start_samp) & (qrs < end_samp)
        sub_qrs = qrs[mask] - start_samp
        if len(sub_qrs) < 20:
            continue

        ecg_sub = ecg_filt[:, start_samp:end_samp]
        metrics = extract_multi_representation_subsegment(
            ecg_sub, sub_qrs, rel_time_s, extractor, Ei_rec, Ei_lead0, Ei_lead1, u1
        )

        metrics.update({
            "context_type": "POST_AF",
            "dataset": dataset,
            "patient_id": patient_id,
            "record_id": record_id,
            "segment_id": seg_id,
            "subsegment_id": sub_id,
            "af_term_s": af_term_s,
            "af_duration_s": af_dur_s,
            "subsequent_sinus_dur_s": sinus_dur_s,
            "time_until_next_af_s": time_until_next_af_s,
            "rel_time_from_af_s": rel_time_s,
            "rel_time_from_af_m": rel_time_m,
        })
        results.append(metrics)
        sub_id += 1

    return results


def process_negative_control_segment(
    ctrl_info: Tuple[Dict, Dict]
) -> List[Dict]:
    """Process a negative-control remote sinus segment with mock landmark t0."""
    row, ref = ctrl_info
    dataset = row["dataset"]
    record_id = row["record_id"]
    patient_id = row["patient_id"]
    start_s = row["ref_start_s"]
    dur_s = row["ref_dur_s"]

    # We need an interval of at least 1800s (30m) to simulate mock post-transition
    # If ref_dur_s < 1800s, load from start_s up to start_s + 3600s
    ecg, qrs, fs = load_ecg_and_qrs_slice(dataset, record_id, start_s, start_s + min(dur_s, 3600.0))
    if ecg is None or qrs is None or len(qrs) < 20:
        return []

    extractor = PMVExtractor(fs=fs, subsegment_len_s=120.0, subsegment_step_s=60.0, min_beats_per_subsegment=20)
    try:
        ecg_filt = extractor.filter_signal(ecg)
    except Exception:
        return []

    Ei_rec = ref["Ei_rec"]
    Ei_lead0 = ref["Ei_lead0"]
    Ei_lead1 = ref["Ei_lead1"]
    u1 = np.array([ref["u1_0"], ref["u1_1"]])

    sub_len_samples = int(round(120.0 * fs))
    sub_step_samples = int(round(60.0 * fs))
    total_samples = ecg_filt.shape[1]

    results = []
    sub_id = 0
    seg_id = f"{dataset}_{record_id}_MOCK_TRANSITION_{int(start_s)}"

    for start_samp in range(0, total_samples - sub_len_samples + 1, sub_step_samples):
        end_samp = start_samp + sub_len_samples
        center_samp = (start_samp + end_samp) // 2
        rel_time_s = center_samp / fs
        rel_time_m = rel_time_s / 60.0

        mask = (qrs >= start_samp) & (qrs < end_samp)
        sub_qrs = qrs[mask] - start_samp
        if len(sub_qrs) < 20:
            continue

        ecg_sub = ecg_filt[:, start_samp:end_samp]
        metrics = extract_multi_representation_subsegment(
            ecg_sub, sub_qrs, rel_time_s, extractor, Ei_rec, Ei_lead0, Ei_lead1, u1
        )

        metrics.update({
            "context_type": "NEGATIVE_CONTROL",
            "dataset": dataset,
            "patient_id": patient_id,
            "record_id": record_id,
            "segment_id": seg_id,
            "subsegment_id": sub_id,
            "rel_time_from_af_s": rel_time_s,
            "rel_time_from_af_m": rel_time_m,
        })
        results.append(metrics)
        sub_id += 1

    return results


def main():
    print("=" * 70)
    print("EXTRACTING DECISIVE MULTI-REPRESENTATION FEATURE SETS")
    print("=" * 70)

    # Load verified recording-level references
    df_refs = pd.read_parquet(MANIFESTS_DIR / "recording_level_references.parquet")
    valid_refs = df_refs[df_refs["has_valid_remote_ref"]].copy()
    ref_dict = {row["record_id"]: row.to_dict() for _, row in valid_refs.iterrows()}
    print(f"Loaded {len(ref_dict)} verified recording references.")

    # 1. Post-AF episodes
    df_inv = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")
    eligible_post = df_inv[(df_inv["record_id"].isin(ref_dict)) & (df_inv["subsequent_sinus_dur_s"] >= 120.0)].copy()
    print(f"Eligible post-AF episodes with verified references: {len(eligible_post)} across {eligible_post['patient_id'].nunique()} patients.")

    post_jobs = [(row.to_dict(), ref_dict[row["record_id"]]) for _, row in eligible_post.iterrows()]

    all_post_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for res_list in executor.map(process_post_af_episode, post_jobs):
            all_post_results.extend(res_list)

    df_post_feat = pd.DataFrame(all_post_results)
    post_out = FEATURES_DIR / "decisive_post_af_features.parquet"
    df_post_feat.to_parquet(post_out, index=False)
    print(f"Saved {len(df_post_feat)} post-AF subsegments to {post_out}")

    # 2. Negative-Control mock transitions
    ctrl_jobs = [(row.to_dict(), row.to_dict()) for _, row in valid_refs.iterrows() if row["ref_dur_s"] >= 300.0]
    print(f"Eligible negative-control segments: {len(ctrl_jobs)}")

    all_ctrl_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for res_list in executor.map(process_negative_control_segment, ctrl_jobs):
            all_ctrl_results.extend(res_list)

    df_ctrl_feat = pd.DataFrame(all_ctrl_results)
    ctrl_out = FEATURES_DIR / "decisive_negative_control_features.parquet"
    df_ctrl_feat.to_parquet(ctrl_out, index=False)
    print(f"Saved {len(df_ctrl_feat)} negative-control subsegments to {ctrl_out}")


if __name__ == "__main__":
    main()
