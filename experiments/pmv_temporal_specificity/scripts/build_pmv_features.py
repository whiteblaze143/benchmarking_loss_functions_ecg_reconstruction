"""Build Unified PMV Feature Sets for Pre-AF, Post-AF, Interepisode, and Remote Sinus Windows.

Extracts PMV metrics strictly following Noriega et al. 2026 across IRIDIA, SHDB, and LTAFDB.
Uses multi-processing for fast, deterministic extraction.
Outputs parquet and CSV files in experiments/pmv_temporal_specificity/features/.
"""

import sys
from pathlib import Path
import time
import os
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
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ecg_and_qrs_slice(
    dataset: str,
    record_id: str,
    start_s: float,
    end_s: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Extract ECG slice and QRS fiducial sample indices relative to slice start."""
    duration_s = end_s - start_s
    if duration_s < 120.0:
        return None, None, 0.0

    if dataset == "LTAFDB":
        fs = 128.0
        rec_base = f"/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference/{record_id}"
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            rec = wfdb.rdrecord(rec_base, sampfrom=s_idx, sampto=e_idx)
            ecg = rec.p_signal.T
            atr = wfdb.rdann(rec_base, "atr", sampfrom=s_idx, sampto=e_idx)
            # Normal sinus beats
            qrs_list = [s - s_idx for s, sym in zip(atr.sample, atr.symbol) if sym in ["N", "B", "L", "R", "j"]]
            qrs = np.array(qrs_list, dtype=int)
            qrs = qrs[(qrs >= 0) & (qrs < ecg.shape[1])]
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    elif dataset == "SHDB-AF":
        fs = 200.0
        rec_base = f"/data/mithunmanivannan/datasets/shdb_af/{record_id}"
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
        rec_dir = Path(f"/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5/{record_id}")
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
            # Detect QRS using xqrs on Lead 0
            qrs = wfdb.processing.xqrs_detect(ecg[0], fs=fs, verbose=False)
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    return None, None, 0.0


def process_segment(
    context_type: str,
    seg_info: Dict
) -> List[Dict]:
    """Process a single continuous segment and return sliding PMV subsegments."""
    dataset = seg_info["dataset"]
    record_id = seg_info["record_id"]
    patient_id = seg_info["patient_id"]
    start_s = seg_info["segment_start_s"]
    end_s = seg_info["segment_end_s"]

    ecg, qrs, fs = load_ecg_and_qrs_slice(dataset, record_id, start_s, end_s)
    if ecg is None or qrs is None or len(qrs) < 20:
        return []

    extractor = PMVExtractor(
        fs=fs,
        subsegment_len_s=120.0,
        subsegment_step_s=60.0,
        min_beats_per_subsegment=20,
        enable_pr_alignment=True,
    )

    try:
        ecg_filt = extractor.filter_signal(ecg)
    except Exception:
        return []

    # 1. Establish fixed baseline reference: compute spatial PCA u1 and fixed reference energy Ei
    # Use first 300 seconds (or entire segment if < 300s) as reference ensemble
    ref_dur_s = min(300.0, end_s - start_s)
    ref_mask = (qrs >= 0) & (qrs < int(round(ref_dur_s * fs)))
    ref_qrs = qrs[ref_mask] if np.sum(ref_mask) >= 15 else qrs[:max(15, len(qrs))]

    if ecg_filt.shape[0] >= 2:
        ref_p_waves, _, _, u1 = extractor.compute_spatial_pca_projection(ecg_filt[:2], ref_qrs)
        fixed_eigenvector = u1 if len(u1) == 2 else None
    else:
        ref_p_waves, _ = extractor.extract_p_wave_ensemble(ecg_filt[0], ref_qrs)
        fixed_eigenvector = None

    if len(ref_p_waves) >= 10:
        fixed_Ei = extractor.compute_reference_energy(ref_p_waves)
    else:
        # Fallback to all valid beats if reference subset had too few beats
        lead = ecg_filt[0] if ecg_filt.ndim == 2 else ecg_filt
        all_p, _ = extractor.extract_p_wave_ensemble(lead, qrs)
        fixed_Ei = extractor.compute_reference_energy(all_p) if len(all_p) > 0 else 1e-6

    fixed_Ei = max(fixed_Ei, 1e-8) if not np.isnan(fixed_Ei) else 1e-6

    # 2. Extract sliding PMV series using fixed PCA vector and fixed reference energy Ei
    results = extractor.extract_sliding_pmv_series(
        ecg_filt,
        qrs,
        start_time_s=0.0,
        end_time_s=(end_s - start_s),
        fixed_eigenvector=fixed_eigenvector,
        fixed_reference_energy=fixed_Ei,
    )

    rows = []
    for res in results:
        abs_center_s = start_s + res.time_center_s

        row = {
            "context_type": context_type,
            "dataset": dataset,
            "patient_id": patient_id,
            "record_id": record_id,
            "segment_id": seg_info.get("segment_id", f"{record_id}_{int(start_s)}"),
            "abs_time_center_s": abs_center_s,
            "time_center_s": res.time_center_s,
            "num_beats": res.num_beats,
            "is_valid": res.is_valid,
            "status": res.status,
            "mad_energy_D": res.mad_energy_D,
            "ref_energy_Ei": res.ref_energy_Ei,
            "pmv_V": res.pmv_V,
            "local_median_energy_Em": res.local_median_energy_Em,
            "pmv_mad_amplitude_mean": res.pmv_mad_amplitude_mean,
            "pmv_mad_amplitude_max": res.pmv_mad_amplitude_max,
            "pr_aligned": res.pr_aligned,
            "pmv_aligned_V": res.pmv_aligned_V,
            "mad_energy_aligned_D": res.pmv_aligned_D,
            "mean_pr_shift_ms": res.mean_abs_pr_shift_ms,
            "pca_lambda2_ratio": res.pca_lambda2_ratio,
            "p_amp_iqr": res.p_amp_dispersion_iqr,
            "p_amp_std": res.p_amp_dispersion_std,
            "p_duration_std": res.p_duration_dispersion_std,
            "p_arc_length_std": res.p_arc_length_dispersion_std,
            "mean_rr_ms": res.mean_rr_ms,
            "sdnn_ms": res.sdnn_ms,
            "rmssd_ms": res.rmssd_ms,
            "mean_hr_bpm": res.mean_hr_bpm,
        }

        # Add context-specific relative time variables
        if context_type == "PRE_AF":
            af_onset_s = seg_info["af_onset_s"]
            row["af_onset_s"] = af_onset_s
            row["rel_time_to_af_s"] = abs_center_s - af_onset_s  # negative
            row["rel_time_to_af_m"] = (abs_center_s - af_onset_s) / 60.0
            row["af_duration_s"] = seg_info["af_duration_s"]
            row["time_since_prev_af_s"] = seg_info.get("time_since_prev_af_s", np.nan)
            row["prev_af_dur_s"] = seg_info.get("prev_af_dur_s", np.nan)
        elif context_type == "POST_AF":
            af_term_s = seg_info["af_term_s"]
            row["af_term_s"] = af_term_s
            row["rel_time_from_af_s"] = abs_center_s - af_term_s  # positive
            row["rel_time_from_af_m"] = (abs_center_s - af_term_s) / 60.0
            row["af_duration_s"] = seg_info["af_duration_s"]
            row["time_until_next_af_s"] = seg_info.get("time_until_next_af_s", np.nan)
            row["next_af_dur_s"] = seg_info.get("next_af_dur_s", np.nan)
        elif context_type == "INTEREPISODE":
            sinus_start_s = seg_info["sinus_start_s"]
            sinus_end_s = seg_info["sinus_end_s"]
            row["sinus_start_s"] = sinus_start_s
            row["sinus_end_s"] = sinus_end_s
            row["sinus_duration_s"] = seg_info["sinus_duration_s"]
            row["prior_af_dur_s"] = seg_info["prior_af_dur_s"]
            row["next_af_dur_s"] = seg_info["next_af_dur_s"]
            row["time_since_prev_af_s"] = abs_center_s - sinus_start_s
            row["time_since_prev_af_m"] = (abs_center_s - sinus_start_s) / 60.0
            row["time_until_next_af_s"] = sinus_end_s - abs_center_s
            row["time_until_next_af_m"] = (sinus_end_s - abs_center_s) / 60.0
        elif context_type == "REMOTE_SINUS":
            row["washout_s"] = seg_info.get("washout_s", 7200.0)

        rows.append(row)

    return rows


def build_all_features():
    print("=" * 70)
    print("PHASE C: EXTRACTING UNIFIED PMV FEATURE SETS")
    print("=" * 70)

    # Load manifests
    df_af = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")
    df_inter = pd.read_parquet(MANIFESTS_DIR / "interepisode_inventory.parquet")
    df_rhythm = pd.read_parquet(MANIFESTS_DIR / "canonical_rhythm_intervals.parquet")

    # 1. Pre-AF Segments (up to 60m pre-AF, eligible >= 15m)
    pre_af_segments = []
    eligible_pre = df_af[df_af["pre_af_eligible_15m"]].copy()
    for idx, row in eligible_pre.iterrows():
        af_onset = row["start_s"]
        avail_sinus = min(row["preceding_sinus_dur_s"], 3600.0)
        seg_start = af_onset - avail_sinus
        seg_end = af_onset
        pre_af_segments.append({
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "record_id": row["record_id"],
            "segment_id": f"PRE_{row['dataset']}_{row['record_id']}_{int(af_onset)}",
            "segment_start_s": seg_start,
            "segment_end_s": seg_end,
            "af_onset_s": af_onset,
            "af_duration_s": row["duration_s"],
            "time_since_prev_af_s": row["time_since_prev_af_s"],
            "prev_af_dur_s": row["prev_af_dur_s"],
        })
    print(f"Constructed {len(pre_af_segments)} Pre-AF extraction segments.")

    # 2. Post-AF Segments (up to 60m post-AF, eligible >= 15m)
    post_af_segments = []
    eligible_post = df_af[df_af["post_af_eligible_15m"]].copy()
    for idx, row in eligible_post.iterrows():
        af_term = row["end_s"]
        avail_sinus = min(row["subsequent_sinus_dur_s"], 3600.0)
        seg_start = af_term
        seg_end = af_term + avail_sinus
        post_af_segments.append({
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "record_id": row["record_id"],
            "segment_id": f"POST_{row['dataset']}_{row['record_id']}_{int(af_term)}",
            "segment_start_s": seg_start,
            "segment_end_s": seg_end,
            "af_term_s": af_term,
            "af_duration_s": row["duration_s"],
            "time_until_next_af_s": row["time_until_next_af_s"],
            "next_af_dur_s": row["next_af_dur_s"],
        })
    print(f"Constructed {len(post_af_segments)} Post-AF extraction segments.")

    # 3. Interepisode Segments (eligible >= 15m)
    inter_segments = []
    eligible_inter = df_inter[df_inter["interepisode_eligible_15m"]].copy()
    for idx, row in eligible_inter.iterrows():
        seg_start = row["sinus_start_s"]
        seg_end = min(row["sinus_end_s"], seg_start + 7200.0)  # up to 2 hours of interepisode
        inter_segments.append({
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "record_id": row["record_id"],
            "segment_id": f"INTER_{row['dataset']}_{row['record_id']}_{int(seg_start)}",
            "segment_start_s": seg_start,
            "segment_end_s": seg_end,
            "sinus_start_s": row["sinus_start_s"],
            "sinus_end_s": row["sinus_end_s"],
            "sinus_duration_s": row["sinus_duration_s"],
            "prior_af_dur_s": row["preceding_af_duration_s"],
            "next_af_dur_s": row["subsequent_af_duration_s"],
        })
    print(f"Constructed {len(inter_segments)} Interepisode extraction segments.")

    # 4. Remote Sinus Control Segments (sinus intervals >= 7200s, slice middle 3600s)
    remote_segments = []
    sinus_long = df_rhythm[
        (df_rhythm["harmonized_rhythm_label"].isin(["CONFIRMED_SINUS", "PROBABLE_SINUS"])) &
        (df_rhythm["duration_seconds"] >= 7200.0)
    ].copy()
    # Sample up to 2 per patient to avoid overrepresentation
    for patient, group in sinus_long.groupby("patient_id"):
        for _, row in group.head(2).iterrows():
            mid_s = 0.5 * (row["start_time_seconds"] + row["end_time_seconds"])
            seg_start = mid_s - 1800.0
            seg_end = mid_s + 1800.0
            remote_segments.append({
                "dataset": row["dataset"],
                "patient_id": row["patient_id"],
                "record_id": row["record_id"],
                "segment_id": f"REMOTE_{row['dataset']}_{row['record_id']}_{int(seg_start)}",
                "segment_start_s": seg_start,
                "segment_end_s": seg_end,
                "washout_s": min(seg_start - row["start_time_seconds"], row["end_time_seconds"] - seg_end),
            })
    print(f"Constructed {len(remote_segments)} Remote Sinus control segments.")

    # Function to run extraction in parallel
    def run_batch(context_name: str, segments: List[Dict], max_workers: int = 16) -> pd.DataFrame:
        print(f"\nExtracting {context_name} ({len(segments)} segments) on {max_workers} processes...")
        t0 = time.time()
        all_rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_segment, context_name, seg) for seg in segments]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    all_rows.extend(res)
                except Exception as e:
                    pass
                completed += 1
                if completed % 50 == 0 or completed == len(segments):
                    print(f"  [{context_name}] Progress: {completed}/{len(segments)} segments completed ({len(all_rows)} subsegments extracted)")

        df_out = pd.DataFrame(all_rows)
        elapsed = time.time() - t0
        print(f"Finished {context_name}: {len(df_out)} total 120s subsegments extracted in {elapsed:.2f}s!")
        return df_out

    # Run extraction across all four sets
    df_pre = run_batch("PRE_AF", pre_af_segments, max_workers=16)
    df_post = run_batch("POST_AF", post_af_segments, max_workers=16)
    df_inter_out = run_batch("INTEREPISODE", inter_segments, max_workers=16)
    df_remote = run_batch("REMOTE_SINUS", remote_segments, max_workers=16)

    # Compute relative PMV (V_rel) for PRE_AF: normalize by median D of [-59m, -55m] for each segment
    if len(df_pre) > 0:
        v_rel_list = []
        for seg_id, group in df_pre.groupby("segment_id"):
            base_rows = group[(group["rel_time_to_af_m"] >= -59.0) & (group["rel_time_to_af_m"] <= -55.0) & (group["is_valid"])]
            if len(base_rows) > 0:
                base_D = base_rows["mad_energy_D"].median()
            else:
                valid_group = group[group["is_valid"]]
                base_D = valid_group["mad_energy_D"].iloc[0] if len(valid_group) > 0 else np.nan
            base_D = max(base_D, 1e-12) if not np.isnan(base_D) else np.nan
            group_v_rel = group["mad_energy_D"] / base_D
            group = group.copy()
            group["relative_pmv_Vrel"] = group_v_rel
            v_rel_list.append(group)
        df_pre = pd.concat(v_rel_list, ignore_index=True)

    # Save feature parquets & CSVs
    print("\nSaving feature manifests to disk...")
    df_pre.to_parquet(FEATURES_DIR / "pmv_pre_af_features.parquet", index=False)
    df_pre.to_csv(FEATURES_DIR / "pmv_pre_af_features.csv", index=False)

    df_post.to_parquet(FEATURES_DIR / "pmv_post_af_features.parquet", index=False)
    df_post.to_csv(FEATURES_DIR / "pmv_post_af_features.csv", index=False)

    df_inter_out.to_parquet(FEATURES_DIR / "pmv_interepisode_features.parquet", index=False)
    df_inter_out.to_csv(FEATURES_DIR / "pmv_interepisode_features.csv", index=False)

    df_remote.to_parquet(FEATURES_DIR / "pmv_remote_sinus_features.parquet", index=False)
    df_remote.to_csv(FEATURES_DIR / "pmv_remote_sinus_features.csv", index=False)

    # Combined master window manifest
    df_master = pd.concat([df_pre, df_post, df_inter_out, df_remote], ignore_index=True)
    df_master.to_parquet(MANIFESTS_DIR / "pmv_window_manifest.parquet", index=False)
    df_master.to_csv(MANIFESTS_DIR / "pmv_window_manifest.csv", index=False)

    print(f"\nSuccessfully generated master window manifest: {len(df_master)} total subsegment windows.")
    print(f"  PRE_AF valid subsegments: {len(df_pre[df_pre['is_valid']]) if len(df_pre)>0 else 0}")
    print(f"  POST_AF valid subsegments: {len(df_post[df_post['is_valid']]) if len(df_post)>0 else 0}")
    print(f"  INTEREPISODE valid subsegments: {len(df_inter_out[df_inter_out['is_valid']]) if len(df_inter_out)>0 else 0}")
    print(f"  REMOTE_SINUS valid subsegments: {len(df_remote[df_remote['is_valid']]) if len(df_remote)>0 else 0}")


if __name__ == "__main__":
    build_all_features()
