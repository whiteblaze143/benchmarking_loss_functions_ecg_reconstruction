#!/usr/bin/env python3
"""Generate a clean, independently reconciled descriptive reference dataset of AF density in IRIDIA-AF.

Scope:
- Pure ground-truth reference analysis: NO machine learning predictions, NO temporal decoding,
  NO sliding-window analysis, and NO artificial noise injection.
- Exactly 1 row per original eligible Holter recording (all 167 recordings across all 152 patients).
- Fully unifies wall-clock observation duration, actual acquired waveform seconds, recording gaps,
  calibration prefix exclusions, episode counts, individual durations, longest episodes, censoring,
  patient multiplicity, exact continuous Charitos density, and author's 30-second binned density.
"""

import sys
import os
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import numba

# Add path for canonical metrics
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/af_density"))
from metrics import charitos_density
from elapsed_evaluation import equal_bin_amounts


@numba.njit
def fast_charitos_binned_density(proportions, minim=0.001, limit=True):
    """Fast compiled implementation of Charitos's original binned density algorithm."""
    N = len(proportions)
    steps = int(round(1.0 / minim))
    area = 0.0
    for s in range(1, steps):
        m = s * minim
        amount = 0.0
        start = 0
        shortest = N + 1
        for end in range(N):
            amount += proportions[end]
            while amount >= m:
                cur_len = end - start + 1
                if cur_len < shortest:
                    shortest = cur_len
                amount -= proportions[start]
                start += 1
        area += abs(m - shortest / N) * minim
    
    total = np.sum(proportions)
    if total == 0:
        return 0.0
    burden = total / N  # Since proportions were normalized to sum to total_af / bin_capacity
    raw = 2.0 * area / (1.0 - burden) if burden < 1.0 else np.inf
    return min(raw, 1.0) if limit else raw


def run():
    print("=" * 80)
    print("GENERATING INDEPENDENT REFERENCE AF-DENSITY DESCRIPTIVE DATASET (IRIDIA-AF)")
    print("=" * 80)
    
    t0 = time.time()
    
    # 1. Load source ground truth data
    iridia_dir = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
    meta_path = iridia_dir / "iridia-af-metadata-v1.0.1.csv"
    audit_path = iridia_dir / "timeline_record_audit.csv"
    intervals_path = iridia_dir / "timeline_intervals.csv"
    calib_path = iridia_dir / "calibration_exclusion_intervals.csv"
    manifest_path = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/file_manifest.csv")
    
    df_meta = pd.read_csv(meta_path)
    df_audit = pd.read_csv(audit_path)
    df_intervals = pd.read_csv(intervals_path)
    df_calib = pd.read_csv(calib_path)
    df_fm = pd.read_csv(manifest_path)
    
    # Partition mapping
    rec_partitions = df_fm.drop_duplicates(subset=["record_id"]).set_index("record_id")["partition"].to_dict()
    
    # Calibration exclusion summary per record
    calib_by_rec = df_calib.groupby("record_id")["excluded_seconds"].sum().to_dict()
    
    # Patient record counts
    pat_rec_counts = df_meta["patient_id"].value_counts().to_dict()
    
    # Sort metadata by patient_id and record_id
    df_meta = df_meta.sort_values(by=["patient_id", "record_id"]).reset_index(drop=True)
    
    rows = []
    
    for idx, row in df_meta.iterrows():
        rec_id = row["record_id"]
        pat_id = row["patient_id"]
        
        # Audit entry
        audit_match = df_audit[df_audit["record_id"] == rec_id]
        if len(audit_match) == 0:
            raise RuntimeError(f"Missing audit entry for {rec_id}")
        audit_row = audit_match.iloc[0]
        
        # Intervals
        sub_intervals = df_intervals[df_intervals["record_id"] == rec_id].sort_values("start_seconds")
        n_episodes = len(sub_intervals)
        
        # Waveform seconds & timedelta
        wallclock_seconds = float(row["record_timedelta"])
        waveform_seconds = float(audit_row["waveform_seconds"])
        waveform_samples = int(audit_row["waveform_samples"])
        gap_seconds = wallclock_seconds - waveform_seconds
        has_gap = bool(gap_seconds > 1.0)
        
        daily_files = int(audit_row["daily_files"])
        calib_prefix_seconds = float(calib_by_rec.get(rec_id, 0.0))
        effective_observation_seconds = max(1.0, waveform_seconds - calib_prefix_seconds)
        
        # Episode durations and statistics
        durations = []
        interval_pairs = []
        for _, int_row in sub_intervals.iterrows():
            st = float(int_row["start_seconds"])
            en = float(int_row["end_seconds"])
            dur = en - st
            durations.append(dur)
            interval_pairs.append((st, en))
            
        total_af_seconds = math.fsum(durations)
        total_af_hours = total_af_seconds / 3600.0
        af_burden = total_af_seconds / waveform_seconds
        af_burden_effective = total_af_seconds / effective_observation_seconds
        
        longest_ep_seconds = max(durations) if durations else 0.0
        longest_ep_fraction = (longest_ep_seconds / total_af_seconds) if total_af_seconds > 0 else 0.0
        shortest_ep_seconds = min(durations) if durations else 0.0
        median_ep_seconds = float(np.median(durations)) if durations else 0.0
        
        # Censoring
        left_censored = bool(
            (sub_intervals["start_sample"] == 0).any() or 
            (sub_intervals["left_observation_censored"] == True).any()
        )
        right_censored = bool(
            (sub_intervals["end_sample_exclusive"] >= waveform_samples).any() or 
            (sub_intervals["end_at_observation_boundary"] == True).any()
        )
        if left_censored and right_censored:
            censoring_status = "both_censored"
        elif left_censored:
            censoring_status = "left_censored"
        elif right_censored:
            censoring_status = "right_censored"
        else:
            censoring_status = "uncensored"
            
        # Continuous Charitos density
        d_cont = charitos_density(interval_pairs, waveform_seconds)
        
        # 30-second binned Charitos density (author's algorithm)
        # equal_bin_amounts returns AF fraction per bin (0 to 1)
        bin_fractions = equal_bin_amounts(interval_pairs, waveform_seconds, 30.0)
        # Sum of fractions is total_af / bin_width
        # For fast_charitos_binned_density, we pass bin_fractions
        d_bin = fast_charitos_binned_density(bin_fractions, minim=0.001, limit=True)
        
        diff_cont_bin = abs(d_cont - d_bin) if (d_cont is not None and d_bin is not None) else np.nan
        
        is_single = bool(n_episodes == 1)
        is_multi_informative = bool(n_episodes >= 2)
        pat_count = pat_rec_counts.get(pat_id, 1)
        is_multi_record_patient = bool(pat_count > 1)
        
        partition = rec_partitions.get(rec_id, "unknown")
        
        rows.append({
            "record_id": rec_id,
            "patient_id": pat_id,
            "split": partition,
            "patient_sex": row["patient_sex"],
            "patient_age": int(row["patient_age"]),
            "record_date": row["record_date"],
            "record_start_time": row["record_start_time"],
            "record_end_time": row["record_end_time"],
            "wallclock_duration_seconds": round(wallclock_seconds, 3),
            "wallclock_duration_hours": round(wallclock_seconds / 3600.0, 3),
            "acquired_waveform_seconds": round(waveform_seconds, 3),
            "acquired_waveform_hours": round(waveform_seconds / 3600.0, 3),
            "recording_gap_seconds": round(gap_seconds, 3),
            "has_recording_gap": has_gap,
            "num_daily_files": daily_files,
            "device_calibration_prefix_seconds": round(calib_prefix_seconds, 3),
            "effective_observation_seconds": round(effective_observation_seconds, 3),
            "num_af_episodes": n_episodes,
            "total_af_seconds": round(total_af_seconds, 3),
            "total_af_hours": round(total_af_hours, 3),
            "af_burden_raw": round(af_burden, 6),
            "af_burden_effective": round(af_burden_effective, 6),
            "longest_af_episode_seconds": round(longest_ep_seconds, 3),
            "longest_af_episode_hours": round(longest_ep_seconds / 3600.0, 3),
            "longest_af_episode_fraction_of_total": round(longest_ep_fraction, 6),
            "shortest_af_episode_seconds": round(shortest_ep_seconds, 3),
            "median_af_episode_seconds": round(median_ep_seconds, 3),
            "individual_episode_durations_seconds_json": json.dumps([round(d, 3) for d in durations]),
            "left_censored": left_censored,
            "right_censored": right_censored,
            "censoring_status": censoring_status,
            "charitos_continuous_density": round(d_cont, 6) if d_cont is not None else np.nan,
            "charitos_binned_30s_density": round(d_bin, 6) if d_bin is not None else np.nan,
            "density_difference_cont_vs_binned": round(diff_cont_bin, 6) if not np.isnan(diff_cont_bin) else np.nan,
            "is_single_episode": is_single,
            "is_multi_episode_informative": is_multi_informative,
            "patient_record_count": pat_count,
            "is_multi_record_patient": is_multi_record_patient,
        })
        
    df_out = pd.DataFrame(rows)
    
    # Save outputs
    out_dir_repo = ROOT / "reports"
    out_dir_repo.mkdir(exist_ok=True, parents=True)
    out_dir_v4 = ROOT / "experiments/af_density_v4/reports"
    out_dir_v4.mkdir(exist_ok=True, parents=True)
    out_dir_codex = Path("/data/mithunmanivannan/codex_artifacts/af_density_v4")
    out_dir_codex.mkdir(exist_ok=True, parents=True)
    
    csv_repo = out_dir_repo / "iridia_reference_af_density_dataset.csv"
    parquet_repo = out_dir_repo / "iridia_reference_af_density_dataset.parquet"
    df_out.to_csv(csv_repo, index=False)
    df_out.to_parquet(parquet_repo, index=False)
    
    csv_v4 = out_dir_v4 / "iridia_reference_af_density_dataset.csv"
    parquet_v4 = out_dir_v4 / "iridia_reference_af_density_dataset.parquet"
    df_out.to_csv(csv_v4, index=False)
    df_out.to_parquet(parquet_v4, index=False)
    
    csv_codex = out_dir_codex / "iridia_reference_af_density_dataset.csv"
    parquet_codex = out_dir_codex / "iridia_reference_af_density_dataset.parquet"
    df_out.to_csv(csv_codex, index=False)
    df_out.to_parquet(parquet_codex, index=False)
    
    print(f"\nSuccessfully generated reference dataset ({len(df_out)} rows) in {time.time() - t0:.2f}s!")
    print(f"Saved to:\n  - {csv_repo}\n  - {parquet_repo}\n  - {csv_v4}\n  - {parquet_v4}\n  - {csv_codex}")
    
    # Print key verification statistics
    print("\n" + "=" * 80)
    print("KEY COHORT & EPISODE VERIFICATION STATISTICS")
    print("=" * 80)
    print(f"Total Holter recordings: {len(df_out)}")
    print(f"Total unique patients: {df_out['patient_id'].nunique()}")
    print(f"Total annotated AF episodes: {df_out['num_af_episodes'].sum()}")
    print(f"Total acquired monitoring hours: {df_out['acquired_waveform_hours'].sum():.1f} hours")
    print(f"Total annotated AF hours: {df_out['total_af_hours'].sum():.1f} hours")
    
    print("\n[Episode Count Distribution per Recording]:")
    print(df_out['num_af_episodes'].value_counts().sort_index())
    
    print("\n[Single vs Multi-Episode Informative Split]:")
    print(f"Single AF episode (Density = 1.0 structurally): {df_out['is_single_episode'].sum()} / {len(df_out)} ({df_out['is_single_episode'].mean()*100:.1f}%)")
    print(f"Two AF episodes: {(df_out['num_af_episodes'] == 2).sum()} / {len(df_out)} ({(df_out['num_af_episodes'] == 2).mean()*100:.1f}%)")
    print(f"Three or more AF episodes: {(df_out['num_af_episodes'] >= 3).sum()} / {len(df_out)} ({(df_out['num_af_episodes'] >= 3).mean()*100:.1f}%)")
    print(f"Multi-episode Informative Total: {df_out['is_multi_episode_informative'].sum()} / {len(df_out)} ({df_out['is_multi_episode_informative'].mean()*100:.1f}%)")
    
    print("\n[Partition Split Breakdown]:")
    for part in ["train", "validation", "test"]:
        sub = df_out[df_out["split"] == part]
        pts = sub["patient_id"].nunique()
        recs = len(sub)
        multi_recs = sub["is_multi_episode_informative"].sum()
        multi_pts = sub[sub["is_multi_episode_informative"]]["patient_id"].nunique()
        print(f"  {part.upper():12s}: {pts:2d} patients | {recs:2d} recordings | {multi_recs:2d} multi-episode recs | {multi_pts:2d} multi-episode patients")
        
    print("\n[Patients with Multiple Recordings]:")
    multi_rec_pts = df_out[df_out["is_multi_record_patient"]]
    print(f"Total multi-record patients: {multi_rec_pts['patient_id'].nunique()}")
    print(f"Total recordings belonging to multi-record patients: {len(multi_rec_pts)}")
    
    print("\n[Continuous vs 30s-Binned Charitos Density Parity]:")
    diffs = df_out["density_difference_cont_vs_binned"]
    print(f"Max absolute discrepancy: {diffs.max():.6f}")
    print(f"Mean absolute discrepancy: {diffs.mean():.6f}")
    print(f"Median absolute discrepancy: {diffs.median():.6f}")
    print(f"95th percentile discrepancy: {np.percentile(diffs, 95):.6f}")
    
    print("\n[Recording Gaps and Censoring]:")
    print(f"Recordings with wall-clock recording gap > 1s: {df_out['has_recording_gap'].sum()} / {len(df_out)}")
    print(f"Left-censored recordings (AF at start): {df_out['left_censored'].sum()} / {len(df_out)}")
    print(f"Right-censored recordings (AF at end): {df_out['right_censored'].sum()} / {len(df_out)}")


if __name__ == "__main__":
    run()
