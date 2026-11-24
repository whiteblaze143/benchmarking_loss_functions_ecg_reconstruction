#!/usr/bin/env python3
"""Compute authentic reference AF density, aggregation, episode topology, and interepisode distributions
across all three eligible clinical Holter datasets: IRIDIA-AF, SHDB-AF, and LTAFDB.

Scope:
- 100% computed directly from expert-annotated reference interval ground truth.
- Zero machine learning model predictions, zero temporal decoding, zero sliding-window approximations.
- Computes:
  1. AF Burden
  2. AF Density (continuous-time exact integral + 30-second binned author port)
  3. AF Aggregation (discrete temporal aggregation index)
  4. Episode Count & Single vs Multi-episode classification
  5. Longest AF Episode duration and fraction of total AF
  6. Episode-duration distribution (quantiles and clinical duration bins)
  7. Interepisode intervals (quantiles and gap duration bins)
  8. Actual monitoring duration
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

# Canonical metrics
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/af_density"))
from metrics import charitos_density, discrete_pattern_from_runs
from elapsed_evaluation import equal_bin_amounts


@numba.njit
def fast_charitos_binned_density(proportions, minim=0.001, limit=True):
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
    burden = total / N
    raw = 2.0 * area / (1.0 - burden) if burden < 1.0 else np.inf
    return min(raw, 1.0) if limit else raw


def process_intervals_to_runs(raw_intervals, duration_seconds):
    """Sort and merge abutting/overlapping intervals, ensuring clean intervals within [0, duration]."""
    if not raw_intervals:
        return []
    sorted_int = sorted(raw_intervals, key=lambda x: x[0])
    merged = []
    for a, b in sorted_int:
        a = max(0.0, float(a))
        b = min(float(duration_seconds), float(b))
        if b <= a:
            continue
        if merged and merged[-1][1] >= a:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def compute_recording_metrics(dataset_name, record_id, subject_id, episodes, duration_seconds, extra_meta=None):
    """Compute all 8 requested reference metrics for a single recording."""
    if extra_meta is None:
        extra_meta = {}
        
    dur = float(duration_seconds)
    ep_count = len(episodes)
    
    if ep_count == 0:
        return {
            "dataset": dataset_name,
            "record_id": str(record_id),
            "subject_id": str(subject_id),
            "monitoring_duration_seconds": round(dur, 2),
            "monitoring_duration_hours": round(dur / 3600.0, 3),
            "episode_count": 0,
            "total_af_seconds": 0.0,
            "total_af_hours": 0.0,
            "af_burden": 0.0,
            "charitos_density_continuous": np.nan,
            "charitos_density_binned_30s": 0.0,
            "af_aggregation": np.nan,
            "longest_episode_seconds": 0.0,
            "longest_episode_hours": 0.0,
            "longest_episode_fraction": 0.0,
            "shortest_episode_seconds": 0.0,
            "median_episode_seconds": 0.0,
            "mean_episode_seconds": 0.0,
            "interepisode_count": 0,
            "interepisode_mean_seconds": np.nan,
            "interepisode_median_seconds": np.nan,
            "interepisode_min_seconds": np.nan,
            "interepisode_max_seconds": np.nan,
            "is_single_episode": False,
            "is_multi_episode": False,
            "has_af": False,
            **extra_meta
        }
        
    # Non-zero AF calculations
    durations = [b - a for a, b in episodes]
    total_af = math.fsum(durations)
    burden = total_af / dur
    
    longest_ep = max(durations)
    longest_frac = longest_ep / total_af if total_af > 0 else 0.0
    shortest_ep = min(durations)
    median_ep = float(np.median(durations))
    mean_ep = float(np.mean(durations))
    
    # Interepisode intervals (gaps between consecutive episodes)
    if ep_count >= 2:
        gaps = [episodes[i+1][0] - episodes[i][1] for i in range(ep_count - 1)]
        inter_count = len(gaps)
        inter_mean = float(np.mean(gaps))
        inter_median = float(np.median(gaps))
        inter_min = float(min(gaps))
        inter_max = float(max(gaps))
    else:
        gaps = []
        inter_count = 0
        inter_mean = np.nan
        inter_median = np.nan
        inter_min = np.nan
        inter_max = np.nan
        
    # Density calculations
    d_cont = charitos_density(episodes, dur) if 0 < burden < 1 else np.nan
    
    # Binned density
    bin_fractions = equal_bin_amounts(episodes, dur, 30.0)
    d_bin = fast_charitos_binned_density(bin_fractions, minim=0.001, limit=True)
    
    # Aggregation calculation via discrete_pattern_from_runs
    # Round to integer seconds for discrete aggregation index
    int_runs_raw = [(int(round(a)), int(round(b))) for a, b in episodes if int(round(b)) > int(round(a))]
    int_runs = []
    for a, b in int_runs_raw:
        if int_runs and int_runs[-1][1] >= a:
            int_runs[-1] = (int_runs[-1][0], max(int_runs[-1][1], b))
        else:
            int_runs.append((a, b))
            
    n_bins = int(round(dur))
    k_af = sum(b - a for a, b in int_runs)
    if 0 < k_af < n_bins:
        try:
            pattern_res = discrete_pattern_from_runs(int_runs, n_bins)
            aggregation = pattern_res.get("aggregation", np.nan)
        except Exception:
            aggregation = np.nan
    else:
        aggregation = np.nan
        
    return {
        "dataset": dataset_name,
        "record_id": str(record_id),
        "subject_id": str(subject_id),
        "monitoring_duration_seconds": round(dur, 2),
        "monitoring_duration_hours": round(dur / 3600.0, 3),
        "episode_count": ep_count,
        "total_af_seconds": round(total_af, 2),
        "total_af_hours": round(total_af / 3600.0, 3),
        "af_burden": round(burden, 6),
        "charitos_density_continuous": round(d_cont, 6) if d_cont is not None and not np.isnan(d_cont) else np.nan,
        "charitos_density_binned_30s": round(d_bin, 6) if d_bin is not None and not np.isnan(d_bin) else np.nan,
        "af_aggregation": round(aggregation, 6) if aggregation is not None and not np.isnan(aggregation) else np.nan,
        "longest_episode_seconds": round(longest_ep, 2),
        "longest_episode_hours": round(longest_ep / 3600.0, 3),
        "longest_episode_fraction": round(longest_frac, 6),
        "shortest_episode_seconds": round(shortest_ep, 2),
        "median_episode_seconds": round(median_ep, 2),
        "mean_episode_seconds": round(mean_ep, 2),
        "interepisode_count": inter_count,
        "interepisode_mean_seconds": round(inter_mean, 2) if not np.isnan(inter_mean) else np.nan,
        "interepisode_median_seconds": round(inter_median, 2) if not np.isnan(inter_median) else np.nan,
        "interepisode_min_seconds": round(inter_min, 2) if not np.isnan(inter_min) else np.nan,
        "interepisode_max_seconds": round(inter_max, 2) if not np.isnan(inter_max) else np.nan,
        "is_single_episode": bool(ep_count == 1),
        "is_multi_episode": bool(ep_count >= 2),
        "has_af": True,
        **extra_meta
    }


def run():
    print("=" * 80)
    print("THREE-DATASET REFERENCE AF-DENSITY & EPISODE TOPOLOGY BENCHMARK")
    print("=" * 80)
    
    t0 = time.time()
    
    all_rows = []
    dataset_episodes = {"IRIDIA-AF": [], "SHDB-AF": [], "LTAFDB": []}
    dataset_gaps = {"IRIDIA-AF": [], "SHDB-AF": [], "LTAFDB": []}
    
    # =========================================================================
    # 1. IRIDIA-AF (167 Holter recordings, 152 patients)
    # =========================================================================
    print("\n[1/3] Processing IRIDIA-AF...")
    iridia_dir = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
    df_iridia_meta = pd.read_csv(iridia_dir / "iridia-af-metadata-v1.0.1.csv")
    df_iridia_audit = pd.read_csv(iridia_dir / "timeline_record_audit.csv")
    df_iridia_int = pd.read_csv(iridia_dir / "timeline_intervals.csv")
    
    for _, meta_row in df_iridia_meta.iterrows():
        rec_id = meta_row["record_id"]
        pat_id = meta_row["patient_id"]
        audit_match = df_iridia_audit[df_iridia_audit["record_id"] == rec_id].iloc[0]
        dur = float(audit_match["waveform_seconds"])
        
        sub = df_iridia_int[df_iridia_int["record_id"] == rec_id].sort_values("start_seconds")
        raw_ints = [(row["start_seconds"], row["end_seconds"]) for _, row in sub.iterrows()]
        merged_ep = process_intervals_to_runs(raw_ints, dur)
        
        res = compute_recording_metrics("IRIDIA-AF", rec_id, pat_id, merged_ep, dur, {
            "patient_sex": meta_row["patient_sex"],
            "patient_age": int(meta_row["patient_age"]),
            "wallclock_seconds": float(meta_row["record_timedelta"]),
            "recording_gap_seconds": float(meta_row["record_timedelta"]) - dur,
        })
        all_rows.append(res)
        
        for a, b in merged_ep:
            dataset_episodes["IRIDIA-AF"].append(b - a)
        if len(merged_ep) >= 2:
            for i in range(len(merged_ep) - 1):
                dataset_gaps["IRIDIA-AF"].append(merged_ep[i+1][0] - merged_ep[i][1])
                
    print(f"  Processed {len(df_iridia_meta)} IRIDIA-AF recordings ({len(dataset_episodes['IRIDIA-AF'])} episodes).")
    
    # =========================================================================
    # 2. SHDB-AF (St. Petersburg INCART 12-lead Holter Database, 98 annotated)
    # =========================================================================
    print("\n[2/3] Processing SHDB-AF...")
    df_shdb_rec = pd.read_csv(iridia_dir / "shdb_record_audit.csv")
    df_shdb_int = pd.read_csv(iridia_dir / "shdb_reference_intervals.csv")
    
    for _, rec_row in df_shdb_rec.iterrows():
        rec_id = str(rec_row["record_id"])
        subj_id = str(rec_row["subject_id"])
        annotated = bool(rec_row["annotated"])
        dur = float(rec_row["duration_seconds"])
        
        if not annotated:
            continue
            
        sub = df_shdb_int[(df_shdb_int["record_id"].astype(str) == rec_id) & (df_shdb_int["rhythm"] == "(AFIB")].sort_values("start_sample")
        fs = float(rec_row["sampling_rate"])
        raw_ints = [(r["start_sample"] / fs, r["end_sample_exclusive"] / fs) for _, r in sub.iterrows()]
        merged_ep = process_intervals_to_runs(raw_ints, dur)
        
        res = compute_recording_metrics("SHDB-AF", rec_id, subj_id, merged_ep, dur, {
            "channel_names": rec_row["channel_names"],
            "initial_unknown_seconds": float(rec_row["initial_unknown_seconds"]),
        })
        all_rows.append(res)
        
        for a, b in merged_ep:
            dataset_episodes["SHDB-AF"].append(b - a)
        if len(merged_ep) >= 2:
            for i in range(len(merged_ep) - 1):
                dataset_gaps["SHDB-AF"].append(merged_ep[i+1][0] - merged_ep[i][1])
                
    print(f"  Processed {len([r for r in all_rows if r['dataset'] == 'SHDB-AF'])} SHDB-AF annotated recordings ({len(dataset_episodes['SHDB-AF'])} episodes).")
    
    # =========================================================================
    # 3. LTAFDB (Long-Term Atrial Fibrillation Database, 84 recordings)
    # =========================================================================
    print("\n[3/3] Processing LTAFDB...")
    ltaf_dir = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
    df_ltaf_rec = pd.read_csv(ltaf_dir / "record_reference_audit.csv")
    df_ltaf_int = pd.read_csv(ltaf_dir / "reference_rhythm_intervals.csv")
    
    for _, rec_row in df_ltaf_rec.iterrows():
        rec_id = str(rec_row["record_id"])
        subj_id = f"ltaf_{rec_id}"
        dur = float(rec_row["raw_duration_seconds"])
        fs = float(rec_row["sampling_rate"])
        
        sub = df_ltaf_int[(df_ltaf_int["record_id"].astype(str) == rec_id) & (df_ltaf_int["rhythm"] == "(AFIB")].sort_values("start_sample")
        raw_ints = [(r["start_sample"] / fs, r["end_sample_exclusive"] / fs) for _, r in sub.iterrows()]
        merged_ep = process_intervals_to_runs(raw_ints, dur)
        
        res = compute_recording_metrics("LTAFDB", rec_id, subj_id, merged_ep, dur, {
            "initial_unknown_seconds": float(rec_row["initial_unknown_seconds"]),
            "annotated_suffix_seconds": float(rec_row["rhythm_annotated_suffix_seconds"]),
        })
        all_rows.append(res)
        
        for a, b in merged_ep:
            dataset_episodes["LTAFDB"].append(b - a)
        if len(merged_ep) >= 2:
            for i in range(len(merged_ep) - 1):
                dataset_gaps["LTAFDB"].append(merged_ep[i+1][0] - merged_ep[i][1])
                
    print(f"  Processed {len([r for r in all_rows if r['dataset'] == 'LTAFDB'])} LTAFDB recordings ({len(dataset_episodes['LTAFDB'])} episodes).")
    
    # Create DataFrame
    df_master = pd.DataFrame(all_rows)
    
    # Save CSV and Parquet
    out_dir_repo = ROOT / "reports"
    out_dir_repo.mkdir(exist_ok=True, parents=True)
    out_dir_v4 = ROOT / "experiments/af_density_v4/reports"
    out_dir_v4.mkdir(exist_ok=True, parents=True)
    out_dir_codex = Path("/data/mithunmanivannan/codex_artifacts/af_density_v4")
    out_dir_codex.mkdir(exist_ok=True, parents=True)
    
    csv_file = out_dir_repo / "three_dataset_reference_af_density_manifest.csv"
    parquet_file = out_dir_repo / "three_dataset_reference_af_density_manifest.parquet"
    df_master.to_csv(csv_file, index=False)
    df_master.to_parquet(parquet_file, index=False)
    
    df_master.to_csv(out_dir_v4 / "three_dataset_reference_af_density_manifest.csv", index=False)
    df_master.to_parquet(out_dir_v4 / "three_dataset_reference_af_density_manifest.parquet", index=False)
    
    df_master.to_csv(out_dir_codex / "three_dataset_reference_af_density_manifest.csv", index=False)
    df_master.to_parquet(out_dir_codex / "three_dataset_reference_af_density_manifest.parquet", index=False)
    
    print(f"\nSaved master manifest ({len(df_master)} total recordings) to:\n  - {csv_file}\n  - {parquet_file}")
    
    # =========================================================================
    # Compute Cross-Dataset Summary Statistics & Generate Markdown Report
    # =========================================================================
    print("\nSynthesizing Comprehensive Cross-Dataset Markdown Report...")
    generate_markdown_report(df_master, dataset_episodes, dataset_gaps, out_dir_repo, out_dir_v4, out_dir_codex)
    
    print(f"\nAll three-dataset reference benchmarking completed in {time.time() - t0:.2f}s!")


def format_stats(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "N/A"
    return f"{np.median(arr):.3f} [{np.percentile(arr, 25):.3f}, {np.percentile(arr, 75):.3f}] (min: {np.min(arr):.3f}, max: {np.max(arr):.3f})"


def bin_distribution(arr, edges, labels):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    total = len(arr)
    if total == 0:
        return {label: "0 (0.0%)" for label in labels}
    counts = np.histogram(arr, bins=edges)[0]
    return {label: f"{count} ({count/total*100:.1f}%)" for label, count in zip(labels, counts)}


def generate_markdown_report(df, episodes_dict, gaps_dict, out_dir1, out_dir2, out_dir3):
    datasets = ["IRIDIA-AF", "SHDB-AF", "LTAFDB"]
    
    report_lines = []
    report_lines.append("# Empirical Characterization of Reference AF Density, Aggregation, and Episode Dynamics Across Three Clinical Cohorts")
    report_lines.append("")
    report_lines.append("**Program:** T-CAIREM Multi-Cohort Continuous AF-Density Research  ")
    report_lines.append("**Date:** November 23, 2026  ")
    report_lines.append("**Investigation:** Independent Empirical Ground-Truth Characterization Across All Three Eligible Datasets  ")
    report_lines.append("**Datasets Evaluated:** IRIDIA-AF, SHDB-AF (St. Petersburg INCART), LTAFDB (Long-Term Atrial Fibrillation Database)  ")
    report_lines.append("**Source Artifacts:**")
    report_lines.append("- Master Manifest: `reports/three_dataset_reference_af_density_manifest.csv`")
    report_lines.append("- Master Parquet: `reports/three_dataset_reference_af_density_manifest.parquet`")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("Prior conclusions regarding AF density in this program were drawn almost exclusively from IRIDIA-AF, which featured only 11 informative multi-episode validation patients in the primary model comparison. To establish an authentic clinical and statistical baseline, we have computed all fundamental temporal AF pattern metrics directly from expert ground-truth annotations across all **three eligible clinical Holter datasets** (349 total recordings, 8,540 annotated AF episodes, and 9,332 hours of continuous monitoring).")
    report_lines.append("")
    report_lines.append("### Key Multi-Cohort Findings")
    report_lines.append("")
    report_lines.append("1. **Structural Contrast in Episode Topology:**")
    report_lines.append("   - **IRIDIA-AF (167 records, 388 episodes):** Characterized by prolonged continuous observation (mean 38.0 hours, up to 95.0 hours) but **very low episode fragmentation** (median 1 episode/record, mean 2.3 episodes). Fully **57.5% (96/167)** of recordings contain only a single solitary AF episode, structurally predisposing them to maximum continuous density ($D_{\\mathrm{AF}} \\equiv 1.0$).")
    report_lines.append("   - **SHDB-AF (98 records, 794 episodes):** Standard 24-hour recordings with moderate paroxysmal bursting (mean 8.1 episodes/record, up to 173 episodes). Only 23.5% of recordings are single-episode, while 55.1% are multi-episode informative.")
    report_lines.append("   - **LTAFDB (84 records, 7,358 episodes):** Standard 24-hour recordings with **extreme episode fragmentation** (median 12 episodes/record, mean 87.6 episodes, max 1,044 episodes). Only 1.2% are single-episode; 97.6% are multi-episode paroxysmal recordings.")
    report_lines.append("")
    report_lines.append("2. **AF Density vs. AF Aggregation Divergence:**")
    report_lines.append("   - In **IRIDIA-AF**, Charitos continuous density is heavily skewed toward unity (median 1.000, mean 0.897) due to single-episode dominance.")
    report_lines.append("   - In **LTAFDB**, where paroxysmal fragmentation dominates, Charitos density drops to a median of **0.865** (IQR: 0.635–0.967, min: 0.126), and temporal aggregation drops to a median of **0.472** (IQR: 0.231–0.781). This proves that highly fragmented AF patterns produce genuinely diverse density and aggregation values that do not collapse to 1.0.")
    report_lines.append("")
    report_lines.append("3. **Episode Duration and Interepisode Dynamics:**")
    report_lines.append("   - In **IRIDIA-AF**, episodes are long (median duration: **2.61 hours**, 43.8% exceeding 1 hour). Interepisode intervals are long and sparse (median gap: **5.67 hours**).")
    report_lines.append("   - In **LTAFDB**, episodes are brief and highly recurrent (median duration: **3.88 minutes**, with 45.2% lasting under 2 minutes). Interepisode intervals are very short (median gap: **1.88 minutes**).")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Cross-Dataset Cohort Characteristics & Monitoring Scope")
    report_lines.append("")
    report_lines.append("| Metric | IRIDIA-AF | SHDB-AF (Annotated) | LTAFDB | Combined All Cohorts |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    n_iridia = len(df[df["dataset"] == "IRIDIA-AF"])
    n_shdb = len(df[df["dataset"] == "SHDB-AF"])
    n_ltaf = len(df[df["dataset"] == "LTAFDB"])
    n_total = len(df)
    
    pts_iridia = df[df["dataset"] == "IRIDIA-AF"]["subject_id"].nunique()
    pts_shdb = df[df["dataset"] == "SHDB-AF"]["subject_id"].nunique()
    pts_ltaf = df[df["dataset"] == "LTAFDB"]["subject_id"].nunique()
    pts_total = pts_iridia + pts_shdb + pts_ltaf
    
    hrs_iridia = df[df["dataset"] == "IRIDIA-AF"]["monitoring_duration_hours"].sum()
    hrs_shdb = df[df["dataset"] == "SHDB-AF"]["monitoring_duration_hours"].sum()
    hrs_ltaf = df[df["dataset"] == "LTAFDB"]["monitoring_duration_hours"].sum()
    hrs_total = hrs_iridia + hrs_shdb + hrs_ltaf
    
    af_hrs_iridia = df[df["dataset"] == "IRIDIA-AF"]["total_af_hours"].sum()
    af_hrs_shdb = df[df["dataset"] == "SHDB-AF"]["total_af_hours"].sum()
    af_hrs_ltaf = df[df["dataset"] == "LTAFDB"]["total_af_hours"].sum()
    af_hrs_total = af_hrs_iridia + af_hrs_shdb + af_hrs_ltaf
    
    ep_iridia = df[df["dataset"] == "IRIDIA-AF"]["episode_count"].sum()
    ep_shdb = df[df["dataset"] == "SHDB-AF"]["episode_count"].sum()
    ep_ltaf = df[df["dataset"] == "LTAFDB"]["episode_count"].sum()
    ep_total = ep_iridia + ep_shdb + ep_ltaf
    
    report_lines.append(f"| **Total Holter Recordings** | {n_iridia} | {n_shdb} | {n_ltaf} | {n_total} |")
    report_lines.append(f"| **Unique Patients / Subjects** | {pts_iridia} | {pts_shdb} | {pts_ltaf} | {pts_total} |")
    report_lines.append(f"| **Total Monitoring Duration** | {hrs_iridia:.1f} hours | {hrs_shdb:.1f} hours | {hrs_ltaf:.1f} hours | {hrs_total:.1f} hours |")
    report_lines.append(f"| **Total Annotated AF Duration** | {af_hrs_iridia:.1f} hours | {af_hrs_shdb:.1f} hours | {af_hrs_ltaf:.1f} hours | {af_hrs_total:.1f} hours |")
    report_lines.append(f"| **Total Annotated AF Episodes** | {ep_iridia} | {ep_shdb} | {ep_ltaf} | {ep_total} |")
    report_lines.append(f"| **Sampling Rate (Hz)** | 200 Hz | 200 Hz | 128 Hz | 128–200 Hz |")
    report_lines.append(f"| **Leads Available** | 2 channels | 12 leads (ECG1/ECG2 audited) | 2 channels | 2–12 channels |")
    
    dur_iridia = df[df["dataset"] == "IRIDIA-AF"]["monitoring_duration_hours"]
    dur_shdb = df[df["dataset"] == "SHDB-AF"]["monitoring_duration_hours"]
    dur_ltaf = df[df["dataset"] == "LTAFDB"]["monitoring_duration_hours"]
    dur_all = df["monitoring_duration_hours"]
    report_lines.append(f"| **Monitoring Duration / Record (h)** | {format_stats(dur_iridia)} | {format_stats(dur_shdb)} | {format_stats(dur_ltaf)} | {format_stats(dur_all)} |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. AF Episode Counts & Single vs. Multi-Episode Informative Breakdown")
    report_lines.append("")
    report_lines.append("| Episode Metric | IRIDIA-AF | SHDB-AF | LTAFDB | Combined All Cohorts |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    ep_cnt_iridia = df[df["dataset"] == "IRIDIA-AF"]["episode_count"]
    ep_cnt_shdb = df[df["dataset"] == "SHDB-AF"]["episode_count"]
    ep_cnt_ltaf = df[df["dataset"] == "LTAFDB"]["episode_count"]
    ep_cnt_all = df["episode_count"]
    report_lines.append(f"| **Episode Count / Record [IQR]** | {format_stats(ep_cnt_iridia)} | {format_stats(ep_cnt_shdb)} | {format_stats(ep_cnt_ltaf)} | {format_stats(ep_cnt_all)} |")
    
    zero_iridia = (ep_cnt_iridia == 0).sum()
    zero_shdb = (ep_cnt_shdb == 0).sum()
    zero_ltaf = (ep_cnt_ltaf == 0).sum()
    zero_all = (ep_cnt_all == 0).sum()
    report_lines.append(f"| **Non-AF Recordings (0 episodes)** | {zero_iridia} ({zero_iridia/n_iridia*100:.1f}%) | {zero_shdb} ({zero_shdb/n_shdb*100:.1f}%) | {zero_ltaf} ({zero_ltaf/n_ltaf*100:.1f}%) | {zero_all} ({zero_all/n_total*100:.1f}%) |")
    
    single_iridia = (ep_cnt_iridia == 1).sum()
    single_shdb = (ep_cnt_shdb == 1).sum()
    single_ltaf = (ep_cnt_ltaf == 1).sum()
    single_all = (ep_cnt_all == 1).sum()
    report_lines.append(f"| **Single AF Episode ($D \\equiv 1.0$)** | **{single_iridia} ({single_iridia/n_iridia*100:.1f}%)** | {single_shdb} ({single_shdb/n_shdb*100:.1f}%) | {single_ltaf} ({single_ltaf/n_ltaf*100:.1f}%) | {single_all} ({single_all/n_total*100:.1f}%) |")
    
    two_iridia = (ep_cnt_iridia == 2).sum()
    two_shdb = (ep_cnt_shdb == 2).sum()
    two_ltaf = (ep_cnt_ltaf == 2).sum()
    two_all = (ep_cnt_all == 2).sum()
    report_lines.append(f"| **Two AF Episodes** | {two_iridia} ({two_iridia/n_iridia*100:.1f}%) | {two_shdb} ({two_shdb/n_shdb*100:.1f}%) | {two_ltaf} ({two_ltaf/n_ltaf*100:.1f}%) | {two_all} ({two_all/n_total*100:.1f}%) |")
    
    multi3_iridia = (ep_cnt_iridia >= 3).sum()
    multi3_shdb = (ep_cnt_shdb >= 3).sum()
    multi3_ltaf = (ep_cnt_ltaf >= 3).sum()
    multi3_all = (ep_cnt_all >= 3).sum()
    report_lines.append(f"| **Three or More AF Episodes** | {multi3_iridia} ({multi3_iridia/n_iridia*100:.1f}%) | {multi3_shdb} ({multi3_shdb/n_shdb*100:.1f}%) | **{multi3_ltaf} ({multi3_ltaf/n_ltaf*100:.1f}%)** | {multi3_all} ({multi3_all/n_total*100:.1f}%) |")
    
    multi_tot_iridia = (ep_cnt_iridia >= 2).sum()
    multi_tot_shdb = (ep_cnt_shdb >= 2).sum()
    multi_tot_ltaf = (ep_cnt_ltaf >= 2).sum()
    multi_tot_all = (ep_cnt_all >= 2).sum()
    report_lines.append(f"| **Multi-Episode Informative Total** | **{multi_tot_iridia} ({multi_tot_iridia/n_iridia*100:.1f}%)** | **{multi_tot_shdb} ({multi_tot_shdb/n_shdb*100:.1f}%)** | **{multi_tot_ltaf} ({multi_tot_ltaf/n_ltaf*100:.1f}%)** | **{multi_tot_all} ({multi_tot_all/n_total*100:.1f}%)** |")
    
    report_lines.append("")
    report_lines.append("> **Clinical Interpretation:**")
    report_lines.append("> IRIDIA-AF is dominated by persistent-like solitary episodes (57.5%), which limits its ability to evaluate pattern dispersion algorithms. In stark contrast, LTAFDB is almost entirely paroxysmal (97.6% multi-episode, with 50% of recordings having $\\ge 12$ episodes and up to 1,044 episodes). LTAFDB and SHDB-AF provide the necessary empirical diversity to validate density and aggregation metrics.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Reference AF Burden, Density, and Aggregation Distributions")
    report_lines.append("")
    report_lines.append("| Metric | IRIDIA-AF | SHDB-AF | LTAFDB | Combined All Cohorts |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    b_iridia = df[df["dataset"] == "IRIDIA-AF"]["af_burden"]
    b_shdb = df[(df["dataset"] == "SHDB-AF") & (df["has_af"])]["af_burden"]
    b_ltaf = df[(df["dataset"] == "LTAFDB") & (df["has_af"])]["af_burden"]
    b_all = df[df["has_af"]]["af_burden"]
    report_lines.append(f"| **AF Burden (AF Records) [IQR]** | {format_stats(b_iridia)} | {format_stats(b_shdb)} | {format_stats(b_ltaf)} | {format_stats(b_all)} |")
    
    d_cont_iridia = df[df["dataset"] == "IRIDIA-AF"]["charitos_density_continuous"]
    d_cont_shdb = df[df["dataset"] == "SHDB-AF"]["charitos_density_continuous"]
    d_cont_ltaf = df[df["dataset"] == "LTAFDB"]["charitos_density_continuous"]
    d_cont_all = df["charitos_density_continuous"]
    report_lines.append(f"| **Continuous Charitos Density (All AF)** | {format_stats(d_cont_iridia)} | {format_stats(d_cont_shdb)} | {format_stats(d_cont_ltaf)} | {format_stats(d_cont_all)} |")
    
    # Continuous density among MULTI-EPISODE records
    d_multi_iridia = df[(df["dataset"] == "IRIDIA-AF") & (df["is_multi_episode"])]["charitos_density_continuous"]
    d_multi_shdb = df[(df["dataset"] == "SHDB-AF") & (df["is_multi_episode"])]["charitos_density_continuous"]
    d_multi_ltaf = df[(df["dataset"] == "LTAFDB") & (df["is_multi_episode"])]["charitos_density_continuous"]
    d_multi_all = df[df["is_multi_episode"]]["charitos_density_continuous"]
    report_lines.append(f"| **Continuous Density (Multi-Episode Only)** | **{format_stats(d_multi_iridia)}** | **{format_stats(d_multi_shdb)}** | **{format_stats(d_multi_ltaf)}** | **{format_stats(d_multi_all)}** |")
    
    d_bin_iridia = df[df["dataset"] == "IRIDIA-AF"]["charitos_density_binned_30s"]
    d_bin_shdb = df[(df["dataset"] == "SHDB-AF") & (df["has_af"])]["charitos_density_binned_30s"]
    d_bin_ltaf = df[(df["dataset"] == "LTAFDB") & (df["has_af"])]["charitos_density_binned_30s"]
    d_bin_all = df[df["has_af"]]["charitos_density_binned_30s"]
    report_lines.append(f"| **Author 30s-Binned Density [IQR]** | {format_stats(d_bin_iridia)} | {format_stats(d_bin_shdb)} | {format_stats(d_bin_ltaf)} | {format_stats(d_bin_all)} |")
    
    agg_iridia = df[df["dataset"] == "IRIDIA-AF"]["af_aggregation"]
    agg_shdb = df[df["dataset"] == "SHDB-AF"]["af_aggregation"]
    agg_ltaf = df[df["dataset"] == "LTAFDB"]["af_aggregation"]
    agg_all = df["af_aggregation"]
    report_lines.append(f"| **AF Aggregation (Discrete Index) [IQR]** | {format_stats(agg_iridia)} | {format_stats(agg_shdb)} | {format_stats(agg_ltaf)} | {format_stats(agg_all)} |")
    
    report_lines.append("")
    report_lines.append("### Burden-Stratified Clinical Breakdown")
    report_lines.append("")
    report_lines.append("| Burden Strata | IRIDIA-AF | SHDB-AF (AF records) | LTAFDB (AF records) | Combined AF Records |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for label, (lo, hi) in [
        ("Low Burden (< 5%)", (0.0, 0.05)),
        ("Mild Burden (5% – 10%)", (0.05, 0.10)),
        ("Moderate Burden (10% – 30%)", (0.10, 0.30)),
        ("High Burden (30% – 60%)", (0.30, 0.60)),
        ("Very High Burden (> 60%)", (0.60, 1.01))
    ]:
        c_iri = ((b_iridia >= lo) & (b_iridia < hi)).sum()
        c_shdb = ((b_shdb >= lo) & (b_shdb < hi)).sum()
        c_ltaf = ((b_ltaf >= lo) & (b_ltaf < hi)).sum()
        c_all = ((b_all >= lo) & (b_all < hi)).sum()
        report_lines.append(f"| **{label}** | {c_iri} ({c_iri/len(b_iridia)*100:.1f}%) | {c_shdb} ({c_shdb/len(b_shdb)*100:.1f}%) | {c_ltaf} ({c_ltaf/len(b_ltaf)*100:.1f}%) | {c_all} ({c_all/len(b_all)*100:.1f}%) |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. Longest AF Episode Distribution & Relative Dominance")
    report_lines.append("")
    report_lines.append("| Longest Episode Characteristic | IRIDIA-AF | SHDB-AF | LTAFDB | Combined All Cohorts |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    long_dur_iridia = df[df["dataset"] == "IRIDIA-AF"]["longest_episode_hours"]
    long_dur_shdb = df[(df["dataset"] == "SHDB-AF") & (df["has_af"])]["longest_episode_hours"]
    long_dur_ltaf = df[(df["dataset"] == "LTAFDB") & (df["has_af"])]["longest_episode_hours"]
    long_dur_all = df[df["has_af"]]["longest_episode_hours"]
    report_lines.append(f"| **Longest Episode Duration (hours) [IQR]** | {format_stats(long_dur_iridia)} | {format_stats(long_dur_shdb)} | {format_stats(long_dur_ltaf)} | {format_stats(long_dur_all)} |")
    
    frac_iridia = df[df["dataset"] == "IRIDIA-AF"]["longest_episode_fraction"]
    frac_shdb = df[(df["dataset"] == "SHDB-AF") & (df["has_af"])]["longest_episode_fraction"]
    frac_ltaf = df[(df["dataset"] == "LTAFDB") & (df["has_af"])]["longest_episode_fraction"]
    frac_all = df[df["has_af"]]["longest_episode_fraction"]
    report_lines.append(f"| **Longest Ep. Fraction of Total AF [IQR]** | {format_stats(frac_iridia)} | {format_stats(frac_shdb)} | {format_stats(frac_ltaf)} | {format_stats(frac_all)} |")
    
    # Stratified longest episode length
    report_lines.append("")
    report_lines.append("| Longest Episode Duration Strata | IRIDIA-AF | SHDB-AF | LTAFDB | Combined |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for label, (lo, hi) in [
        ("< 10 minutes", (0.0, 10/60)),
        ("10 – 60 minutes", (10/60, 1.0)),
        ("1 – 6 hours", (1.0, 6.0)),
        ("6 – 24 hours", (6.0, 24.0)),
        ("> 24 hours", (24.0, 1000.0))
    ]:
        c_iri = ((long_dur_iridia >= lo) & (long_dur_iridia < hi)).sum()
        c_shdb = ((long_dur_shdb >= lo) & (long_dur_shdb < hi)).sum()
        c_ltaf = ((long_dur_ltaf >= lo) & (long_dur_ltaf < hi)).sum()
        c_all = ((long_dur_all >= lo) & (long_dur_all < hi)).sum()
        report_lines.append(f"| **{label}** | {c_iri} ({c_iri/len(long_dur_iridia)*100:.1f}%) | {c_shdb} ({c_shdb/len(long_dur_shdb)*100:.1f}%) | {c_ltaf} ({c_ltaf/len(long_dur_ltaf)*100:.1f}%) | {c_all} ({c_all/len(long_dur_all)*100:.1f}%) |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 5. Individual Episode Duration Distribution (All 8,540 Episodes)")
    report_lines.append("")
    report_lines.append("Analyzing all individual contiguous AF episodes identified by expert annotations across the cohorts:")
    report_lines.append("")
    report_lines.append("| Episode Duration Statistic | IRIDIA-AF (388 episodes) | SHDB-AF (794 episodes) | LTAFDB (7,358 episodes) | Combined (8,540 episodes) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    ep_iri_m = np.array(episodes_dict["IRIDIA-AF"]) / 60.0  # in minutes
    ep_shdb_m = np.array(episodes_dict["SHDB-AF"]) / 60.0
    ep_ltaf_m = np.array(episodes_dict["LTAFDB"]) / 60.0
    ep_all_m = np.concatenate([ep_iri_m, ep_shdb_m, ep_ltaf_m])
    
    report_lines.append(f"| **Episode Duration in Minutes [IQR]** | {format_stats(ep_iri_m)} | {format_stats(ep_shdb_m)} | {format_stats(ep_ltaf_m)} | {format_stats(ep_all_m)} |")
    report_lines.append(f"| **10th Percentile (minutes)** | {np.percentile(ep_iri_m, 10):.2f} min | {np.percentile(ep_shdb_m, 10):.2f} min | {np.percentile(ep_ltaf_m, 10):.2f} min | {np.percentile(ep_all_m, 10):.2f} min |")
    report_lines.append(f"| **25th Percentile (minutes)** | {np.percentile(ep_iri_m, 25):.2f} min | {np.percentile(ep_shdb_m, 25):.2f} min | {np.percentile(ep_ltaf_m, 25):.2f} min | {np.percentile(ep_all_m, 25):.2f} min |")
    report_lines.append(f"| **50th Percentile (Median)** | **{np.percentile(ep_iri_m, 50):.2f} min** | **{np.percentile(ep_shdb_m, 50):.2f} min** | **{np.percentile(ep_ltaf_m, 50):.2f} min** | **{np.percentile(ep_all_m, 50):.2f} min** |")
    report_lines.append(f"| **75th Percentile (minutes)** | {np.percentile(ep_iri_m, 75):.2f} min | {np.percentile(ep_shdb_m, 75):.2f} min | {np.percentile(ep_ltaf_m, 75):.2f} min | {np.percentile(ep_all_m, 75):.2f} min |")
    report_lines.append(f"| **90th Percentile (minutes)** | {np.percentile(ep_iri_m, 90):.2f} min | {np.percentile(ep_shdb_m, 90):.2f} min | {np.percentile(ep_ltaf_m, 90):.2f} min | {np.percentile(ep_all_m, 90):.2f} min |")
    report_lines.append(f"| **Maximum Episode Duration (hours)** | {np.max(ep_iri_m)/60.0:.2f} h | {np.max(ep_shdb_m)/60.0:.2f} h | {np.max(ep_ltaf_m)/60.0:.2f} h | {np.max(ep_all_m)/60.0:.2f} h |")
    
    # Clinical bins for episode durations
    report_lines.append("")
    report_lines.append("| Clinical Episode Duration Buckets | IRIDIA-AF | SHDB-AF | LTAFDB | Combined |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    ep_bins = [0.0, 2.0, 10.0, 60.0, 360.0, 10000.0]
    ep_labels = [
        "< 2 minutes (Micro-episodes)",
        "2 – 10 minutes (Short bursts)",
        "10 – 60 minutes (Intermediate)",
        "1 – 6 hours (Prolonged paroxysm)",
        "> 6 hours (Sustained / Persistent)"
    ]
    b_bins_iri = bin_distribution(ep_iri_m, ep_bins, ep_labels)
    b_bins_shdb = bin_distribution(ep_shdb_m, ep_bins, ep_labels)
    b_bins_ltaf = bin_distribution(ep_ltaf_m, ep_bins, ep_labels)
    b_bins_all = bin_distribution(ep_all_m, ep_bins, ep_labels)
    
    for lbl in ep_labels:
        report_lines.append(f"| **{lbl}** | {b_bins_iri[lbl]} | {b_bins_shdb[lbl]} | {b_bins_ltaf[lbl]} | {b_bins_all[lbl]} |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 6. Interepisode Intervals (Non-AF Gap Dynamics in Multi-Episode Recordings)")
    report_lines.append("")
    report_lines.append("Analyzing all intervals between consecutive AF episodes $(\\Delta t_{\\mathrm{inter}} = \\mathrm{start}_{i+1} - \\mathrm{end}_i)$ in multi-episode recordings:")
    report_lines.append("")
    report_lines.append("| Interepisode Gap Statistic | IRIDIA-AF (221 gaps) | SHDB-AF (717 gaps) | LTAFDB (7,275 gaps) | Combined (8,213 gaps) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    g_iri_m = np.array(gaps_dict["IRIDIA-AF"]) / 60.0  # in minutes
    g_shdb_m = np.array(gaps_dict["SHDB-AF"]) / 60.0
    g_ltaf_m = np.array(gaps_dict["LTAFDB"]) / 60.0
    g_all_m = np.concatenate([g_iri_m, g_shdb_m, g_ltaf_m])
    
    report_lines.append(f"| **Interepisode Gap in Minutes [IQR]** | {format_stats(g_iri_m)} | {format_stats(g_shdb_m)} | {format_stats(g_ltaf_m)} | {format_stats(g_all_m)} |")
    report_lines.append(f"| **10th Percentile (minutes)** | {np.percentile(g_iri_m, 10):.2f} min | {np.percentile(g_shdb_m, 10):.2f} min | {np.percentile(g_ltaf_m, 10):.2f} min | {np.percentile(g_all_m, 10):.2f} min |")
    report_lines.append(f"| **25th Percentile (minutes)** | {np.percentile(g_iri_m, 25):.2f} min | {np.percentile(g_shdb_m, 25):.2f} min | {np.percentile(g_ltaf_m, 25):.2f} min | {np.percentile(g_all_m, 25):.2f} min |")
    report_lines.append(f"| **50th Percentile (Median Gap)** | **{np.percentile(g_iri_m, 50):.2f} min ({np.percentile(g_iri_m, 50)/60.0:.2f} h)** | **{np.percentile(g_shdb_m, 50):.2f} min** | **{np.percentile(g_ltaf_m, 50):.2f} min** | **{np.percentile(g_all_m, 50):.2f} min** |")
    report_lines.append(f"| **75th Percentile (minutes)** | {np.percentile(g_iri_m, 75):.2f} min | {np.percentile(g_shdb_m, 75):.2f} min | {np.percentile(g_ltaf_m, 75):.2f} min | {np.percentile(g_all_m, 75):.2f} min |")
    report_lines.append(f"| **90th Percentile (minutes)** | {np.percentile(g_iri_m, 90):.2f} min | {np.percentile(g_shdb_m, 90):.2f} min | {np.percentile(g_ltaf_m, 90):.2f} min | {np.percentile(g_all_m, 90):.2f} min |")
    report_lines.append(f"| **Maximum Interepisode Gap (hours)** | {np.max(g_iri_m)/60.0:.2f} h | {np.max(g_shdb_m)/60.0:.2f} h | {np.max(g_ltaf_m)/60.0:.2f} h | {np.max(g_all_m)/60.0:.2f} h |")
    
    # Clinical bins for gaps
    report_lines.append("")
    report_lines.append("| Clinical Interepisode Gap Buckets | IRIDIA-AF | SHDB-AF | LTAFDB | Combined |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    gap_bins = [0.0, 5.0, 30.0, 120.0, 360.0, 10000.0]
    gap_labels = [
        "< 5 minutes (Frequent clustered re-triggering)",
        "5 – 30 minutes (Short recurrence)",
        "30 – 120 minutes (Intermediate recovery)",
        "2 – 6 hours (Prolonged sinus interval)",
        "> 6 hours (Sparse / Isolated episodes)"
    ]
    bg_iri = bin_distribution(g_iri_m, gap_bins, gap_labels)
    bg_shdb = bin_distribution(g_shdb_m, gap_bins, gap_labels)
    bg_ltaf = bin_distribution(g_ltaf_m, gap_bins, gap_labels)
    bg_all = bin_distribution(g_all_m, gap_bins, gap_labels)
    
    for lbl in gap_labels:
        report_lines.append(f"| **{lbl}** | {bg_iri[lbl]} | {bg_shdb[lbl]} | {bg_ltaf[lbl]} | {bg_all[lbl]} |")
        
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 7. Mathematical & Clinical Synthesis")
    report_lines.append("")
    report_lines.append("### Why Multicohort Reference Characterization Reshapes Density Research")
    report_lines.append("")
    report_lines.append("1. **The Single-Episode Selection Bias in IRIDIA-AF:**")
    report_lines.append("   - Because 57.5% of IRIDIA-AF recordings contain only a single episode, evaluating density estimators on the entire IRIDIA cohort produces misleadingly inflated benchmark scores: any model that roughly predicts AF within the episode region achieves a predicted density near 1.0, matching the reference density of 1.0.")
    report_lines.append("   - When evaluated only on **multi-episode recordings**, the median density drops from **1.000** to **0.762** in IRIDIA-AF, **0.865** in LTAFDB, and **0.840** in SHDB-AF.")
    report_lines.append("")
    report_lines.append("2. **Density ($D_{\\mathrm{AF}}$) vs. Aggregation ($A$) Orthogonality:**")
    report_lines.append("   - In LTAFDB, where episode counts reach into the hundreds, Charitos density and Butkuviene aggregation provide distinct topological views:")
    report_lines.append("     - Charitos density measures the normalized area between the lower envelope $L(m)$ and the diagonal, scaling by $2 / (1 - b)$.")
    report_lines.append("     - Aggregation measures the maximum AF concentration curve $M(w)$ relative to uniform spread.")
    report_lines.append("     - In highly fragmented records (e.g. record 74 with 1,044 episodes), Charitos density is **0.296**, while Aggregation is **0.021**, demonstrating that aggregation penalizes fine fragmentation more heavily than Charitos density.")
    report_lines.append("")
    report_lines.append("3. **Recommendations for Future 2026 Metrics:**")
    report_lines.append("   - Any new temporal AF dispersion metric proposed in 2026 must be evaluated across **all three datasets** rather than IRIDIA alone.")
    report_lines.append("   - Models and metrics must report performance stratified by **episode fragmentation strata** (single episode vs 2–10 episodes vs >10 episodes).")
    report_lines.append("   - All recording-level reference metrics from this study are permanently committed and queryable in `reports/three_dataset_reference_af_density_manifest.csv`.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 8. Verification & Provenance Manifest")
    report_lines.append("")
    report_lines.append("| Dataset | Total Records | AF Records | Multi-Episode Records | Total Episodes | Total Monitoring Hours | Manifest File |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    report_lines.append(f"| **IRIDIA-AF** | {n_iridia} | {n_iridia} | {multi_tot_iridia} | {ep_iridia} | {hrs_iridia:.1f} h | `reports/three_dataset_reference_af_density_manifest.csv` |")
    report_lines.append(f"| **SHDB-AF** | {n_shdb} | {len(b_shdb)} | {multi_tot_shdb} | {ep_shdb} | {hrs_shdb:.1f} h | `reports/three_dataset_reference_af_density_manifest.csv` |")
    report_lines.append(f"| **LTAFDB** | {n_ltaf} | {len(b_ltaf)} | {multi_tot_ltaf} | {ep_ltaf} | {hrs_ltaf:.1f} h | `reports/three_dataset_reference_af_density_manifest.csv` |")
    report_lines.append(f"| **Total** | **{n_total}** | **{len(b_all)}** | **{multi_tot_all}** | **{ep_total}** | **{hrs_total:.1f} h** | `reports/three_dataset_reference_af_density_manifest.parquet` |")
    
    text = "\n".join(report_lines) + "\n"
    
    report_path1 = out_dir1 / "three_dataset_reference_af_density_report.md"
    report_path2 = out_dir2 / "three_dataset_reference_af_density_report.md"
    report_path3 = out_dir3 / "three_dataset_reference_af_density_report.md"
    
    report_path1.write_text(text)
    report_path2.write_text(text)
    report_path3.write_text(text)
    
    print(f"Report saved to:\n  - {report_path1}\n  - {report_path2}\n  - {report_path3}")


if __name__ == "__main__":
    run()
