#!/usr/bin/env python3
"""Script 10: Experiment E03 - Temporal Localization of Rapid Ventricular Response.
Extracts density-defining shortest intervals I_p (for p in 0.50, 0.75, 0.80, 0.90),
computes rapid-AF capture C_rapid(p), localization contrast Delta C(p) = C_rapid(p) - p,
missed rapid-AF minutes, and benchmarks against alternative interval-selection rules.
Generates:
  - results/E03_concentration_intervals.parquet
  - results/E03_rapid_af_capture.csv
  - results/E03_interval_benchmark.csv
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import find_shortest_interval_for_fraction

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    manifests_dir = root_dir / "manifests"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE F: EXPERIMENT E03 - TEMPORAL LOCALIZATION OF RAPID VENTRICULAR RESPONSE")
    print("=" * 80)

    patient_win_parquet = features_dir / "patient_window_features.parquet"
    rate_itvs_parquet = features_dir / "ventricular_rate_intervals.parquet"
    runs_parquet = manifests_dir / "canonical_af_runs.parquet"

    df_pwin = pd.read_parquet(patient_win_parquet)
    df_rates = pd.read_parquet(rate_itvs_parquet)
    df_runs = pd.read_parquet(runs_parquet)

    # Deduplicate rate intervals across overlapping window definitions
    df_rates = df_rates.drop_duplicates(subset=["dataset", "record_id", "start_time_s", "end_time_s"]).copy()

    # Filter to primary 12h horizon windows with rapid AF active (Q120 > 0)
    df_12h = df_pwin[
        (df_pwin["horizon"] == "12h") &
        (df_pwin["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df_pwin["is_density_defined"]) &
        (df_pwin["q_120"] > 0)
    ].copy()

    print(f"Analyzing rapid-AF localization across {len(df_12h)} windows with active rapid AF...")

    coverages = [0.50, 0.75, 0.80, 0.90]
    concentration_intervals = []
    capture_records = []
    benchmark_records = []

    for idx, w in df_12h.iterrows():
        dataset = w["dataset"]
        rec_id = w["record_id"]
        pat_id = w["patient_id"]
        w_start = float(w["window_start_s"])
        w_end = float(w["window_end_s"])
        w_dur = float(w["window_duration_s"])
        total_af_s = float(w["total_af_s"])
        total_r120_s = float(w["r_120_s"])

        if total_r120_s <= 0:
            continue

        # Fetch clipped AF runs relative to window start
        rec_runs = df_runs[(df_runs["dataset"] == dataset) & (df_runs["record_id"] == rec_id)]
        w_runs = []
        for _, r in rec_runs.iterrows():
            rs = max(w_start, float(r["start_time_s"]))
            re = min(w_end, float(r["end_time_s"]))
            if re > rs:
                w_runs.append((rs - w_start, re - w_start))
        w_runs.sort()

        # Fetch rate intervals in this window
        w_rate_itvs = df_rates[
            (df_rates["dataset"] == dataset) &
            (df_rates["record_id"] == rec_id) &
            (df_rates["start_time_s"] >= w_start) &
            (df_rates["end_time_s"] <= w_end)
        ].copy()
        if w_rate_itvs.empty:
            continue

        # Convert rate intervals to relative coordinates [0, w_dur]
        w_rate_itvs["rel_start"] = w_rate_itvs["start_time_s"] - w_start
        w_rate_itvs["rel_end"] = w_rate_itvs["end_time_s"] - w_start

        # For each target coverage fraction p
        for p in coverages:
            # 1. Density-defining shortest interval I_p
            cand_len, rel_s, rel_e = find_shortest_interval_for_fraction(w_runs, w_dur, p)
            abs_s = w_start + rel_s
            abs_e = w_start + rel_e

            # Rapid AF inside I_p
            in_ip = w_rate_itvs[
                (w_rate_itvs["rel_start"] >= rel_s) &
                (w_rate_itvs["rel_end"] <= rel_e) &
                (w_rate_itvs["is_rapid_120"])
            ]
            r120_inside_s = in_ip["duration_s"].sum()
            r120_outside_s = max(0.0, total_r120_s - r120_inside_s)

            c_rapid = r120_inside_s / total_r120_s if total_r120_s > 0 else 0.0
            delta_c = c_rapid - p
            missed_min = r120_outside_s / 60.0

            concentration_intervals.append({
                "dataset": dataset,
                "record_id": rec_id,
                "patient_id": pat_id,
                "target_fraction_p": p,
                "interval_length_s": cand_len,
                "interval_fraction_of_window": cand_len / w_dur,
                "start_time_s": abs_s,
                "end_time_s": abs_e,
                "total_af_s": total_af_s,
                "total_rapid_af_s": total_r120_s,
                "rapid_af_inside_s": r120_inside_s,
                "rapid_af_outside_s": r120_outside_s,
                "c_rapid": c_rapid,
                "delta_c": delta_c,
                "missed_rapid_af_min": missed_min
            })

            capture_records.append({
                "dataset": dataset,
                "record_id": rec_id,
                "patient_id": pat_id,
                "target_p": p,
                "c_rapid": c_rapid,
                "delta_c": delta_c,
                "missed_min": missed_min,
                "interval_duration_min": cand_len / 60.0
            })

            # Benchmark alternative strategies at primary p=0.80
            if math.isclose(p, 0.80):
                budget_s = cand_len

                # Strategy 1: Density interval
                s1_c = c_rapid

                # Strategy 2: Longest episode anchor (centered or started at longest run)
                longest_run = max(w_runs, key=lambda x: x[1] - x[0]) if w_runs else (0.0, 0.0)
                l_start = longest_run[0]
                l_s2_start = max(0.0, min(w_dur - budget_s, l_start))
                l_s2_end = l_s2_start + budget_s
                in_s2 = w_rate_itvs[
                    (w_rate_itvs["rel_start"] >= l_s2_start) &
                    (w_rate_itvs["rel_end"] <= l_s2_end) &
                    (w_rate_itvs["is_rapid_120"])
                ]["duration_s"].sum()
                s2_c = in_s2 / total_r120_s

                # Strategy 3: Acquisition-anchored interval [0, budget_s]
                in_s3 = w_rate_itvs[
                    (w_rate_itvs["rel_start"] >= 0.0) &
                    (w_rate_itvs["rel_end"] <= budget_s) &
                    (w_rate_itvs["is_rapid_120"])
                ]["duration_s"].sum()
                s3_c = in_s3 / total_r120_s

                # Strategy 4: Oracle interval maximizing directly measured rapid AF of duration budget_s
                # Sliding window of length budget_s across 30s bins
                rapid_bins = w_rate_itvs.sort_values("rel_start").copy()
                max_oracle_r120 = 0.0
                for _, b_start_row in rapid_bins.iterrows():
                    o_s = b_start_row["rel_start"]
                    o_e = o_s + budget_s
                    r_sum = rapid_bins[
                        (rapid_bins["rel_start"] >= o_s) &
                        (rapid_bins["rel_end"] <= o_e) &
                        (rapid_bins["is_rapid_120"])
                    ]["duration_s"].sum()
                    max_oracle_r120 = max(max_oracle_r120, r_sum)
                s5_oracle_c = max_oracle_r120 / total_r120_s

                benchmark_records.append({
                    "dataset": dataset,
                    "record_id": rec_id,
                    "patient_id": pat_id,
                    "budget_minutes": budget_s / 60.0,
                    "density_interval_capture": s1_c,
                    "longest_episode_anchor_capture": s2_c,
                    "acquisition_anchor_capture": s3_c,
                    "oracle_capture": s5_oracle_c,
                    "gap_to_oracle": s5_oracle_c - s1_c
                })

    df_ci = pd.DataFrame(concentration_intervals)
    df_cap = pd.DataFrame(capture_records)
    df_bm = pd.DataFrame(benchmark_records)

    out_ci = results_dir / "E03_concentration_intervals.parquet"
    out_cap = results_dir / "E03_rapid_af_capture.csv"
    out_bm = results_dir / "E03_interval_benchmark.csv"

    df_ci.to_parquet(out_ci, index=False)
    df_cap.to_csv(out_cap, index=False)
    df_bm.to_csv(out_bm, index=False)

    print(f"\nSaved concentration intervals to {out_ci}")
    print(f"Saved capture summary to {out_cap}")
    print(f"Saved interval benchmark to {out_bm}")

    # Summary for primary p=0.80
    p80 = df_cap[df_cap["target_p"] == 0.80]
    print("\n" + "=" * 80)
    print("PRIMARY E03 LOCALIZATION CONTRAST SUMMARY (p = 0.80)")
    print("=" * 80)
    print(f"Total windows with active rapid AF: {len(p80)}")
    print(f"Median Rapid AF Capture C_rapid(0.80): {p80['c_rapid'].median()*100:.1f}%")
    print(f"Median Localization Contrast Delta C(0.80): {p80['delta_c'].median()*100:.1f} percentage points")
    print(f"Median Missed Rapid AF: {p80['missed_min'].median():.2f} minutes")
    print(f"Oracle Capture Benchmark: {df_bm['oracle_capture'].median()*100:.1f}% vs Density Interval: {df_bm['density_interval_capture'].median()*100:.1f}%")

if __name__ == "__main__":
    main()
