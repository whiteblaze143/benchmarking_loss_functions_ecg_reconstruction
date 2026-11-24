#!/usr/bin/env python3
"""Script 18: Rosman Factorial Horizon x Resolution (R03) & Comparative Metric Stability (R05).
Generates:
  - results/R03_factorial_horizon_resolution.csv
  - results/R05_metric_stability_comparison.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density
from src.density_binned import compute_binned_density_from_episodes
from src.rosman_phenotypes import (
    compute_rank_stability,
    compute_intraclass_correlation
)

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    manifests_dir = root_dir / "manifests"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ROSMAN 2026: R03 (FACTORIAL HORIZON x RESOLUTION) & R05 (METRIC STABILITY)")
    print("=" * 80)

    df_intervals = pd.read_parquet(manifests_dir / "canonical_rhythm_intervals.parquet")
    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")

    # Filter to recordings with at least 24 hours of monitoring and some AF (0 < b < 1)
    rec_meta = df_intervals.groupby(["dataset", "record_id", "patient_id"]).agg(
        t_start=("start_time_s", "min"),
        t_end=("end_time_s", "max")
    ).reset_index()
    rec_meta["monitoring_dur_h"] = (rec_meta["t_end"] - rec_meta["t_start"]) / 3600.0

    # Eligible recordings with >= 24h monitoring
    eligible_recs = rec_meta[rec_meta["monitoring_dur_h"] >= 24.0].copy()
    print(f"Eligible continuous recordings with >= 24h monitoring: {len(eligible_recs)}")

    # R03 Factorial Grid
    horizons = {
        "6h": 6.0 * 3600.0,
        "12h": 12.0 * 3600.0,
        "24h": 24.0 * 3600.0
    }
    resolutions = {
        "native_continuous": None,
        "1m": 60.0,
        "5m": 300.0,
        "15m": 900.0,
        "60m": 3600.0,
        "30_equal_bins": "30_bins"
    }

    factorial_records = []
    stability_records = []

    for idx, rec in eligible_recs.iterrows():
        ds = rec["dataset"]
        r_id = rec["record_id"]
        p_id = rec["patient_id"]
        t0 = rec["t_start"]

        rec_rhythm = df_intervals[(df_intervals["dataset"] == ds) & (df_intervals["record_id"] == r_id)]
        rec_af = rec_rhythm[rec_rhythm["rhythm_label_harmonized"] == "AF"]

        # Metrics for R05 stability across horizons
        horiz_metrics = {}

        for h_name, h_dur in horizons.items():
            w_start = t0
            w_end = t0 + h_dur

            # Extract clipped AF runs
            w_runs = []
            total_af_s = 0.0
            for _, r in rec_af.iterrows():
                cs = max(w_start, float(r["start_time_s"]))
                ce = min(w_end, float(r["end_time_s"]))
                if ce > cs:
                    w_runs.append((cs - w_start, ce - w_start))
                    total_af_s += (ce - cs)
            w_runs.sort()

            b = total_af_s / h_dur
            if b <= 0.0 or b >= 1.0:
                continue

            # Native density
            d_native = compute_continuous_density(w_runs, h_dur)
            run_durs = [(e - s) for s, e in w_runs]
            l_max_min = max(run_durs) / 60.0 if run_durs else 0.0
            n_episodes = len(w_runs)
            ep_rate_per_h = n_episodes / (h_dur / 3600.0)

            # AF aggregation (1 - normalized dispersion of durations)
            if len(run_durs) >= 2:
                mean_dur = np.mean(run_durs)
                std_dur = np.std(run_durs)
                cv_dur = std_dur / mean_dur if mean_dur > 0 else 0.0
                af_agg = float(np.clip(1.0 - 1.0 / (cv_dur + 1.0), 0.0, 1.0))
            elif len(run_durs) == 1:
                af_agg = 1.0
            else:
                af_agg = 0.0

            horiz_metrics[h_name] = {
                "burden": b,
                "density": d_native,
                "l_max_min": l_max_min,
                "episode_rate": ep_rate_per_h,
                "af_aggregation": af_agg
            }

            # Factorial resolution evaluation
            for res_name, res_val in resolutions.items():
                if res_name == "native_continuous":
                    d_val = d_native
                elif res_val == "30_bins":
                    # 30 equal bins over horizon
                    bin_dur = h_dur / 30.0
                    d_val = compute_binned_density_from_episodes(w_runs, h_dur, bin_width_s=bin_dur)
                else:
                    d_val = compute_binned_density_from_episodes(w_runs, h_dur, bin_width_s=float(res_val))

                factorial_records.append({
                    "dataset": ds,
                    "record_id": r_id,
                    "patient_id": p_id,
                    "horizon": h_name,
                    "horizon_duration_h": h_dur / 3600.0,
                    "resolution": res_name,
                    "af_burden": b,
                    "af_density": d_val,
                    "density_error_vs_native": abs(d_val - d_native) if (d_val is not None and d_native is not None) else np.nan
                })

        if len(horiz_metrics) >= 2:
            stability_records.append({
                "dataset": ds,
                "record_id": r_id,
                "patient_id": p_id,
                "horiz_metrics": horiz_metrics
            })

    df_factorial = pd.DataFrame(factorial_records)
    out_fact = results_dir / "R03_factorial_horizon_resolution.csv"
    df_factorial.to_csv(out_fact, index=False)
    print(f"Saved {len(df_factorial)} factorial measurements to {out_fact}")

    # Summarize factorial results
    print("\nR03: Mean Absolute Density Discretization Error vs Native by Horizon and Resolution:")
    fact_pivot = df_factorial.groupby(["horizon", "resolution"])["density_error_vs_native"].agg(["mean", "median", "std"]).reset_index()
    print(fact_pivot.to_string(index=False))

    # R05: Comparative Metric Stability
    print("\n--- R05: Comparative Metric Stability Across Horizons ---")
    metric_names = ["burden", "density", "l_max_min", "episode_rate", "af_aggregation"]
    stability_results = []

    # Prepare wide tables per metric across 6h, 12h, 24h
    for m in metric_names:
        rows = []
        for s in stability_records:
            h_m = s["horiz_metrics"]
            if "6h" in h_m and "12h" in h_m and "24h" in h_m:
                rows.append({
                    "record_id": s["record_id"],
                    "val_6h": h_m["6h"][m],
                    "val_12h": h_m["12h"][m],
                    "val_24h": h_m["24h"][m]
                })

        df_m = pd.DataFrame(rows).dropna()
        if len(df_m) < 5:
            continue

        # 1. Rank stability (6h vs 24h, and 12h vs 24h)
        rho_6_24, _ = compute_rank_stability(df_m["val_6h"], df_m["val_24h"])
        rho_12_24, _ = compute_rank_stability(df_m["val_12h"], df_m["val_24h"])

        # 2. Intraclass correlation ICC(3,1)
        icc = compute_intraclass_correlation(df_m[["val_6h", "val_12h", "val_24h"]])

        # 3. Within-subject CV across horizons
        vals = df_m[["val_6h", "val_12h", "val_24h"]].values
        means = np.mean(vals, axis=1)
        stds = np.std(vals, axis=1)
        within_cvs = np.where(means > 0, stds / means, 0.0)
        median_cv = float(np.median(within_cvs))
        mean_cv = float(np.mean(within_cvs))

        stability_results.append({
            "metric": m,
            "n_recordings_complete": len(df_m),
            "icc_3_1": icc,
            "rank_stability_rho_6h_to_24h": rho_6_24,
            "rank_stability_rho_12h_to_24h": rho_12_24,
            "mean_within_subject_cv": mean_cv,
            "median_within_subject_cv": median_cv
        })

    df_stab = pd.DataFrame(stability_results)
    out_stab = results_dir / "R05_metric_stability_comparison.csv"
    df_stab.to_csv(out_stab, index=False)
    print("\nR05 Metric Stability Comparison (6h vs 12h vs 24h):")
    print(df_stab.to_string(index=False))

if __name__ == "__main__":
    main()
