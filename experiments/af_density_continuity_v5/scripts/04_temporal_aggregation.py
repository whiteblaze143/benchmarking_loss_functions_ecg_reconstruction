#!/usr/bin/env python3
"""Script 04: Experiment E01 - Controlled Temporal Aggregation Analysis.
Evaluates the impact of reducing continuous rhythm timelines to exact bin-level durations.
Bin resolutions: 30s, 1m, 5m, 15m, 30m, 60m, 2h, 6h.
Generates:
  - results/E01_resolution_sensitivity.csv
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density
from src.density_binned import charitos_binned_density
from src.density_bounds import compute_coarsening_bound

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE C02: EXPERIMENT E01 - CONTROLLED TEMPORAL AGGREGATION")
    print("=" * 80)

    feats_parquet = features_dir / "density_reference_features.parquet"
    runs_parquet = root_dir / "manifests" / "canonical_af_runs.parquet"

    df_feats = pd.read_parquet(feats_parquet)
    df_runs = pd.read_parquet(runs_parquet)

    # Filter to primary 12h horizon density-defined windows
    df_12h = df_feats[
        (df_feats["horizon"] == "12h") &
        (df_feats["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df_feats["is_density_defined"])
    ].copy()

    print(f"Analyzing {len(df_12h)} density-defined 12h windows...")

    bin_resolutions = [
        ("30s", 30.0),
        ("1m", 60.0),
        ("5m", 300.0),
        ("15m", 900.0),
        ("30m", 1800.0),
        ("60m", 3600.0),
        ("2h", 7200.0),
        ("6h", 21600.0)
    ]

    sensitivity_rows = []

    for b_name, delta in bin_resolutions:
        diffs = []
        signed_diffs = []
        d_cont_list = []
        d_binned_list = []
        bound_widths = []

        for _, w in df_12h.iterrows():
            dataset = w["dataset"]
            rec_id = w["record_id"]
            w_start = float(w["window_start_s"])
            w_end = float(w["window_end_s"])
            w_dur = float(w["window_duration_s"])
            burden = float(w["af_burden"])
            d_cont = float(w["density_continuous"])

            # Fetch runs
            rec_runs = df_runs[(df_runs["dataset"] == dataset) & (df_runs["record_id"] == rec_id)]
            w_runs = []
            for _, r in rec_runs.iterrows():
                rs = max(w_start, float(r["start_time_s"]))
                re = min(w_end, float(r["end_time_s"]))
                if re > rs:
                    w_runs.append((rs - w_start, re - w_start))

            # Discretize into bins
            n_bins = int(math.ceil(w_dur / delta))
            bin_masses = np.zeros(n_bins, dtype=np.float64)
            for s, e in w_runs:
                sb = int(s // delta)
                eb = int(min(n_bins - 1, e // delta))
                for b_i in range(sb, eb + 1):
                    bs = b_i * delta
                    be = (b_i + 1) * delta
                    overlap = max(0.0, min(e, be) - max(s, bs))
                    bin_masses[b_i] += overlap

            # Exact mass conservation assertion
            assert math.isclose(sum(bin_masses), sum(e - s for s, e in w_runs), abs_tol=1e-8), "Mass conservation failed!"

            try:
                d_b = charitos_binned_density(bin_masses, delta)
            except Exception:
                d_b = np.nan

            if np.isfinite(d_b):
                d_cont_list.append(d_cont)
                d_binned_list.append(d_b)
                diff = abs(d_cont - d_b)
                signed = d_b - d_cont
                diffs.append(diff)
                signed_diffs.append(signed)

            c_bound = compute_coarsening_bound(delta, w_dur, burden)
            bound_widths.append(c_bound)

        d_cont_arr = np.array(d_cont_list)
        d_binned_arr = np.array(d_binned_list)
        diffs_arr = np.array(diffs)
        signed_arr = np.array(signed_diffs)

        spearman_rho, _ = spearmanr(d_cont_arr, d_binned_arr) if len(d_cont_arr) > 1 else (1.0, 0.0)

        sensitivity_rows.append({
            "bin_resolution": b_name,
            "delta_seconds": delta,
            "n_windows": len(diffs_arr),
            "median_abs_error": float(np.median(diffs_arr)),
            "iqr_abs_error": float(np.percentile(diffs_arr, 75) - np.percentile(diffs_arr, 25)),
            "mean_abs_error": float(np.mean(diffs_arr)),
            "max_abs_error": float(np.max(diffs_arr)),
            "mean_signed_error": float(np.mean(signed_arr)),
            "rank_correlation_spearman": float(spearman_rho),
            "median_theoretical_coarsening_bound": float(np.median(bound_widths)),
            "mean_theoretical_coarsening_bound": float(np.mean(bound_widths))
        })
        print(f"Res {b_name:5s} (Delta={delta:5.0f}s): Median |D_cont - D_binned| = {np.median(diffs_arr):.4f} | Spearman rho = {spearman_rho:.4f}")

    df_sens = pd.DataFrame(sensitivity_rows)
    out_csv = results_dir / "E01_resolution_sensitivity.csv"
    df_sens.to_csv(out_csv, index=False)
    print(f"\nSaved resolution sensitivity table to {out_csv}")

if __name__ == "__main__":
    main()
