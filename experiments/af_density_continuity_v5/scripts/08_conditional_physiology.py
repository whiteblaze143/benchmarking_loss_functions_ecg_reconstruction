#!/usr/bin/env python3
"""Script 08: Experiment E02 - Physiological Heterogeneity Conditional on AF Burden and Density.
Executes:
  - Descriptive matched pair analysis (Burden <= 0.025, Density <= 0.025, Delta S_max >= 0.30)
  - Nested fractional logistic models (M0 through M3) with patient-clustered bootstrap
  - Cohort-specific conditional models
  - Head-to-head comparison with simpler descriptors (B0 through B6) via 5-fold patient-grouped CV
Generates:
  - results/E02_matched_pairs.parquet
  - results/E02_matched_effects.csv
  - results/E02_nested_models.csv
  - results/E02_cohort_specific_results.csv
  - results/E02_out_of_fold_predictions.parquet
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.matching import find_matched_pairs
from src.statistics import (
    fit_fractional_logit,
    estimate_standardized_smax_contrast,
    run_patient_bootstrap_smax_effect,
    run_patient_grouped_cv
)

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE E: EXPERIMENT E02 - PHYSIOLOGICAL HETEROGENEITY (Q120)")
    print("=" * 80)

    patient_win_parquet = features_dir / "patient_window_features.parquet"
    df = pd.read_parquet(patient_win_parquet)

    # Filter to primary 12h horizon density-defined windows with valid Q120
    df_12h = df[
        (df["horizon"] == "12h") &
        (df["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df["is_density_defined"]) &
        (df["q_120"].notna())
    ].copy()

    # One-hot encode cohort
    cohort_dummies = pd.get_dummies(df_12h["dataset"], prefix="cohort", drop_first=True, dtype=float)
    df_12h = pd.concat([df_12h, cohort_dummies], axis=1)
    cohort_cols = cohort_dummies.columns.tolist()

    # Compute additional baseline descriptors
    df_12h["log_longest_run_s"] = np.log1p(df_12h["longest_af_run_s"].values.astype(float))
    df_12h["log_n_episodes"] = np.log1p(df_12h["n_af_episodes"].values.astype(float))
    # Approximation of AF aggregation (normalized dispersion)
    df_12h["af_aggregation"] = 1.0 - (df_12h["n_af_episodes"] / (df_12h["total_af_s"] / 60.0 + 1.0)).clip(0.0, 1.0)

    print(f"Total eligible windows for E02: {len(df_12h)} across {df_12h['patient_id'].nunique()} patients.")

    # 1. Matched Pair Analysis (E04)
    print("\n--- 1. Matched Pair Analysis ---")
    df_pairs, match_diag = find_matched_pairs(
        df_12h,
        burden_caliper=0.025,
        density_caliper=0.025,
        min_smax_contrast=0.30,
        seed=42
    )

    out_pairs = results_dir / "E02_matched_pairs.parquet"
    df_pairs.to_parquet(out_pairs, index=False)
    print(f"Saved {len(df_pairs)} matched pairs to {out_pairs}")

    matched_effects = []
    if not df_pairs.empty:
        delta_q120 = df_pairs["delta_q_120"].values
        matched_effects.append({
            "contrast": "S_max >= 0.50 vs S_max <= 0.20 (|Delta S_max| >= 0.30)",
            "n_pairs": len(df_pairs),
            "n_unique_patients": match_diag["n_unique_patients_matched"],
            "mean_delta_q120_pct_points": float(np.mean(delta_q120) * 100.0),
            "median_delta_q120_pct_points": float(np.median(delta_q120) * 100.0),
            "std_delta_q120_pct_points": float(np.std(delta_q120) * 100.0),
            "pct_pairs_exceeding_10pp": float(df_pairs["diff_exceeds_10pct"].mean() * 100.0),
            "pct_pairs_exceeding_20pp": float(df_pairs["diff_exceeds_20pct"].mean() * 100.0),
            "mean_delta_burden": match_diag["mean_delta_burden"],
            "mean_delta_density": match_diag["mean_delta_density"]
        })
        print(f"Matched Effect: Mean Delta Q120 = {np.mean(delta_q120)*100.0:.2f} percentage points (Median = {np.median(delta_q120)*100.0:.2f} pp)")
    else:
        matched_effects.append({
            "contrast": "No pairs found within calipers",
            "n_pairs": 0, "n_unique_patients": 0,
            "mean_delta_q120_pct_points": np.nan, "median_delta_q120_pct_points": np.nan,
            "std_delta_q120_pct_points": np.nan, "pct_pairs_exceeding_10pp": np.nan,
            "pct_pairs_exceeding_20pp": np.nan, "mean_delta_burden": np.nan, "mean_delta_density": np.nan
        })

    df_match_eff = pd.DataFrame(matched_effects)
    out_match_eff = results_dir / "E02_matched_effects.csv"
    df_match_eff.to_csv(out_match_eff, index=False)
    print(f"Saved matched effects to {out_match_eff}")

    # 2. Nested Fractional Logistic Regression Models (E05)
    print("\n--- 2. Nested Fractional Logistic Regression Models ---")
    model_definitions = {
        "M0": cohort_cols,
        "M1": cohort_cols + ["af_burden"],
        "M2": cohort_cols + ["af_burden", "density_continuous"],
        "M3": cohort_cols + ["af_burden", "density_continuous", "s_max"]
    }

    nested_rows = []
    m3_res = None
    for m_name, f_cols in model_definitions.items():
        try:
            res, coefs = fit_fractional_logit(df_12h, f_cols, target_col="q_120", cluster_col="patient_id")
            coefs["model"] = m_name
            coefs["features"] = "+".join(f_cols)
            nested_rows.append(coefs)
            if m_name == "M3":
                m3_res = res
        except Exception as e:
            print(f"Model {m_name} fit error: {e}")

    df_nested = pd.DataFrame(nested_rows)
    out_nested = results_dir / "E02_nested_models.csv"
    df_nested.to_csv(out_nested, index=False)
    print(f"Saved nested models to {out_nested}")

    # Patient-clustered bootstrap for M3 standardized S_max contrast (+0.25 S_max)
    if m3_res is not None:
        print("\nRunning patient-clustered bootstrap (B=2000 replicates) for S_max contrast...")
        m3_features = model_definitions["M3"]
        boot_est, boot_ci_l, boot_ci_u = run_patient_bootstrap_smax_effect(
            df_12h, m3_features, target_col="q_120", cluster_col="patient_id", n_bootstrap=2000, seed=42
        )
        print(f"Standardized S_max (+0.25) Effect on Q120: {boot_est:.2f} percentage points [95% CI: {boot_ci_l:.2f}, {boot_ci_u:.2f}]")

    # 3. Cohort-Specific Models
    print("\n--- 3. Cohort-Specific Regressions ---")
    cohort_specific_records = []
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        sub = df_12h[df_12h["dataset"] == ds].copy()
        if len(sub) < 10 or sub["patient_id"].nunique() < 5:
            continue
        try:
            res_c, coefs_c = fit_fractional_logit(sub, ["af_burden", "density_continuous", "s_max"], target_col="q_120", cluster_col="patient_id")
            contrast_c = estimate_standardized_smax_contrast(res_c, sub, ["af_burden", "density_continuous", "s_max"])
            cohort_specific_records.append({
                "cohort": ds,
                "n_windows": len(sub),
                "n_patients": sub["patient_id"].nunique(),
                "s_max_coef": coefs_c.get("s_max_coef", np.nan),
                "s_max_pval": coefs_c.get("s_max_pval", np.nan),
                "density_coef": coefs_c.get("density_continuous_coef", np.nan),
                "density_pval": coefs_c.get("density_continuous_pval", np.nan),
                "s_max_effect_pct_points": contrast_c["mean_adjusted_diff_pct_points"],
                "aic": coefs_c.get("aic", np.nan)
            })
            print(f"Cohort {ds:10s}: S_max coef = {coefs_c.get('s_max_coef', np.nan):.4f} (p={coefs_c.get('s_max_pval', np.nan):.4f}) | Effect = {contrast_c['mean_adjusted_diff_pct_points']:.2f} pp")
        except Exception as e:
            print(f"Cohort {ds} regression failed: {e}")

    df_cohort = pd.DataFrame(cohort_specific_records)
    out_cohort = results_dir / "E02_cohort_specific_results.csv"
    df_cohort.to_csv(out_cohort, index=False)
    print(f"Saved cohort-specific results to {out_cohort}")

    # 4. Head-to-Head Comparison with Simpler Descriptors (E06)
    print("\n--- 4. Head-to-Head Comparison with Simpler Descriptors (5-Fold Grouped CV) ---")
    baseline_models = {
        "B0_Burden_Only": cohort_cols + ["af_burden"],
        "B1_Burden_Density": cohort_cols + ["af_burden", "density_continuous"],
        "B2_Burden_Longest_Run": cohort_cols + ["af_burden", "log_longest_run_s"],
        "B3_Burden_Episode_Count": cohort_cols + ["af_burden", "log_n_episodes"],
        "B4_Burden_S_max": cohort_cols + ["af_burden", "s_max"],
        "B5_Burden_Density_S_max": cohort_cols + ["af_burden", "density_continuous", "s_max"],
        "B6_Burden_Aggregation": cohort_cols + ["af_burden", "af_aggregation"]
    }

    df_cv, df_oof = run_patient_grouped_cv(
        df_12h,
        baseline_models,
        target_col="q_120",
        group_col="patient_id",
        n_splits=5,
        seed=42
    )

    out_oof = results_dir / "E02_out_of_fold_predictions.parquet"
    df_oof.to_parquet(out_oof, index=False)
    print(f"Saved out-of-fold predictions to {out_oof}")
    print("\nHead-to-head out-of-fold performance:")
    print(df_cv.to_string(index=False))

if __name__ == "__main__":
    main()
