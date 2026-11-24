#!/usr/bin/env python3
"""Script 21: Experiment A - SHDB Structural Association Feasibility (LAD).
Analyzes the association between AF density, burden, longest episode duration,
and left atrial diameter (LAD) using authentic SHDB Holter and echo metadata.
Generates:
  - results/shdb_structural_manifest.csv
  - results/shdb_density_joint_support.csv
  - results/shdb_structural_associations.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density

def main():
    manifests_dir = root_dir / "manifests"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT A: SHDB STRUCTURAL ASSOCIATION (LAD & ECHOCARDIOGRAPHY)")
    print("=" * 80)

    # 1. Load SHDB rhythm intervals and canonical runs
    df_intervals = pd.read_parquet(manifests_dir / "canonical_rhythm_intervals.parquet")
    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")
    shdb_csv = "/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv"
    df_meta = pd.read_csv(shdb_csv)

    shdb_meta = df_meta[df_meta["Annotated"] == True].copy()
    shdb_meta["record_id"] = shdb_meta["Data_ID"].astype(str).str.zfill(3)
    shdb_meta["patient_id"] = "shdb_" + shdb_meta["Subject_ID"].astype(str)

    # Filter intervals to SHDB
    shdb_intervals = df_intervals[df_intervals["dataset"] == "SHDB-AF"].copy()
    shdb_runs = df_runs[df_runs["dataset"] == "SHDB-AF"].copy()

    # 2. Extract 24-hour common observation horizon (or full available recording)
    rec_features = []
    for idx, row in shdb_meta.iterrows():
        r_id = row["record_id"]
        p_id = row["patient_id"]

        r_itvs = shdb_intervals[shdb_intervals["record_id"] == r_id]
        if r_itvs.empty:
            continue

        t_start = float(r_itvs["start_time_s"].min())
        t_end = float(r_itvs["end_time_s"].max())
        monitoring_dur = t_end - t_start

        # 24-hour horizon anchored at t0
        h_dur = min(monitoring_dur, 86400.0)
        w_start = t_start
        w_end = t_start + h_dur

        # Runs in 24h horizon
        r_af = shdb_runs[shdb_runs["record_id"] == r_id]
        w_runs = []
        total_af_s = 0.0
        run_durs = []
        for _, r in r_af.iterrows():
            cs = max(w_start, float(r["start_time_s"]))
            ce = min(w_end, float(r["end_time_s"]))
            if ce > cs:
                dur = ce - cs
                w_runs.append((cs - w_start, ce - w_start))
                total_af_s += dur
                run_durs.append(dur)
        w_runs.sort()

        burden = total_af_s / h_dur if h_dur > 0 else 0.0
        n_episodes = len(w_runs)
        longest_run_s = max(run_durs) if run_durs else 0.0
        s_max = longest_run_s / total_af_s if total_af_s > 0 else 0.0

        density = compute_continuous_density(w_runs, h_dur) if (0.0 < burden < 1.0) else np.nan

        # Interepisode gaps
        gap_durs = []
        for i in range(len(w_runs) - 1):
            gap = w_runs[i + 1][0] - w_runs[i][1]
            if gap > 0:
                gap_durs.append(gap)
        median_gap_s = float(np.median(gap_durs)) if gap_durs else np.nan

        # Clinical metadata
        lad = pd.to_numeric(row["Echo_LAD"], errors="coerce")
        lvef = pd.to_numeric(row["Echo_LVEF"], errors="coerce")
        age = pd.to_numeric(row["Age_at_Holter"], errors="coerce")
        female = 1.0 if str(row["Sex"]).upper() == "F" else 0.0
        af_type = str(row["AF_Type"]).strip()
        is_paf = (af_type == "PAF")
        prev_ablation = bool(row["Previous_AF_Ablation"])
        bb_use = 1.0 if pd.notna(row["Antiarrhythmic_Drug_BB"]) and str(row["Antiarrhythmic_Drug_BB"]).strip() != "" else 0.0

        # Echo date delta
        date_holter = pd.to_datetime(row["Date_Holter"], errors="coerce")
        date_echo = pd.to_datetime(row["Echo_Date"], errors="coerce")
        delta_echo_days = (date_echo - date_holter).days if pd.notna(date_holter) and pd.notna(date_echo) else np.nan

        rec_features.append({
            "record_id": r_id,
            "patient_id": p_id,
            "subject_id": row["Subject_ID"],
            "monitoring_duration_h": monitoring_dur / 3600.0,
            "horizon_duration_h": h_dur / 3600.0,
            "af_burden": burden,
            "density_continuous": density,
            "is_density_defined": pd.notna(density),
            "n_af_episodes": n_episodes,
            "longest_af_run_s": longest_run_s,
            "s_max": s_max,
            "median_gap_s": median_gap_s,
            "echo_lad_mm": lad,
            "has_valid_lad": pd.notna(lad),
            "echo_lvef_pct": lvef,
            "age": age,
            "female": female,
            "af_type": af_type,
            "is_paroxysmal": is_paf,
            "previous_ablation": prev_ablation,
            "beta_blocker": bb_use,
            "days_between_holter_and_echo": delta_echo_days
        })

    df_manifest = pd.DataFrame(rec_features)
    out_manifest = results_dir / "shdb_structural_manifest.csv"
    df_manifest.to_csv(out_manifest, index=False)
    print(f"Saved structural manifest to {out_manifest} (N={len(df_manifest)} recordings)")

    # 3. Patient Counts and Joint Support
    n_pats_total = df_manifest["patient_id"].nunique()
    n_pats_lad = df_manifest[df_manifest["has_valid_lad"]]["patient_id"].nunique()
    n_pats_dens = df_manifest[df_manifest["is_density_defined"]]["patient_id"].nunique()
    n_pats_both = df_manifest[df_manifest["has_valid_lad"] & df_manifest["is_density_defined"]]["patient_id"].nunique()
    n_pats_multi = df_manifest[df_manifest["has_valid_lad"] & df_manifest["is_density_defined"] & (df_manifest["n_af_episodes"] >= 2)]["patient_id"].nunique()
    n_pats_paf = df_manifest[df_manifest["has_valid_lad"] & df_manifest["is_density_defined"] & df_manifest["is_paroxysmal"]]["patient_id"].nunique()

    print(f"\n--- Independent Patient Sample Sizes ---")
    print(f"Total annotated recordings: {len(df_manifest)} across {n_pats_total} unique patients")
    print(f"Patients with valid Echo LAD: {n_pats_lad}")
    print(f"Patients with defined AF density (0 < b < 1): {n_pats_dens}")
    print(f"Patients with BOTH valid LAD and defined Density: {n_pats_both}")
    print(f"Multi-episode patients with LAD and Density: {n_pats_multi}")
    print(f"Clinically documented PAF patients with LAD and Density: {n_pats_paf}")

    # Check collinearity and joint support
    df_analysis = df_manifest[df_manifest["has_valid_lad"] & df_manifest["is_density_defined"]].copy()
    # If repeated recordings, take average per patient
    df_patient = df_analysis.groupby("patient_id").agg({
        "echo_lad_mm": "first",
        "af_burden": "mean",
        "density_continuous": "mean",
        "longest_af_run_s": "mean",
        "s_max": "mean",
        "n_af_episodes": "mean",
        "age": "first",
        "female": "first",
        "beta_blocker": "first",
        "is_paroxysmal": "first"
    }).reset_index()

    df_patient["log_longest_run_s"] = np.log1p(df_patient["longest_af_run_s"])

    corr_matrix = df_patient[["echo_lad_mm", "af_burden", "density_continuous", "s_max", "log_longest_run_s", "age"]].corr()
    out_support = results_dir / "shdb_density_joint_support.csv"
    corr_matrix.to_csv(out_support)
    print("\n--- Correlation and Collinearity Matrix ---")
    print(corr_matrix.round(3).to_string())

    # 4. Parsimonious Regression Models for Left Atrial Diameter (LAD)
    print("\n--- Parsimonious Regression Models for LAD (mm) ---")
    reg_records = []

    # Model specifications
    models = {
        "M0_Demographics": ["age", "female"],
        "M1_Demo_Burden": ["age", "female", "af_burden"],
        "M2_Demo_Burden_Density": ["age", "female", "af_burden", "density_continuous"],
        "M3_Demo_Burden_LongestRun": ["age", "female", "af_burden", "log_longest_run_s"],
        "M4_Demo_Burden_Smax": ["age", "female", "af_burden", "s_max"],
        "M5_Full_Temporal": ["age", "female", "af_burden", "density_continuous", "log_longest_run_s"]
    }

    y = df_patient["echo_lad_mm"]
    m1_r2 = None

    for m_name, covs in models.items():
        X = sm.add_constant(df_patient[covs])
        fit = sm.OLS(y, X).fit()
        r2 = fit.rsquared
        r2_adj = fit.rsquared_adj
        f_stat = fit.fvalue
        p_val = fit.f_pvalue

        if m_name == "M1_Demo_Burden":
            m1_r2 = r2

        partial_r2 = (r2 - m1_r2) / (1.0 - m1_r2) if (m1_r2 is not None and m_name not in ["M0_Demographics", "M1_Demo_Burden"]) else np.nan

        row = {
            "model": m_name,
            "n_patients": len(df_patient),
            "covariates": "+".join(covs),
            "r_squared": r2,
            "adj_r_squared": r2_adj,
            "partial_r2_over_burden": partial_r2,
            "aic": fit.aic,
            "bic": fit.bic
        }
        # Coefficients
        if "density_continuous" in covs:
            row["density_coef_mm"] = fit.params.get("density_continuous", np.nan)
            row["density_ci_low"] = fit.conf_int().loc["density_continuous", 0]
            row["density_ci_high"] = fit.conf_int().loc["density_continuous", 1]
            row["density_pval"] = fit.pvalues.get("density_continuous", np.nan)
        if "af_burden" in covs:
            row["burden_coef_mm"] = fit.params.get("af_burden", np.nan)
            row["burden_pval"] = fit.pvalues.get("af_burden", np.nan)

        reg_records.append(row)
        print(f"Model {m_name:30s} | R² = {r2:.4f} (Adj R² = {r2_adj:.4f}) | Partial R² = {partial_r2 if pd.notna(partial_r2) else 0.0:.4f}")

    df_reg = pd.DataFrame(reg_records)
    out_reg = results_dir / "shdb_structural_associations.csv"
    df_reg.to_csv(out_reg, index=False)
    print(f"Saved structural regression results to {out_reg}")

if __name__ == "__main__":
    main()
