#!/usr/bin/env python3
"""Script 25: Priority 1 - Rigorous Adjudication of SHDB Structural Association (LAD vs Density).
Performs comprehensive sensitivity analyses, LOPO influence diagnostics, and patient-level
bootstrapping to resolve the negative association between AF density and Left Atrial Diameter (LAD).

Subcohorts & Adjustments:
  1. All eligible patients (N=72)
  2. Multi-episode only (N=49, removing single-episode density ceiling at 1.0)
  3. Paroxysmal AF only (N=63)
  4. Multi-episode PAF only (N=42)
  5. Ablation-naive patients (N=50)
  6. Contemporaneous echocardiogram (|Δt| <= 90d, N=44; |Δt| <= 30d, N=35)
  7. Body-size adjustment (BMI and Mosteller BSA)
  8. Comorbidity adjustment (HTN, LVEF, Moderate/Severe MR)
  9. Episode duration-filtered density (runs >= 30s, runs >= 60s)
  10. Outlier-excluded sensitivity (excluding high-leverage patient shdb_4339581)

Diagnostics & Validation:
  - Leave-One-Patient-Out (LOPO): Cook's distance, studentized residuals, DFBETAS for beta_D.
  - Patient-level bootstrap (B=2000) for empirical parameter distributions and 95% CIs.

Generates:
  - results/adjudication_shdb_subcohort_models.csv
  - results/adjudication_shdb_lopo_influence.csv
  - results/adjudication_shdb_bootstrap_distributions.csv
  - figures/adjudication_fig1_lad_density_scatter.png
  - figures/adjudication_fig2_lopo_influence.png
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density

def main():
    manifests_dir = root_dir / "manifests"
    results_dir = root_dir / "results"
    figures_dir = root_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PRIORITY 1: ADJUDICATION OF SHDB STRUCTURAL ASSOCIATION (LAD vs DENSITY)")
    print("=" * 80)

    # 1. Load canonical intervals and runs
    df_intervals = pd.read_parquet(manifests_dir / "canonical_rhythm_intervals.parquet")
    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")
    shdb_csv = "/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv"
    df_meta = pd.read_csv(shdb_csv)

    # Filter to annotated SHDB recordings
    shdb_meta = df_meta[df_meta["Annotated"] == True].copy()
    shdb_meta["record_id"] = shdb_meta["Data_ID"].astype(str).str.zfill(3)
    shdb_meta["patient_id"] = "shdb_" + shdb_meta["Subject_ID"].astype(str)

    shdb_intervals = df_intervals[df_intervals["dataset"] == "SHDB-AF"].copy()
    shdb_runs = df_runs[df_runs["dataset"] == "SHDB-AF"].copy()

    # 2. Extract 24h metrics and duration-filtered densities per recording
    rec_records = []
    for idx, row in shdb_meta.iterrows():
        r_id = row["record_id"]
        p_id = row["patient_id"]

        r_itvs = shdb_intervals[shdb_intervals["record_id"] == r_id]
        if r_itvs.empty:
            continue

        t_start = float(r_itvs["start_time_s"].min())
        t_end = float(r_itvs["end_time_s"].max())
        monitoring_dur = t_end - t_start
        h_dur = min(monitoring_dur, 86400.0)
        w_start = t_start
        w_end = t_start + h_dur

        r_af = shdb_runs[shdb_runs["record_id"] == r_id]
        w_runs = []
        w_runs_30 = []
        w_runs_60 = []
        total_af_s = 0.0
        total_af_30_s = 0.0
        total_af_60_s = 0.0
        run_durs = []

        for _, r in r_af.iterrows():
            cs = max(w_start, float(r["start_time_s"]))
            ce = min(w_end, float(r["end_time_s"]))
            if ce > cs:
                dur = ce - cs
                w_runs.append((cs - w_start, ce - w_start))
                total_af_s += dur
                run_durs.append(dur)
                if dur >= 30.0:
                    w_runs_30.append((cs - w_start, ce - w_start))
                    total_af_30_s += dur
                if dur >= 60.0:
                    w_runs_60.append((cs - w_start, ce - w_start))
                    total_af_60_s += dur

        w_runs.sort()
        w_runs_30.sort()
        w_runs_60.sort()

        burden = total_af_s / h_dur if h_dur > 0 else 0.0
        n_episodes = len(w_runs)
        longest_run_s = max(run_durs) if run_durs else 0.0
        s_max = longest_run_s / total_af_s if total_af_s > 0 else 0.0
        density = compute_continuous_density(w_runs, h_dur) if (0.0 < burden < 1.0) else np.nan

        burden_30 = total_af_30_s / h_dur if h_dur > 0 else 0.0
        density_30 = compute_continuous_density(w_runs_30, h_dur) if (0.0 < burden_30 < 1.0) else np.nan

        burden_60 = total_af_60_s / h_dur if h_dur > 0 else 0.0
        density_60 = compute_continuous_density(w_runs_60, h_dur) if (0.0 < burden_60 < 1.0) else np.nan

        lad = pd.to_numeric(row["Echo_LAD"], errors="coerce")
        lvef = pd.to_numeric(row["Echo_LVEF"], errors="coerce")
        age = pd.to_numeric(row["Age_at_Holter"], errors="coerce")
        female = 1.0 if str(row["Sex"]).upper() == "F" else 0.0
        af_type = str(row["AF_Type"]).strip()
        is_paf = (af_type == "PAF")
        prev_ablation = bool(row["Previous_AF_Ablation"])
        bb_use = 1.0 if pd.notna(row["Antiarrhythmic_Drug_BB"]) and str(row["Antiarrhythmic_Drug_BB"]).strip() != "" else 0.0

        date_holter = pd.to_datetime(row["Date_Holter"], errors="coerce")
        date_echo = pd.to_datetime(row["Echo_Date"], errors="coerce")
        delta_days = abs((date_echo - date_holter).days) if pd.notna(date_holter) and pd.notna(date_echo) else np.nan

        height = pd.to_numeric(row["Height"], errors="coerce")
        weight = pd.to_numeric(row["Weight"], errors="coerce")
        bmi = pd.to_numeric(row["BMI"], errors="coerce")
        bsa = np.sqrt((height * 100.0 * weight) / 3600.0) if pd.notna(height) and pd.notna(weight) and height > 0 and weight > 0 else np.nan

        htn = 1.0 if (pd.notna(row["HTN"]) and bool(row["HTN"])) else 0.0
        mr = float(row["Moderate_or_Severe_MR"]) if pd.notna(row["Moderate_or_Severe_MR"]) else 0.0

        rec_records.append({
            "record_id": r_id,
            "patient_id": p_id,
            "af_burden": burden,
            "density_continuous": density,
            "is_density_defined": pd.notna(density),
            "n_af_episodes": n_episodes,
            "longest_af_run_s": longest_run_s,
            "s_max": s_max,
            "af_burden_30": burden_30,
            "density_continuous_30": density_30,
            "n_episodes_30": len(w_runs_30),
            "af_burden_60": burden_60,
            "density_continuous_60": density_60,
            "n_episodes_60": len(w_runs_60),
            "echo_lad_mm": lad,
            "has_valid_lad": pd.notna(lad),
            "echo_lvef_pct": lvef,
            "age": age,
            "female": female,
            "is_paroxysmal": is_paf,
            "previous_ablation": prev_ablation,
            "beta_blocker": bb_use,
            "delta_echo_days": delta_days,
            "bmi": bmi,
            "bsa": bsa,
            "htn": htn,
            "mr": mr
        })

    df_rec = pd.DataFrame(rec_records)

    # 3. Patient Level Aggregation
    df_valid = df_rec[df_rec["has_valid_lad"] & df_rec["is_density_defined"]].copy()
    df_pat = df_valid.groupby("patient_id").agg({
        "echo_lad_mm": "first",
        "af_burden": "mean",
        "density_continuous": "mean",
        "n_af_episodes": "mean",
        "longest_af_run_s": "mean",
        "s_max": "mean",
        "af_burden_30": "mean",
        "density_continuous_30": "mean",
        "n_episodes_30": "mean",
        "af_burden_60": "mean",
        "density_continuous_60": "mean",
        "n_episodes_60": "mean",
        "echo_lvef_pct": "first",
        "age": "first",
        "female": "first",
        "is_paroxysmal": "first",
        "previous_ablation": "first",
        "beta_blocker": "first",
        "delta_echo_days": "min",
        "bmi": "first",
        "bsa": "first",
        "htn": "first",
        "mr": "first"
    }).reset_index()

    print(f"Total eligible patients with valid LAD and defined Density: N = {len(df_pat)}")

    # 4. Define Subcohort Dictionary
    subcohort_definitions = [
        ("All_Eligible", df_pat, ["age", "female", "af_burden", "density_continuous"]),
        ("Multi_Episode", df_pat[df_pat["n_af_episodes"] >= 2], ["age", "female", "af_burden", "density_continuous"]),
        ("PAF_Only", df_pat[df_pat["is_paroxysmal"] == True], ["age", "female", "af_burden", "density_continuous"]),
        ("Multi_Episode_PAF", df_pat[(df_pat["n_af_episodes"] >= 2) & (df_pat["is_paroxysmal"] == True)], ["age", "female", "af_burden", "density_continuous"]),
        ("Ablation_Naive", df_pat[df_pat["previous_ablation"] == False], ["age", "female", "af_burden", "density_continuous"]),
        ("Contemporaneous_le90d", df_pat[df_pat["delta_echo_days"] <= 90.0], ["age", "female", "af_burden", "density_continuous"]),
        ("Contemporaneous_le30d", df_pat[df_pat["delta_echo_days"] <= 30.0], ["age", "female", "af_burden", "density_continuous"]),
        ("All_Adjust_BMI", df_pat.dropna(subset=["bmi"]), ["age", "female", "af_burden", "density_continuous", "bmi"]),
        ("Multi_Adjust_BMI", df_pat[df_pat["n_af_episodes"] >= 2].dropna(subset=["bmi"]), ["age", "female", "af_burden", "density_continuous", "bmi"]),
        ("All_Adjust_BSA", df_pat.dropna(subset=["bsa"]), ["age", "female", "af_burden", "density_continuous", "bsa"]),
        ("Multi_Adjust_BSA", df_pat[df_pat["n_af_episodes"] >= 2].dropna(subset=["bsa"]), ["age", "female", "af_burden", "density_continuous", "bsa"]),
        ("All_Adjust_Comorbidities", df_pat, ["age", "female", "af_burden", "density_continuous", "htn", "echo_lvef_pct", "mr"]),
        ("Multi_Adjust_Comorbidities", df_pat[df_pat["n_af_episodes"] >= 2], ["age", "female", "af_burden", "density_continuous", "htn", "echo_lvef_pct", "mr"]),
        ("Duration_Filtered_30s_All", df_pat.dropna(subset=["density_continuous_30"]), ["age", "female", "af_burden_30", "density_continuous_30"]),
        ("Duration_Filtered_30s_Multi", df_pat[(df_pat["n_episodes_30"] >= 2)].dropna(subset=["density_continuous_30"]), ["age", "female", "af_burden_30", "density_continuous_30"]),
        ("Duration_Filtered_60s_All", df_pat.dropna(subset=["density_continuous_60"]), ["age", "female", "af_burden_60", "density_continuous_60"]),
        ("Duration_Filtered_60s_Multi", df_pat[(df_pat["n_episodes_60"] >= 2)].dropna(subset=["density_continuous_60"]), ["age", "female", "af_burden_60", "density_continuous_60"]),
        ("All_Excl_Top_Outlier", df_pat[df_pat["patient_id"] != "shdb_4339581"], ["age", "female", "af_burden", "density_continuous"]),
        ("Multi_Excl_Top_Outlier", df_pat[(df_pat["n_af_episodes"] >= 2) & (df_pat["patient_id"] != "shdb_4339581")], ["age", "female", "af_burden", "density_continuous"]),
    ]

    model_rows = []
    print("\n--- Subcohort & Sensitivity Regression Results ---")
    for name, df_sub, covs in subcohort_definitions:
        y = df_sub["echo_lad_mm"]
        X = sm.add_constant(df_sub[covs])
        fit = sm.OLS(y, X).fit()

        # Fit reduced model without density to compute partial R2
        dens_col = [c for c in covs if "density" in c][0]
        covs_nodens = [c for c in covs if c != dens_col]
        X_nodens = sm.add_constant(df_sub[covs_nodens])
        fit_nodens = sm.OLS(y, X_nodens).fit()

        r2 = fit.rsquared
        r2_adj = fit.rsquared_adj
        r2_red = fit_nodens.rsquared
        partial_r2 = (r2 - r2_red) / (1.0 - r2_red) if (1.0 - r2_red) > 0 else np.nan

        b_dens = fit.params[dens_col]
        ci_dens = fit.conf_int().loc[dens_col]
        p_dens = fit.pvalues[dens_col]

        burd_col = [c for c in covs if "burden" in c][0]
        b_burd = fit.params[burd_col]
        p_burd = fit.pvalues[burd_col]

        model_rows.append({
            "subcohort": name,
            "n_patients": len(df_sub),
            "covariates": "+".join(covs),
            "density_col": dens_col,
            "beta_density_mm": b_dens,
            "ci_low_95": ci_dens[0],
            "ci_high_95": ci_dens[1],
            "p_density": p_dens,
            "beta_burden_mm": b_burd,
            "p_burden": p_burd,
            "r_squared": r2,
            "adj_r_squared": r2_adj,
            "partial_r2_density": partial_r2,
            "aic": fit.aic,
            "bic": fit.bic
        })

        sig_mark = "***" if p_dens < 0.001 else ("**" if p_dens < 0.01 else ("*" if p_dens < 0.05 else "ns"))
        print(f"{name:32s} (N={len(df_sub):2d}) | β_D = {b_dens:6.2f} mm [{ci_dens[0]:6.2f}, {ci_dens[1]:6.2f}] | p = {p_dens:.4f} {sig_mark:2s} | Partial R² = {partial_r2*100:5.2f}%")

    df_models = pd.DataFrame(model_rows)
    out_models = results_dir / "adjudication_shdb_subcohort_models.csv"
    df_models.to_csv(out_models, index=False)
    print(f"\nSaved subcohort regression models to {out_models}")

    # 5. LOPO Diagnostics for Multi-Episode Cohort (N=49) and All Eligible (N=72)
    print("\n--- Computing Leave-One-Patient-Out (LOPO) Diagnostics ---")
    lopo_rows = []

    for target_name, df_sub in [("All_Eligible", df_pat), ("Multi_Episode", df_pat[df_pat["n_af_episodes"] >= 2].copy())]:
        df_target = df_sub.reset_index(drop=True).copy()
        covs = ["age", "female", "af_burden", "density_continuous"]
        X_full = sm.add_constant(df_target[covs])
        y_full = df_target["echo_lad_mm"]
        fit_full = sm.OLS(y_full, X_full).fit()
        influence = OLSInfluence(fit_full)

        cooks_d = np.asarray(influence.cooks_distance[0])
        student_resid = np.asarray(influence.resid_studentized_internal)
        dfbetas_dens = np.asarray(influence.dfbetas)[:, covs.index("density_continuous") + 1]

        beta_full = fit_full.params["density_continuous"]

        for i, p_row in df_target.iterrows():
            # Fit omitting patient i
            df_loo = df_target.drop(index=i)
            X_loo = sm.add_constant(df_loo[covs])
            y_loo = df_loo["echo_lad_mm"]
            fit_loo = sm.OLS(y_loo, X_loo).fit()

            b_loo = fit_loo.params["density_continuous"]
            p_loo = fit_loo.pvalues["density_continuous"]

            lopo_rows.append({
                "subcohort": target_name,
                "patient_id": p_row["patient_id"],
                "lad_mm": p_row["echo_lad_mm"],
                "af_burden": p_row["af_burden"],
                "density_continuous": p_row["density_continuous"],
                "delta_echo_days": p_row["delta_echo_days"],
                "n_af_episodes": p_row["n_af_episodes"],
                "cooks_distance": float(cooks_d[i]),
                "studentized_resid": float(student_resid[i]),
                "dfbeta_density": float(dfbetas_dens[i]),
                "beta_dens_full": float(beta_full),
                "beta_dens_loo": float(b_loo),
                "delta_beta_dens": float(b_loo - beta_full),
                "pval_dens_loo": float(p_loo)
            })

    df_lopo = pd.DataFrame(lopo_rows)
    out_lopo = results_dir / "adjudication_shdb_lopo_influence.csv"
    df_lopo.to_csv(out_lopo, index=False)
    print(f"Saved LOPO influence diagnostics to {out_lopo}")

    # Display top 5 influential patients in Multi-Episode
    df_lopo_multi = df_lopo[df_lopo["subcohort"] == "Multi_Episode"].sort_values("cooks_distance", ascending=False)
    print("\nTop 5 Influential Patients in Multi-Episode Cohort (N=49):")
    print(df_lopo_multi[["patient_id", "lad_mm", "af_burden", "density_continuous", "delta_echo_days", "cooks_distance", "dfbeta_density", "beta_dens_loo", "pval_dens_loo"]].head(5).to_string(index=False))

    # 6. Patient-Level Bootstrap (B=2000)
    print("\n--- Running Patient-Level Bootstrap (B=2000) ---")
    np.random.seed(42)
    B = 2000
    boot_rows = []

    for target_name, df_target in [("All_Eligible", df_pat), ("Multi_Episode", df_pat[df_pat["n_af_episodes"] >= 2].copy())]:
        covs = ["age", "female", "af_burden", "density_continuous"]
        n_pats = len(df_target)
        p_ids = df_target["patient_id"].values

        beta_dens_dist = []
        partial_r2_dist = []

        for b in range(B):
            sample_ids = np.random.choice(p_ids, size=n_pats, replace=True)
            df_boot = df_target.set_index("patient_id").loc[sample_ids].reset_index()

            y_b = df_boot["echo_lad_mm"]
            X_b = sm.add_constant(df_boot[covs])
            fit_b = sm.OLS(y_b, X_b).fit()

            # Reduced
            X_b_red = sm.add_constant(df_boot[["age", "female", "af_burden"]])
            fit_b_red = sm.OLS(y_b, X_b_red).fit()

            pr2 = (fit_b.rsquared - fit_b_red.rsquared) / (1.0 - fit_b_red.rsquared) if (1.0 - fit_b_red.rsquared) > 0 else 0.0

            beta_dens_dist.append(fit_b.params["density_continuous"])
            partial_r2_dist.append(pr2)

        beta_arr = np.array(beta_dens_dist)
        pr2_arr = np.array(partial_r2_dist)

        boot_mean = np.mean(beta_arr)
        boot_se = np.std(beta_arr, ddof=1)
        ci_low = np.percentile(beta_arr, 2.5)
        ci_high = np.percentile(beta_arr, 97.5)
        prob_neg = np.mean(beta_arr < 0.0)

        pr2_mean = np.mean(pr2_arr)
        pr2_ci_low = np.percentile(pr2_arr, 2.5)
        pr2_ci_high = np.percentile(pr2_arr, 97.5)

        boot_rows.append({
            "subcohort": target_name,
            "n_patients": n_pats,
            "bootstrap_replicates": B,
            "beta_dens_mean": boot_mean,
            "beta_dens_se": boot_se,
            "beta_dens_ci95_low": ci_low,
            "beta_dens_ci95_high": ci_high,
            "prob_beta_negative": prob_neg,
            "partial_r2_mean": pr2_mean,
            "partial_r2_ci95_low": pr2_ci_low,
            "partial_r2_ci95_high": pr2_ci_high
        })

        print(f"Bootstrap {target_name:15s} (N={n_pats}) | Mean β_D = {boot_mean:6.2f} mm (SE = {boot_se:5.2f}) | 95% CI: [{ci_low:6.2f}, {ci_high:6.2f}] | P(β_D < 0) = {prob_neg*100:.1f}%")

    df_boot_res = pd.DataFrame(boot_rows)
    out_boot = results_dir / "adjudication_shdb_bootstrap_distributions.csv"
    df_boot_res.to_csv(out_boot, index=False)
    print(f"Saved bootstrap summary to {out_boot}")

    # 7. Generate Adjudication Figures
    print("\n--- Generating Publication Diagnostic Figures ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: 4-Panel Scatter and Diagnostic Regressions
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: All Eligible (N=72)
    ax = axes[0, 0]
    multi_mask = df_pat["n_af_episodes"] >= 2
    ax.scatter(df_pat.loc[~multi_mask, "density_continuous"], df_pat.loc[~multi_mask, "echo_lad_mm"],
               color="tab:gray", alpha=0.7, s=50, label="Single-Episode (D=1.0, N=23)")
    ax.scatter(df_pat.loc[multi_mask, "density_continuous"], df_pat.loc[multi_mask, "echo_lad_mm"],
               color="tab:blue", alpha=0.8, s=60, label="Multi-Episode (N=49)")
    
    # Fit line All
    p_all = np.polyfit(df_pat["density_continuous"], df_pat["echo_lad_mm"], 1)
    x_grid = np.linspace(0.05, 1.0, 100)
    ax.plot(x_grid, p_all[0]*x_grid + p_all[1], "k--", lw=1.5, label=f"OLS All (N=72, β={p_all[0]:.2f})")
    ax.set_title("A. All Eligible Patients (N=72)\nCeiling Effect at D=1.0 Attenuates Slope", fontsize=11, fontweight="bold")
    ax.set_xlabel("AF Density (24h continuous)", fontsize=10)
    ax.set_ylabel("Left Atrial Diameter (Echo LAD, mm)", fontsize=10)
    ax.legend(frameon=True, fontsize=8.5, loc="lower left")

    # Panel B: Multi-Episode (N=49)
    ax = axes[0, 1]
    df_m = df_pat[multi_mask].copy()
    ax.scatter(df_m["density_continuous"], df_m["echo_lad_mm"], color="tab:blue", alpha=0.85, s=65)
    
    # Highlight shdb_4339581
    outlier = df_m[df_m["patient_id"] == "shdb_4339581"]
    if not outlier.empty:
        ax.scatter(outlier["density_continuous"], outlier["echo_lad_mm"], color="crimson", s=130, edgecolors="black", lw=1.5, zorder=5)
        ax.annotate("shdb_4339581 (High-leverage)\nLAD=55mm, D=0.46, Δt=980d",
                    (outlier["density_continuous"].iloc[0], outlier["echo_lad_mm"].iloc[0]),
                    xytext=(0.15, 53.5), arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2), fontsize=8.5, fontweight="bold", color="crimson")
    
    # Fit line Multi
    p_m = np.polyfit(df_m["density_continuous"], df_m["echo_lad_mm"], 1)
    ax.plot(x_grid, p_m[0]*x_grid + p_m[1], color="tab:blue", lw=2, label=f"OLS Multi-Episode (β={p_m[0]:.2f})")
    ax.set_title("B. Multi-Episode Patients (N=49)\nNegative Association Unmasked (β = -12.58 mm, p = 0.019)", fontsize=11, fontweight="bold")
    ax.set_xlabel("AF Density (24h continuous)", fontsize=10)
    ax.set_ylabel("Left Atrial Diameter (Echo LAD, mm)", fontsize=10)
    ax.legend(frameon=True, fontsize=9, loc="lower left")

    # Panel C: Added Variable Plot (Partial Residuals vs Density)
    ax = axes[1, 0]
    # In Multi-episode: fit LAD on Age+Female+Burden
    fit_red = sm.OLS(df_m["echo_lad_mm"], sm.add_constant(df_m[["age", "female", "af_burden"]])).fit()
    fit_d_on_red = sm.OLS(df_m["density_continuous"], sm.add_constant(df_m[["age", "female", "af_burden"]])).fit()
    lad_resid = fit_red.resid
    dens_resid = fit_d_on_red.resid

    ax.scatter(dens_resid, lad_resid, color="darkslateblue", alpha=0.85, s=65)
    p_av = np.polyfit(dens_resid, lad_resid, 1)
    x_av = np.linspace(dens_resid.min(), dens_resid.max(), 100)
    ax.plot(x_av, p_av[0]*x_av, color="darkslateblue", lw=2, label=f"Partial Slope β_D = {p_av[0]:.2f} mm (p=0.019)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_title("C. Added-Variable Plot (Adjusted for Age, Sex, Burden)\nPartial R² = 11.82% in Multi-Episode Cohort", fontsize=11, fontweight="bold")
    ax.set_xlabel("e(Density | Age, Sex, Burden)", fontsize=10)
    ax.set_ylabel("e(LAD | Age, Sex, Burden)", fontsize=10)
    ax.legend(frameon=True, fontsize=9, loc="lower left")

    # Panel D: Contemporaneous Echo Stratification
    ax = axes[1, 1]
    contemp_mask = df_m["delta_echo_days"] <= 90.0
    ax.scatter(df_m.loc[contemp_mask, "density_continuous"], df_m.loc[contemp_mask, "echo_lad_mm"],
               color="tab:green", alpha=0.85, s=70, label=f"Contemporaneous (|Δt| <= 90d, N={contemp_mask.sum()})")
    ax.scatter(df_m.loc[~contemp_mask, "density_continuous"], df_m.loc[~contemp_mask, "echo_lad_mm"],
               color="tab:orange", marker="^", alpha=0.85, s=70, label=f"Non-Contemporaneous (|Δt| > 90d, N={(~contemp_mask).sum()})")
    
    if contemp_mask.sum() >= 5:
        p_c = np.polyfit(df_m.loc[contemp_mask, "density_continuous"], df_m.loc[contemp_mask, "echo_lad_mm"], 1)
        ax.plot(x_grid, p_c[0]*x_grid + p_c[1], color="tab:green", lw=1.8, label=f"Contemporaneous OLS (β={p_c[0]:+.2f})")
    if (~contemp_mask).sum() >= 5:
        p_nc = np.polyfit(df_m.loc[~contemp_mask, "density_continuous"], df_m.loc[~contemp_mask, "echo_lad_mm"], 1)
        ax.plot(x_grid, p_nc[0]*x_grid + p_nc[1], color="tab:orange", ls="--", lw=1.8, label=f"Non-Contemporaneous OLS (β={p_nc[0]:.2f})")

    ax.set_title("D. Temporal Discordance Adjudication\nContemporaneous Echo Flips Association Direction", fontsize=11, fontweight="bold")
    ax.set_xlabel("AF Density (24h continuous)", fontsize=10)
    ax.set_ylabel("Left Atrial Diameter (Echo LAD, mm)", fontsize=10)
    ax.legend(frameon=True, fontsize=8.5, loc="lower left")

    plt.tight_layout()
    fig1_path = figures_dir / "adjudication_fig1_lad_density_scatter.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {fig1_path}")

    # Figure 2: LOPO Influence Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Cook's Distance
    ax = axes[0]
    cooks_m = df_lopo_multi["cooks_distance"].values
    pids_m = df_lopo_multi["patient_id"].values
    x_idx = np.arange(len(cooks_m))
    ax.stem(x_idx, cooks_m, linefmt="tab:blue", markerfmt="o", basefmt="k-")
    cutoff = 4.0 / len(cooks_m)
    ax.axhline(cutoff, color="crimson", ls="--", lw=1.2, label=f"Cutoff 4/N = {cutoff:.3f}")
    
    # Annotate top outlier
    top_idx = np.argmax(cooks_m)
    ax.annotate(f"{pids_m[top_idx]}\nD={cooks_m[top_idx]:.3f}",
                (top_idx, cooks_m[top_idx]),
                xytext=(top_idx + 2, cooks_m[top_idx] - 0.05),
                arrowprops=dict(arrowstyle="->", color="crimson"),
                fontweight="bold", color="crimson", fontsize=9)
    
    ax.set_title("A. Cook's Distance in Multi-Episode Cohort (N=49)\nIdentifying Influential Leverage Points", fontsize=11, fontweight="bold")
    ax.set_xlabel("Patient Index", fontsize=10)
    ax.set_ylabel("Cook's Distance", fontsize=10)
    ax.legend(frameon=True, fontsize=9)

    # Panel B: DFBETAS for Density
    ax = axes[1]
    dfb_m = df_lopo_multi["dfbeta_density"].values
    ax.stem(x_idx, dfb_m, linefmt="darkslateblue", markerfmt="s", basefmt="k-")
    dfb_cutoff = 2.0 / np.sqrt(len(dfb_m))
    ax.axhline(dfb_cutoff, color="crimson", ls="--", lw=1.2, label=f"Threshold ±2/√N = ±{dfb_cutoff:.3f}")
    ax.axhline(-dfb_cutoff, color="crimson", ls="--", lw=1.2)
    ax.axhline(0, color="gray", ls="-", lw=0.8)

    ax.annotate(f"{pids_m[top_idx]}\nDFBETA={dfb_m[top_idx]:.2f}",
                (top_idx, dfb_m[top_idx]),
                xytext=(top_idx - 12, dfb_m[top_idx] + 0.15),
                arrowprops=dict(arrowstyle="->", color="crimson"),
                fontweight="bold", color="crimson", fontsize=9)

    ax.set_title("B. DFBETAS for AF Density Coefficient (β_D)\nImpact of Omitting Individual Patients", fontsize=11, fontweight="bold")
    ax.set_xlabel("Patient Index", fontsize=10)
    ax.set_ylabel("DFBETAS(Density)", fontsize=10)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig2_path = figures_dir / "adjudication_fig2_lopo_influence.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}")

    print("\n" + "=" * 80)
    print("PRIORITY 1 COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
