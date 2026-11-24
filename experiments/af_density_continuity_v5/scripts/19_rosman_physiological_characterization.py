#!/usr/bin/env python3
"""Script 19: Physiological Characterization of Rosman Phenotypes (R04).
Characterizes ventricular rate, rapid AF, irregularity, episode fragmentation,
and intervening gaps across Rosman 4x4 cells, and evaluates incremental value of density.
Generates:
  - results/R04_rosman_cell_physiology.csv
  - results/R04_incremental_value_models.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.rosman_phenotypes import ROSMAN_TIERS, assign_rosman_tier

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ROSMAN 2026: R04 (PHYSIOLOGICAL CHARACTERIZATION OF PHENOTYPES)")
    print("=" * 80)

    pwin_parquet = features_dir / "patient_window_features.parquet"
    if not os.path.exists(pwin_parquet):
        print(f"Error: {pwin_parquet} does not exist yet. Please wait for ventricular rate extraction.")
        return

    df_pwin = pd.read_parquet(pwin_parquet)

    # Primary 12h horizon density-defined windows with rate features
    df_12h = df_pwin[
        (df_pwin["horizon"] == "12h") &
        (df_pwin["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df_pwin["is_density_defined"])
    ].copy()

    # Assign Rosman Tiers
    df_12h["burden_tier"] = df_12h["af_burden"].apply(assign_rosman_tier)
    df_12h["density_tier"] = df_12h["density_continuous"].apply(assign_rosman_tier)

    # 1. Rosman Cell Characterization
    print("\n--- 1. Physiological Profile per Supported 4x4 Cell ---")
    cell_records = []

    for b_tier in ROSMAN_TIERS:
        for d_tier in ROSMAN_TIERS:
            sub = df_12h[(df_12h["burden_tier"] == b_tier) & (df_12h["density_tier"] == d_tier)]
            n_wins = len(sub)
            n_pats = sub["patient_id"].nunique()

            if n_wins == 0:
                cell_records.append({
                    "burden_tier": b_tier,
                    "density_tier": d_tier,
                    "n_windows": 0,
                    "n_patients": 0,
                    "supported": False,
                    "median_burden": np.nan,
                    "median_density": np.nan,
                    "median_longest_af_min": np.nan,
                    "median_n_episodes": np.nan,
                    "mean_ventricular_rate_bpm": np.nan,
                    "median_q_120": np.nan,
                    "pct_windows_with_rapid_af": np.nan,
                    "mean_cv_rr": np.nan
                })
                continue

            cell_records.append({
                "burden_tier": b_tier,
                "density_tier": d_tier,
                "n_windows": n_wins,
                "n_patients": n_pats,
                "supported": (n_pats >= 5),
                "median_burden": float(sub["af_burden"].median()),
                "median_density": float(sub["density_continuous"].median()),
                "median_longest_af_min": float(sub["longest_af_run_s"].median() / 60.0),
                "median_n_episodes": float(sub["n_af_episodes"].median()),
                "mean_ventricular_rate_bpm": float(sub["mean_af_rate_bpm"].mean()) if "mean_af_rate_bpm" in sub and sub["mean_af_rate_bpm"].notna().any() else np.nan,
                "median_q_120": float(sub["q_120"].median()) if "q_120" in sub and sub["q_120"].notna().any() else np.nan,
                "pct_windows_with_rapid_af": float((sub["q_120"] > 0).mean() * 100.0) if "q_120" in sub else np.nan,
                "mean_cv_rr": float(sub["mean_cv_rr"].mean()) if "mean_cv_rr" in sub and sub["mean_cv_rr"].notna().any() else np.nan
            })

    df_cells = pd.DataFrame(cell_records)
    out_cells = results_dir / "R04_rosman_cell_physiology.csv"
    df_cells.to_csv(out_cells, index=False)
    print(f"Saved cell-level physiology table to {out_cells}")
    print(df_cells[df_cells["supported"]][["burden_tier", "density_tier", "n_patients", "median_longest_af_min", "median_q_120"]].to_string(index=False))

    # 2. Incremental Information of Density Beyond Burden and Longest Episode
    print("\n--- 2. Incremental Value Models (Target: Rapid AF Exposure Q120) ---")
    valid_reg = df_12h.dropna(subset=["q_120", "af_burden", "longest_af_run_s", "density_continuous"]).copy()
    valid_reg["log_longest_af_s"] = np.log1p(valid_reg["longest_af_run_s"].values.astype(float))

    # Cohort dummies
    cdummies = pd.get_dummies(valid_reg["dataset"], prefix="cohort", drop_first=True, dtype=float)
    valid_reg = pd.concat([valid_reg, cdummies], axis=1)
    base_covs = cdummies.columns.tolist()

    models = {
        "Mod1_Burden_Only": base_covs + ["af_burden"],
        "Mod2_Burden_Longest_Episode": base_covs + ["af_burden", "log_longest_af_s"],
        "Mod3_Burden_Longest_Continuous_Density": base_covs + ["af_burden", "log_longest_af_s", "density_continuous"],
    }

    # Add density tier dummies
    d_dummies = pd.get_dummies(valid_reg["density_tier"], prefix="dens_tier", drop_first=True, dtype=float)
    valid_reg = pd.concat([valid_reg, d_dummies], axis=1)
    models["Mod4_Burden_Longest_Density_Tiers"] = base_covs + ["af_burden", "log_longest_af_s"] + d_dummies.columns.tolist()

    model_rows = []
    y = valid_reg["q_120"].clip(1e-4, 1.0 - 1e-4)

    for m_name, covs in models.items():
        X = sm.add_constant(valid_reg[covs])
        try:
            glm = sm.GLM(y, X, family=Binomial(link=sm.genmod.families.links.Logit()))
            fit = glm.fit(cov_type="cluster", cov_kwds={"groups": valid_reg["patient_id"]})
            llf = fit.llf
            aic = fit.aic
            bic = fit.bic

            row = {
                "model": m_name,
                "n_obs": len(valid_reg),
                "n_clusters": valid_reg["patient_id"].nunique(),
                "log_likelihood": float(llf),
                "aic": float(aic),
                "bic": float(bic),
                "covariates": "+".join(covs)
            }
            # Record density p-value if present
            if "density_continuous" in covs:
                row["density_continuous_coef"] = float(fit.params.get("density_continuous", np.nan))
                row["density_continuous_pval"] = float(fit.pvalues.get("density_continuous", np.nan))

            model_rows.append(row)
            print(f"Model {m_name:40s} | AIC = {aic:.2f} | LogLik = {llf:.2f}")
        except Exception as e:
            print(f"Model {m_name} failed: {e}")

    df_models = pd.DataFrame(model_rows)
    out_models = results_dir / "R04_incremental_value_models.csv"
    df_models.to_csv(out_models, index=False)
    print(f"Saved incremental value models to {out_models}")

if __name__ == "__main__":
    main()
