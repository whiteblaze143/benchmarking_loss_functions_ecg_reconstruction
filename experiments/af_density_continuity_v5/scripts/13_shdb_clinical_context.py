#!/usr/bin/env python3
"""Script 13: Phase H - SHDB Clinical and Pharmacological Metadata Context.
Adjusts the association between continuity (S_max), density, and ventricular physiology (Q120)
for clinical covariates: age, sex, left atrial diameter (LAD), LVEF, and beta-blocker use.
Generates:
  - results/shdb_clinical_context.csv
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

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE H: SHDB CLINICAL AND PHARMACOLOGICAL METADATA CONTEXT")
    print("=" * 80)

    pwin_parquet = features_dir / "patient_window_features.parquet"
    shdb_meta_csv = "/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv"

    df_pwin = pd.read_parquet(pwin_parquet)
    df_meta = pd.read_csv(shdb_meta_csv)

    # Filter to SHDB 12h windows
    shdb_12h = df_pwin[
        (df_pwin["dataset"] == "SHDB-AF") &
        (df_pwin["horizon"] == "12h") &
        (df_pwin["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df_pwin["is_density_defined"]) &
        (df_pwin["q_120"].notna())
    ].copy()

    # Map Data_ID to match record_id
    df_meta["record_id"] = df_meta["Data_ID"].astype(str).str.zfill(3)
    shdb_merged = pd.merge(shdb_12h, df_meta, on="record_id", how="inner")

    print(f"SHDB windows with merged clinical metadata: {len(shdb_merged)}")

    # Clean variables
    shdb_merged["age"] = pd.to_numeric(shdb_merged["Age_at_Holter"], errors="coerce")
    shdb_merged["female"] = (shdb_merged["Sex"].astype(str).str.upper() == "F").astype(float)
    shdb_merged["beta_blocker"] = pd.to_numeric(shdb_merged["Antiarrhythmic_Drug_BB"], errors="coerce").fillna(0.0)
    shdb_merged["lad_mm"] = pd.to_numeric(shdb_merged["Echo_LAD"], errors="coerce")
    shdb_merged["lvef_pct"] = pd.to_numeric(shdb_merged["Echo_LVEF"], errors="coerce")

    # Fit adjusted models
    # Unadjusted: Q120 ~ af_burden + density_continuous + s_max
    # Adjusted: Q120 ~ af_burden + density_continuous + s_max + beta_blocker + age + female
    shdb_clean = shdb_merged.dropna(subset=["q_120", "af_burden", "density_continuous", "s_max", "beta_blocker", "age"]).copy()
    print(f"Complete cases for multivariable adjustment: {len(shdb_clean)}")

    y = np.clip(shdb_clean["q_120"].values.astype(float), 1e-6, 1.0 - 1e-6)

    # Model 1: Temporal only
    X1 = sm.add_constant(shdb_clean[["af_burden", "density_continuous", "s_max"]].values.astype(float))
    res1 = sm.GLM(y, X1, family=Binomial()).fit()

    # Model 2: Temporal + Medication (Beta-blocker)
    X2 = sm.add_constant(shdb_clean[["af_burden", "density_continuous", "s_max", "beta_blocker"]].values.astype(float))
    res2 = sm.GLM(y, X2, family=Binomial()).fit()

    # Model 3: Temporal + Beta-blocker + Demographics (Age, Female)
    X3 = sm.add_constant(shdb_clean[["af_burden", "density_continuous", "s_max", "beta_blocker", "age", "female"]].values.astype(float))
    res3 = sm.GLM(y, X3, family=Binomial()).fit()

    records = [
        {
            "model": "Temporal_Only",
            "s_max_coef": float(res1.params[3]),
            "s_max_se": float(res1.bse[3]),
            "s_max_pval": float(res1.pvalues[3]),
            "density_coef": float(res1.params[2]),
            "density_pval": float(res1.pvalues[2]),
            "bb_coef": np.nan, "bb_pval": np.nan,
            "aic": float(res1.aic)
        },
        {
            "model": "Temporal_Plus_BetaBlocker",
            "s_max_coef": float(res2.params[3]),
            "s_max_se": float(res2.bse[3]),
            "s_max_pval": float(res2.pvalues[3]),
            "density_coef": float(res2.params[2]),
            "density_pval": float(res2.pvalues[2]),
            "bb_coef": float(res2.params[4]),
            "bb_pval": float(res2.pvalues[4]),
            "aic": float(res2.aic)
        },
        {
            "model": "Temporal_Plus_BB_Age_Sex",
            "s_max_coef": float(res3.params[3]),
            "s_max_se": float(res3.bse[3]),
            "s_max_pval": float(res3.pvalues[3]),
            "density_coef": float(res3.params[2]),
            "density_pval": float(res3.pvalues[2]),
            "bb_coef": float(res3.params[4]),
            "bb_pval": float(res3.pvalues[4]),
            "aic": float(res3.aic)
        }
    ]

    df_shdb_res = pd.DataFrame(records)
    out_csv = results_dir / "shdb_clinical_context.csv"
    df_shdb_res.to_csv(out_csv, index=False)
    print(f"Saved SHDB clinical adjustment table to {out_csv}")
    print(df_shdb_res.to_string(index=False))

if __name__ == "__main__":
    main()
