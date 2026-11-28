"""Priority 4: Independent E03 SHDB-AF Clinical & Structural Adjudication (V7-R)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def patient_bootstrap_ols(df, predictors, outcome, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    pts = df["patient_id"].unique()
    n_pts = len(pts)
    pt_map = {p: df[df["patient_id"] == p] for p in pts}

    X_full = sm.add_constant(df[predictors].astype(float))
    y_full = df[outcome].astype(float)
    fit_full = sm.OLS(y_full, X_full).fit()

    dens_col = "density_12h" if "density_12h" in predictors else None
    dens_betas = []
    for _ in range(n_boot):
        samp = rng.choice(pts, size=n_pts, replace=True)
        bdf = pd.concat([pt_map[p] for p in samp], ignore_index=True)
        try:
            Xb = sm.add_constant(bdf[predictors].astype(float))
            yb = bdf[outcome].astype(float)
            fb = sm.OLS(yb, Xb).fit()
            if dens_col and dens_col in fb.params:
                dens_betas.append(float(fb.params[dens_col]) * 0.10)
        except Exception:
            continue

    arr = np.array(dens_betas) if dens_betas else np.array([np.nan])
    est_010 = float(fit_full.params[dens_col]) * 0.10 if dens_col else np.nan
    p_ols = float(fit_full.pvalues[dens_col]) if dens_col else np.nan
    return {
        "n_records": len(df),
        "n_patients": int(n_pts),
        "r2": float(fit_full.rsquared),
        "r2_adj": float(fit_full.rsquared_adj),
        "effect_per_0_10_density": est_010,
        "ols_p_value": p_ols,
        "boot_ci95_low": float(np.nanpercentile(arr, 2.5)),
        "boot_ci95_high": float(np.nanpercentile(arr, 97.5)),
        "n_boot_valid": int(len(dens_betas)),
    }


def main():
    print("=== Priority 4: Independent E03 SHDB-AF Clinical & Structural Audit ===")
    raw_meta = pd.read_csv("/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv")
    coh = pd.read_csv("experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_12h_standardized_cohort.csv")

    print(f"SHDB AdditionalData.csv: {len(raw_meta)} rows, {raw_meta['Subject_ID'].nunique()} unique subjects")
    print(f"Annotated SHDB records: {(coh['Annotated'] == True).sum()} rows, "
          f"{coh[coh['Annotated'] == True]['patient_id'].nunique()} unique patients")
    print(f"Repeated patients in SHDB metadata: {len(raw_meta) - raw_meta['Subject_ID'].nunique()} duplicate subject records")

    # Deduplicate to 1 recording per patient (earliest annotated / smallest echo latency) for strict independence
    ann = coh[coh["Annotated"] == True].copy()
    ann["echo_latency_abs_days"] = pd.to_numeric(ann["echo_latency_abs_days"], errors="coerce")

    rows = []
    for outcome in ["Echo_LAD", "Echo_LVEF"]:
        for lat_name, max_d in [("<=30d", 30.0), ("<=90d", 90.0), ("<=180d", 180.0), ("Unrestricted", np.inf)]:
            # Multi-run PAF (where density is strictly defined and non-trivial)
            sub_base = ann[(ann["AF_Type"] == "PAF") & (ann["n_runs_12h"] >= 2) &
                           ann[outcome].notna() & ann["density_12h"].notna() &
                           ann["Age_at_Holter"].notna() & ann["sex_male"].notna()].copy()
            if np.isfinite(max_d):
                sub_base = sub_base[sub_base["echo_latency_abs_days"] <= max_d]

            # Deduplicate by patient_id keeping smallest echo_latency_abs_days
            sub_dedup = sub_base.sort_values("echo_latency_abs_days").drop_duplicates("patient_id")

            # Model 2: Age + Sex + Burden
            m2 = patient_bootstrap_ols(sub_dedup, ["Age_at_Holter", "sex_male", "af_burden_12h"], outcome)
            # Model 3A: Age + Sex + Burden + Density
            m3a = patient_bootstrap_ols(sub_dedup, ["Age_at_Holter", "sex_male", "af_burden_12h", "density_12h"], outcome)
            rows.append({
                "cohort_subset": "PAF_MultiRun_PatientDedup",
                "outcome": outcome,
                "echo_latency_window": lat_name,
                "n_patients": m3a["n_patients"],
                "m2_burden_r2": round(m2["r2"], 4),
                "m3a_plus_density_r2": round(m3a["r2"], 4),
                "partial_r2_density": round(m3a["r2"] - m2["r2"], 4),
                "effect_per_0_10_density": round(m3a["effect_per_0_10_density"], 4),
                "boot_ci95_low": round(m3a["boot_ci95_low"], 4),
                "boot_ci95_high": round(m3a["boot_ci95_high"], 4),
                "ols_p_value": round(m3a["ols_p_value"], 4),
                "excludes_zero": bool(m3a["boot_ci95_low"] > 0 or m3a["boot_ci95_high"] < 0),
            })

            # Also check ALL PAF (including single-run where density=1.0 or undefined)
            sub_all = ann[(ann["AF_Type"] == "PAF") & ann[outcome].notna() & ann["density_12h"].notna() &
                          ann["Age_at_Holter"].notna() & ann["sex_male"].notna()].copy()
            if np.isfinite(max_d):
                sub_all = sub_all[sub_all["echo_latency_abs_days"] <= max_d]
            sub_all = sub_all.sort_values("echo_latency_abs_days").drop_duplicates("patient_id")
            if len(sub_all) >= 10:
                m2_all = patient_bootstrap_ols(sub_all, ["Age_at_Holter", "sex_male", "af_burden_12h"], outcome)
                m3_all = patient_bootstrap_ols(sub_all, ["Age_at_Holter", "sex_male", "af_burden_12h", "density_12h"], outcome)
                rows.append({
                    "cohort_subset": "PAF_AllDefinedDensity_PatientDedup",
                    "outcome": outcome,
                    "echo_latency_window": lat_name,
                    "n_patients": m3_all["n_patients"],
                    "m2_burden_r2": round(m2_all["r2"], 4),
                    "m3a_plus_density_r2": round(m3_all["r2"], 4),
                    "partial_r2_density": round(m3_all["r2"] - m2_all["r2"], 4),
                    "effect_per_0_10_density": round(m3_all["effect_per_0_10_density"], 4),
                    "boot_ci95_low": round(m3_all["boot_ci95_low"], 4),
                    "boot_ci95_high": round(m3_all["boot_ci95_high"], 4),
                    "ols_p_value": round(m3_all["ols_p_value"], 4),
                    "excludes_zero": bool(m3_all["boot_ci95_low"] > 0 or m3_all["boot_ci95_high"] < 0),
                })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_DIR / "V7R_e03_reconciled.csv", index=False)
    print(res_df.to_string(index=False))

    # Check if <1% burden 38-patient claim exists anywhere in 12h cohort
    lt1 = ann[ann["af_burden_12h"] < 0.01]
    lt5 = ann[ann["af_burden_12h"] < 0.05]
    print(f"\nSHDB <1% burden in 12h annotated cohort: {len(lt1)} records, {lt1['patient_id'].nunique()} patients (AF+ only: {(lt1['af_burden_12h'] > 0).sum()})")
    print(f"SHDB <5% burden (AF+ only): {((ann['af_burden_12h'] > 0) & (ann['af_burden_12h'] < 0.05)).sum()} records")


if __name__ == "__main__":
    main()
