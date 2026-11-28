"""Priority 3: Independent E02 Matched-Panel, Density-Fill Audit & Out-of-Patient GAM (V7-R)."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from pygam import LinearGAM, s
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Priority 3: Independent E02 Physiological Panel, Fill Audit & Patient-CV GAM ===")
    rr = pd.read_parquet("experiments/af_density_temporal_phenotypes_v7/results/E02_fixed_elapsed_rr_features.parquet")
    man = pd.read_parquet("experiments/af_density_temporal_phenotypes_v7/results/E02_historical_exposure_manifest.parquet")

    # 1. Strictly matched episode panels (no survivorship bias!)
    # Require episode survival through END of window (window_rel_end_s) and valid physiology across ALL landmarks
    v = rr[rr["is_valid_physio"] == True].copy()
    wins_17 = ["win_30_180s", "win_5_7_5m", "win_15_17_5m"]
    wins_32 = ["win_30_180s", "win_5_7_5m", "win_15_17_5m", "win_30_32_5m"]

    panel_rows = []
    for panel_name, w_list in [("Matched_Through_17_5m", wins_17), ("Matched_Through_32_5m", wins_32)]:
        sub = v[v["window_name"].isin(w_list)].copy()
        ep_ok = sub.groupby("episode_id")["window_name"].nunique()
        keep_eps = ep_ok[ep_ok == len(w_list)].index
        matched = sub[sub["episode_id"].isin(keep_eps)].copy()
        for w in w_list:
            wm = matched[matched["window_name"] == w]
            panel_rows.append({
                "panel": panel_name,
                "window_name": w,
                "n_episodes": int(wm["episode_id"].nunique()),
                "n_patients": int(wm["patient_id"].nunique()),
                "mean_hr_bpm": round(float(wm["rate_bpm"].mean()), 3),
                "se_hr_bpm": round(float(wm["rate_bpm"].sem()), 3),
                "mean_cvrr": round(float(wm["cv_rr"].mean()), 4),
                "se_cvrr": round(float(wm["cv_rr"].sem()), 4),
                "mean_rmssd_ms": round(float(wm["rmssd_ms"].mean()), 2),
                "se_rmssd_ms": round(float(wm["rmssd_ms"].sem()), 2),
            })
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(OUT_DIR / "V7R_e02_matched_panel.csv", index=False)
    print("Matched Episode Panel (Zero Survivorship Bias):")
    print(panel_df.to_string(index=False))

    # 2. Audit hist_density_filled vs strict complete-case defined density in Cox survival
    pop_a = man[man["is_population_a"]].copy()
    pop_a["event"] = pop_a["has_observed_termination"].astype(float)
    pop_a["time_to_event_m"] = pop_a["duration_m"].clip(lower=0.5)
    pop_a["last_gap_h"] = pop_a["h6_most_recent_non_af_gap_s"] / 3600.0
    pop_a["last_dur_h"] = pop_a["h6_immediately_preceding_dur_s"] / 3600.0

    n_undef = int(pop_a["h6_density"].isna().sum())
    print(f"\nPop A total episodes: {len(pop_a)} | Undefined h6_density filled as 0.0 previously: {n_undef} ({100*n_undef/len(pop_a):.2f}%)")

    surv_rows = []
    # Restrict strictly to h6_density.notna() (NO 0.0 filling!)
    pop_a_def = pop_a[pop_a["h6_density"].notna() & pop_a["last_gap_h"].notna() & pop_a["last_dur_h"].notna()].copy()
    for ds_lbl, sub_s in [("Pooled_DefinedDensity", pop_a_def),
                          ("LTAFDB_DefinedDensity", pop_a_def[pop_a_def["dataset"] == "LTAFDB"]),
                          ("SHDB_DefinedDensity", pop_a_def[pop_a_def["dataset"] == "SHDB-AF"]),
                          ("IRIDIA_DefinedDensity", pop_a_def[pop_a_def["dataset"] == "IRIDIA-AF"])]:
        if len(sub_s) < 30 or sub_s["patient_id"].nunique() < 5:
            continue
        cols = ["last_gap_h", "last_dur_h", "h6_burden", "h6_run_count", "h6_density"]
        try:
            ph = PHReg(sub_s["time_to_event_m"], sub_s[cols], status=sub_s["event"]).fit(groups=sub_s["patient_id"].astype(str))
            idx_d = cols.index("h6_density")
            coef, se, p = float(ph.params[idx_d]), float(ph.bse[idx_d]), float(ph.pvalues[idx_d])
            surv_rows.append({
                "analysis": "Cox_PH_CompleteDefinedDensity",
                "cohort": ds_lbl,
                "n_episodes": len(sub_s),
                "n_patients": int(sub_s["patient_id"].nunique()),
                "density_hr": round(float(np.exp(coef)), 4),
                "ci_lo": round(float(np.exp(coef - 1.96 * se)), 4),
                "ci_hi": round(float(np.exp(coef + 1.96 * se)), 4),
                "p_value": p,
            })
        except Exception as e:
            print(f"Cox failed on {ds_lbl}: {e}")

    # 3. Patient-disjoint 5-fold CV for 6h GAM with vs without density (isolating incremental value!)
    sub_gam = pop_a_def.copy()
    sub_gam["log_dur_s"] = np.log(sub_gam["duration_s"].clip(lower=1.0))
    # One-hot cohort control + recent/burden/arch vs +density
    for ds in ["LTAFDB", "SHDB-AF"]:
        sub_gam[f"ds_{ds}"] = (sub_gam["dataset"] == ds).astype(float)

    base_cols = ["ds_LTAFDB", "ds_SHDB-AF", "last_gap_h", "last_dur_h", "h6_burden", "h6_run_count"]
    full_cols = base_cols + ["h6_density"]
    y_g = sub_gam["log_dur_s"].values
    pts_g = sub_gam["patient_id"].values
    gkf = GroupKFold(n_splits=5)

    oof_base = np.zeros(len(y_g))
    oof_full = np.zeros(len(y_g))
    for tr, va in gkf.split(sub_gam, y_g, groups=pts_g):
        g_base = LinearGAM(s(0, n_splines=4) + s(1, n_splines=4) + s(2, n_splines=8) + s(3, n_splines=8) + s(4, n_splines=8) + s(5, n_splines=8)).fit(sub_gam[base_cols].values[tr], y_g[tr])
        g_full = LinearGAM(s(0, n_splines=4) + s(1, n_splines=4) + s(2, n_splines=8) + s(3, n_splines=8) + s(4, n_splines=8) + s(5, n_splines=8) + s(6, n_splines=8)).fit(sub_gam[full_cols].values[tr], y_g[tr])
        oof_base[va] = g_base.predict(sub_gam[base_cols].values[va])
        oof_full[va] = g_full.predict(sub_gam[full_cols].values[va])

    r2_oof_base = float(r2_score(y_g, oof_base))
    r2_oof_full = float(r2_score(y_g, oof_full))
    print(f"\n5-Fold Patient-Disjoint Out-of-Fold GAM R^2 (6h): Base={r2_oof_base:.4f}, +Density={r2_oof_full:.4f}, Delta={r2_oof_full - r2_oof_base:+.5f}")

    # Save unified E02 reconciliation CSV
    e02_rec = pd.DataFrame(surv_rows)
    e02_rec["gam_5fold_oof_r2_base"] = round(r2_oof_base, 5)
    e02_rec["gam_5fold_oof_r2_plus_density"] = round(r2_oof_full, 5)
    e02_rec["gam_5fold_oof_delta_density"] = round(r2_oof_full - r2_oof_base, 5)
    e02_rec.to_csv(OUT_DIR / "V7R_e02_reconciled.csv", index=False)
    print(e02_rec.to_string(index=False))


if __name__ == "__main__":
    main()
