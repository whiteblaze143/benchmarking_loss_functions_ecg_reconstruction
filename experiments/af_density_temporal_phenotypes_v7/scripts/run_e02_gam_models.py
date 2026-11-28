"""
Experiment E02: Generalized Additive Models (GAMs) vs Linear OLS Models
Evaluating non-linear spline relationships of historical AF concentration
and continuous Charitos density with:
1. Within-episode ventricular trajectory (Delta CVRR, Delta Rate, Delta RMSSD)
   across duration landmarks (7.5m, 17.5m, 32.5m).
2. Subsequent episode duration (log duration) in Population A across lookback horizons.
"""

import os
import json
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
import statsmodels.api as sm
import statsmodels.formula.api as smf


def evaluate_gam_vs_ols_trajectory(sub: pd.DataFrame, outcome: str, n_sp: int = 6):
    """
    Fits nested OLS and GAM models (M1, M2, M3, M4) on paired trajectory data.
    """
    y = sub[outcome].values
    n = len(sub)
    n_pts = sub["patient_id"].nunique()
    
    # Feature matrices
    sub_c = sub.copy()
    sub_c["gap_h"] = sub_c["hist_last_gap_m"] / 60.0
    sub_c["last_dur_h"] = sub_c["hist_last_dur_m"] / 60.0
    sub_c["longest_h"] = sub_c["hist_longest_m"] / 60.0
    
    # OLS Models
    ols_m1 = smf.ols(f"{outcome} ~ gap_h + last_dur_h", data=sub_c).fit()
    ols_m2 = smf.ols(f"{outcome} ~ gap_h + last_dur_h + hist_burden", data=sub_c).fit()
    ols_m3 = smf.ols(f"{outcome} ~ gap_h + last_dur_h + hist_burden + hist_run_count + longest_h", data=sub_c).fit()
    ols_m4 = smf.ols(f"{outcome} ~ gap_h + last_dur_h + hist_burden + hist_run_count + longest_h + hist_density", data=sub_c).fit()
    
    # GAM feature arrays
    X_m1 = sub_c[["gap_h", "last_dur_h"]].values
    X_m2 = sub_c[["gap_h", "last_dur_h", "hist_burden"]].values
    X_m3 = sub_c[["gap_h", "last_dur_h", "hist_burden", "hist_run_count", "longest_h"]].values
    X_m4 = sub_c[["gap_h", "last_dur_h", "hist_burden", "hist_run_count", "longest_h", "hist_density"]].values
    
    # Fit GAMs
    sp = min(n_sp, max(4, n // 15))
    gam_m1 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp)).fit(X_m1, y)
    gam_m2 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp) + s(2, n_splines=sp)).fit(X_m2, y)
    gam_m3 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp) + s(2, n_splines=sp) + s(3, n_splines=sp) + s(4, n_splines=sp)).fit(X_m3, y)
    gam_m4 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp) + s(2, n_splines=sp) + s(3, n_splines=sp) + s(4, n_splines=sp) + s(5, n_splines=sp)).fit(X_m4, y)
    
    m2_dev = gam_m2.statistics_["pseudo_r2"]["explained_deviance"]
    m3_dev = gam_m3.statistics_["pseudo_r2"]["explained_deviance"]
    m4_dev = gam_m4.statistics_["pseudo_r2"]["explained_deviance"]
    
    m2_burden_p = gam_m2.statistics_["p_values"][2] if len(gam_m2.statistics_["p_values"]) > 2 else np.nan
    m4_density_p = gam_m4.statistics_["p_values"][5] if len(gam_m4.statistics_["p_values"]) > 5 else np.nan
    
    return {
        "n_observations": n,
        "n_patients": n_pts,
        "n_splines": sp,
        "ols_m1_r2": float(ols_m1.rsquared),
        "ols_m1_aic": float(ols_m1.aic),
        "gam_m1_dev_expl": float(gam_m1.statistics_["pseudo_r2"]["explained_deviance"]),
        "gam_m1_aic": float(gam_m1.statistics_["AIC"]),
        
        "ols_m2_r2": float(ols_m2.rsquared),
        "ols_m2_aic": float(ols_m2.aic),
        "ols_m2_burden_p": float(ols_m2.pvalues.get("hist_burden", np.nan)),
        "gam_m2_dev_expl": float(m2_dev),
        "gam_m2_aic": float(gam_m2.statistics_["AIC"]),
        "gam_m2_burden_p": float(m2_burden_p),
        
        "ols_m3_r2": float(ols_m3.rsquared),
        "ols_m3_aic": float(ols_m3.aic),
        "gam_m3_dev_expl": float(m3_dev),
        "gam_m3_aic": float(gam_m3.statistics_["AIC"]),
        "gam_m3_delta_dev": float(m3_dev - m2_dev),
        
        "ols_m4_r2": float(ols_m4.rsquared),
        "ols_m4_aic": float(ols_m4.aic),
        "ols_m4_density_p": float(ols_m4.pvalues.get("hist_density", np.nan)),
        "gam_m4_dev_expl": float(m4_dev),
        "gam_m4_aic": float(gam_m4.statistics_["AIC"]),
        "gam_m4_density_p": float(m4_density_p),
        "gam_m4_delta_dev": float(m4_dev - m3_dev),
        "m4_dev_gain_over_ols": float(m4_dev - ols_m4.rsquared)
    }


def run_trajectory_gam_benchmarks(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs GAM benchmarks across multiple landmarks and outcomes.
    """
    records = []
    
    landmarks = [
        ("win_5_7_5m", "7.5m (450s)", 450),
        ("win_15_17_5m", "17.5m (1050s)", 1050),
        ("win_30_32_5m", "32.5m (1950s)", 1950)
    ]
    
    init = feat_df[feat_df["window_name"] == "win_30_180s"].copy()
    
    for win_name, land_label, land_sec in landmarks:
        land = feat_df[feat_df["window_name"] == win_name].copy()
        paired = pd.merge(
            init, land,
            on=["episode_id", "dataset", "patient_id", "record_id"],
            suffixes=("_init", "_land")
        )
        paired["delta_cv_rr"] = paired["cv_rr_land"] - paired["cv_rr_init"]
        paired["delta_rate_bpm"] = paired["rate_bpm_land"] - paired["rate_bpm_init"]
        paired["delta_rmssd_ms"] = paired["rmssd_ms_land"] - paired["rmssd_ms_init"]
        paired["hist_burden"] = paired["h6_burden_init"]
        paired["hist_density"] = paired["h6_density_init"]
        paired["hist_run_count"] = paired["h6_run_count_init"]
        paired["hist_last_dur_m"] = paired["h6_immediately_preceding_dur_s_init"] / 60.0
        paired["hist_last_gap_m"] = paired["h6_most_recent_non_af_gap_s_init"] / 60.0
        paired["hist_longest_m"] = paired["h6_longest_preceding_run_s_init"] / 60.0
        
        cohorts = {
            "ALL": paired,
            "IRIDIA-AF": paired[paired["dataset"] == "IRIDIA-AF"],
            "LTAFDB": paired[paired["dataset"] == "LTAFDB"],
            "SHDB-AF": paired[paired["dataset"] == "SHDB-AF"]
        }
        
        for cohort_name, c_df in cohorts.items():
            valid = c_df[c_df["hist_density"].notna()].dropna(
                subset=["delta_cv_rr", "delta_rate_bpm", "hist_burden", "hist_run_count", "hist_longest_m"]
            ).copy()
            
            if len(valid) < 20 or valid["patient_id"].nunique() < 5:
                continue
                
            for outcome in ["delta_cv_rr", "delta_rate_bpm"]:
                metrics = evaluate_gam_vs_ols_trajectory(valid, outcome, n_sp=6)
                metrics.update({
                    "duration_landmark": land_label,
                    "landmark_seconds": land_sec,
                    "cohort": cohort_name,
                    "outcome": outcome
                })
                records.append(metrics)
                
    return pd.DataFrame(records)


def evaluate_gam_vs_ols_duration(exp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits nested OLS and GAM models for log episode duration on Population A.
    """
    records = []
    
    horizons = [
        ("h1", "H1 (1h)", 3600),
        ("h3", "H3 (3h)", 10800),
        ("h6", "H6 (6h, Primary)", 21600)
    ]
    
    for h_prefix, h_label, h_sec in horizons:
        burden_col = f"{h_prefix}_burden"
        dens_col = f"{h_prefix}_density"
        
        if burden_col not in exp_df.columns or dens_col not in exp_df.columns:
            continue
            
        sub = exp_df[
            exp_df["is_population_a"] & 
            exp_df[f"has_{h_prefix[1:]}h_history"] & 
            exp_df[dens_col].notna() &
            (exp_df["duration_s"] > 0)
        ].copy()
        
        sub["log_dur"] = np.log(sub["duration_s"].clip(lower=1.0))
        sub["log_gap"] = np.log(sub["h6_most_recent_non_af_gap_s"].clip(lower=1.0))
        sub["log_last_dur"] = np.log(sub["h6_immediately_preceding_dur_s"].clip(lower=1.0))
        sub["log_longest"] = np.log(sub["h6_longest_preceding_run_s"].clip(lower=1.0))
        sub["hist_burden"] = sub[burden_col]
        sub["hist_density"] = sub[dens_col]
        sub["hist_run_count"] = sub["h6_run_count"]
        
        cohorts = {
            "ALL": sub,
            "IRIDIA-AF": sub[sub["dataset"] == "IRIDIA-AF"],
            "LTAFDB": sub[sub["dataset"] == "LTAFDB"],
            "SHDB-AF": sub[sub["dataset"] == "SHDB-AF"]
        }
        
        for c_name, c_sub in cohorts.items():
            valid = c_sub.dropna(subset=["log_dur", "log_gap", "log_last_dur", "hist_burden", "hist_run_count", "log_longest", "hist_density"]).copy()
            n = len(valid)
            if n < 30:
                continue
                
            n_pts = valid["patient_id"].nunique()
            y = valid["log_dur"].values
            
            # OLS
            ols_m1 = smf.ols("log_dur ~ log_gap + log_last_dur", data=valid).fit()
            ols_m2 = smf.ols("log_dur ~ log_gap + log_last_dur + hist_burden", data=valid).fit()
            ols_m3 = smf.ols("log_dur ~ log_gap + log_last_dur + hist_burden + hist_run_count + log_longest", data=valid).fit()
            ols_m4 = smf.ols("log_dur ~ log_gap + log_last_dur + hist_burden + hist_run_count + log_longest + hist_density", data=valid).fit()
            
            # GAM
            X_m1 = valid[["log_gap", "log_last_dur"]].values
            X_m2 = valid[["log_gap", "log_last_dur", "hist_burden"]].values
            X_m3 = valid[["log_gap", "log_last_dur", "hist_burden", "hist_run_count", "log_longest"]].values
            X_m4 = valid[["log_gap", "log_last_dur", "hist_burden", "hist_run_count", "log_longest", "hist_density"]].values
            
            sp = 8 if n > 500 else (6 if n > 100 else 4)
            gam_m1 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp)).fit(X_m1, y)
            gam_m2 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp) + s(2, n_splines=sp)).fit(X_m2, y)
            gam_m3 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp) + s(2, n_splines=sp) + s(3, n_splines=sp) + s(4, n_splines=sp)).fit(X_m3, y)
            gam_m4 = LinearGAM(s(0, n_splines=sp) + s(1, n_splines=sp) + s(2, n_splines=sp) + s(3, n_splines=sp) + s(4, n_splines=sp) + s(5, n_splines=sp)).fit(X_m4, y)
            
            m2_dev = gam_m2.statistics_["pseudo_r2"]["explained_deviance"]
            m3_dev = gam_m3.statistics_["pseudo_r2"]["explained_deviance"]
            m4_dev = gam_m4.statistics_["pseudo_r2"]["explained_deviance"]
            
            records.append({
                "history_horizon": h_label,
                "cohort": c_name,
                "outcome": "log_episode_duration_s",
                "n_observations": n,
                "n_patients": n_pts,
                "n_splines": sp,
                
                "ols_m1_r2": float(ols_m1.rsquared),
                "ols_m1_aic": float(ols_m1.aic),
                "gam_m1_dev_expl": float(gam_m1.statistics_["pseudo_r2"]["explained_deviance"]),
                "gam_m1_aic": float(gam_m1.statistics_["AIC"]),
                
                "ols_m2_r2": float(ols_m2.rsquared),
                "ols_m2_aic": float(ols_m2.aic),
                "ols_m2_burden_p": float(ols_m2.pvalues.get("hist_burden", np.nan)),
                "gam_m2_dev_expl": float(m2_dev),
                "gam_m2_aic": float(gam_m2.statistics_["AIC"]),
                "gam_m2_burden_p": float(gam_m2.statistics_["p_values"][2]) if len(gam_m2.statistics_["p_values"]) > 2 else np.nan,
                
                "ols_m3_r2": float(ols_m3.rsquared),
                "ols_m3_aic": float(ols_m3.aic),
                "gam_m3_dev_expl": float(m3_dev),
                "gam_m3_aic": float(gam_m3.statistics_["AIC"]),
                "gam_m3_delta_dev": float(m3_dev - m2_dev),
                
                "ols_m4_r2": float(ols_m4.rsquared),
                "ols_m4_aic": float(ols_m4.aic),
                "ols_m4_density_p": float(ols_m4.pvalues.get("hist_density", np.nan)),
                "gam_m4_dev_expl": float(m4_dev),
                "gam_m4_aic": float(gam_m4.statistics_["AIC"]),
                "gam_m4_density_p": float(gam_m4.statistics_["p_values"][5]) if len(gam_m4.statistics_["p_values"]) > 5 else np.nan,
                "gam_m4_delta_dev": float(m4_dev - m3_dev),
                "m4_dev_gain_over_ols": float(m4_dev - ols_m4.rsquared)
            })
            
    return pd.DataFrame(records)


def main():
    res_dir = "experiments/af_density_temporal_phenotypes_v7/results"
    codex_dir = "/data/mithunmanivannan/codex_artifacts/af_density_temporal_phenotypes_v7/results"
    os.makedirs(codex_dir, exist_ok=True)
    
    print("Loading E02 feature and exposure datasets...")
    feat_df = pd.read_parquet(os.path.join(res_dir, "E02_fixed_elapsed_rr_features.parquet"))
    exp_df = pd.read_parquet(os.path.join(res_dir, "E02_historical_exposure_manifest.parquet"))
    
    print("\n[1/2] Running GAM vs Linear OLS for Ventricular Trajectory Models...")
    df_traj_gam = run_trajectory_gam_benchmarks(feat_df)
    traj_path = os.path.join(res_dir, "E02_gam_vs_ols_trajectory.csv")
    df_traj_gam.to_csv(traj_path, index=False)
    df_traj_gam.to_csv(os.path.join(codex_dir, "E02_gam_vs_ols_trajectory.csv"), index=False)
    print(f"Generated {traj_path} ({len(df_traj_gam)} rows)")
    
    print("\n[2/2] Running GAM vs Linear OLS for Subsequent Episode Duration...")
    df_dur_gam = evaluate_gam_vs_ols_duration(exp_df)
    dur_path = os.path.join(res_dir, "E02_gam_vs_ols_duration.csv")
    df_dur_gam.to_csv(dur_path, index=False)
    df_dur_gam.to_csv(os.path.join(codex_dir, "E02_gam_vs_ols_duration.csv"), index=False)
    print(f"Generated {dur_path} ({len(df_dur_gam)} rows)")
    
    summary = {
        "status": "SUCCESS",
        "n_trajectory_benchmarks": len(df_traj_gam),
        "n_duration_benchmarks": len(df_dur_gam),
        "trajectory_landmarks": df_traj_gam["duration_landmark"].unique().tolist(),
        "trajectory_outcomes": df_traj_gam["outcome"].unique().tolist(),
        "duration_horizons": df_dur_gam["history_horizon"].unique().tolist(),
        "primary_m4_dev_gain_cvrr": float(df_traj_gam[(df_traj_gam["duration_landmark"] == "17.5m (1050s)") & (df_traj_gam["outcome"] == "delta_cv_rr") & (df_traj_gam["cohort"] == "ALL")]["m4_dev_gain_over_ols"].values[0]),
        "primary_m4_dev_gain_duration": float(df_dur_gam[(df_dur_gam["history_horizon"] == "H6 (6h, Primary)") & (df_dur_gam["cohort"] == "ALL")]["m4_dev_gain_over_ols"].values[0])
    }
    
    summary_path = os.path.join(res_dir, "E02_gam_models_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(codex_dir, "E02_gam_models_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Summary JSON saved: {summary_path}")
    print("\nSummary Metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
