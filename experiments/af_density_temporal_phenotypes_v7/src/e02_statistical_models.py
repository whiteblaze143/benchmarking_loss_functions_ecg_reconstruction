"""E02: Nested Trajectory Models (M0-M4), Within/Between Decomposition, and Subsequent Episode Survival.
Implements:
1. Nested linear models M0 to M4 for ventricular trajectory Delta CVRR.
2. Within-patient vs Between-patient decomposition of historical density effects.
3. Unadjusted vs Rate-adjusted electrophysiological models.
4. Cox Proportional Hazards survival models for subsequent episode duration on Population A.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


def build_trajectory_dataset(df_physio: pd.DataFrame) -> pd.DataFrame:
    """Builds paired episode trajectory dataset with initial (30-180s) and landmark (15-17.5m) features."""
    init_df = df_physio[df_physio["window_name"] == "win_30_180s"].copy()
    land_df = df_physio[df_physio["window_name"] == "win_15_17_5m"].copy()
    
    paired = pd.merge(
        init_df, land_df,
        on=["episode_id", "dataset", "patient_id", "record_id"],
        suffixes=("_init", "_land")
    )
    
    # Require valid physiology in both windows
    paired = paired[paired["is_valid_physio_init"] & paired["is_valid_physio_land"]].copy()
    
    paired["delta_cv_rr"] = paired["cv_rr_land"] - paired["cv_rr_init"]
    paired["delta_rate_bpm"] = paired["rate_bpm_land"] - paired["rate_bpm_init"]
    paired["delta_rmssd_ms"] = paired["rmssd_ms_land"] - paired["rmssd_ms_init"]
    
    # Historical variables from initial window
    paired["hist_burden"] = paired["h6_burden_init"]
    paired["hist_density"] = paired["h6_density_init"]
    paired["hist_run_count"] = paired["h6_run_count_init"]
    paired["hist_last_dur_m"] = paired["h6_immediately_preceding_dur_s_init"] / 60.0
    paired["hist_last_gap_m"] = paired["h6_most_recent_non_af_gap_s_init"] / 60.0
    paired["hist_longest_m"] = paired["h6_longest_preceding_run_s_init"] / 60.0
    
    return paired


def fit_nested_trajectory_models(df_traj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fits nested models M0 to M4 for Delta CVRR.
    
    Returns:
      (model_summary_df, nested_comparison_df)
    """
    # Restrict to complete cases for density to ensure identical analytical populations
    sub = df_traj[df_traj["hist_density"].notna()].copy()
    
    if len(sub) < 15 or sub["patient_id"].nunique() < 5:
        # Fallback to full sample with density imputation/zero flag if sparse
        sub = df_traj.copy()
        sub["hist_density"] = sub["hist_density"].fillna(0.0)
        
    y = sub["delta_cv_rr"]
    groups = sub["patient_id"]
    
    # Specification of nested predictors:
    # M0: Intercept only
    X0 = pd.DataFrame({"const": 1.0}, index=sub.index)
    
    # M1: M0 + recent history
    X1 = X0.copy()
    X1["last_gap_h"] = sub["hist_last_gap_m"] / 60.0
    X1["last_dur_h"] = sub["hist_last_dur_m"] / 60.0
    
    # M2: M1 + cumulative burden
    X2 = X1.copy()
    X2["h6_burden"] = sub["hist_burden"]
    
    # M3: M2 + architecture
    X3 = X2.copy()
    X3["h6_run_count"] = sub["hist_run_count"]
    X3["h6_longest_h"] = sub["hist_longest_m"] / 60.0
    
    # M4: M3 + historical continuous density
    X4 = X3.copy()
    X4["h6_density"] = sub["hist_density"]
    
    models = [("M0", X0), ("M1", X1), ("M2", X2), ("M3", X3), ("M4", X4)]
    model_records = []
    comp_records = []
    
    prev_llf = None
    prev_df = None
    
    for name, X_mat in models:
        try:
            ols = sm.OLS(y, X_mat).fit(cov_type="cluster", cov_kwds={"groups": groups})
            
            dens_beta = ols.params.get("h6_density", np.nan)
            dens_se = ols.bse.get("h6_density", np.nan)
            dens_pval = ols.pvalues.get("h6_density", np.nan)
            
            rec = {
                "model_name": name,
                "n_observations": int(ols.nobs),
                "n_patients": sub["patient_id"].nunique(),
                "r_squared": float(ols.rsquared),
                "r_squared_adj": float(ols.rsquared_adj),
                "log_likelihood": float(ols.llf),
                "aic": float(ols.aic),
                "bic": float(ols.bic),
                "density_coef_per_0_10": float(dens_beta * 0.10) if np.isfinite(dens_beta) else np.nan,
                "density_se": float(dens_se),
                "density_p_value": float(dens_pval),
                "model_f_pvalue": float(ols.f_pvalue) if hasattr(ols, "f_pvalue") and np.isfinite(ols.f_pvalue) else np.nan
            }
            model_records.append(rec)
            
            # Nested model comparison
            if prev_llf is not None:
                lr_stat = 2.0 * (ols.llf - prev_llf)
                df_diff = X_mat.shape[1] - prev_df
                lr_pval = float(1.0 - stats.chi2.cdf(lr_stat, df_diff)) if df_diff > 0 else np.nan
                comp_records.append({
                    "comparison": f"{name}_vs_prev",
                    "lr_chi2_stat": float(lr_stat),
                    "df_diff": df_diff,
                    "p_value": lr_pval,
                    "r2_increase": float(ols.rsquared - model_records[-2]["r_squared"])
                })
                
            prev_llf = ols.llf
            prev_df = X_mat.shape[1]
            
        except Exception as e:
            model_records.append({
                "model_name": name,
                "n_observations": len(sub),
                "n_patients": sub["patient_id"].nunique(),
                "r_squared": np.nan,
                "r_squared_adj": np.nan,
                "log_likelihood": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "density_coef_per_0_10": np.nan,
                "density_se": np.nan,
                "density_p_value": np.nan,
                "model_f_pvalue": np.nan
            })
            
    return pd.DataFrame(model_records), pd.DataFrame(comp_records)


def fit_within_between_decomposition(df_traj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Decomposes historical density into within-patient and between-patient components."""
    sub = df_traj[df_traj["hist_density"].notna()].copy()
    
    # Patient means
    p_means = sub.groupby("patient_id")["hist_density"].mean().to_dict()
    sub["density_between"] = sub["patient_id"].map(p_means)
    sub["density_within"] = sub["hist_density"] - sub["density_between"]
    
    groups = sub["patient_id"]
    y = sub["delta_cv_rr"]
    
    # 1. Unadjusted within-between model
    X_decomp = pd.DataFrame({
        "const": 1.0,
        "density_between": sub["density_between"],
        "density_within": sub["density_within"]
    }, index=sub.index)
    
    # 2. Rate-adjusted within-between model
    X_decomp_adj = X_decomp.copy()
    X_decomp_adj["delta_rate_bpm"] = sub["delta_rate_bpm"]
    
    within_records = []
    between_records = []
    
    for adj_name, X_m in [("Unadjusted", X_decomp), ("Rate_Adjusted", X_decomp_adj)]:
        try:
            ols = sm.OLS(y, X_m).fit(cov_type="cluster", cov_kwds={"groups": groups})
            
            # Within effect
            w_b = ols.params["density_within"]
            w_se = ols.bse["density_within"]
            w_p = ols.pvalues["density_within"]
            
            within_records.append({
                "model_adjustment": adj_name,
                "effect_type": "within_patient",
                "coef_per_0_10_density": float(w_b * 0.10),
                "ci_lower_per_0_10": float((w_b - 1.96 * w_se) * 0.10),
                "ci_upper_per_0_10": float((w_b + 1.96 * w_se) * 0.10),
                "p_value": float(w_p),
                "n_observations": int(ols.nobs),
                "n_patients": sub["patient_id"].nunique()
            })
            
            # Between effect
            b_b = ols.params["density_between"]
            b_se = ols.bse["density_between"]
            b_p = ols.pvalues["density_between"]
            
            between_records.append({
                "model_adjustment": adj_name,
                "effect_type": "between_patient",
                "coef_per_0_10_density": float(b_b * 0.10),
                "ci_lower_per_0_10": float((b_b - 1.96 * b_se) * 0.10),
                "ci_upper_per_0_10": float((b_b + 1.96 * b_se) * 0.10),
                "p_value": float(b_p),
                "n_observations": int(ols.nobs),
                "n_patients": sub["patient_id"].nunique()
            })
        except Exception:
            continue
            
    return pd.DataFrame(within_records), pd.DataFrame(between_records)


def fit_subsequent_episode_survival(df_manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fits Cox Proportional Hazards regression on Population A for time to episode termination.
    
    Returns:
      (survival_models_df, risk_sets_df)
    """
    # Population A: episodes with >=6h preceding monitoring
    sub = df_manifest[df_manifest["is_population_a"]].copy()
    
    # Event: observed termination = 1; right-censored = 0
    sub["event"] = sub["has_observed_termination"].astype(float)
    sub["time_to_event_m"] = sub["duration_m"].clip(lower=0.5) # Minimum 30s
    
    # Fill density for zero-history if necessary
    sub["hist_density_filled"] = sub["h6_density"].fillna(0.0)
    
    # Summary of risk sets
    risk_summary = []
    time_pts = [5.0, 15.0, 30.0, 60.0, 120.0, 360.0]
    for tp in time_pts:
        at_risk = (sub["time_to_event_m"] >= tp).sum()
        events_by_tp = ((sub["time_to_event_m"] < tp) & (sub["event"] == 1)).sum()
        cens_by_tp = ((sub["time_to_event_m"] < tp) & (sub["event"] == 0)).sum()
        risk_summary.append({
            "landmark_duration_m": tp,
            "n_at_risk": int(at_risk),
            "cumulative_terminations": int(events_by_tp),
            "cumulative_censored": int(cens_by_tp),
            "survival_probability": float(at_risk / len(sub)) if len(sub) > 0 else 0.0
        })
    df_risk = pd.DataFrame(risk_summary)
    
    # Survival Models:
    # Model 1: Recent history
    sub["last_gap_h"] = sub["h6_most_recent_non_af_gap_s"] / 3600.0
    sub["last_dur_h"] = sub["h6_immediately_preceding_dur_s"] / 3600.0
    
    # Predictor sets
    specs = [
        ("Model_1_Recent_History", ["last_gap_h", "last_dur_h"]),
        ("Model_2_Recent_Plus_Burden", ["last_gap_h", "last_dur_h", "h6_burden"]),
        ("Model_3_Recent_Burden_Architecture", ["last_gap_h", "last_dur_h", "h6_burden", "h6_run_count"]),
        ("Model_4_Full_With_Density", ["last_gap_h", "last_dur_h", "h6_burden", "h6_run_count", "hist_density_filled"])
    ]
    
    surv_records = []
    groups = sub["patient_id"].astype(str)
    
    for name, pred_cols in specs:
        X_df = sub[pred_cols].copy()
        
        try:
            # Statsmodels PHReg
            ph = PHReg(sub["time_to_event_m"], X_df, status=sub["event"])
            res = ph.fit(groups=groups)
            
            for p_idx, p_col in enumerate(pred_cols):
                coef = res.params[p_idx]
                se = res.bse[p_idx]
                pval = res.pvalues[p_idx]
                hr = float(np.exp(coef))
                ci_l = float(np.exp(coef - 1.96 * se))
                ci_u = float(np.exp(coef + 1.96 * se))
                
                surv_records.append({
                    "model_name": name,
                    "predictor": p_col,
                    "hazard_ratio": hr,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "coef": float(coef),
                    "se": float(se),
                    "p_value": float(pval),
                    "log_likelihood": float(res.llf),
                    "n_events": int(sub["event"].sum()),
                    "n_total": len(sub),
                    "n_patients": sub["patient_id"].nunique()
                })
        except Exception as e:
            # Robust fallback record
            surv_records.append({
                "model_name": name,
                "predictor": "fit_error",
                "hazard_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "coef": np.nan,
                "se": np.nan,
                "p_value": np.nan,
                "log_likelihood": np.nan,
                "n_events": int(sub["event"].sum()),
                "n_total": len(sub),
                "n_patients": sub["patient_id"].nunique()
            })
            
    df_surv = pd.DataFrame(surv_records)
    return df_surv, df_risk
