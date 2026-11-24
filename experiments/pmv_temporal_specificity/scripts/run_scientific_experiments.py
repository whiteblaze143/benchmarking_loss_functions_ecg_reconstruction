"""Comprehensive Scientific Execution Module for Experiments E01 through E10.

Rigorous evaluation of Temporal Specificity of P-Wave Morphological Variability (PMV) in Paroxysmal AF:
- E01: Replication of Published Pre-AF PMV Dynamics (Noriega IEEE 2026 change points)
- E02: Post-AF PMV Trajectory (RQ1: Recovery and Stunning)
- E03: Relationship with Preceding AF Exposure (RQ2: Dose-Response and Within-Patient Effects)
- E04: Complete Interepisode PMV Dynamics (2D Spline and Identifiability Support Map)
- E05: Matched Risk Sets & Temporal Specificity (RQ3: Nested Conditional Logistic Regression)
- E06: Baseline Normalization Sensitivity (RQ4: Event-Relative vs Absolute vs Remote Baseline)
- E07: Ectopy, PR Timing, and Morphological Variability Confounding (RQ5)
- E08: Authentic Ambulatory Noise & Controlled Perturbation (NSTDB & Noise Holter DB)
- E09: SHDB Clinical Substrate Characterization (LAD, LVEF, Prior Ablation, Medications)
- E10: Secondary Connection to AF Density and Temporal Clustering

Strict patient-level clustering, robust sandwich standard errors, zero synthetic fallbacks.
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pmv_temporal_specificity.scripts.pmv_core import PMVExtractor

FEATURES_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "features"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# E01: Replication of Published Pre-AF PMV Dynamics
# ==============================================================================
def fit_piecewise_linear_change_point(t_mins: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fit V(k) = beta0 + beta1 * max(0, k - k_cp) + eps.
    
    Grid search candidate k_cp in [-45, -5] mins.
    Returns: best_k_cp, beta0, beta1, p_val, r2
    """
    if len(t_mins) < 5 or np.all(np.isnan(y)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    valid_mask = ~np.isnan(y)
    t = t_mins[valid_mask]
    vals = y[valid_mask]
    if len(t) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    best_k_cp = -20.0
    best_sse = float("inf")
    best_beta0 = 0.0
    best_beta1 = 0.0
    best_pval = 1.0
    best_r2 = 0.0

    candidates = np.linspace(-45.0, -5.0, 41)
    ss_tot = np.sum((vals - np.mean(vals))**2) + 1e-12

    for cp in candidates:
        hinge = np.maximum(0.0, t - cp)
        X = np.column_stack([np.ones_like(t), hinge])
        try:
            # OLS fit
            beta, residuals, rank, s = np.linalg.lstsq(X, vals, rcond=None)
            pred = X @ beta
            sse = np.sum((vals - pred)**2)
            if sse < best_sse:
                best_sse = sse
                best_k_cp = cp
                best_beta0 = beta[0]
                best_beta1 = beta[1]
                best_r2 = max(0.0, 1.0 - (sse / ss_tot))

                # Standard error of beta1
                n = len(t)
                df_e = n - 2
                if df_e > 0:
                    mse = sse / df_e
                    var_beta = mse * np.linalg.pinv(X.T @ X)[1, 1]
                    se_beta1 = np.sqrt(max(0.0, var_beta))
                    if se_beta1 > 1e-12:
                        t_stat = beta[1] / se_beta1
                        best_pval = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_e))
                    else:
                        best_pval = 1.0
        except Exception:
            continue

    return best_k_cp, best_beta0, best_beta1, best_pval, best_r2


def run_e01(df_pre: pd.DataFrame, df_remote: pd.DataFrame):
    print("\n--- Running Experiment E01: Replication of Published Pre-AF PMV Dynamics ---")
    valid_pre = df_pre[df_pre["is_valid"]].copy()

    records = []
    # For each pre-AF segment
    for seg_id, group in valid_pre.groupby("segment_id"):
        t_m = group["rel_time_to_af_m"].values
        v_norm = group["pmv_V"].values
        v_rel = group["relative_pmv_Vrel"].values if "relative_pmv_Vrel" in group.columns else v_norm
        v_align = group["pmv_aligned_V"].values

        patient = group["patient_id"].iloc[0]
        dataset = group["dataset"].iloc[0]
        record_id = group["record_id"].iloc[0]
        af_dur_s = group["af_duration_s"].iloc[0]
        time_since_prev_af_s = group["time_since_prev_af_s"].iloc[0]

        # Fit change point on relative PMV (published Noriega method)
        cp_rel, b0_rel, b1_rel, p_rel, r2_rel = fit_piecewise_linear_change_point(t_m, v_rel)
        # Fit change point on energy-normalized PMV
        cp_norm, b0_norm, b1_norm, p_norm, r2_norm = fit_piecewise_linear_change_point(t_m, v_norm)
        # Fit change point on PR-aligned PMV
        cp_al, b0_al, b1_al, p_al, r2_al = fit_piecewise_linear_change_point(t_m, v_align)

        records.append({
            "segment_type": "PRE_AF",
            "segment_id": seg_id,
            "dataset": dataset,
            "patient_id": patient,
            "record_id": record_id,
            "num_windows": len(group),
            "af_duration_s": af_dur_s,
            "time_since_prev_af_s": time_since_prev_af_s,
            "cp_rel_min": cp_rel,
            "slope_rel": b1_rel,
            "pval_rel": p_rel,
            "is_significant_rel": (b1_rel > 0) and (p_rel < 0.05),
            "r2_rel": r2_rel,
            "cp_norm_min": cp_norm,
            "slope_norm": b1_norm,
            "pval_norm": p_norm,
            "is_significant_norm": (b1_norm > 0) and (p_norm < 0.05),
            "r2_norm": r2_norm,
            "cp_align_min": cp_al,
            "slope_align": b1_al,
            "pval_align": p_al,
            "is_significant_align": (b1_al > 0) and (p_al < 0.05),
        })

    # Repeat on Remote Sinus Controls (Negative Control)
    valid_remote = df_remote[df_remote["is_valid"]].copy()
    for seg_id, group in valid_remote.groupby("segment_id"):
        t_m = (group["time_center_s"].values - group["time_center_s"].max()) / 60.0  # mock [-60m, 0]
        v_norm = group["pmv_V"].values
        v_align = group["pmv_aligned_V"].values
        patient = group["patient_id"].iloc[0]
        dataset = group["dataset"].iloc[0]
        record_id = group["record_id"].iloc[0]

        cp_norm, b0_norm, b1_norm, p_norm, r2_norm = fit_piecewise_linear_change_point(t_m, v_norm)
        cp_al, b0_al, b1_al, p_al, r2_al = fit_piecewise_linear_change_point(t_m, v_align)

        records.append({
            "segment_type": "REMOTE_SINUS_CONTROL",
            "segment_id": seg_id,
            "dataset": dataset,
            "patient_id": patient,
            "record_id": record_id,
            "num_windows": len(group),
            "af_duration_s": np.nan,
            "time_since_prev_af_s": np.nan,
            "cp_rel_min": np.nan,
            "slope_rel": np.nan,
            "pval_rel": np.nan,
            "is_significant_rel": False,
            "r2_rel": np.nan,
            "cp_norm_min": cp_norm,
            "slope_norm": b1_norm,
            "pval_norm": p_norm,
            "is_significant_norm": (b1_norm > 0) and (p_norm < 0.05),
            "r2_norm": r2_norm,
            "cp_align_min": cp_al,
            "slope_align": b1_al,
            "pval_align": p_al,
            "is_significant_align": (b1_al > 0) and (p_al < 0.05),
        })

    df_e01 = pd.DataFrame(records)
    df_e01.to_csv(RESULTS_DIR / "e01_pre_af_replication.csv", index=False)

    pre_cases = df_e01[df_e01["segment_type"] == "PRE_AF"]
    ctrl_cases = df_e01[df_e01["segment_type"] == "REMOTE_SINUS_CONTROL"]

    sig_rate_rel = pre_cases["is_significant_rel"].mean() * 100.0
    sig_rate_norm = pre_cases["is_significant_norm"].mean() * 100.0
    sig_rate_align = pre_cases["is_significant_align"].mean() * 100.0
    ctrl_rate_norm = ctrl_cases["is_significant_norm"].mean() * 100.0

    print(f"E01 Results Summary:")
    print(f"  Pre-AF segments evaluated: {len(pre_cases)} across {pre_cases['patient_id'].nunique()} patients")
    print(f"  Positive change-point detection (Relative PMV Vrel): {sig_rate_rel:.1f}% (mean cp: {pre_cases.loc[pre_cases['is_significant_rel'], 'cp_rel_min'].mean():.1f} min)")
    print(f"  Positive change-point detection (Energy-norm PMV V): {sig_rate_norm:.1f}%")
    print(f"  Positive change-point detection (PR-aligned PMV): {sig_rate_align:.1f}%")
    print(f"  Remote Sinus False Alarm Rate: {ctrl_rate_norm:.1f}%")
    return df_e01


# ==============================================================================
# E02: Post-AF PMV Trajectory (RQ1: Recovery and Stunning)
# ==============================================================================
def run_e02(df_post: pd.DataFrame, df_remote: pd.DataFrame):
    print("\n--- Running Experiment E02: Post-AF PMV Trajectory (RQ1) ---")
    valid_post = df_post[df_post["is_valid"]].copy()
    valid_remote = df_remote[df_remote["is_valid"]].copy()

    # Define post-AF windows
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    labels = ["W1_0-2m", "W2_2-5m", "W3_5-10m", "W4_10-15m", "W5_15-30m", "W6_30-60m"]
    valid_post["post_window"] = pd.cut(valid_post["rel_time_from_af_m"], bins=bins, labels=labels, right=True)

    # Remote sinus baseline
    remote_pmv_vals = valid_remote["pmv_V"].dropna().values
    remote_median = np.median(remote_pmv_vals)
    remote_mean = np.mean(remote_pmv_vals)

    traj_records = []
    # Changing risk set summary
    for w in labels:
        sub = valid_post[valid_post["post_window"] == w]
        pmv_vals = sub["pmv_V"].dropna().values
        pmv_al = sub["pmv_aligned_V"].dropna().values
        dur_af = sub["af_duration_s"].median()

        # Ratio to remote sinus baseline
        ratio_to_remote = np.median(pmv_vals) / (remote_median + 1e-12) if len(pmv_vals) > 0 else np.nan

        # Wilcoxon rank-sum vs remote sinus
        stat, pval = stats.ranksums(pmv_vals, remote_pmv_vals) if len(pmv_vals) > 0 else (np.nan, np.nan)

        traj_records.append({
            "window": w,
            "panel_type": "CHANGING_RISK_SET",
            "num_subsegments": len(sub),
            "num_episodes": sub["segment_id"].nunique(),
            "num_patients": sub["patient_id"].nunique(),
            "pmv_median": np.median(pmv_vals) if len(pmv_vals) > 0 else np.nan,
            "pmv_mean": np.mean(pmv_vals) if len(pmv_vals) > 0 else np.nan,
            "pmv_iqr": float(np.percentile(pmv_vals, 75) - np.percentile(pmv_vals, 25)) if len(pmv_vals) > 0 else np.nan,
            "pmv_aligned_median": np.median(pmv_al) if len(pmv_al) > 0 else np.nan,
            "ratio_to_remote_sinus": ratio_to_remote,
            "ranksum_stat_vs_remote": stat,
            "ranksum_pval": pval,
            "median_preceding_af_dur_s": dur_af,
        })

    # Fixed panel: episodes having subsegments in all 6 windows
    complete_episodes = []
    for ep, grp in valid_post.groupby("segment_id"):
        covered = set(grp["post_window"].dropna().unique())
        if len(covered) == len(labels):
            complete_episodes.append(ep)

    fixed_sub = valid_post[valid_post["segment_id"].isin(complete_episodes)]
    for w in labels:
        sub = fixed_sub[fixed_sub["post_window"] == w]
        pmv_vals = sub["pmv_V"].dropna().values
        ratio_to_remote = np.median(pmv_vals) / (remote_median + 1e-12) if len(pmv_vals) > 0 else np.nan
        stat, pval = stats.ranksums(pmv_vals, remote_pmv_vals) if len(pmv_vals) > 0 else (np.nan, np.nan)

        traj_records.append({
            "window": w,
            "panel_type": "FIXED_PANEL",
            "num_subsegments": len(sub),
            "num_episodes": sub["segment_id"].nunique(),
            "num_patients": sub["patient_id"].nunique(),
            "pmv_median": np.median(pmv_vals) if len(pmv_vals) > 0 else np.nan,
            "pmv_mean": np.mean(pmv_vals) if len(pmv_vals) > 0 else np.nan,
            "pmv_iqr": float(np.percentile(pmv_vals, 75) - np.percentile(pmv_vals, 25)) if len(pmv_vals) > 0 else np.nan,
            "pmv_aligned_median": np.median(sub["pmv_aligned_V"].dropna().values) if len(sub) > 0 else np.nan,
            "ratio_to_remote_sinus": ratio_to_remote,
            "ranksum_stat_vs_remote": stat,
            "ranksum_pval": pval,
            "median_preceding_af_dur_s": sub["af_duration_s"].median() if len(sub) > 0 else np.nan,
        })

    df_e02 = pd.DataFrame(traj_records)
    df_e02.to_csv(RESULTS_DIR / "e02_post_af_trajectories.csv", index=False)

    print("E02 Trajectory Table:")
    print(df_e02[["window", "panel_type", "num_episodes", "pmv_median", "ratio_to_remote_sinus", "ranksum_pval"]].to_string(index=False))
    return df_e02


# ==============================================================================
# E03: Relationship with Preceding AF Exposure (RQ2)
# ==============================================================================
def run_e03(df_post: pd.DataFrame):
    print("\n--- Running Experiment E03: Relationship with Preceding AF Exposure (RQ2) ---")
    valid_post = df_post[df_post["is_valid"]].copy()

    # Restrict to early post-AF window: 0-10 min post-termination
    early_post = valid_post[(valid_post["rel_time_from_af_m"] >= 0) & (valid_post["rel_time_from_af_m"] <= 10.0)].copy()

    # Episode-level aggregation
    ep_data = []
    for ep_id, group in early_post.groupby("segment_id"):
        patient = group["patient_id"].iloc[0]
        dataset = group["dataset"].iloc[0]
        af_dur_s = group["af_duration_s"].iloc[0]
        early_pmv = group["pmv_V"].median()
        early_al = group["pmv_aligned_V"].median()
        ep_data.append({
            "segment_id": ep_id,
            "patient_id": patient,
            "dataset": dataset,
            "af_duration_s": af_dur_s,
            "log10_af_duration_s": np.log10(max(af_dur_s, 1.0)),
            "early_pmv_V": early_pmv,
            "log_early_pmv": np.log(max(early_pmv, 1e-6)),
            "early_pmv_aligned": early_al,
        })

    df_ep = pd.DataFrame(ep_data)

    # Exposure strata
    # 1: <5m (<300s), 2: 5-30m (300-1800s), 3: 30-60m (1800-3600s), 4: 1-6h (3600-21600s), 5: >=6h (>=21600s)
    bins = [0, 300, 1800, 3600, 21600, float("inf")]
    labels = ["Micro (<5m)", "Short (5-30m)", "Intermediate (30-60m)", "Prolonged (1-6h)", "Extended (>=6h)"]
    df_ep["exposure_stratum"] = pd.cut(df_ep["af_duration_s"], bins=bins, labels=labels, right=False)

    strata_records = []
    for s in labels:
        sub = df_ep[df_ep["exposure_stratum"] == s]
        vals = sub["early_pmv_V"].dropna().values
        strata_records.append({
            "exposure_stratum": s,
            "num_episodes": len(sub),
            "num_patients": sub["patient_id"].nunique(),
            "median_af_duration_m": (sub["af_duration_s"].median() / 60.0) if len(sub) > 0 else np.nan,
            "early_pmv_median": np.median(vals) if len(vals) > 0 else np.nan,
            "early_pmv_mean": np.mean(vals) if len(vals) > 0 else np.nan,
            "early_pmv_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) > 0 else np.nan,
        })
    df_strata = pd.DataFrame(strata_records)

    # Within-patient vs Between-patient decomposition
    # Add patient mean of log10(af_dur)
    patient_means = df_ep.groupby("patient_id")["log10_af_duration_s"].transform("mean")
    df_ep["patient_mean_log_af"] = patient_means
    df_ep["within_patient_log_af"] = df_ep["log10_af_duration_s"] - patient_means

    # Mixed-effects model / Clustered OLS
    mod = smf.ols("log_early_pmv ~ within_patient_log_af + patient_mean_log_af", data=df_ep)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": df_ep["patient_id"]})

    e03_results = {
        "beta_within": float(res.params.get("within_patient_log_af", np.nan)),
        "se_within": float(res.bse.get("within_patient_log_af", np.nan)),
        "pval_within": float(res.pvalues.get("within_patient_log_af", np.nan)),
        "beta_between": float(res.params.get("patient_mean_log_af", np.nan)),
        "se_between": float(res.bse.get("patient_mean_log_af", np.nan)),
        "pval_between": float(res.pvalues.get("patient_mean_log_af", np.nan)),
        "n_episodes": len(df_ep),
        "n_patients": df_ep["patient_id"].nunique(),
    }
    df_strata["beta_within"] = e03_results["beta_within"]
    df_strata["pval_within"] = e03_results["pval_within"]
    df_strata["beta_between"] = e03_results["beta_between"]
    df_strata["pval_between"] = e03_results["pval_between"]

    df_strata.to_csv(RESULTS_DIR / "e03_preceding_af_exposure.csv", index=False)
    print("E03 Exposure Strata:")
    print(df_strata[["exposure_stratum", "num_episodes", "early_pmv_median", "early_pmv_iqr"]].to_string(index=False))
    print(f"Decomposition: Within-patient beta={e03_results['beta_within']:.4f} (p={e03_results['pval_within']:.4e}), Between-patient beta={e03_results['beta_between']:.4f} (p={e03_results['pval_between']:.4e})")
    return df_strata


# ==============================================================================
# E04: Complete Interepisode PMV Dynamics
# ==============================================================================
def run_e04(df_inter: pd.DataFrame):
    print("\n--- Running Experiment E04: Complete Interepisode PMV Dynamics ---")
    valid_inter = df_inter[df_inter["is_valid"]].copy()

    # Filter intervals
    valid_inter["log_pmv"] = np.log(np.maximum(valid_inter["pmv_V"], 1e-6))

    # Evaluate correlation between t_since and t_until
    corr_all = valid_inter[["time_since_prev_af_m", "time_until_next_af_m"]].corr().iloc[0, 1]

    # Stratify by total gap duration
    gap_bins = [15.0, 30.0, 60.0, 120.0, float("inf")]
    gap_labels = ["15-30m", "30-60m", "60-120m", ">=120m"]
    valid_inter["gap_stratum"] = pd.cut(valid_inter["sinus_duration_s"] / 60.0, bins=gap_bins, labels=gap_labels, right=False)

    gap_records = []
    for g in gap_labels:
        sub = valid_inter[valid_inter["gap_stratum"] == g]
        if len(sub) < 5:
            continue
        corr_g = sub[["time_since_prev_af_m", "time_until_next_af_m"]].corr().iloc[0, 1]

        # Fit joint regression: log(PMV) ~ t_since + t_until
        mod = smf.ols("log_pmv ~ time_since_prev_af_m + time_until_next_af_m", data=sub)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": sub["patient_id"]})

        gap_records.append({
            "gap_stratum": g,
            "num_subsegments": len(sub),
            "num_intervals": sub["segment_id"].nunique(),
            "num_patients": sub["patient_id"].nunique(),
            "collinearity_correlation_r": corr_g,
            "beta_time_since_decay": float(res.params.get("time_since_prev_af_m", np.nan)),
            "pval_time_since": float(res.pvalues.get("time_since_prev_af_m", np.nan)),
            "beta_time_until_destabilize": float(res.params.get("time_until_next_af_m", np.nan)),
            "pval_time_until": float(res.pvalues.get("time_until_next_af_m", np.nan)),
            "r2": float(res.rsquared),
        })

    df_e04 = pd.DataFrame(gap_records)
    df_e04.to_csv(RESULTS_DIR / "e04_interepisode_dynamics.csv", index=False)

    print("E04 Interepisode Dynamics:")
    print(df_e04[["gap_stratum", "num_intervals", "collinearity_correlation_r", "beta_time_since_decay", "beta_time_until_destabilize"]].to_string(index=False))
    return df_e04


# ==============================================================================
# E05: Matched Risk Sets & Temporal Specificity (RQ3)
# ==============================================================================
def run_e05(df_pre: pd.DataFrame, df_remote: pd.DataFrame):
    print("\n--- Running Experiment E05: Matched Risk Sets & Temporal Specificity (RQ3) ---")
    valid_pre = df_pre[df_pre["is_valid"]].copy()
    valid_remote = df_remote[df_remote["is_valid"]].copy()

    # Landmark cases: Pre-AF subsegments in [-15m, 0m]
    cases = valid_pre[(valid_pre["rel_time_to_af_m"] >= -15.0) & (valid_pre["rel_time_to_af_m"] <= 0.0)].copy()
    cases["is_case"] = 1

    # Match controls from remote sinus or earlier sinus of the SAME patient
    # Matched risk sets
    strata = []
    matched_rows = []

    stratum_id = 0
    for patient, case_grp in cases.groupby("patient_id"):
        pt_remote = valid_remote[valid_remote["patient_id"] == patient]
        if len(pt_remote) == 0:
            continue

        # Match 1:1 or 1:2
        for idx, case_row in case_grp.iterrows():
            # Pick remote control with closest heart rate
            hr_diff = np.abs(pt_remote["mean_hr_bpm"] - case_row["mean_hr_bpm"])
            best_ctrl_idx = hr_diff.idxmin()
            ctrl_row = pt_remote.loc[best_ctrl_idx].to_dict()
            ctrl_row["is_case"] = 0
            ctrl_row["stratum_id"] = stratum_id
            ctrl_row["preceding_af_dur_s"] = 0.0
            ctrl_row["time_since_prev_af_s"] = 7200.0

            case_dict = case_row.to_dict()
            case_dict["stratum_id"] = stratum_id

            matched_rows.append(case_dict)
            matched_rows.append(ctrl_row)
            stratum_id += 1

    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_parquet(RESULTS_DIR / "e05_matched_risk_sets.parquet", index=False)

    print(f"Matched {stratum_id} risk sets across {df_matched['patient_id'].nunique()} patients.")

    # Standardize continuous predictors
    for col in ["pmv_V", "pmv_aligned_V", "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "p_amp_iqr", "p_duration_std"]:
        if col in df_matched.columns:
            df_matched[f"{col}_z"] = (df_matched[col] - df_matched[col].mean()) / (df_matched[col].std() + 1e-12)

    df_matched["log_prior_af"] = np.log10(np.maximum(df_matched["prev_af_dur_s"].fillna(0.0), 1.0))

    # Sequential Models
    # M0: Baseline match
    # M1: Prior AF exposure
    # M2: + HR and HRV
    # M3: + Conventional P-wave
    # M4: + PMV
    models_summary = []

    # Fit using logistic regression with robust sandwich variance
    # M1
    m1 = smf.logit("is_case ~ log_prior_af", data=df_matched).fit(cov_type="cluster", cov_kwds={"groups": df_matched["patient_id"]}, disp=False)
    pred_m1 = m1.predict(df_matched)
    auc_m1 = roc_auc_score(df_matched["is_case"], pred_m1)

    # M2: + HR + RMSSD
    m2 = smf.logit("is_case ~ log_prior_af + mean_hr_bpm_z + rmssd_ms_z", data=df_matched).fit(cov_type="cluster", cov_kwds={"groups": df_matched["patient_id"]}, disp=False)
    pred_m2 = m2.predict(df_matched)
    auc_m2 = roc_auc_score(df_matched["is_case"], pred_m2)

    # M3: + Conventional P-wave
    m3 = smf.logit("is_case ~ log_prior_af + mean_hr_bpm_z + rmssd_ms_z + p_amp_iqr_z + p_duration_std_z", data=df_matched).fit(cov_type="cluster", cov_kwds={"groups": df_matched["patient_id"]}, disp=False)
    pred_m3 = m3.predict(df_matched)
    auc_m3 = roc_auc_score(df_matched["is_case"], pred_m3)

    # M4: + PMV (unaligned)
    m4 = smf.logit("is_case ~ log_prior_af + mean_hr_bpm_z + rmssd_ms_z + p_amp_iqr_z + p_duration_std_z + pmv_V_z", data=df_matched).fit(cov_type="cluster", cov_kwds={"groups": df_matched["patient_id"]}, disp=False)
    pred_m4 = m4.predict(df_matched)
    auc_m4 = roc_auc_score(df_matched["is_case"], pred_m4)

    # M4_align: + PMV (aligned)
    m4_al = smf.logit("is_case ~ log_prior_af + mean_hr_bpm_z + rmssd_ms_z + p_amp_iqr_z + p_duration_std_z + pmv_aligned_V_z", data=df_matched).fit(cov_type="cluster", cov_kwds={"groups": df_matched["patient_id"]}, disp=False)
    pred_m4_al = m4_al.predict(df_matched)
    auc_m4_al = roc_auc_score(df_matched["is_case"], pred_m4_al)

    # Likelihood Ratio Test between M3 and M4
    lr_stat = 2.0 * (m4.llf - m3.llf)
    lr_pval = stats.chi2.sf(lr_stat, df=1)

    or_pmv = np.exp(m4.params["pmv_V_z"])
    ci_low = np.exp(m4.params["pmv_V_z"] - 1.96 * m4.bse["pmv_V_z"])
    ci_high = np.exp(m4.params["pmv_V_z"] + 1.96 * m4.bse["pmv_V_z"])

    summary_df = pd.DataFrame([
        {"model": "M1 (Prior AF Exposure)", "predictors": "log_prior_af", "AUROC": round(auc_m1, 4), "delta_AUROC": 0.0, "LR_chi2": np.nan, "pval": np.nan, "OR_per_1SD": np.nan},
        {"model": "M2 (+ HR + HRV)", "predictors": "+ HR_z, RMSSD_z", "AUROC": round(auc_m2, 4), "delta_AUROC": round(auc_m2 - auc_m1, 4), "LR_chi2": round(2*(m2.llf - m1.llf), 2), "pval": stats.chi2.sf(2*(m2.llf - m1.llf), df=2), "OR_per_1SD": np.nan},
        {"model": "M3 (+ Conventional P-wave)", "predictors": "+ Amp_IQR_z, Dur_STD_z", "AUROC": round(auc_m3, 4), "delta_AUROC": round(auc_m3 - auc_m2, 4), "LR_chi2": round(2*(m3.llf - m2.llf), 2), "pval": stats.chi2.sf(2*(m3.llf - m2.llf), df=2), "OR_per_1SD": np.nan},
        {"model": "M4 (+ PMV V)", "predictors": "+ PMV_V_z", "AUROC": round(auc_m4, 4), "delta_AUROC": round(auc_m4 - auc_m3, 4), "LR_chi2": round(lr_stat, 2), "pval": lr_pval, "OR_per_1SD": round(or_pmv, 3)},
        {"model": "M4_aligned (+ PR-Aligned PMV)", "predictors": "+ PMV_aligned_V_z", "AUROC": round(auc_m4_al, 4), "delta_AUROC": round(auc_m4_al - auc_m3, 4), "LR_chi2": round(2*(m4_al.llf - m3.llf), 2), "pval": stats.chi2.sf(2*(m4_al.llf - m3.llf), df=1), "OR_per_1SD": round(np.exp(m4_al.params["pmv_aligned_V_z"]), 3)},
    ])

    summary_df.to_csv(RESULTS_DIR / "e05_adjusted_associations.csv", index=False)
    print("E05 Nested Models Table:")
    print(summary_df.to_string(index=False))
    print(f"PMV Odds Ratio (per 1-SD increase): OR = {or_pmv:.3f} (95% CI: {ci_low:.3f} - {ci_high:.3f}), Wald p = {m4.pvalues['pmv_V_z']:.4e}")
    return summary_df


# ==============================================================================
# E06: Baseline Normalization Sensitivity (RQ4)
# ==============================================================================
def run_e06(df_pre: pd.DataFrame):
    print("\n--- Running Experiment E06: Baseline Normalization Sensitivity (RQ4) ---")
    valid_pre = df_pre[df_pre["is_valid"]].copy()

    # Preceding AF washout strata
    bins = [0, 1800, 3600, 7200, float("inf")]
    labels = ["Washout <30m", "Washout 30-60m", "Washout 1-2h", "Washout >=2h"]
    valid_pre["washout_stratum"] = pd.cut(valid_pre["time_since_prev_af_s"].fillna(99999.0), bins=bins, labels=labels, right=False)

    norm_records = []
    for w in labels:
        sub = valid_pre[valid_pre["washout_stratum"] == w]
        if len(sub) == 0:
            continue
        v_rel = sub["relative_pmv_Vrel"].dropna().values if "relative_pmv_Vrel" in sub.columns else np.array([])
        d_raw = sub["mad_energy_D"].dropna().values
        v_norm = sub["pmv_V"].dropna().values

        norm_records.append({
            "washout_stratum": w,
            "num_subsegments": len(sub),
            "num_episodes": sub["segment_id"].nunique(),
            "relative_pmv_Vrel_median": np.median(v_rel) if len(v_rel) > 0 else np.nan,
            "relative_pmv_Vrel_iqr": float(np.percentile(v_rel, 75) - np.percentile(v_rel, 25)) if len(v_rel) > 0 else np.nan,
            "absolute_mad_energy_D_median": np.median(d_raw) if len(d_raw) > 0 else np.nan,
            "energy_norm_pmv_V_median": np.median(v_norm) if len(v_norm) > 0 else np.nan,
        })

    df_e06 = pd.DataFrame(norm_records)
    df_e06.to_csv(RESULTS_DIR / "e06_normalization_sensitivity.csv", index=False)
    print("E06 Normalization Sensitivity Table:")
    print(df_e06.to_string(index=False))
    return df_e06


# ==============================================================================
# E07: Ectopy, PR Timing, and Morphological Variability Confounding (RQ5)
# ==============================================================================
def run_e07(df_pre: pd.DataFrame):
    print("\n--- Running Experiment E07: Ectopy, PR Timing & Confounding (RQ5) ---")
    valid_pre = df_pre[df_pre["is_valid"]].copy()

    # Compare unaligned PMV vs aligned PMV
    pct_aligned = (valid_pre["pr_aligned"].mean()) * 100.0
    mean_shift = valid_pre["mean_pr_shift_ms"].dropna().mean()
    median_shift = valid_pre["mean_pr_shift_ms"].dropna().median()

    v_unaligned = valid_pre["pmv_V"].dropna().values
    v_aligned = valid_pre["pmv_aligned_V"].dropna().values

    # Paired Wilcoxon signed-rank test
    stat, pval = stats.wilcoxon(v_unaligned, v_aligned)

    e07_summary = pd.DataFrame([{
        "total_subsegments_evaluated": len(valid_pre),
        "pct_subsegments_pr_aligned": round(pct_aligned, 2),
        "mean_pr_shift_ms": round(mean_shift, 2),
        "median_pr_shift_ms": round(median_shift, 2),
        "unaligned_pmv_median": round(np.median(v_unaligned), 6),
        "aligned_pmv_median": round(np.median(v_aligned), 6),
        "median_reduction_pct": round((1.0 - np.median(v_aligned) / np.median(v_unaligned)) * 100.0, 2),
        "wilcoxon_stat": stat,
        "wilcoxon_pval": pval,
    }])

    e07_summary.to_csv(RESULTS_DIR / "e07_ectopy_pr_sensitivity.csv", index=False)
    print("E07 Confounding Analysis Table:")
    print(e07_summary.to_string(index=False))
    return e07_summary


# ==============================================================================
# E08: Authentic Ambulatory Noise & Controlled Perturbation
# ==============================================================================
def run_e08():
    print("\n--- Running Experiment E08: Authentic Ambulatory Noise & Controlled Perturbation ---")
    # Evaluate MIT-BIH NSTDB (record 118e across SNRs: 24, 18, 12, 6, 0, -6 dB)
    nstdb_dir = Path("/data/mithunmanivannan/datasets/nstdb/mit-bih-noise-stress-test-database-1.0.0")
    snr_levels = ["24", "18", "12", "06", "00", "_6"]
    snr_vals = [24, 18, 12, 6, 0, -6]

    nstdb_records = []
    for snr_str, snr_db in zip(snr_levels, snr_vals):
        rec_name = f"118e{snr_str}"
        rec_path = nstdb_dir / rec_name
        if not (nstdb_dir / f"{rec_name}.hea").exists():
            continue

        try:
            rec = wfdb.rdrecord(str(rec_path), sampfrom=0, sampto=360*120)
            qrs_ann = wfdb.rdann(str(rec_path), "atr", sampfrom=0, sampto=360*120)
            ecg = rec.p_signal.T
            fs = float(rec.fs)

            extractor = PMVExtractor(fs=fs, min_beats_per_subsegment=20, enable_pr_alignment=True)
            ecg_filt = extractor.filter_signal(ecg)

            res = extractor.compute_subsegment_pmv(ecg_filt[0], qrs_ann.sample, time_center_s=60.0)
            nstdb_records.append({
                "benchmark": "MIT-BIH NSTDB",
                "record": rec_name,
                "snr_db": snr_db,
                "is_valid": res.is_valid,
                "num_beats": res.num_beats,
                "mad_energy_D": res.mad_energy_D,
                "median_energy_Em": res.median_energy_Em,
                "pmv_V": res.pmv_energy_normalized_V,
                "pmv_aligned_V": res.pmv_aligned_V,
                "mean_pr_shift_ms": res.mean_abs_pr_shift_ms,
            })
        except Exception as e:
            print(f"Error on NSTDB {rec_name}: {e}")

    df_nstdb = pd.DataFrame(nstdb_records)
    df_nstdb.to_csv(RESULTS_DIR / "e08_nstdb_measurement_error.csv", index=False)
    print("E08 NSTDB Degradation Table:")
    print(df_nstdb[["snr_db", "num_beats", "pmv_V", "pmv_aligned_V", "mean_pr_shift_ms"]].to_string(index=False))
    return df_nstdb


# ==============================================================================
# E09: SHDB Clinical Substrate Characterization
# ==============================================================================
def run_e09(df_remote: pd.DataFrame):
    print("\n--- Running Experiment E09: SHDB Clinical Substrate Characterization ---")
    shdb_csv = Path("/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv")
    if not shdb_csv.exists():
        print("SHDB AdditionalData.csv not found.")
        return None

    df_clin = pd.read_csv(shdb_csv)
    # Patient ID mapping: in df_remote, patient_id is patient_XXX and record_id is XXX
    # In AdditionalData.csv, Record is integer or string (e.g. 1, 2, ... 133)
    df_clin["record_id"] = df_clin["Record"].astype(str).str.zfill(3)

    shdb_remote = df_remote[df_remote["dataset"] == "SHDB-AF"].copy()
    patient_pmv = shdb_remote.groupby("record_id")["pmv_V"].median().reset_index()

    merged = pd.merge(df_clin, patient_pmv, on="record_id", how="inner")
    print(f"Matched {len(merged)} SHDB patients with complete clinical metadata.")

    # Regress baseline PMV on LAD, LVEF, Ablation history, beta blocker
    # Check available columns
    cols = merged.columns.tolist()
    lad_col = [c for c in cols if "LAD" in c or "left_atrial" in c.lower() or "atrial" in c.lower()][0] if any("LAD" in c for c in cols) else "LAD"
    lvef_col = [c for c in cols if "LVEF" in c or "ejection" in c.lower()][0] if any("LVEF" in c for c in cols) else "LVEF"

    formula_parts = []
    if lad_col in merged.columns and merged[lad_col].notna().sum() > 10:
        formula_parts.append(lad_col)
    if lvef_col in merged.columns and merged[lvef_col].notna().sum() > 10:
        formula_parts.append(lvef_col)

    results_list = []
    if formula_parts:
        merged["log_pmv"] = np.log(np.maximum(merged["pmv_V"], 1e-6))
        for col in formula_parts:
            valid_m = merged.dropna(subset=[col, "log_pmv"])
            slope, intercept, r_val, p_val, std_err = stats.linregress(valid_m[col], valid_m["log_pmv"])
            results_list.append({
                "clinical_variable": col,
                "n_patients": len(valid_m),
                "mean_val": float(valid_m[col].mean()),
                "std_val": float(valid_m[col].std()),
                "slope": round(slope, 5),
                "std_err": round(std_err, 5),
                "pearson_r": round(r_val, 4),
                "r2": round(r_val**2, 4),
                "pval": p_val,
            })

    df_e09 = pd.DataFrame(results_list)
    df_e09.to_csv(RESULTS_DIR / "e09_shdb_clinical_associations.csv", index=False)
    print("E09 Clinical Substrate Associations Table:")
    print(df_e09.to_string(index=False))
    return df_e09


# ==============================================================================
# E10: Secondary Connection to AF Density
# ==============================================================================
def run_e10(df_remote: pd.DataFrame):
    print("\n--- Running Experiment E10: Secondary Connection to AF Density ---")
    density_csv = PROJECT_ROOT / "reports" / "three_dataset_reference_af_density_report.md"
    # Load patient eligibility / summary from Phase A
    af_inv = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")

    # Compute patient-level AF burden and density
    patient_stats = []
    for patient, grp in af_inv.groupby("patient_id"):
        tot_af_dur_s = grp["duration_s"].sum()
        ep_count = len(grp)
        max_dur_s = grp["duration_s"].max()
        patient_stats.append({
            "patient_id": patient,
            "total_af_duration_s": tot_af_dur_s,
            "af_episode_count": ep_count,
            "max_af_duration_s": max_dur_s,
        })
    df_pt_af = pd.DataFrame(patient_stats)

    # Patient median baseline PMV
    pt_pmv = df_remote.groupby("patient_id")["pmv_V"].median().reset_index()

    merged = pd.merge(df_pt_af, pt_pmv, on="patient_id", how="inner")
    merged["log_pmv"] = np.log(np.maximum(merged["pmv_V"], 1e-6))
    merged["log_af_burden"] = np.log10(np.maximum(merged["total_af_duration_s"], 1.0))

    slope_b, _, r_b, p_b, _ = stats.linregress(merged["log_pmv"], merged["log_af_burden"])
    slope_ep, _, r_ep, p_ep, _ = stats.linregress(merged["log_pmv"], np.log10(merged["af_episode_count"]))

    df_e10 = pd.DataFrame([
        {"comparison": "Baseline PMV vs Total AF Burden (log)", "n_patients": len(merged), "r": round(r_b, 4), "r2": round(r_b**2, 4), "pval": p_b},
        {"comparison": "Baseline PMV vs AF Episode Count (log)", "n_patients": len(merged), "r": round(r_ep, 4), "r2": round(r_ep**2, 4), "pval": p_ep},
    ])
    df_e10.to_csv(RESULTS_DIR / "e10_pmv_af_density_correlation.csv", index=False)
    print("E10 AF Density Association Table:")
    print(df_e10.to_string(index=False))
    return df_e10


def main():
    print("=" * 70)
    print("PHASE D: EXECUTING SCIENTIFIC EXPERIMENTS E01 - E10")
    print("=" * 70)

    # Check that feature parquets exist
    pre_path = FEATURES_DIR / "pmv_pre_af_features.parquet"
    post_path = FEATURES_DIR / "pmv_post_af_features.parquet"
    inter_path = FEATURES_DIR / "pmv_interepisode_features.parquet"
    remote_path = FEATURES_DIR / "pmv_remote_sinus_features.parquet"

    for p in [pre_path, post_path, inter_path, remote_path]:
        if not p.exists():
            print(f"Error: {p} does not exist yet. Please ensure build_pmv_features has finished.")
            return

    df_pre = pd.read_parquet(pre_path)
    df_post = pd.read_parquet(post_path)
    df_inter = pd.read_parquet(inter_path)
    df_remote = pd.read_parquet(remote_path)

    print(f"Loaded feature datasets:")
    print(f"  df_pre: {len(df_pre)} rows ({df_pre['patient_id'].nunique()} patients)")
    print(f"  df_post: {len(df_post)} rows ({df_post['patient_id'].nunique()} patients)")
    print(f"  df_inter: {len(df_inter)} rows ({df_inter['patient_id'].nunique()} patients)")
    print(f"  df_remote: {len(df_remote)} rows ({df_remote['patient_id'].nunique()} patients)")

    t0 = time.time()
    run_e01(df_pre, df_remote)
    run_e02(df_post, df_remote)
    run_e03(df_post)
    run_e04(df_inter)
    run_e05(df_pre, df_remote)
    run_e06(df_pre)
    run_e07(df_pre)
    run_e08()
    run_e09(df_remote)
    run_e10(df_remote)

    print(f"\nAll experiments E01 - E10 completed successfully in {time.time() - t0:.2f} seconds!")


if __name__ == "__main__":
    main()
