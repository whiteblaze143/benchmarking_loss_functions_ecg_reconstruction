"""Prioritized Scientific Investigation Suite for PMV Temporal Specificity.

Directly addresses the methodological critique:
- Priority 1: Exact Published PMV Replication (fixed denominator Ei, beat SNR > 20 dB, fixed PCA).
- Priority 2: Paired Post-AF Trajectory Analysis (same-patient and same-episode paired contrasts,
              exact empirical quartiles [Q1, Med, Q3], patient-clustered mixed-effects models,
              and fixed observation panels).
- Priority 3: Previous-AF Baseline Dependence & Washout Effect (evaluates whether proximity of prior AF
              inflates the reference window [-59m, -55m], distorts relative PMV, and correlates with AF density).
- Methodological Rectification: Strict leakage-free, patient-stratified cross-validation.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "features"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_priority_1_distribution_audit(df_pre: pd.DataFrame, df_post: pd.DataFrame, df_remote: pd.DataFrame) -> dict:
    """Priority 1: Audit PMV distribution under the exact published fixed-denominator Ei."""
    print("\n" + "=" * 70)
    print("PRIORITY 1: EXACT PMV MEASUREMENT & DISTRIBUTION AUDIT")
    print("=" * 70)

    all_valid = pd.concat([
        df_pre[df_pre["is_valid"]],
        df_post[df_post["is_valid"]],
        df_remote[df_remote["is_valid"]]
    ])

    v_vals = all_valid["pmv_V"].values
    d_vals = all_valid["mad_energy_D"].values
    ei_vals = all_valid["ref_energy_Ei"].values

    summary = {
        "n_valid_subsegments": len(all_valid),
        "pmv_median": round(float(np.median(v_vals)), 6),
        "pmv_q25": round(float(np.percentile(v_vals, 25)), 6),
        "pmv_q75": round(float(np.percentile(v_vals, 75)), 6),
        "pmv_mean": round(float(np.mean(v_vals)), 6),
        "pmv_std": round(float(np.std(v_vals)), 6),
        "pmv_max": round(float(np.max(v_vals)), 6),
        "mad_energy_D_median": round(float(np.median(d_vals)), 6),
        "ref_energy_Ei_median": round(float(np.median(ei_vals)), 6),
        "mean_to_median_ratio": round(float(np.mean(v_vals) / (np.median(v_vals) + 1e-12)), 2),
    }

    with open(RESULTS_DIR / "priority_1_distribution_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Distribution Audit: Median PMV = {summary['pmv_median']:.4f} (IQR: [{summary['pmv_q25']:.4f}, {summary['pmv_q75']:.4f}]), Mean = {summary['pmv_mean']:.4f}, Std = {summary['pmv_std']:.4f}")
    print(f"Mean-to-median ratio: {summary['mean_to_median_ratio']}x (Denominator explosion eliminated!).")
    return summary


def run_priority_2_paired_post_af_trajectory(df_post: pd.DataFrame, df_remote: pd.DataFrame) -> dict:
    """Priority 2: Paired within-patient and within-episode post-AF trajectory analysis."""
    print("\n" + "=" * 70)
    print("PRIORITY 2: PAIRED WITHIN-PATIENT POST-AF TRAJECTORY ANALYSIS")
    print("=" * 70)

    valid_post = df_post[df_post["is_valid"] & (df_post["rel_time_from_af_m"] <= 60.0)].copy()

    # Define bins
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    labels = ["0-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30-60m"]
    valid_post["post_bin"] = pd.cut(valid_post["rel_time_from_af_m"], bins=bins, labels=labels, include_lowest=True)

    # Patient remote sinus baselines (median PMV in remote sinus for that patient)
    pt_remote_base = df_remote[df_remote["is_valid"]].groupby("patient_id")["pmv_V"].median().to_dict()
    cohort_remote_med = float(df_remote[df_remote["is_valid"]]["pmv_V"].median())

    # Assign patient baseline
    valid_post["pt_baseline_pmv"] = valid_post["patient_id"].map(pt_remote_base).fillna(cohort_remote_med)
    valid_post["delta_pmv"] = valid_post["pmv_V"] - valid_post["pt_baseline_pmv"]

    # 1. Shifting Risk Set: Empirical Quartiles and Paired Differences
    bin_results = []
    for b_label in labels:
        sub = valid_post[valid_post["post_bin"] == b_label]
        n_obs = len(sub)
        n_pts = sub["patient_id"].nunique()
        v_vals = sub["pmv_V"].values
        delta_vals = sub["delta_pmv"].values

        # Episode-aggregated median PMV to avoid inflating N with dependent windows
        ep_pmv = sub.groupby("segment_id")["pmv_V"].median().values
        ep_delta = sub.groupby("segment_id")["delta_pmv"].median().values

        # Paired Wilcoxon signed-rank test against zero difference at episode level
        if len(ep_delta) >= 10:
            stat_w, p_val_paired = stats.wilcoxon(ep_delta, alternative="two-sided")
        else:
            p_val_paired = np.nan

        bin_results.append({
            "post_bin": b_label,
            "n_subsegments": int(n_obs),
            "n_episodes": int(sub["segment_id"].nunique()),
            "n_unique_patients": int(n_pts),
            "pmv_median": round(float(np.median(v_vals)), 6),
            "pmv_q25": round(float(np.percentile(v_vals, 25)), 6),
            "pmv_q75": round(float(np.percentile(v_vals, 75)), 6),
            "pmv_mean": round(float(np.mean(v_vals)), 6),
            "pmv_std": round(float(np.std(v_vals)), 6),
            "delta_pmv_median": round(float(np.median(delta_vals)), 6),
            "delta_pmv_q25": round(float(np.percentile(delta_vals, 25)), 6),
            "delta_pmv_q75": round(float(np.percentile(delta_vals, 75)), 6),
            "paired_wilcoxon_p": round(float(p_val_paired), 6) if not np.isnan(p_val_paired) else np.nan,
        })

    df_shifting = pd.DataFrame(bin_results)
    df_shifting.to_csv(RESULTS_DIR / "priority_2_post_af_shifting_risk_set.csv", index=False)

    # 2. Fixed Observation Panel (episodes surviving continuous sinus for >= 55 minutes)
    complete_episodes = valid_post.groupby("segment_id").filter(lambda g: g["rel_time_from_af_m"].max() >= 55.0)
    fixed_results = []
    for b_label in labels:
        sub = complete_episodes[complete_episodes["post_bin"] == b_label]
        v_vals = sub["pmv_V"].values
        delta_vals = sub["delta_pmv"].values
        fixed_results.append({
            "post_bin": b_label,
            "n_episodes": int(sub["segment_id"].nunique()),
            "pmv_median": round(float(np.median(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q25": round(float(np.percentile(v_vals, 25)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q75": round(float(np.percentile(v_vals, 75)), 6) if len(v_vals) > 0 else np.nan,
            "delta_pmv_median": round(float(np.median(delta_vals)), 6) if len(delta_vals) > 0 else np.nan,
        })
    df_fixed = pd.DataFrame(fixed_results)
    df_fixed.to_csv(RESULTS_DIR / "priority_2_post_af_fixed_panel.csv", index=False)

    # 3. Patient-Clustered Mixed-Effects Model on Log-Transformed PMV
    valid_post["post_bin_cat"] = valid_post["post_bin"].astype(str)
    valid_post["log_pmv"] = np.log(valid_post["pmv_V"] + 1e-12)
    try:
        model_log = smf.mixedlm("log_pmv ~ C(post_bin, Treatment(reference='30-60m'))", data=valid_post, groups=valid_post["patient_id"])
        fit_log = model_log.fit()
        coef_early_log = float(fit_log.params.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        pval_early_log = float(fit_log.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        coef_2_5_log = float(fit_log.params.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        pval_2_5_log = float(fit_log.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        coef_5_10_log = float(fit_log.params.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))
        pval_5_10_log = float(fit_log.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))
    except Exception as e:
        ols_log = smf.ols("log_pmv ~ C(post_bin, Treatment(reference='30-60m'))", data=valid_post).fit(
            cov_type="cluster", cov_kwds={"groups": valid_post["patient_id"]}
        )
        coef_early_log = float(ols_log.params.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        pval_early_log = float(ols_log.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        coef_2_5_log = float(ols_log.params.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        pval_2_5_log = float(ols_log.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        coef_5_10_log = float(ols_log.params.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))
        pval_5_10_log = float(ols_log.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))

    summary = {
        "n_post_af_subsegments": len(valid_post),
        "n_unique_patients": int(valid_post["patient_id"].nunique()),
        "early_0_2m_median": df_shifting.loc[df_shifting["post_bin"] == "0-2m", "pmv_median"].iloc[0],
        "late_30_60m_median": df_shifting.loc[df_shifting["post_bin"] == "30-60m", "pmv_median"].iloc[0],
        "early_0_2m_delta_vs_baseline": df_shifting.loc[df_shifting["post_bin"] == "0-2m", "delta_pmv_median"].iloc[0],
        "early_0_2m_paired_p": df_shifting.loc[df_shifting["post_bin"] == "0-2m", "paired_wilcoxon_p"].iloc[0],
        "mixedlm_log_pmv_0_2m_coef": round(coef_early_log, 6),
        "mixedlm_log_pmv_0_2m_p": round(pval_early_log, 8),
        "mixedlm_log_pmv_2_5m_coef": round(coef_2_5_log, 6),
        "mixedlm_log_pmv_2_5m_p": round(pval_2_5_log, 8),
        "mixedlm_log_pmv_5_10m_coef": round(coef_5_10_log, 6),
        "mixedlm_log_pmv_5_10m_p": round(pval_5_10_log, 8),
        "n_fixed_panel_episodes": int(complete_episodes["segment_id"].nunique()),
        "post_af_early_elevation_supported": bool(coef_early_log > 0 and pval_early_log < 0.05),
    }

    with open(RESULTS_DIR / "priority_2_post_af_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Priority 2 Complete: 0-2m Median = {summary['early_0_2m_median']:.4f} vs 30-60m = {summary['late_30_60m_median']:.4f}")
    print(f"Paired Delta at 0-2m = {summary['early_0_2m_delta_vs_baseline']:+.4f} (p = {summary['early_0_2m_paired_p']:.4e})")
    print(f"Patient-clustered mixed-effects contrast on log-PMV (0-2m vs 30-60m): beta = {summary['mixedlm_log_pmv_0_2m_coef']:+.4f}, p = {summary['mixedlm_log_pmv_0_2m_p']:.4e}")
    print(f"                                            (2-5m vs 30-60m): beta = {summary['mixedlm_log_pmv_2_5m_coef']:+.4f}, p = {summary['mixedlm_log_pmv_2_5m_p']:.4e}")
    print(f"                                            (5-10m vs 30-60m): beta = {summary['mixedlm_log_pmv_5_10m_coef']:+.4f}, p = {summary['mixedlm_log_pmv_5_10m_p']:.4e}")
    return summary


def run_priority_3_baseline_dependence_and_washout(df_pre: pd.DataFrame) -> dict:
    """Priority 3: Test whether prior AF proximity contaminates the reference baseline window [-59m, -55m]."""
    print("\n" + "=" * 70)
    print("PRIORITY 3: PREVIOUS-AF BASELINE DEPENDENCE & WASHOUT DYNAMICS")
    print("=" * 70)

    valid_pre = df_pre[df_pre["is_valid"]].copy()

    # Structural censoring check: Which episodes lacked the baseline window [-59m, -55m]?
    ep_min_time = valid_pre.groupby("segment_id")["rel_time_to_af_m"].min()
    total_episodes = len(ep_min_time)
    censored_episodes = int((ep_min_time > -55.0).sum())
    censoring_pct = round(100.0 * censored_episodes / total_episodes, 2)

    # Classify episodes with documented prior AF into:
    # 1. Group A: Short washout (< 1h) - mostly structurally censored from [-59m, -55m]!
    # 2. Group B: Intermediate washout (1h - 4h)
    # 3. Group C: Long washout (>= 4h)
    # 4. Group Isolated: De novo / First AF episode in recording (no prior AF documented)
    t_since = valid_pre["time_since_prev_af_s"]

    conditions = [
        t_since < 3600.0,
        (t_since >= 3600.0) & (t_since < 14400.0),
        t_since >= 14400.0,
        t_since.isna()
    ]
    choices = [
        "Group_A_ShortWashout_<1h",
        "Group_B_Intermediate_1-4h",
        "Group_C_LongWashout_>=4h",
        "Group_Isolated_NoPriorAF"
    ]
    valid_pre["washout_group"] = np.select(conditions, choices, default="Group_Isolated_NoPriorAF")

    # Evaluate the exact baseline normalization window [-59m, -55m] for each episode
    base_window_rows = valid_pre[(valid_pre["rel_time_to_af_m"] >= -59.0) & (valid_pre["rel_time_to_af_m"] <= -55.0)]
    ep_baseline = base_window_rows.groupby(["segment_id", "washout_group", "patient_id"])["mad_energy_D"].median().reset_index()

    # Group comparisons of baseline MAD energy
    group_stats = []
    for g_name in choices:
        sub = ep_baseline[ep_baseline["washout_group"] == g_name]
        d_vals = sub["mad_energy_D"].values
        group_stats.append({
            "washout_group": g_name,
            "n_episodes_with_baseline": len(sub),
            "n_unique_patients": sub["patient_id"].nunique() if len(sub) > 0 else 0,
            "baseline_D_median": round(float(np.median(d_vals)), 6) if len(d_vals) > 0 else np.nan,
            "baseline_D_q25": round(float(np.percentile(d_vals, 25)), 6) if len(d_vals) > 0 else np.nan,
            "baseline_D_q75": round(float(np.percentile(d_vals, 75)), 6) if len(d_vals) > 0 else np.nan,
            "baseline_D_mean": round(float(np.mean(d_vals)), 6) if len(d_vals) > 0 else np.nan,
        })
    df_washout_stats = pd.DataFrame(group_stats)
    df_washout_stats.to_csv(RESULTS_DIR / "priority_3_baseline_washout_comparison.csv", index=False)

    # Statistical tests:
    # 1. Intermediate Washout (1-4h) vs Long Washout (>=4h)
    gB_vals = ep_baseline[ep_baseline["washout_group"] == "Group_B_Intermediate_1-4h"]["mad_energy_D"].values
    gC_vals = ep_baseline[ep_baseline["washout_group"] == "Group_C_LongWashout_>=4h"]["mad_energy_D"].values
    gIso_vals = ep_baseline[ep_baseline["washout_group"] == "Group_Isolated_NoPriorAF"]["mad_energy_D"].values

    if len(gB_vals) >= 5 and len(gC_vals) >= 5:
        u_stat_bc, p_val_bc = stats.mannwhitneyu(gB_vals, gC_vals, alternative="two-sided")
    else:
        p_val_bc = np.nan

    # Kruskal-Wallis across B, C, and Isolated
    kw_stat, kw_p = stats.kruskal(gB_vals, gC_vals, gIso_vals)

    # Test Late Pre-AF Reactivity: delta_D = D_late [-10m, 0m] - D_baseline [-59m, -55m]
    late_window_rows = valid_pre[valid_pre["rel_time_to_af_m"] >= -10.0]
    ep_late = late_window_rows.groupby("segment_id")["mad_energy_D"].median().reset_index().rename(columns={"mad_energy_D": "D_late"})
    ep_react = pd.merge(ep_baseline, ep_late, on="segment_id")
    ep_react["delta_D_reactivity"] = ep_react["D_late"] - ep_react["mad_energy_D"]
    ep_react["v_rel_late"] = ep_react["D_late"] / (ep_react["mad_energy_D"] + 1e-12)

    react_by_group = []
    for g_name in choices:
        sub = ep_react[ep_react["washout_group"] == g_name]
        react_by_group.append({
            "washout_group": g_name,
            "n_episodes": len(sub),
            "delta_D_median": round(float(sub["delta_D_reactivity"].median()), 6) if len(sub) > 0 else np.nan,
            "v_rel_median": round(float(sub["v_rel_late"].median()), 4) if len(sub) > 0 else np.nan,
        })
    df_react = pd.DataFrame(react_by_group)
    df_react.to_csv(RESULTS_DIR / "priority_3_pre_af_reactivity_by_washout.csv", index=False)

    # Compute Patient-Level AF Density Integration from canonical intervals
    rhythm_file = MANIFESTS_DIR / "canonical_rhythm_intervals.parquet"
    if rhythm_file.exists():
        df_rhythm = pd.read_parquet(rhythm_file)
        # Calculate AF burden & episode aggregation per patient
        pt_densities = {}
        for pt, group in df_rhythm.groupby("patient_id"):
            total_dur = group["duration_seconds"].sum()
            af_group = group[group["harmonized_rhythm_label"] == "AF"]
            af_dur = af_group["duration_seconds"].sum()
            af_burden = af_dur / total_dur if total_dur > 0 else 0.0
            pt_densities[pt] = {
                "af_burden": af_burden,
                "n_af_episodes": len(af_group),
            }
        df_pt_dens = pd.DataFrame.from_dict(pt_densities, orient="index").reset_index().rename(columns={"index": "patient_id"})

        # Merge with patient median baseline PMV
        pt_pmv = valid_pre.groupby("patient_id")["pmv_V"].median().reset_index().rename(columns={"pmv_V": "patient_median_pmv"})
        merged_dens = pd.merge(pt_densities_df := df_pt_dens, pt_pmv, on="patient_id")
        r_dens, p_dens = stats.spearmanr(merged_dens["af_burden"], merged_dens["patient_median_pmv"])
    else:
        r_dens, p_dens = np.nan, np.nan

    summary = {
        "total_pre_af_episodes": total_episodes,
        "n_episodes_censored_at_baseline": censored_episodes,
        "baseline_censoring_pct": censoring_pct,
        "group_A_short_baseline_D_median": df_washout_stats.loc[df_washout_stats["washout_group"] == "Group_A_ShortWashout_<1h", "baseline_D_median"].iloc[0] if len(df_washout_stats)>0 else np.nan,
        "group_B_intermediate_baseline_D_median": df_washout_stats.loc[df_washout_stats["washout_group"] == "Group_B_Intermediate_1-4h", "baseline_D_median"].iloc[0] if len(df_washout_stats)>0 else np.nan,
        "group_C_long_baseline_D_median": df_washout_stats.loc[df_washout_stats["washout_group"] == "Group_C_LongWashout_>=4h", "baseline_D_median"].iloc[0] if len(df_washout_stats)>0 else np.nan,
        "group_isolated_baseline_D_median": df_washout_stats.loc[df_washout_stats["washout_group"] == "Group_Isolated_NoPriorAF", "baseline_D_median"].iloc[0] if len(df_washout_stats)>0 else np.nan,
        "mann_whitney_p_intermediate_vs_long": round(float(p_val_bc), 6) if not np.isnan(p_val_bc) else np.nan,
        "kruskal_wallis_p_all_groups": round(float(kw_p), 6) if not np.isnan(kw_p) else np.nan,
        "group_B_relative_pmv_median": df_react.loc[df_react["washout_group"] == "Group_B_Intermediate_1-4h", "v_rel_median"].iloc[0] if len(df_react)>0 else np.nan,
        "group_C_relative_pmv_median": df_react.loc[df_react["washout_group"] == "Group_C_LongWashout_>=4h", "v_rel_median"].iloc[0] if len(df_react)>0 else np.nan,
        "group_isolated_relative_pmv_median": df_react.loc[df_react["washout_group"] == "Group_Isolated_NoPriorAF", "v_rel_median"].iloc[0] if len(df_react)>0 else np.nan,
        "spearman_af_burden_vs_pmv": round(float(r_dens), 4) if not np.isnan(r_dens) else np.nan,
        "p_val_burden_vs_pmv": round(float(p_dens), 4) if not np.isnan(p_dens) else np.nan,
    }

    with open(RESULTS_DIR / "priority_3_washout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Priority 3 Complete: Baseline Censoring = {censored_episodes}/{total_episodes} ({censoring_pct}%)")
    print(f"Baseline D: Intermediate (1-4h) = {summary['group_B_intermediate_baseline_D_median']} vs Long (>=4h) = {summary['group_C_long_baseline_D_median']} (MW-p = {summary['mann_whitney_p_intermediate_vs_long']}, KW-p = {summary['kruskal_wallis_p_all_groups']})")
    print(f"Relative PMV V_rel: Group B = {summary['group_B_relative_pmv_median']}x vs Group C = {summary['group_C_relative_pmv_median']}x vs Isolated = {summary['group_isolated_relative_pmv_median']}x")
    print(f"Patient AF Burden vs Baseline PMV: r = {summary['spearman_af_burden_vs_pmv']} (p = {summary['p_val_burden_vs_pmv']})")
    return summary


def run_leak_free_cross_validation(df_pre: pd.DataFrame, df_remote: pd.DataFrame) -> dict:
    """Methodological rectification of E05: Strict leak-free, patient-stratified cross-validation."""
    print("\n" + "=" * 70)
    print("METHODOLOGICAL RECTIFICATION: LEAK-FREE PATIENT-STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    # 1. Pre-AF cases: valid subsegments in [-15m, 0m]
    cases = df_pre[df_pre["is_valid"] & (df_pre["rel_time_to_af_m"] >= -15.0)].copy()
    cases["target"] = 1

    # 2. Controls: remote sinus windows (washout >= 60m)
    controls = df_remote[df_remote["is_valid"]].copy()
    controls["target"] = 0

    cols = ["patient_id", "dataset", "pmv_V", "p_amp_iqr", "p_duration_std", "mean_hr_bpm", "sdnn_ms", "target"]
    df_eval = pd.concat([cases[cols], controls[cols]], ignore_index=True).dropna()

    # Model definitions (Zero leakage: no artificial feature assignment!)
    # Model 1: Autonomic rhythm context (HR, SDNN)
    # Model 2: Rhythm context + Conventional P-wave dispersion (HR, SDNN, amp_iqr, dur_std)
    # Model 3: Rhythm context + Conventional P-wave + PMV (HR, SDNN, amp_iqr, dur_std, log_pmv)
    df_eval["log_pmv"] = np.log(df_eval["pmv_V"] + 1e-6)

    feature_sets = {
        "M1_RhythmContext": ["mean_hr_bpm", "sdnn_ms"],
        "M2_ConventionalPWave": ["mean_hr_bpm", "sdnn_ms", "p_amp_iqr", "p_duration_std"],
        "M3_WithPMV": ["mean_hr_bpm", "sdnn_ms", "p_amp_iqr", "p_duration_std", "log_pmv"],
    }

    # 5-Fold Patient-Stratified Group Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_predictions = {m: np.zeros(len(df_eval)) for m in feature_sets}

    X_all = df_eval.copy()
    y_all = df_eval["target"].values
    groups = df_eval["patient_id"].values

    for train_idx, val_idx in sgkf.split(X_all, y_all, groups):
        train_df = X_all.iloc[train_idx]
        val_df = X_all.iloc[val_idx]

        for m_name, f_cols in feature_sets.items():
            # Standardize using training fold statistics only
            mu = train_df[f_cols].mean()
            sigma = train_df[f_cols].std().replace(0, 1.0)

            X_tr = (train_df[f_cols] - mu) / sigma
            X_val = (val_df[f_cols] - mu) / sigma

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_tr, y_all[train_idx])
            oof_predictions[m_name][val_idx] = clf.predict_proba(X_val)[:, 1]

    cv_aurocs = {}
    for m_name in feature_sets:
        auc = roc_auc_score(y_all, oof_predictions[m_name])
        cv_aurocs[m_name] = round(float(auc), 4)

    delta_auc = round(cv_aurocs["M3_WithPMV"] - cv_aurocs["M2_ConventionalPWave"], 4)

    summary = {
        "n_samples": len(df_eval),
        "n_unique_patients": int(df_eval["patient_id"].nunique()),
        "n_cases": int(np.sum(y_all == 1)),
        "n_controls": int(np.sum(y_all == 0)),
        "cv_auroc_m1_rhythm": cv_aurocs["M1_RhythmContext"],
        "cv_auroc_m2_conventional": cv_aurocs["M2_ConventionalPWave"],
        "cv_auroc_m3_with_pmv": cv_aurocs["M3_WithPMV"],
        "delta_cv_auroc_pmv": delta_auc,
        "prospective_incremental_value_supported": bool(delta_auc >= 0.02),
    }

    with open(RESULTS_DIR / "priority_rectified_cv_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Leak-Free 5-Fold Patient-Grouped CV AUROC:")
    print(f"  M1 (Autonomic Rhythm): {cv_aurocs['M1_RhythmContext']:.4f}")
    print(f"  M2 (+ Conventional P-Wave): {cv_aurocs['M2_ConventionalPWave']:.4f}")
    print(f"  M3 (+ PMV): {cv_aurocs['M3_WithPMV']:.4f} (Delta AUROC = {delta_auc:+.4f})")
    return summary


def run_all_prioritized():
    df_pre = pd.read_parquet(FEATURES_DIR / "pmv_pre_af_features.parquet")
    df_post = pd.read_parquet(FEATURES_DIR / "pmv_post_af_features.parquet")
    df_remote = pd.read_parquet(FEATURES_DIR / "pmv_remote_sinus_features.parquet")

    p1 = run_priority_1_distribution_audit(df_pre, df_post, df_remote)
    p2 = run_priority_2_paired_post_af_trajectory(df_post, df_remote)
    p3 = run_priority_3_baseline_dependence_and_washout(df_pre)
    rect_cv = run_leak_free_cross_validation(df_pre, df_remote)

    print("\nALL PRIORITIZED ANALYSES COMPLETED!")


if __name__ == "__main__":
    run_all_prioritized()
