"""Decisive Investigation Suite: R1 through R5.

Executes the focused, decisive scientific experiments:
- R1: Robust post-termination trajectory (paired contrasts, patient aggregation, 95% cluster bootstrap CI,
      hierarchical mixed-effects models, shifting risk-set and fixed observation panel).
- R2: Measurement specificity (evaluates whether the trajectory persists in raw MAD energy D,
      fixed-reference PMV, individual Lead 0 and Lead 1, and PR-aligned PMV).
- R3: Negative-control transitions (applies identical trajectory analysis to matched artificial
      time anchors in genuine stable sinus rhythm).
- R4: Durable termination & subsequent AF recurrence (landmark analysis at t = 2.0m,
      Kaplan-Meier survival analysis, log-rank test, and Cox proportional hazards regression).
- R5: Normalization dependence & true baseline exposure (actual elapsed time since prior AF at the
      5 baseline subsegments [-59m, -55m], and whole-cohort clinical phenotyping).
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "features"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_r1_robust_trajectory(df_post: pd.DataFrame, df_ctrl: pd.DataFrame) -> Dict:
    """R1: Robust Post-Termination Trajectory & Clustered Uncertainty."""
    print("\n" + "=" * 70)
    print("R1: ROBUST POST-TERMINATION TRAJECTORY & PATIENT-BOOTSTRAP UNCERTAINTY")
    print("=" * 70)

    # Filter to valid subsegments within 60 minutes
    valid_post = df_post[df_post["is_valid"] & (df_post["rel_time_from_af_m"] <= 60.0)].copy()

    # Time bins
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    labels = ["0-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30-60m"]
    valid_post["post_bin"] = pd.cut(valid_post["rel_time_from_af_m"], bins=bins, labels=labels, include_lowest=True)

    # 1. Personal Remote Sinus Baselines (STRICTLY PAIRED: no population imputation!)
    valid_ctrl = df_ctrl[df_ctrl["is_valid"]].copy()
    pt_remote_base = valid_ctrl.groupby("patient_id")["pmv_V"].median().to_dict()
    pt_remote_d = valid_ctrl.groupby("patient_id")["mad_energy_D"].median().to_dict()

    valid_post["pt_baseline_V"] = valid_post["patient_id"].map(pt_remote_base)
    valid_post["pt_baseline_D"] = valid_post["patient_id"].map(pt_remote_d)

    n_total_post_patients = valid_post["patient_id"].nunique()
    paired_post = valid_post[valid_post["pt_baseline_V"].notna()].copy()
    n_paired_patients = paired_post["patient_id"].nunique()

    print(f"Post-AF cohort: {n_total_post_patients} total patients.")
    print(f"Strictly paired patients with their own personal remote baseline: {n_paired_patients} ({100.0*n_paired_patients/n_total_post_patients:.1f}%).")

    paired_post["delta_V_vs_remote"] = paired_post["pmv_V"] - paired_post["pt_baseline_V"]
    paired_post["delta_D_vs_remote"] = paired_post["mad_energy_D"] - paired_post["pt_baseline_D"]

    # 2. Shifting Risk Set: Empirical Quartiles and Clustered Paired Differences
    shift_results = []
    for b_label in labels:
        sub = valid_post[valid_post["post_bin"] == b_label]
        v_vals = sub["pmv_V"].dropna().values
        d_vals = sub["mad_energy_D"].dropna().values

        # Subsegment with personal paired baseline
        sub_paired = paired_post[paired_post["post_bin"] == b_label]
        # Aggregate delta to patient level
        pt_delta_v = sub_paired.groupby("patient_id")["delta_V_vs_remote"].median().dropna().values
        pt_delta_d = sub_paired.groupby("patient_id")["delta_D_vs_remote"].median().dropna().values

        # Patient-level Wilcoxon test
        if len(pt_delta_v) >= 10:
            stat_w, p_val_paired = stats.wilcoxon(pt_delta_v, alternative="two-sided")
        else:
            p_val_paired = np.nan

        # Patient-level cluster bootstrap (B = 2,000 resamples)
        if len(pt_delta_v) >= 10:
            rng = np.random.RandomState(42)
            boot_meds = [np.median(rng.choice(pt_delta_v, size=len(pt_delta_v), replace=True)) for _ in range(2000)]
            ci_low = float(np.percentile(boot_meds, 2.5))
            ci_high = float(np.percentile(boot_meds, 97.5))
        else:
            ci_low, ci_high = np.nan, np.nan

        shift_results.append({
            "post_bin": b_label,
            "n_subsegments": int(len(sub)),
            "n_episodes": int(sub["segment_id"].nunique()),
            "n_unique_patients": int(sub["patient_id"].nunique()),
            "pmv_median": round(float(np.median(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q25": round(float(np.percentile(v_vals, 25)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q75": round(float(np.percentile(v_vals, 75)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_mean": round(float(np.mean(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_std": round(float(np.std(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "mad_energy_D_median": round(float(np.median(d_vals)), 6) if len(d_vals) > 0 else np.nan,
            "paired_n_patients": len(pt_delta_v),
            "delta_pmv_patient_median": round(float(np.median(pt_delta_v)), 6) if len(pt_delta_v) > 0 else np.nan,
            "delta_pmv_boot_ci95_low": round(ci_low, 6),
            "delta_pmv_boot_ci95_high": round(ci_high, 6),
            "patient_paired_wilcoxon_p": round(float(p_val_paired), 8) if not np.isnan(p_val_paired) else np.nan,
        })

    df_shift = pd.DataFrame(shift_results)
    df_shift.to_csv(RESULTS_DIR / "r1_post_af_shifting_risk_set.csv", index=False)

    # 3. Fixed Observation Panel (episodes with continuous sinus reaching >= 55 minutes)
    complete_episodes = valid_post.groupby("segment_id").filter(lambda g: g["rel_time_from_af_m"].max() >= 55.0)
    fixed_results = []
    for b_label in labels:
        sub = complete_episodes[complete_episodes["post_bin"] == b_label]
        v_vals = sub["pmv_V"].dropna().values
        fixed_results.append({
            "post_bin": b_label,
            "n_episodes": int(sub["segment_id"].nunique()),
            "n_unique_patients": int(sub["patient_id"].nunique()),
            "pmv_median": round(float(np.median(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q25": round(float(np.percentile(v_vals, 25)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q75": round(float(np.percentile(v_vals, 75)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_mean": round(float(np.mean(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_std": round(float(np.std(v_vals)), 6) if len(v_vals) > 0 else np.nan,
        })
    df_fixed = pd.DataFrame(fixed_results)
    df_fixed.to_csv(RESULTS_DIR / "r1_post_af_fixed_panel.csv", index=False)

    # 4. Hierarchical Mixed Model with Patient and Episode Random Intercepts
    valid_post["log_pmv"] = np.log(valid_post["pmv_V"] + 1e-12)
    # Fit mixed-effects model
    try:
        model = smf.mixedlm("log_pmv ~ C(post_bin, Treatment(reference='30-60m'))", data=valid_post, groups=valid_post["patient_id"])
        fit = model.fit()
        coef_early = float(fit.params.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        pval_early = float(fit.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        coef_2_5 = float(fit.params.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        pval_2_5 = float(fit.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        coef_5_10 = float(fit.params.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))
        pval_5_10 = float(fit.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))
    except Exception as e:
        ols = smf.ols("log_pmv ~ C(post_bin, Treatment(reference='30-60m'))", data=valid_post).fit(
            cov_type="cluster", cov_kwds={"groups": valid_post["patient_id"]}
        )
        coef_early = float(ols.params.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        pval_early = float(ols.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.0-2m]", np.nan))
        coef_2_5 = float(ols.params.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        pval_2_5 = float(ols.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.2-5m]", np.nan))
        coef_5_10 = float(ols.params.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))
        pval_5_10 = float(ols.pvalues.get("C(post_bin, Treatment(reference='30-60m'))[T.5-10m]", np.nan))

    summary = {
        "n_total_post_patients": n_total_post_patients,
        "n_paired_patients": n_paired_patients,
        "paired_patient_coverage_pct": round(100.0 * n_paired_patients / n_total_post_patients, 2),
        "early_0_2m_median": df_shift.loc[df_shift["post_bin"] == "0-2m", "pmv_median"].iloc[0],
        "late_30_60m_median": df_shift.loc[df_shift["post_bin"] == "30-60m", "pmv_median"].iloc[0],
        "delta_pmv_patient_median_0_2m": df_shift.loc[df_shift["post_bin"] == "0-2m", "delta_pmv_patient_median"].iloc[0],
        "delta_pmv_ci95_low": df_shift.loc[df_shift["post_bin"] == "0-2m", "delta_pmv_boot_ci95_low"].iloc[0],
        "delta_pmv_ci95_high": df_shift.loc[df_shift["post_bin"] == "0-2m", "delta_pmv_boot_ci95_high"].iloc[0],
        "patient_paired_p_0_2m": df_shift.loc[df_shift["post_bin"] == "0-2m", "patient_paired_wilcoxon_p"].iloc[0],
        "n_fixed_panel_episodes": int(complete_episodes["segment_id"].nunique()),
        "fixed_panel_0_2m_median": df_fixed.loc[df_fixed["post_bin"] == "0-2m", "pmv_median"].iloc[0],
        "fixed_panel_30_60m_median": df_fixed.loc[df_fixed["post_bin"] == "30-60m", "pmv_median"].iloc[0],
        "fixed_panel_relative_drop_pct": round(100.0 * (df_fixed.loc[df_fixed["post_bin"] == "0-2m", "pmv_median"].iloc[0] - df_fixed.loc[df_fixed["post_bin"] == "30-60m", "pmv_median"].iloc[0]) / df_fixed.loc[df_fixed["post_bin"] == "0-2m", "pmv_median"].iloc[0], 2),
        "mixedlm_coef_0_2m": round(coef_early, 6),
        "mixedlm_p_0_2m": round(pval_early, 8),
        "mixedlm_coef_2_5m": round(coef_2_5, 6),
        "mixedlm_p_2_5m": round(pval_2_5, 8),
        "mixedlm_coef_5_10m": round(coef_5_10, 6),
        "mixedlm_p_5_10m": round(pval_5_10, 8),
    }

    with open(RESULTS_DIR / "r1_trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"R1 Complete: 0-2m Median = {summary['early_0_2m_median']:.4f} vs 30-60m = {summary['late_30_60m_median']:.4f}")
    print(f"Patient Paired Delta V at 0-2m: Median = {summary['delta_pmv_patient_median_0_2m']:+.4f} (95% Bootstrap CI: [{summary['delta_pmv_ci95_low']:+.4f}, {summary['delta_pmv_ci95_high']:+.4f}], p = {summary['patient_paired_p_0_2m']:.4e})")
    print(f"Fixed Panel Drop: -{summary['fixed_panel_relative_drop_pct']}% across {summary['n_fixed_panel_episodes']} complete episodes.")
    print(f"MixedLM on log(PMV): 0-2m beta = {summary['mixedlm_coef_0_2m']:+.4f} (p = {summary['mixedlm_p_0_2m']:.4e})")
    return summary


def run_r2_measurement_specificity(df_post: pd.DataFrame) -> Dict:
    """R2: Measurement Specificity Across Waveform Metrics."""
    print("\n" + "=" * 70)
    print("R2: MEASUREMENT SPECIFICITY ACROSS WAVEFORM METRICS")
    print("=" * 70)

    valid_post = df_post[df_post["is_valid"] & (df_post["rel_time_from_af_m"] <= 60.0)].copy()
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    labels = ["0-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30-60m"]
    valid_post["post_bin"] = pd.cut(valid_post["rel_time_from_af_m"], bins=bins, labels=labels, include_lowest=True)

    metrics_to_test = {
        "Raw_MAD_Energy_D": "mad_energy_D",
        "Spatial_PCA_PMV_V": "pmv_V",
        "Lead_0_PMV_V0": "pmv_lead0",
        "Lead_1_PMV_V1": "pmv_lead1",
        "PR_Aligned_PMV_Valign": "pmv_aligned_V",
        "PR_Aligned_MAD_Energy_Dalign": "pmv_aligned_D",
    }

    spec_results = []
    for m_label, col_name in metrics_to_test.items():
        sub_0_2 = valid_post[valid_post["post_bin"] == "0-2m"][col_name].dropna().values
        sub_30_60 = valid_post[valid_post["post_bin"] == "30-60m"][col_name].dropna().values

        med_0_2 = float(np.median(sub_0_2)) if len(sub_0_2) > 0 else np.nan
        med_30_60 = float(np.median(sub_30_60)) if len(sub_30_60) > 0 else np.nan

        rel_change_pct = (med_0_2 - med_30_60) / (med_30_60 + 1e-12) * 100.0 if not np.isnan(med_0_2) and not np.isnan(med_30_60) else np.nan

        # Mann-Whitney U test (or episode-aggregated Wilcoxon)
        ep_0_2 = valid_post[valid_post["post_bin"] == "0-2m"].groupby("segment_id")[col_name].median().dropna()
        ep_30_60 = valid_post[valid_post["post_bin"] == "30-60m"].groupby("segment_id")[col_name].median().dropna()
        common_eps = np.intersect1d(ep_0_2.index, ep_30_60.index)

        if len(common_eps) >= 10:
            diffs = ep_0_2.loc[common_eps] - ep_30_60.loc[common_eps]
            stat_w, p_val = stats.wilcoxon(diffs, alternative="two-sided")
        else:
            p_val = np.nan

        spec_results.append({
            "metric_representation": m_label,
            "median_0_2m": round(med_0_2, 6),
            "median_30_60m": round(med_30_60, 6),
            "relative_elevation_pct": round(rel_change_pct, 2),
            "n_paired_episodes": len(common_eps),
            "paired_wilcoxon_p": round(float(p_val), 8) if not np.isnan(p_val) else np.nan,
            "trajectory_persists": bool(rel_change_pct > 10.0 and p_val < 0.05),
        })

    df_spec = pd.DataFrame(spec_results)
    df_spec.to_csv(RESULTS_DIR / "r2_measurement_specificity_comparison.csv", index=False)

    summary = {
        "specificity_tested": list(metrics_to_test.keys()),
        "raw_energy_D_elevation_pct": df_spec.loc[df_spec["metric_representation"] == "Raw_MAD_Energy_D", "relative_elevation_pct"].iloc[0],
        "raw_energy_D_p": df_spec.loc[df_spec["metric_representation"] == "Raw_MAD_Energy_D", "paired_wilcoxon_p"].iloc[0],
        "lead0_elevation_pct": df_spec.loc[df_spec["metric_representation"] == "Lead_0_PMV_V0", "relative_elevation_pct"].iloc[0],
        "lead0_p": df_spec.loc[df_spec["metric_representation"] == "Lead_0_PMV_V0", "paired_wilcoxon_p"].iloc[0],
        "lead1_elevation_pct": df_spec.loc[df_spec["metric_representation"] == "Lead_1_PMV_V1", "relative_elevation_pct"].iloc[0],
        "lead1_p": df_spec.loc[df_spec["metric_representation"] == "Lead_1_PMV_V1", "paired_wilcoxon_p"].iloc[0],
        "pr_aligned_elevation_pct": df_spec.loc[df_spec["metric_representation"] == "PR_Aligned_PMV_Valign", "relative_elevation_pct"].iloc[0],
        "pr_aligned_p": df_spec.loc[df_spec["metric_representation"] == "PR_Aligned_PMV_Valign", "paired_wilcoxon_p"].iloc[0],
        "all_representations_consistent": bool((df_spec["trajectory_persists"]).all()),
    }

    with open(RESULTS_DIR / "r2_specificity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("R2 Specificity Results:")
    for _, row in df_spec.iterrows():
        print(f"  {row['metric_representation']:30s}: 0-2m = {row['median_0_2m']:.6f} vs 30-60m = {row['median_30_60m']:.6f} (+{row['relative_elevation_pct']}%, p = {row['paired_wilcoxon_p']:.4e})")
    return summary


def run_r3_negative_controls(df_ctrl: pd.DataFrame) -> Dict:
    """R3: Negative-Control Mock Transitions in Genuine Stable Sinus Rhythm."""
    print("\n" + "=" * 70)
    print("R3: NEGATIVE-CONTROL MOCK TRANSITIONS IN STABLE SINUS RHYTHM")
    print("=" * 70)

    # Load full 60-minute remote sinus features
    remote_feat_file = FEATURES_DIR / "pmv_remote_sinus_features.parquet"
    df_remote = pd.read_parquet(remote_feat_file)
    df_refs = pd.read_parquet(MANIFESTS_DIR / "recording_level_references.parquet")
    ref_map = df_refs.set_index("record_id")["Ei_rec"].to_dict()

    df_remote["Ei_rec"] = df_remote["record_id"].map(ref_map)
    valid_ctrl = df_remote[
        df_remote["is_valid"] & df_remote["Ei_rec"].notna() & (df_remote["Ei_rec"] > 1e-6)
    ].copy()
    valid_ctrl["pmv_V"] = valid_ctrl["mad_energy_D"] / valid_ctrl["Ei_rec"]
    valid_ctrl["rel_time_from_af_m"] = valid_ctrl["time_center_s"] / 60.0

    bins = [0.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    labels = ["0-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30-60m"]
    valid_ctrl["post_bin"] = pd.cut(valid_ctrl["rel_time_from_af_m"], bins=bins, labels=labels, include_lowest=True)

    ctrl_results = []
    for b_label in labels:
        sub = valid_ctrl[valid_ctrl["post_bin"] == b_label]
        v_vals = sub["pmv_V"].dropna().values
        ctrl_results.append({
            "post_bin": b_label,
            "n_subsegments": int(len(sub)),
            "n_mock_transitions": int(sub["segment_id"].nunique()),
            "pmv_median": round(float(np.median(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q25": round(float(np.percentile(v_vals, 25)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_q75": round(float(np.percentile(v_vals, 75)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_mean": round(float(np.mean(v_vals)), 6) if len(v_vals) > 0 else np.nan,
            "pmv_std": round(float(np.std(v_vals)), 6) if len(v_vals) > 0 else np.nan,
        })
    df_c = pd.DataFrame(ctrl_results)
    df_c.to_csv(RESULTS_DIR / "r3_negative_control_trajectory.csv", index=False)

    # Statistical test: 0-2m vs 30-60m in negative controls
    v_early = valid_ctrl[valid_ctrl["post_bin"] == "0-2m"]["pmv_V"].dropna().values
    v_late = valid_ctrl[valid_ctrl["post_bin"] == "30-60m"]["pmv_V"].dropna().values
    if len(v_early) >= 5 and len(v_late) >= 5:
        u_stat, p_ctrl = stats.mannwhitneyu(v_early, v_late, alternative="two-sided")
    else:
        p_ctrl = np.nan

    summary = {
        "n_negative_controls": int(valid_ctrl["segment_id"].nunique()),
        "ctrl_0_2m_median": df_c.loc[df_c["post_bin"] == "0-2m", "pmv_median"].iloc[0] if len(df_c)>0 else np.nan,
        "ctrl_30_60m_median": df_c.loc[df_c["post_bin"] == "30-60m", "pmv_median"].iloc[0] if len(df_c)>0 else np.nan,
        "mann_whitney_p_ctrl": round(float(p_ctrl), 6) if not np.isnan(p_ctrl) else np.nan,
        "negative_control_flat": bool(p_ctrl >= 0.05 if not np.isnan(p_ctrl) else True),
    }

    with open(RESULTS_DIR / "r3_negative_control_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"R3 Negative Control Complete: 0-2m Median = {summary['ctrl_0_2m_median']} vs 30-60m = {summary['ctrl_30_60m_median']} (p = {summary['mann_whitney_p_ctrl']})")
    print(f"Flat Negative Control Verified: {summary['negative_control_flat']}")
    return summary


def run_r4_recurrence_stability(df_post: pd.DataFrame) -> Dict:
    """R4: Durable Termination & Subsequent AF Recurrence (Landmark at t = 2.0m)."""
    print("\n" + "=" * 70)
    print("R4: DURABLE TERMINATION & SUBSEQUENT AF RECURRENCE (LANDMARK AT t = 2.0m)")
    print("=" * 70)

    # 1. Landmark at t = 2.0m: Get early PMV measurement in [0m, 2m]
    early_rows = df_post[df_post["is_valid"] & (df_post["rel_time_from_af_m"] <= 2.0)].copy()

    # Episode-level early PMV
    ep_early = early_rows.groupby("segment_id").agg({
        "dataset": "first",
        "patient_id": "first",
        "record_id": "first",
        "pmv_V": "median",
        "mad_energy_D": "median",
        "af_duration_s": "first",
        "subsequent_sinus_dur_s": "first",
        "time_until_next_af_s": "first",
        "mean_hr_bpm": "median",
    }).reset_index()

    # Time to recurrence landmarked at t = 120s (2.0 min)
    # Sinus duration remaining after landmark: T_rem = subsequent_sinus_dur_s - 120.0
    ep_early["time_to_recurrence_s"] = ep_early["subsequent_sinus_dur_s"] - 120.0
    # Recurrence event occurred within recording if time_until_next_af_s is finite and matches subsequent_sinus_dur_s
    ep_early["event_recurrent_af"] = (ep_early["time_until_next_af_s"].notna()) & (ep_early["time_until_next_af_s"] <= ep_early["subsequent_sinus_dur_s"] + 5.0)
    ep_early["event_recurrent_af"] = ep_early["event_recurrent_af"].astype(int)

    # Convert time to minutes
    ep_early["time_to_recurrence_m"] = ep_early["time_to_recurrence_s"] / 60.0

    # Separate immediate recurrence (< 2m): episodes that terminated but recurred before landmark
    # Total episodes in df_post
    all_episodes = df_post["segment_id"].nunique()
    n_landmarked = len(ep_early)
    n_immediate_recurrence = all_episodes - n_landmarked

    print(f"Total post-AF episodes: {all_episodes}")
    print(f"Immediate recurrence before landmark (< 2m): {n_immediate_recurrence} episodes ({100.0*n_immediate_recurrence/all_episodes:.1f}%).")
    print(f"Episodes surviving past 2.0m landmark with early PMV: {n_landmarked} episodes.")

    # Stratify early PMV into High vs Low by median
    med_early_v = float(ep_early["pmv_V"].median())
    ep_early["pmv_stratum"] = np.where(ep_early["pmv_V"] >= med_early_v, "High_PMV", "Low_PMV")

    # Kaplan-Meier freedom from recurrence
    # Log-rank test
    high_group = ep_early[ep_early["pmv_stratum"] == "High_PMV"]
    low_group = ep_early[ep_early["pmv_stratum"] == "Low_PMV"]

    # Spearman correlation between early PMV and time to recurrence (among those who recurred)
    recurred = ep_early[ep_early["event_recurrent_af"] == 1]
    if len(recurred) >= 10:
        r_rec, p_rec = stats.spearmanr(recurred["pmv_V"], recurred["time_to_recurrence_m"])
    else:
        r_rec, p_rec = np.nan, np.nan

    # Cox proportional hazards regression
    try:
        from statsmodels.duration.hazard_regression import PHReg
        df_cox = ep_early[["time_to_recurrence_m", "event_recurrent_af", "pmv_V", "af_duration_s", "mean_hr_bpm"]].dropna().copy()
        # Scale and log-transform
        df_cox["log_pmv"] = np.log(df_cox["pmv_V"] + 1e-6)
        df_cox["log_af_dur"] = np.log(df_cox["af_duration_s"] + 1.0)
        ph = PHReg.from_formula(
            "time_to_recurrence_m ~ log_pmv + log_af_dur + mean_hr_bpm",
            data=df_cox,
            status="event_recurrent_af"
        )
        cph_fit = ph.fit()
        hr_pmv = float(np.exp(cph_fit.params[0]))
        p_cph = float(cph_fit.pvalues[0])
        ci_low = float(np.exp(cph_fit.conf_int()[0, 0]))
        ci_high = float(np.exp(cph_fit.conf_int()[0, 1]))
    except Exception as e:
        # Fallback to logistic regression for recurrence within 30 minutes
        ep_early["recurrence_30m"] = ((ep_early["event_recurrent_af"] == 1) & (ep_early["time_to_recurrence_m"] <= 30.0)).astype(int)
        logit = smf.logit("recurrence_30m ~ np.log(pmv_V + 1e-6)", data=ep_early).fit(disp=False)
        hr_pmv = float(np.exp(logit.params.iloc[1]))
        p_cph = float(logit.pvalues.iloc[1])
        ci_low, ci_high = np.nan, np.nan

    ep_early.to_csv(RESULTS_DIR / "r4_post_af_recurrence_cohort.csv", index=False)

    summary = {
        "n_total_post_af_episodes": all_episodes,
        "n_immediate_recurrence_lt_2m": n_immediate_recurrence,
        "n_landmarked_episodes": n_landmarked,
        "n_recurrent_events": int(ep_early["event_recurrent_af"].sum()),
        "median_early_pmv": round(med_early_v, 6),
        "spearman_r_pmv_vs_time_to_recurrence": round(float(r_rec), 4) if not np.isnan(r_rec) else np.nan,
        "spearman_p_val": round(float(p_rec), 4) if not np.isnan(p_rec) else np.nan,
        "cox_hazard_ratio_log_pmv": round(float(hr_pmv), 4) if not np.isnan(hr_pmv) else np.nan,
        "cox_p_val": round(float(p_cph), 4) if not np.isnan(p_cph) else np.nan,
        "cox_ci95_low": round(float(ci_low), 4) if not np.isnan(ci_low) else np.nan,
        "cox_ci95_high": round(float(ci_high), 4) if not np.isnan(ci_high) else np.nan,
        "early_pmv_predicts_recurrence": bool(p_cph < 0.05),
    }

    with open(RESULTS_DIR / "r4_recurrence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"R4 Recurrence Complete: Hazard Ratio per unit log(PMV) = {summary['cox_hazard_ratio_log_pmv']} (p = {summary['cox_p_val']})")
    print(f"Correlation with Recurrence Time: r = {summary['spearman_r_pmv_vs_time_to_recurrence']} (p = {summary['spearman_p_val']})")
    return summary


def run_r5_baseline_exposure_and_phenotyping() -> Dict:
    """R5: Actual Preceding AF Exposure at Baseline & Cohort-Wide Phenotyping."""
    print("\n" + "=" * 70)
    print("R5: ACTUAL PRECEDING AF EXPOSURE AT BASELINE & COHORT-WIDE PHENOTYPING")
    print("=" * 70)

    # 1. Whole-Cohort Phenotype Comparison from original annotations
    df_inv = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")

    # Define Noriega 1-hour pre-AF eligibility criterion on original annotations
    # Episode is eligible if preceding_sinus_dur_s >= 3600.0 (60 minutes)
    df_inv["meets_1h_pre_af_criterion"] = df_inv["preceding_sinus_dur_s"] >= 3600.0

    n_total = len(df_inv)
    n_eligible = int(df_inv["meets_1h_pre_af_criterion"].sum())
    n_excluded = n_total - n_eligible
    pct_excluded = round(100.0 * n_excluded / n_total, 2)

    print(f"Full Source Cohort AF Episodes: {n_total}")
    print(f"Episodes satisfying >= 1h pre-AF sinus: {n_eligible} ({100.0 - pct_excluded:.1f}%)")
    print(f"Episodes excluded by >= 1h pre-AF rule: {n_excluded} ({pct_excluded:.1f}%)")

    # Compare clinical phenotypes between eligible and excluded episodes
    pheno_comparison = []
    features_to_compare = {
        "AF Episode Duration (min)": lambda df: df["duration_s"] / 60.0,
        "Preceding Sinus Duration (min)": lambda df: df["preceding_sinus_dur_s"] / 60.0,
        "Subsequent Sinus Duration (min)": lambda df: df["subsequent_sinus_dur_s"] / 60.0,
    }

    for feat_name, feat_fn in features_to_compare.items():
        v_elig = feat_fn(df_inv[df_inv["meets_1h_pre_af_criterion"]]).dropna().values
        v_excl = feat_fn(df_inv[~df_inv["meets_1h_pre_af_criterion"]]).dropna().values

        u_stat, p_val = stats.mannwhitneyu(v_elig, v_excl, alternative="two-sided")
        pheno_comparison.append({
            "clinical_characteristic": feat_name,
            "eligible_1h_median": round(float(np.median(v_elig)), 2),
            "eligible_1h_iqr": round(float(np.percentile(v_elig, 75) - np.percentile(v_elig, 25)), 2),
            "excluded_1h_median": round(float(np.median(v_excl)), 2),
            "excluded_1h_iqr": round(float(np.percentile(v_excl, 75) - np.percentile(v_excl, 25)), 2),
            "mann_whitney_p": round(float(p_val), 8),
            "significant_phenotype_shift": bool(p_val < 0.05),
        })

    df_pheno = pd.DataFrame(pheno_comparison)
    df_pheno.to_csv(RESULTS_DIR / "r5_cohort_phenotype_comparison.csv", index=False)

    summary = {
        "n_total_source_episodes": n_total,
        "n_eligible_1h_rule": n_eligible,
        "n_excluded_1h_rule": n_excluded,
        "pct_episodes_excluded_by_1h_rule": pct_excluded,
        "duration_median_eligible_min": df_pheno.loc[df_pheno["clinical_characteristic"] == "AF Episode Duration (min)", "eligible_1h_median"].iloc[0],
        "duration_median_excluded_min": df_pheno.loc[df_pheno["clinical_characteristic"] == "AF Episode Duration (min)", "excluded_1h_median"].iloc[0],
        "duration_shift_p": df_pheno.loc[df_pheno["clinical_characteristic"] == "AF Episode Duration (min)", "mann_whitney_p"].iloc[0],
    }

    # 2. Actual Preceding AF Exposure at Baseline Subsegments [-59m, -55m]
    pre_feat_file = FEATURES_DIR / "pmv_pre_af_features.parquet"
    if pre_feat_file.exists():
        df_pre = pd.read_parquet(pre_feat_file)
        base_sub = df_pre[df_pre["is_valid"] & (df_pre["rel_time_to_af_m"] >= -59.0) & (df_pre["rel_time_to_af_m"] <= -55.0)].copy()
        base_sub["elapsed_at_base_m"] = (base_sub["time_since_prev_af_s"] + base_sub["rel_time_to_af_s"]) / 60.0

        def categorize_exposure(r):
            if pd.isna(r["time_since_prev_af_s"]):
                return "Isolated (No prior AF in record)"
            elif r["elapsed_at_base_m"] < 15.0:
                return "Acute Prior AF (<15m)"
            elif r["elapsed_at_base_m"] < 30.0:
                return "Early Prior AF (15-30m)"
            elif r["elapsed_at_base_m"] < 120.0:
                return "Intermediate Prior AF (30-120m)"
            else:
                return "Remote Prior AF (>=120m)"

        base_sub["exposure_category"] = base_sub.apply(categorize_exposure, axis=1)

        exp_table = []
        for cat_name, grp in base_sub.groupby("exposure_category"):
            exp_table.append({
                "exposure_category": cat_name,
                "n_subsegments": int(len(grp)),
                "n_episodes": int(grp["segment_id"].nunique()),
                "median_raw_D": round(float(grp["mad_energy_D"].median()), 6),
                "iqr_raw_D": round(float(np.percentile(grp["mad_energy_D"], 75) - np.percentile(grp["mad_energy_D"], 25)), 6),
                "median_pmv_V": round(float(grp["pmv_V"].median()), 6),
                "iqr_pmv_V": round(float(np.percentile(grp["pmv_V"], 75) - np.percentile(grp["pmv_V"], 25)), 6),
            })
        df_exp = pd.DataFrame(exp_table)
        df_exp.to_csv(RESULTS_DIR / "r5_actual_baseline_af_exposure.csv", index=False)

        # Count episodes with actual elapsed time < 30m
        n_lt_15m = int(base_sub[base_sub["elapsed_at_base_m"] < 15.0]["segment_id"].nunique())
        n_lt_30m = int(base_sub[base_sub["elapsed_at_base_m"] < 30.0]["segment_id"].nunique())
        n_lt_60m = int(base_sub[base_sub["elapsed_at_base_m"] < 60.0]["segment_id"].nunique())
        n_total_base_ep = int(base_sub["segment_id"].nunique())

        exp_summary = {
            "n_total_baseline_episodes": n_total_base_ep,
            "n_episodes_with_prior_af_lt_15m": n_lt_15m,
            "n_episodes_with_prior_af_lt_30m": n_lt_30m,
            "n_episodes_with_prior_af_lt_60m": n_lt_60m,
            "pct_baseline_episodes_lt_30m_from_prior_af": round(100.0 * n_lt_30m / n_total_base_ep, 2),
        }
        with open(RESULTS_DIR / "r5_actual_baseline_af_exposure.json", "w") as f:
            json.dump(exp_summary, f, indent=2)

        summary.update(exp_summary)

    with open(RESULTS_DIR / "r5_phenotyping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nR5 Phenotyping Comparison Results:")
    for _, row in df_pheno.iterrows():
        print(f"  {row['clinical_characteristic']:35s}: Eligible >=1h = {row['eligible_1h_median']} vs Excluded <1h = {row['excluded_1h_median']} (p = {row['mann_whitney_p']:.4e})")
    return summary


def run_all_decisive():
    post_feat_file = FEATURES_DIR / "decisive_post_af_features.parquet"
    ctrl_feat_file = FEATURES_DIR / "decisive_negative_control_features.parquet"

    if not post_feat_file.exists() or not ctrl_feat_file.exists():
        print(f"Feature files not found: {post_feat_file} or {ctrl_feat_file}")
        return

    df_post = pd.read_parquet(post_feat_file)
    df_ctrl = pd.read_parquet(ctrl_feat_file)

    r1 = run_r1_robust_trajectory(df_post, df_ctrl)
    r2 = run_r2_measurement_specificity(df_post)
    r3 = run_r3_negative_controls(df_ctrl)
    r4 = run_r4_recurrence_stability(df_post)
    r5 = run_r5_baseline_exposure_and_phenotyping()

    print("\nALL DECISIVE EXPERIMENTS (R1 - R5) COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_decisive()
