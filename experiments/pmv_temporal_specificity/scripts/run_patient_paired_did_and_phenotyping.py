"""Patient-Paired Ratio, Difference-in-Differences, and Phenotype Harmonization.

Implements:
1. Exact patient-level paired ratio: V_early / V_late == D_early / D_late within recording.
2. Patient-paired Difference-in-Differences:
   delta_i = (V_AF_early - V_AF_late)_i - (V_ctrl_early - V_ctrl_late)_i
   calculated within the same patient/recording with 95% cluster bootstrap CI.
3. Stratified Eligibility Breakdown:
   Eligibility by dataset (IRIDIA, SHDB, LTAFDB), by episode duration threshold (native, >=30s, >=2m),
   and patient-balanced vs episode-weighted estimates.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "features"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"


def analyze_patient_paired_ratios():
    print("\n1. EXACT PATIENT-LEVEL PAIRED RATIOS (RAW D VS NORMALIZED V)")
    df_post = pd.read_parquet(FEATURES_DIR / "decisive_post_af_features.parquet")
    valid = df_post[df_post["is_valid"] & (df_post["rel_time_from_af_m"] <= 60.0)].copy()

    # Early (center <= 2.0m) and Late (center >= 30.0m)
    early = valid[valid["rel_time_from_af_m"] <= 2.0].groupby("segment_id").agg({
        "patient_id": "first",
        "record_id": "first",
        "mad_energy_D": "median",
        "pmv_V": "median"
    }).rename(columns={"mad_energy_D": "D_early", "pmv_V": "V_early"})

    late = valid[valid["rel_time_from_af_m"] >= 30.0].groupby("segment_id").agg({
        "mad_energy_D": "median",
        "pmv_V": "median"
    }).rename(columns={"mad_energy_D": "D_late", "pmv_V": "V_late"})

    paired = early.join(late, how="inner").dropna()
    print(f"Paired episodes with both early (<=2m) and late (>=30m) windows: {len(paired)} across {paired['patient_id'].nunique()} patients.")

    # Episode-level relative change
    paired["ratio_D"] = paired["D_early"] / paired["D_late"]
    paired["ratio_V"] = paired["V_early"] / paired["V_late"]
    paired["pct_change_D"] = (paired["ratio_D"] - 1.0) * 100.0
    paired["pct_change_V"] = (paired["ratio_V"] - 1.0) * 100.0

    # Aggregate to patient level
    pt_paired = paired.groupby("patient_id").agg({
        "ratio_D": "median",
        "ratio_V": "median",
        "pct_change_D": "median",
        "pct_change_V": "median"
    })

    # Bootstrap CI for patient median paired change
    rng = np.random.RandomState(42)
    boot_pct = [np.median(rng.choice(pt_paired["pct_change_V"], size=len(pt_paired), replace=True)) for _ in range(2000)]
    ci_low = float(np.percentile(boot_pct, 2.5))
    ci_high = float(np.percentile(boot_pct, 97.5))

    stat_w, p_val = stats.wilcoxon(pt_paired["pct_change_V"], alternative="two-sided")

    print(f"Patient-level median ratio D_early / D_late = {pt_paired['ratio_D'].median():.4f}")
    print(f"Patient-level median ratio V_early / V_late = {pt_paired['ratio_V'].median():.4f}")
    print(f"Patient-level median paired elevation = +{pt_paired['pct_change_V'].median():.2f}% (95% CI: [{ci_low:+.2f}%, {ci_high:+.2f}%], Wilcoxon p = {p_val:.4e})")

    res = {
        "n_paired_episodes": len(paired),
        "n_paired_patients": int(paired["patient_id"].nunique()),
        "patient_median_ratio_D": round(float(pt_paired["ratio_D"].median()), 4),
        "patient_median_ratio_V": round(float(pt_paired["ratio_V"].median()), 4),
        "patient_median_pct_elevation": round(float(pt_paired["pct_change_V"].median()), 2),
        "bootstrap_ci95_low": round(ci_low, 2),
        "bootstrap_ci95_high": round(ci_high, 2),
        "wilcoxon_p": round(float(p_val), 8),
    }
    with open(RESULTS_DIR / "patient_paired_ratio_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def analyze_patient_paired_difference_in_differences():
    print("\n2. PATIENT-PAIRED DIFFERENCE-IN-DIFFERENCES (POST-AF VS STABLE SINUS)")
    df_post = pd.read_parquet(FEATURES_DIR / "decisive_post_af_features.parquet")
    df_ctrl = pd.read_parquet(FEATURES_DIR / "pmv_remote_sinus_features.parquet")
    df_refs = pd.read_parquet(MANIFESTS_DIR / "recording_level_references.parquet")
    ref_map = df_refs.set_index("record_id")["Ei_rec"].to_dict()

    df_ctrl["Ei_rec"] = df_ctrl["record_id"].map(ref_map)
    valid_ctrl = df_ctrl[df_ctrl["is_valid"] & df_ctrl["Ei_rec"].notna() & (df_ctrl["Ei_rec"] > 1e-6)].copy()
    valid_ctrl["pmv_V"] = valid_ctrl["mad_energy_D"] / valid_ctrl["Ei_rec"]
    valid_ctrl["rel_time_m"] = valid_ctrl["time_center_s"] / 60.0

    valid_post = df_post[df_post["is_valid"] & (df_post["rel_time_from_af_m"] <= 60.0)].copy()

    # AF temporal delta per patient: Delta_AF = V(0-2m) - V(30-60m)
    af_early = valid_post[valid_post["rel_time_from_af_m"] <= 2.0].groupby(["patient_id", "record_id"])["pmv_V"].median()
    af_late = valid_post[valid_post["rel_time_from_af_m"] >= 30.0].groupby(["patient_id", "record_id"])["pmv_V"].median()
    df_af_delta = (af_early - af_late).dropna().rename("delta_AF").reset_index()

    # Control temporal delta per patient: Delta_ctrl = V(0-2m) - V(30-60m)
    ctrl_early = valid_ctrl[valid_ctrl["rel_time_m"] <= 2.0].groupby(["patient_id", "record_id"])["pmv_V"].median()
    ctrl_late = valid_ctrl[valid_ctrl["rel_time_m"] >= 30.0].groupby(["patient_id", "record_id"])["pmv_V"].median()
    df_ctrl_delta = (ctrl_early - ctrl_late).dropna().rename("delta_ctrl").reset_index()

    # Merge within same patient and recording
    did_merged = pd.merge(df_af_delta, df_ctrl_delta, on=["patient_id", "record_id"], how="inner")
    print(f"Recordings with BOTH post-AF trajectory and complete 60m stable sinus control: {len(did_merged)} across {did_merged['patient_id'].nunique()} patients.")

    did_merged["DiD_delta"] = did_merged["delta_AF"] - did_merged["delta_ctrl"]

    # Aggregate to patient level
    pt_did = did_merged.groupby("patient_id")["DiD_delta"].median()

    # Patient bootstrap CI
    rng = np.random.RandomState(42)
    boot_did = [np.median(rng.choice(pt_did, size=len(pt_did), replace=True)) for _ in range(2000)]
    ci_low = float(np.percentile(boot_did, 2.5))
    ci_high = float(np.percentile(boot_did, 97.5))

    stat_w, p_did = stats.wilcoxon(pt_did, alternative="two-sided")

    print(f"Within-patient Delta_AF median = {did_merged['delta_AF'].median():+.6f}")
    print(f"Within-patient Delta_ctrl median = {did_merged['delta_ctrl'].median():+.6f}")
    print(f"Patient-paired DiD (Delta_AF - Delta_ctrl): Median = {pt_did.median():+.6f} (95% CI: [{ci_low:+.6f}, {ci_high:+.6f}], Wilcoxon p = {p_did:.4e})")

    res = {
        "n_recordings_with_paired_control": len(did_merged),
        "n_patients_with_paired_control": int(did_merged["patient_id"].nunique()),
        "patient_median_delta_AF": round(float(did_merged["delta_AF"].median()), 6),
        "patient_median_delta_ctrl": round(float(did_merged["delta_ctrl"].median()), 6),
        "patient_paired_DiD_median": round(float(pt_did.median()), 6),
        "did_ci95_low": round(ci_low, 6),
        "did_ci95_high": round(ci_high, 6),
        "did_wilcoxon_p": round(float(p_did), 8),
        "did_significant": bool(p_did < 0.05 and ci_low > 0),
    }
    with open(RESULTS_DIR / "patient_paired_did_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def analyze_cohort_eligibility_breakdown():
    print("\n3. STRATIFIED ELIGIBILITY BREAKDOWN BY DATASET AND THRESHOLDS")
    df_inv = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")

    thresholds = [
        ("Native (All runs)", 0.0),
        ("Clinical AF (>=30s)", 30.0),
        ("Sustained AF (>=2m)", 120.0),
    ]

    breakdown = []
    for th_name, min_dur in thresholds:
        sub_df = df_inv[df_inv["duration_s"] >= min_dur].copy()
        for ds in ["IRIDIA-AF", "SHDB-AF", "LTAFDB", "Total Cohort"]:
            if ds == "Total Cohort":
                ds_df = sub_df
            else:
                ds_df = sub_df[sub_df["dataset"] == ds]

            tot_runs = len(ds_df)
            if tot_runs == 0:
                continue

            elig_runs = ds_df["preceding_sinus_dur_s"] >= 3600.0
            n_elig = int(elig_runs.sum())
            pct_elig = n_elig / tot_runs * 100.0

            # Patient-level eligibility: patients with at least 1 eligible run
            tot_pts = ds_df["patient_id"].nunique()
            elig_pts = ds_df[elig_runs]["patient_id"].nunique()
            pct_elig_pts = elig_pts / tot_pts * 100.0 if tot_pts > 0 else 0.0

            breakdown.append({
                "duration_threshold": th_name,
                "dataset": ds,
                "total_runs": tot_runs,
                "eligible_runs_ge_1h": n_elig,
                "pct_eligible_runs": round(pct_elig, 2),
                "total_patients": tot_pts,
                "eligible_patients": elig_pts,
                "pct_eligible_patients": round(pct_elig_pts, 2),
                "median_episode_dur_min": round(float(ds_df["duration_s"].median() / 60.0), 2),
            })

    df_breakdown = pd.DataFrame(breakdown)
    df_breakdown.to_csv(RESULTS_DIR / "cohort_eligibility_stratified_breakdown.csv", index=False)
    print(df_breakdown.to_string())
    return df_breakdown


def main():
    analyze_patient_paired_ratios()
    analyze_patient_paired_difference_in_differences()
    analyze_cohort_eligibility_breakdown()


if __name__ == "__main__":
    main()
