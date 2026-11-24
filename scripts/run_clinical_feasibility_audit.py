#!/usr/bin/env python3
"""Clinical Feasibility Audit: Episode Reinitiation, 6-Minute Trial Discordance,
SHDB Phenotypes, LTAFDB Fragmentation, and Follow-Up Adequacy.

Resolves the 5 foundational reference feasibility questions before committing to new model training:
1. AF reinitiation within 30s, 2m, 6m, 30m, and 1h of termination.
2. Discordance between cumulative AF exposure and the October 2026 HRS/AFA-ARC 6-minute trial endpoint.
3. Completeness of clinical phenotype, echocardiography, and medication data in SHDB-AF.
4. Concentration of LTAFDB extreme fragmentation and identity of competing rhythms during gaps.
5. Adequacy of post-termination observation follow-up to distinguish durable termination from right-censoring.
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/data/mithunmanivannan/datasets")
LTAF_DIR = DATA_DIR / "ltafdb_1.0.0_reference"
SHDB_DIR = DATA_DIR / "shdb_af"
IRIDIA_DIR = DATA_DIR / "iridia_af_8405941_v1.0.1"
REPORTS_DIR = ROOT / "reports"
OUT_DIR = ROOT / "experiments/af_entrenchment/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_q1_and_q5_reinitiation_and_censoring(df_ep):
    """Q1: AF reinitiation horizons post-termination.
    Q5: Censoring and follow-up adequacy.
    """
    print("\n" + "=" * 80)
    print("QUESTION 1 & 5: TERMINATION DURABILITY, REINITIATION & FOLLOW-UP ADEQUACY")
    print("=" * 80)

    # Sort episodes by record and start time
    df = df_ep.sort_values(["dataset", "record_id", "start_s"]).copy()
    
    # Identify terminating episodes (not right-censored at recording end)
    df_term = df[df["event"] == 1].copy()
    print(f"Total terminating episodes with observed stop: {len(df_term)} (out of {len(df)})")

    # Compute gap to next episode in the same record
    df["next_start_s"] = df.groupby(["dataset", "record_id"])["start_s"].shift(-1)
    df["gap_to_next_s"] = df["next_start_s"] - df["end_s"]
    df["has_subsequent_af"] = ~df["next_start_s"].isna()
    df["post_term_obs_s"] = df["rec_duration_s"] - df["end_s"]

    # Filter to terminating episodes
    df_term = df[df["event"] == 1].copy()

    # Reinitiation horizons
    horizons = [
        ("30s", 30.0),
        ("2m", 120.0),
        ("6m", 360.0),
        ("30m", 1800.0),
        ("1h", 3600.0),
    ]

    q1_episode_results = []
    q1_patient_results = []

    for label, h_sec in horizons:
        # Episode-level: reinitiates within h_sec
        reinit_ep = df_term["has_subsequent_af"] & (df_term["gap_to_next_s"] <= h_sec)
        pct_ep = reinit_ep.mean() * 100.0
        n_ep = reinit_ep.sum()

        # Patient-level: patient has AT LEAST ONE reinitiation within h_sec
        pts_with_reinit = df_term[reinit_ep]["patient_id"].nunique()
        total_pts = df_term["patient_id"].nunique()
        pct_pt = pts_with_reinit / total_pts * 100.0

        q1_episode_results.append({
            "horizon": label,
            "horizon_seconds": h_sec,
            "reinitiating_episodes": int(n_ep),
            "total_terminating_episodes": len(df_term),
            "reinitiation_rate_episodes_pct": round(pct_ep, 2),
            "reinitiating_patients": int(pts_with_reinit),
            "total_patients": total_pts,
            "reinitiation_rate_patients_pct": round(pct_pt, 2)
        })

    df_q1_summary = pd.DataFrame(q1_episode_results)
    print("\nOverall AF Reinitiation Horizons Post-Termination:")
    print(df_q1_summary.to_string(index=False))

    # Cohort-stratified reinitiation
    cohort_reinit = []
    for d_name, d_df in df_term.groupby("dataset"):
        row = {"dataset": d_name, "total_terminating_episodes": len(d_df), "total_patients": d_df["patient_id"].nunique()}
        for label, h_sec in horizons:
            reinit_m = d_df["has_subsequent_af"] & (d_df["gap_to_next_s"] <= h_sec)
            pts = d_df[reinit_m]["patient_id"].nunique()
            row[f"ep_reinit_{label}_pct"] = round(reinit_m.mean() * 100.0, 2)
            row[f"pt_reinit_{label}"] = pts
            row[f"pt_reinit_{label}_pct"] = round(pts / row["total_patients"] * 100.0, 2)
        cohort_reinit.append(row)

    df_cohort_reinit = pd.DataFrame(cohort_reinit)
    print("\nCohort-Stratified Reinitiation Horizons:")
    print(df_cohort_reinit.to_string(index=False))

    # Reinitiation by prior episode duration (Does longer AF lead to durable or transient termination?)
    dur_bins = [0, 30, 120, 360, 1800, 3600, np.inf]
    dur_labels = ["<30s", "30s-2m", "2m-6m", "6m-30m", "30m-1h", ">=1h"]
    df_term["dur_bin"] = pd.cut(df_term["duration_s"], bins=dur_bins, labels=dur_labels, right=False)

    reinit_by_dur = df_term.groupby("dur_bin", observed=True).agg(
        n_episodes=("duration_s", "count"),
        n_patients=("patient_id", "nunique"),
        pct_reinit_30s=("gap_to_next_s", lambda g: np.mean((g <= 30.0) & df_term.loc[g.index, "has_subsequent_af"]) * 100.0),
        pct_reinit_2m=("gap_to_next_s", lambda g: np.mean((g <= 120.0) & df_term.loc[g.index, "has_subsequent_af"]) * 100.0),
        pct_reinit_6m=("gap_to_next_s", lambda g: np.mean((g <= 360.0) & df_term.loc[g.index, "has_subsequent_af"]) * 100.0),
        pct_reinit_1h=("gap_to_next_s", lambda g: np.mean((g <= 3600.0) & df_term.loc[g.index, "has_subsequent_af"]) * 100.0),
        pct_durable_term_rem=("has_subsequent_af", lambda s: np.mean(~s) * 100.0)
    ).reset_index()

    print("\nReinitiation Likelihood by Prior Episode Duration:")
    print(reinit_by_dur.to_string(index=False))

    # Q5: Follow-up adequacy
    # Last episode in each recording: does it have adequate post-termination follow-up?
    last_episodes = df_term[~df_term["has_subsequent_af"]].copy()
    print(f"\nLast Terminating Episode in Recording (Durable Candidates): {len(last_episodes)} episodes")
    followup_thresholds = [("30s", 30.0), ("2m", 120.0), ("6m", 360.0), ("30m", 1800.0), ("1h", 3600.0), ("6h", 21600.0)]
    
    q5_results = []
    for label, h_sec in followup_thresholds:
        adequate = last_episodes["post_term_obs_s"] >= h_sec
        q5_results.append({
            "followup_threshold": label,
            "threshold_seconds": h_sec,
            "episodes_with_sufficient_followup": int(adequate.sum()),
            "total_candidate_episodes": len(last_episodes),
            "percentage_sufficient": round(adequate.mean() * 100.0, 2),
            "right_censored_potential_pct": round((1.0 - adequate.mean()) * 100.0, 2)
        })

    df_q5_summary = pd.DataFrame(q5_results)
    print("\nFollow-up Adequacy for Apparent Durable Terminations:")
    print(df_q5_summary.to_string(index=False))

    return df_q1_summary, df_cohort_reinit, reinit_by_dur, df_q5_summary, df


def audit_q2_arc_6min_discordance(df_ep):
    """Q2: Discordance between cumulative AF exposure and the October 2026 HRS/AFA-ARC 6-minute trial endpoint."""
    print("\n" + "=" * 80)
    print("QUESTION 2: DISCORDANCE WITH OCTOBER 2026 HRS/AFA-ARC 6-MINUTE ENDPOINT")
    print("=" * 80)

    # Per-recording aggregation
    rec_agg = df_ep.groupby(["dataset", "record_id", "patient_id"]).agg(
        rec_duration_h=("rec_duration_s", lambda x: x.iloc[0] / 3600.0),
        total_af_episodes=("duration_s", "count"),
        total_af_hours=("duration_s", lambda x: x.sum() / 3600.0),
        max_episode_dur_s=("duration_s", "max"),
        has_ep_ge_30s=("duration_s", lambda x: (x >= 30.0).any()),
        has_ep_ge_6m=("duration_s", lambda x: (x >= 360.0).any()),
        n_ep_ge_30s=("duration_s", lambda x: (x >= 30.0).sum()),
        n_ep_ge_6m=("duration_s", lambda x: (x >= 360.0).sum()),
        burden_ge_30s_h=("duration_s", lambda x: x[x >= 30.0].sum() / 3600.0)
    ).reset_index()

    rec_agg["af_burden_pct"] = (rec_agg["total_af_hours"] / rec_agg["rec_duration_h"]) * 100.0

    # Categorize trial endpoint status:
    # Category 1: Recurrence under BOTH (has >= 30s AND has >= 6m)
    # Category 2: DISCORDANT (has >= 30s BUT NO >= 6m) -> Declared "Success" by 2026 ARC, but "Recurrence" by traditional trials!
    # Category 3: Micro-AF only (has < 30s only)
    # Category 4: No AF
    rec_agg["trial_category"] = "No_AF"
    rec_agg.loc[rec_agg["total_af_episodes"] > 0, "trial_category"] = "Micro_AF_Only_lt30s"
    rec_agg.loc[rec_agg["has_ep_ge_30s"], "trial_category"] = "DISCORDANT_30s_to_6m"
    rec_agg.loc[rec_agg["has_ep_ge_6m"], "trial_category"] = "Recurrence_Both_ge6m"

    print("\nTrial Classification Across All 215 AF Recordings:")
    cat_summary = rec_agg.groupby("trial_category").agg(
        n_recordings=("record_id", "count"),
        n_patients=("patient_id", "nunique"),
        mean_burden_pct=("af_burden_pct", "mean"),
        median_burden_pct=("af_burden_pct", "median"),
        mean_af_hours=("total_af_hours", "mean"),
        total_af_hours=("total_af_hours", "sum"),
        mean_episodes=("total_af_episodes", "mean"),
        median_episodes=("total_af_episodes", "median"),
        max_ep_dur_s_median=("max_episode_dur_s", "median")
    ).reset_index()
    print(cat_summary.to_string(index=False))

    # Cohort breakdown of discordant recordings
    discordant_by_cohort = rec_agg.groupby(["dataset", "trial_category"]).agg(
        n_recordings=("record_id", "count"),
        n_patients=("patient_id", "nunique"),
        median_burden_pct=("af_burden_pct", "median"),
        total_af_hours=("total_af_hours", "sum")
    ).reset_index()
    print("\nTrial Category by Cohort:")
    print(discordant_by_cohort.to_string(index=False))

    # Deep dive into DISCORDANT recordings:
    discordant_recs = rec_agg[rec_agg["trial_category"] == "DISCORDANT_30s_to_6m"].copy()
    print(f"\nDetailed Profile of {len(discordant_recs)} DISCORDANT Recordings (AF >= 30s but NO episode >= 6m):")
    print(f"  Total Unique Patients: {discordant_recs['patient_id'].nunique()}")
    print(f"  Cumulative AF Exposure: {discordant_recs['total_af_hours'].sum():.2f} hours")
    print(f"  Median AF Burden: {discordant_recs['af_burden_pct'].median():.2f}% [IQR: {discordant_recs['af_burden_pct'].quantile(0.25):.2f}%, {discordant_recs['af_burden_pct'].quantile(0.75):.2f}%]")
    print(f"  Maximum Burden in a Discordant Record: {discordant_recs['af_burden_pct'].max():.2f}% ({discordant_recs.loc[discordant_recs['af_burden_pct'].idxmax(), 'record_id']} in {discordant_recs.loc[discordant_recs['af_burden_pct'].idxmax(), 'dataset']})")
    print(f"  Median Episode Count: {discordant_recs['total_af_episodes'].median():.0f} [IQR: {discordant_recs['total_af_episodes'].quantile(0.25):.0f}, {discordant_recs['total_af_episodes'].quantile(0.75):.0f}]")
    print(f"  Maximum Episode Count in a Discordant Record: {discordant_recs['total_af_episodes'].max()} episodes")

    return rec_agg, cat_summary, discordant_by_cohort, discordant_recs


def audit_q3_shdb_clinical_metadata():
    """Q3: Completeness of clinical phenotype, echocardiography, and medication data in SHDB-AF."""
    print("\n" + "=" * 80)
    print("QUESTION 3: SHDB-AF CLINICAL METADATA COMPLETENESS AUDIT")
    print("=" * 80)

    meta_path = SHDB_DIR / "AdditionalData.csv"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found!")
        return None

    df_meta = pd.read_csv(meta_path)
    print(f"Total entries in AdditionalData.csv: {len(df_meta)} rows ({df_meta['Subject_ID'].nunique()} unique subjects)")

    # Filter to annotated recordings (the 98 Holters with rhythm annotations)
    df_ann = df_meta[df_meta["Annotated"] == True].copy()
    print(f"Rhythm-annotated recordings: {len(df_ann)} ({df_ann['Subject_ID'].nunique()} unique subjects)")

    # Key clinical variables to audit
    vars_to_audit = [
        ("Demographics", "Age_at_Holter"),
        ("Demographics", "Sex"),
        ("Demographics", "BMI"),
        ("AF History", "AF_Type"),
        ("AF History", "AF_Duration_Months"),
        ("AF History", "Previous_AF_Ablation"),
        ("AF History", "Previously_Documented_AFL"),
        ("Medications", "Antiarrhythmic_Drug_nonBB"),
        ("Medications", "Antiarrhythmic_Drug_BB"),
        ("Medications", "Anticoagulation"),
        ("Echocardiography", "Echo_LAD"),
        ("Echocardiography", "Echo_LVEF"),
        ("Echocardiography", "Moderate_or_Severe_MR"),
        ("Comorbidities", "CHF"),
        ("Comorbidities", "HTN"),
        ("Comorbidities", "DM"),
        ("Comorbidities", "Stroke"),
        ("Comorbidities", "Vascular_Diseases"),
        ("PPM", "PPM_on_Holter")
    ]

    completeness_rows = []
    for domain, col in vars_to_audit:
        if col in df_ann.columns:
            # Check non-null and non-empty
            valid = df_ann[col].notna() & (df_ann[col].astype(str).str.strip() != "")
            n_valid = valid.sum()
            pct_valid = n_valid / len(df_ann) * 100.0
            
            # Sample unique values or mean/std
            val_sample = ""
            if np.issubdtype(df_ann[col].dtype, np.number):
                val_sample = f"Mean: {df_ann[col].mean():.1f} +/- {df_ann[col].std():.1f}"
            else:
                top_vals = df_ann[col].value_counts().head(3).to_dict()
                val_sample = str(top_vals)

            completeness_rows.append({
                "Domain": domain,
                "Variable": col,
                "Valid_Entries": int(n_valid),
                "Total_Annotated": len(df_ann),
                "Completeness_Pct": round(pct_valid, 2),
                "Summary_or_Top_Values": val_sample
            })

    df_comp = pd.DataFrame(completeness_rows)
    print("\nSHDB-AF Clinical Metadata Completeness:")
    print(df_comp.to_string(index=False))

    # Cross-tabulation: AF Type and Previous Ablation
    print("\nAF Type Breakdown (Annotated Cohort):")
    print(df_ann["AF_Type"].value_counts(dropna=False))

    print("\nPrevious Ablation History:")
    print(df_ann["Previous_AF_Ablation"].value_counts(dropna=False))

    print("\nBeta-Blocker / Antiarrhythmic Use:")
    has_bb = df_ann["Antiarrhythmic_Drug_BB"].notna() & (df_ann["Antiarrhythmic_Drug_BB"].str.strip() != "")
    has_aad = df_ann["Antiarrhythmic_Drug_nonBB"].notna() & (df_ann["Antiarrhythmic_Drug_nonBB"].str.strip() != "")
    print(f"  Beta-blocker prescribed: {has_bb.sum()} / {len(df_ann)} ({has_bb.mean()*100:.1f}%)")
    print(f"  Non-BB Antiarrhythmic prescribed: {has_aad.sum()} / {len(df_ann)} ({has_aad.mean()*100:.1f}%)")

    return df_meta, df_ann, df_comp


def audit_q4_ltafdb_fragmentation():
    """Q4: Concentration of LTAFDB extreme fragmentation and competing rhythms during gaps."""
    print("\n" + "=" * 80)
    print("QUESTION 4: LTAFDB FRAGMENTATION CONCENTRATION & COMPETING RHYTHMS")
    print("=" * 80)

    out_dir = ROOT / "experiments/af_entrenchment/reports"
    df_ep = pd.read_csv(out_dir / "master_episode_survival_manifest.csv")
    ltaf_ep = df_ep[df_ep["dataset"] == "LTAFDB"].copy()

    print(f"Total LTAFDB AF Episodes: {len(ltaf_ep)} across {ltaf_ep['patient_id'].nunique()} patients ({ltaf_ep['record_id'].nunique()} recordings)")

    # 1. Episode concentration by patient
    pt_counts = ltaf_ep.groupby("patient_id")["duration_s"].agg(
        n_episodes="count",
        total_af_hours=lambda s: s.sum() / 3600.0,
        median_dur_s="median"
    ).sort_values("n_episodes", ascending=False).reset_index()

    pt_counts["cum_episodes"] = pt_counts["n_episodes"].cumsum()
    pt_counts["cum_pct_episodes"] = pt_counts["cum_episodes"] / len(ltaf_ep) * 100.0

    print("\nTop 10 Most Fragmented LTAFDB Patients:")
    print(pt_counts.head(10).to_string(index=False))

    top1 = pt_counts.iloc[0]["cum_pct_episodes"]
    top3 = pt_counts.iloc[2]["cum_pct_episodes"]
    top5 = pt_counts.iloc[4]["cum_pct_episodes"]
    top10 = pt_counts.iloc[9]["cum_pct_episodes"]
    top20 = pt_counts.iloc[19]["cum_pct_episodes"]

    print("\nPareto Concentration Analysis:")
    print(f"  Top 1 patient accounts for: {top1:.2f}% of all episodes ({pt_counts.iloc[0]['n_episodes']} ep)")
    print(f"  Top 3 patients account for: {top3:.2f}% of all episodes ({pt_counts.iloc[2]['cum_episodes']} ep)")
    print(f"  Top 5 patients account for: {top5:.2f}% of all episodes ({pt_counts.iloc[4]['cum_episodes']} ep)")
    print(f"  Top 10 patients account for: {top10:.2f}% of all episodes ({pt_counts.iloc[9]['cum_episodes']} ep)")
    print(f"  Top 20 patients account for: {top20:.2f}% of all episodes ({pt_counts.iloc[19]['cum_episodes']} ep)")

    # 2. Competing rhythm audit in LTAFDB annotations
    print("\nAuditing WFDB Rhythm Annotations During Non-AF Gaps in LTAFDB...")
    rhythm_transitions = Counter()
    gap_rhythms = Counter()

    # Sample 15 LTAFDB records with high and moderate fragmentation
    sample_records = pt_counts["patient_id"].head(15).tolist()
    
    for pt in sample_records:
        rec_id = pt.replace("ltaf_", "")
        rec_str = f"{int(rec_id):02d}"
        p = LTAF_DIR / rec_str
        try:
            ann = wfdb.rdann(str(p), "atr")
            # Extract rhythm change annotations (aux_note starting with '(')
            rhythm_events = []
            for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                if aux and aux.startswith("("):
                    rhythm_events.append((s, aux.strip()))
            
            for i in range(len(rhythm_events) - 1):
                s1, r1 = rhythm_events[i]
                s2, r2 = rhythm_events[i+1]
                if "AFIB" in r1 or "AF" in r1:
                    transition = f"{r1} -> {r2}"
                    rhythm_transitions[transition] += 1
                    gap_rhythms[r2] += 1
        except Exception:
            pass

    print("\nRhythms Directly Following AF Termination in LTAFDB:")
    total_transitions = sum(gap_rhythms.values())
    gap_rhythm_table = []
    for r, count in gap_rhythms.most_common():
        gap_rhythm_table.append({
            "Rhythm_Label": r,
            "Description": "Normal Sinus Rhythm" if r == "(N" else ("Atrial Flutter" if r == "(AFL" else ("Supraventricular Tachycardia" if r == "(SVTA" else ("Nodal/Junctional" if r == "(J" else "Other Arrhythmia"))),
            "Transition_Count": count,
            "Percentage": round(count / total_transitions * 100.0, 2)
        })
    df_gap_rhythms = pd.DataFrame(gap_rhythm_table)
    print(df_gap_rhythms.to_string(index=False))

    return pt_counts, df_gap_rhythms


def run_all_feasibility_audits():
    t0 = time.time()
    out_dir = ROOT / "experiments/af_entrenchment/reports"
    df_ep = pd.read_csv(out_dir / "master_episode_survival_manifest.csv")

    # Q1 & Q5
    df_q1, df_cohort_q1, reinit_dur, df_q5, df_annotated = audit_q1_and_q5_reinitiation_and_censoring(df_ep)
    df_q1.to_csv(out_dir / "feasibility_q1_reinitiation_horizons.csv", index=False)
    df_cohort_q1.to_csv(out_dir / "feasibility_q1_cohort_reinitiation.csv", index=False)
    reinit_dur.to_csv(out_dir / "feasibility_q1_reinitiation_by_duration.csv", index=False)
    df_q5.to_csv(out_dir / "feasibility_q5_followup_censoring.csv", index=False)

    # Q2
    rec_agg, cat_sum, disc_cohort, disc_recs = audit_q2_arc_6min_discordance(df_ep)
    cat_sum.to_csv(out_dir / "feasibility_q2_arc_trial_classification.csv", index=False)
    disc_cohort.to_csv(out_dir / "feasibility_q2_cohort_trial_categories.csv", index=False)
    disc_recs.to_csv(out_dir / "feasibility_q2_discordant_recordings_detail.csv", index=False)

    # Q3
    res_q3 = audit_q3_shdb_clinical_metadata()
    if res_q3:
        df_meta, df_ann, df_comp = res_q3
        df_comp.to_csv(out_dir / "feasibility_q3_shdb_metadata_completeness.csv", index=False)

    # Q4
    pt_counts, df_gap_rhythms = audit_q4_ltafdb_fragmentation()
    pt_counts.to_csv(out_dir / "feasibility_q4_ltafdb_patient_concentration.csv", index=False)
    df_gap_rhythms.to_csv(out_dir / "feasibility_q4_ltafdb_gap_rhythms.csv", index=False)

    # Summary synthesis
    feasibility_summary = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": round(time.time() - t0, 2),
        "q1_reinitiation_within_30s_episodes_pct": float(df_q1.loc[df_q1["horizon"] == "30s", "reinitiation_rate_episodes_pct"].iloc[0]),
        "q1_reinitiation_within_1h_episodes_pct": float(df_q1.loc[df_q1["horizon"] == "1h", "reinitiation_rate_episodes_pct"].iloc[0]),
        "q1_patients_with_reinit_within_6m_pct": float(df_q1.loc[df_q1["horizon"] == "6m", "reinitiation_rate_patients_pct"].iloc[0]),
        "q2_discordant_recordings_count": int(len(disc_recs)),
        "q2_discordant_patients_count": int(disc_recs["patient_id"].nunique()),
        "q2_discordant_total_af_exposure_hours": round(float(disc_recs["total_af_hours"].sum()), 2),
        "q2_discordant_median_af_burden_pct": round(float(disc_recs["af_burden_pct"].median()), 2),
        "q3_shdb_annotated_records": 98,
        "q3_shdb_lad_completeness_pct": float(df_comp.loc[df_comp["Variable"] == "Echo_LAD", "Completeness_Pct"].iloc[0]),
        "q3_shdb_lvef_completeness_pct": float(df_comp.loc[df_comp["Variable"] == "Echo_LVEF", "Completeness_Pct"].iloc[0]),
        "q4_ltafdb_top10_patient_concentration_pct": round(float(pt_counts.iloc[9]["cum_pct_episodes"]), 2),
        "q4_ltafdb_sinus_gap_pct": float(df_gap_rhythms.loc[df_gap_rhythms["Rhythm_Label"] == "(N", "Percentage"].iloc[0]),
        "q5_durable_candidates_with_ge_1h_followup_pct": float(df_q5.loc[df_q5["followup_threshold"] == "1h", "percentage_sufficient"].iloc[0])
    }

    (out_dir / "feasibility_reference_audit_summary.json").write_text(json.dumps(feasibility_summary, indent=2))
    print(f"\nAll feasibility audits completed and saved in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    run_all_feasibility_audits()
