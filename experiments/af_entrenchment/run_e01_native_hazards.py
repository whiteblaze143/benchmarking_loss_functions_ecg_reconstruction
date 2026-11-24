#!/usr/bin/env python3
"""Experiment E01: Native AF Episode Entrenchment Hazard Estimation.

Directly tests the Spitaler et al. (Heart Rhythm 2026) early entrenchment hypothesis:
Does the conditional probability of AF self-termination P(T <= t + Delta | T > t)
decrease as elapsed episode duration t increases?

Evaluated across:
1. All 8,540 native episodes.
2. Dataset-stratified cohorts (IRIDIA-AF, SHDB-AF, LTAFDB).
3. Patient-weighted cohort (equal weight per patient, eliminating LTAFDB sample dominance).

Outputs:
- experiments/af_entrenchment/reports/e01_native_entrenchment_hazards.csv
- experiments/af_entrenchment/reports/e01_landmark_termination_probabilities.csv
- experiments/af_entrenchment/reports/e01_summary.json
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def load_episodes():
    iridia_dir = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
    df_iridia_meta = pd.read_csv(iridia_dir / "iridia-af-metadata-v1.0.1.csv")
    df_iridia_audit = pd.read_csv(iridia_dir / "timeline_record_audit.csv")
    df_iridia_int = pd.read_csv(iridia_dir / "timeline_intervals.csv")

    df_shdb_rec = pd.read_csv(iridia_dir / "shdb_record_audit.csv")
    df_shdb_int = pd.read_csv(iridia_dir / "shdb_reference_intervals.csv")

    ltaf_dir = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
    df_ltaf_rec = pd.read_csv(ltaf_dir / "record_reference_audit.csv")
    df_ltaf_int = pd.read_csv(ltaf_dir / "reference_rhythm_intervals.csv")

    episodes = []

    # 1. IRIDIA
    for _, meta_row in df_iridia_meta.iterrows():
        rec_id = meta_row["record_id"]
        pat_id = meta_row["patient_id"]
        dur = float(df_iridia_audit[df_iridia_audit["record_id"] == rec_id].iloc[0]["waveform_seconds"])
        sub = df_iridia_int[df_iridia_int["record_id"] == rec_id].sort_values("start_seconds")
        raw_ints = [(r["start_seconds"], r["end_seconds"]) for _, r in sub.iterrows()]
        
        # Merge
        sorted_int = sorted(raw_ints, key=lambda x: x[0])
        merged = []
        for a, b in sorted_int:
            a, b = max(0.0, float(a)), min(dur, float(b))
            if b <= a: continue
            if merged and merged[-1][1] >= a:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
                
        for st, en in merged:
            is_left = bool(st <= 1.0)
            is_right = bool(en >= dur - 1.0)
            episodes.append({
                "dataset": "IRIDIA-AF",
                "record_id": str(rec_id),
                "patient_id": str(pat_id),
                "duration_s": en - st,
                "start_s": st,
                "end_s": en,
                "rec_duration_s": dur,
                "is_left_censored": is_left,
                "is_right_censored": is_right,
                "event": 0 if is_right else 1 # Event = 1 means termination observed
            })

    # 2. SHDB
    for _, rec_row in df_shdb_rec.iterrows():
        if not bool(rec_row["annotated"]): continue
        rec_id = str(rec_row["record_id"])
        subj_id = str(rec_row["subject_id"])
        dur = float(rec_row["duration_seconds"])
        fs = float(rec_row["sampling_rate"])
        t_start = float(rec_row["initial_unknown_seconds"])
        sub = df_shdb_int[(df_shdb_int["record_id"].astype(str) == rec_id) & (df_shdb_int["rhythm"] == "(AFIB")].sort_values("start_sample")
        raw_ints = [(r["start_sample"]/fs, r["end_sample_exclusive"]/fs) for _, r in sub.iterrows()]
        
        sorted_int = sorted(raw_ints, key=lambda x: x[0])
        merged = []
        for a, b in sorted_int:
            a, b = max(0.0, float(a)), min(dur, float(b))
            if b <= a: continue
            if merged and merged[-1][1] >= a:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
                
        for st, en in merged:
            is_left = bool(st <= t_start + 1.0)
            is_right = bool(en >= dur - 1.0)
            episodes.append({
                "dataset": "SHDB-AF",
                "record_id": str(rec_id),
                "patient_id": str(subj_id),
                "duration_s": en - st,
                "start_s": st,
                "end_s": en,
                "rec_duration_s": dur,
                "is_left_censored": is_left,
                "is_right_censored": is_right,
                "event": 0 if is_right else 1
            })

    # 3. LTAFDB
    for _, rec_row in df_ltaf_rec.iterrows():
        rec_id = str(rec_row["record_id"])
        subj_id = f"ltaf_{rec_id}"
        dur = float(rec_row["raw_duration_seconds"])
        fs = float(rec_row["sampling_rate"])
        t_start = float(rec_row["initial_unknown_seconds"])
        sub = df_ltaf_int[(df_ltaf_int["record_id"].astype(str) == rec_id) & (df_ltaf_int["rhythm"] == "(AFIB")].sort_values("start_sample")
        raw_ints = [(r["start_sample"]/fs, r["end_sample_exclusive"]/fs) for _, r in sub.iterrows()]
        
        sorted_int = sorted(raw_ints, key=lambda x: x[0])
        merged = []
        for a, b in sorted_int:
            a, b = max(0.0, float(a)), min(dur, float(b))
            if b <= a: continue
            if merged and merged[-1][1] >= a:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
                
        for st, en in merged:
            is_left = bool(st <= t_start + 1.0)
            is_right = bool(en >= dur - 1.0)
            episodes.append({
                "dataset": "LTAFDB",
                "record_id": str(rec_id),
                "patient_id": str(subj_id),
                "duration_s": en - st,
                "start_s": st,
                "end_s": en,
                "rec_duration_s": dur,
                "is_left_censored": is_left,
                "is_right_censored": is_right,
                "event": 0 if is_right else 1
            })

    df = pd.DataFrame(episodes)
    # Calculate patient weights
    patient_counts = df.groupby("patient_id")["duration_s"].transform("count")
    df["patient_weight"] = 1.0 / patient_counts
    return df

def run_e01():
    print("=" * 80)
    print("EXPERIMENT E01: NATIVE AF EPISODE ENTRENCHMENT HAZARDS")
    print("=" * 80)
    
    t0 = time.time()
    df_ep = load_episodes()
    print(f"Loaded {len(df_ep)} episodes across {df_ep['patient_id'].nunique()} patients.")

    landmarks_s = [15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0, 21600.0]
    landmark_labels = ["15s", "30s", "1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h", "6h"]
    deltas_s = [60.0, 300.0, 900.0, 1800.0, 3600.0]
    delta_labels = ["1m", "5m", "15m", "30m", "1h"]

    results = []

    # Strata to evaluate: Pooled, Patient-Weighted, IRIDIA, SHDB, LTAFDB
    strata = [
        ("Pooled (All Unweighted)", df_ep, False),
        ("Patient-Weighted (Balanced)", df_ep, True),
        ("IRIDIA-AF", df_ep[df_ep["dataset"] == "IRIDIA-AF"], False),
        ("SHDB-AF (Saitama)", df_ep[df_ep["dataset"] == "SHDB-AF"], False),
        ("LTAFDB", df_ep[df_ep["dataset"] == "LTAFDB"], False),
    ]

    for stratum_name, sub_df, is_weighted in strata:
        for t_val, t_lbl in zip(landmarks_s, landmark_labels):
            # Risk set: episodes surviving strictly past landmark t
            # For pure native duration analysis, consider episodes with duration > t
            risk = sub_df[sub_df["duration_s"] > t_val]
            n_risk = len(risk)
            if n_risk == 0:
                continue
                
            n_patients_at_risk = risk["patient_id"].nunique()

            for d_val, d_lbl in zip(deltas_s, delta_labels):
                # Terminations in [t, t + delta]
                # In native analysis: uncensored terminations that terminate <= t + d_val
                terminations = risk[(risk["duration_s"] <= t_val + d_val) & (risk["event"] == 1)]
                
                if is_weighted:
                    w_risk = risk["patient_weight"].sum()
                    w_term = terminations["patient_weight"].sum()
                    p_term = w_term / w_risk if w_risk > 0 else np.nan
                else:
                    p_term = len(terminations) / n_risk if n_risk > 0 else np.nan

                # Standard error using binomial / Wilson
                se = np.sqrt(p_term * (1.0 - p_term) / n_risk) if n_risk > 0 and not np.isnan(p_term) else np.nan
                ci_low = max(0.0, p_term - 1.96 * se) if not np.isnan(se) else np.nan
                ci_high = min(1.0, p_term + 1.96 * se) if not np.isnan(se) else np.nan

                results.append({
                    "stratum": stratum_name,
                    "landmark_label": t_lbl,
                    "landmark_seconds": t_val,
                    "delta_label": d_lbl,
                    "delta_seconds": d_val,
                    "n_at_risk": n_risk,
                    "n_patients_at_risk": n_patients_at_risk,
                    "n_terminations": len(terminations),
                    "p_termination": round(p_term, 6),
                    "ci_95_low": round(ci_low, 6),
                    "ci_95_high": round(ci_high, 6),
                    "empirical_hazard_per_min": round(p_term / (d_val / 60.0), 6)
                })

    df_res = pd.DataFrame(results)
    out_dir = ROOT / "experiments/af_entrenchment/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "e01_landmark_termination_probabilities.csv", index=False)
    
    # Save raw episode manifest with event flags for downstream experiments
    df_ep.to_csv(out_dir / "master_episode_survival_manifest.csv", index=False)
    
    # Generate human-readable summary
    print("\n--- E01 RESULTS SUMMARY (Forward Window Delta = 15m) ---")
    summary_15m = df_res[df_res["delta_label"] == "15m"].pivot(index="landmark_label", columns="stratum", values="p_termination")
    print(summary_15m)
    
    print("\n--- E01 RESULTS SUMMARY (Forward Window Delta = 5m) ---")
    summary_5m = df_res[df_res["delta_label"] == "5m"].pivot(index="landmark_label", columns="stratum", values="p_termination")
    print(summary_5m)

    # Statistical test of monotonic trend (Spearman rank correlation of P(term) vs landmark t)
    from scipy.stats import spearmanr
    corrs = {}
    for stratum_name in df_res["stratum"].unique():
        sub = df_res[(df_res["stratum"] == stratum_name) & (df_res["delta_label"] == "15m")].dropna(subset=["p_termination"])
        if len(sub) >= 4:
            rho, pval = spearmanr(sub["landmark_seconds"], sub["p_termination"])
            corrs[stratum_name] = {"spearman_rho": round(rho, 4), "p_value": float(pval), "entrenchment_detected": bool(rho < 0 and pval < 0.05)}
            print(f"Stratum '{stratum_name}': Spearman rho={rho:.4f}, p={pval:.4e} -> Entrenchment={'CONFIRMED' if rho < 0 and pval < 0.05 else 'NOT DETECTED'}")

    summary_json = {
        "experiment": "E01_native_entrenchment_hazards",
        "total_episodes": len(df_ep),
        "total_patients": int(df_ep["patient_id"].nunique()),
        "landmark_correlation_15m": corrs,
        "runtime_s": round(time.time() - t0, 2)
    }
    (out_dir / "e01_summary.json").write_text(json.dumps(summary_json, indent=2))
    print(f"\nSaved E01 artifacts to {out_dir} in {time.time() - t0:.2f}s!")

if __name__ == "__main__":
    run_e01()
