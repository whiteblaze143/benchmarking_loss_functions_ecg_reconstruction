#!/usr/bin/env python3
"""Experiment E02: Harmonized Definitions and Rigorous Censoring Survival Analysis.

Evaluates AF Episode Entrenchment under:
1. Four clinical duration thresholds: Native, >=30s (HRS/EHRA), >=2min, >=6min (AHRE).
2. Rigorous censoring treatment:
   - Excludes left-censored episodes (incident episodes only with confirmed onset).
   - Right-censoring properly incorporated into Kaplan-Meier survival S(t) and Nelson-Aalen H(t).
   - Conditional termination probability: P(Term in [t, t+Delta] | T > t) = 1 - S(t+Delta)/S(t).

Outputs:
- experiments/af_entrenchment/reports/e02_kaplan_meier_survival.csv
- experiments/af_entrenchment/reports/e02_harmonized_entrenchment_comparison.csv
- experiments/af_entrenchment/reports/e02_summary.json
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]

def kaplan_meier_estimator(durations, events):
    """Compute non-parametric Kaplan-Meier survival S(t)."""
    df = pd.DataFrame({"time": durations, "event": events}).sort_values("time")
    unique_times = df["time"].unique()
    
    n_at_risk = len(df)
    s_curr = 1.0
    
    km_times = [0.0]
    km_surv = [1.0]
    km_hazard = [0.0]
    cum_hazard = [0.0]
    h_cum = 0.0
    
    for t in unique_times:
        sub = df[df["time"] == t]
        d_events = sub["event"].sum()
        c_censored = len(sub) - d_events
        
        if n_at_risk > 0:
            p_death = d_events / n_at_risk
            s_curr *= (1.0 - p_death)
            h_inc = d_events / n_at_risk
            h_cum += h_inc
            
            km_times.append(t)
            km_surv.append(s_curr)
            km_hazard.append(h_inc)
            cum_hazard.append(h_cum)
            
            n_at_risk -= (d_events + c_censored)
            
    return np.array(km_times), np.array(km_surv), np.array(cum_hazard)

def get_surv_at_t(km_times, km_surv, t_target):
    idx = np.searchsorted(km_times, t_target, side="right") - 1
    idx = max(0, min(idx, len(km_surv) - 1))
    return km_surv[idx]

def run_e02():
    print("=" * 80)
    print("EXPERIMENT E02: HARMONIZED DEFINITIONS AND CENSORED SURVIVAL ANALYSIS")
    print("=" * 80)
    
    t0 = time.time()
    out_dir = ROOT / "experiments/af_entrenchment/reports"
    df_ep = pd.read_csv(out_dir / "master_episode_survival_manifest.csv")
    print(f"Loaded master manifest: {len(df_ep)} total episodes.")
    
    # Restrict to incident episodes (exclude left-censored, where onset is unobserved)
    df_incident = df_ep[~df_ep["is_left_censored"]].copy()
    print(f"Incident episodes (confirmed observed onset): {len(df_incident)} episodes across {df_incident['patient_id'].nunique()} patients.")

    thresholds_s = [0.0, 30.0, 120.0, 360.0]
    thresh_labels = ["Native (0s)", ">= 30s (HRS)", ">= 2min", ">= 6min (AHRE)"]

    landmarks_s = [30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0]
    landmark_labels = ["30s", "1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h"]
    deltas_s = [300.0, 900.0, 1800.0, 3600.0]
    delta_labels = ["5m", "15m", "30m", "1h"]

    km_curves_out = []
    e02_results = []
    entrenchment_tests = {}

    for t_filter, t_lbl in zip(thresholds_s, thresh_labels):
        sub_filter = df_incident[df_incident["duration_s"] >= t_filter].copy()
        n_episodes = len(sub_filter)
        n_patients = sub_filter["patient_id"].nunique()
        n_events = sub_filter["event"].sum()
        
        # Fit Kaplan-Meier
        km_times, km_surv, km_cumhaz = kaplan_meier_estimator(sub_filter["duration_s"].values, sub_filter["event"].values)
        
        # Save sampled KM curve
        sample_indices = np.linspace(0, len(km_times) - 1, min(50, len(km_times)), dtype=int)
        for s_idx in sample_indices:
            km_curves_out.append({
                "threshold_label": t_lbl,
                "time_seconds": round(km_times[s_idx], 2),
                "time_minutes": round(km_times[s_idx] / 60.0, 3),
                "survival_probability": round(km_surv[s_idx], 6),
                "cumulative_hazard": round(km_cumhaz[s_idx], 6)
            })

        # Calculate conditional termination probabilities P(T <= t + Delta | T > t) = 1 - S(t+Delta)/S(t)
        p_terms_15m = []
        l_secs_15m = []

        for l_s, l_lbl in zip(landmarks_s, landmark_labels):
            if l_s < t_filter:
                continue # Cannot evaluate landmark shorter than filter threshold
                
            # Number surviving past landmark
            n_surv_l = len(sub_filter[sub_filter["duration_s"] > l_s])
            if n_surv_l < 10:
                continue
                
            s_l = get_surv_at_t(km_times, km_surv, l_s)
            if s_l <= 0.0:
                continue

            for d_s, d_lbl in zip(deltas_s, delta_labels):
                s_l_plus_d = get_surv_at_t(km_times, km_surv, l_s + d_s)
                # Conditional survival S(t+Delta) / S(t)
                cond_surv = s_l_plus_d / s_l
                cond_term = max(0.0, min(1.0, 1.0 - cond_surv))
                
                # Empirical hazard rate
                haz_rate = cond_term / (d_s / 60.0)

                e02_results.append({
                    "duration_filter": t_lbl,
                    "filter_threshold_s": t_filter,
                    "total_cohort_episodes": n_episodes,
                    "total_cohort_patients": n_patients,
                    "landmark_label": l_lbl,
                    "landmark_seconds": l_s,
                    "delta_label": d_lbl,
                    "delta_seconds": d_s,
                    "n_surviving_landmark": n_surv_l,
                    "km_s_t": round(s_l, 6),
                    "km_s_t_plus_delta": round(s_l_plus_d, 6),
                    "km_conditional_termination": round(cond_term, 6),
                    "km_hazard_per_min": round(haz_rate, 6)
                })

                if d_lbl == "15m":
                    p_terms_15m.append(cond_term)
                    l_secs_15m.append(l_s)

        # Test monotonicity
        if len(p_terms_15m) >= 4:
            rho, pval = spearmanr(l_secs_15m, p_terms_15m)
            entrenchment_tests[t_lbl] = {
                "n_episodes": n_episodes,
                "n_patients": n_patients,
                "spearman_rho": round(rho, 4),
                "p_value": float(pval),
                "entrenchment_confirmed": bool(rho < 0 and pval < 0.05)
            }
            print(f"Threshold '{t_lbl}': episodes={n_episodes}, Spearman rho={rho:.4f}, p={pval:.4e} -> Entrenchment={'CONFIRMED' if rho < 0 and pval < 0.05 else 'NOT DETECTED'}")

    df_km = pd.DataFrame(km_curves_out)
    df_e02 = pd.DataFrame(e02_results)
    
    df_km.to_csv(out_dir / "e02_kaplan_meier_survival.csv", index=False)
    df_e02.to_csv(out_dir / "e02_harmonized_entrenchment_comparison.csv", index=False)

    print("\n--- E02 CONDITIONAL TERMINATION PROBABILITY TABLE (Forward Window Delta = 15m) ---")
    piv_15m = df_e02[df_e02["delta_label"] == "15m"].pivot(index="landmark_label", columns="duration_filter", values="km_conditional_termination")
    print(piv_15m)

    summary = {
        "experiment": "E02_harmonized_and_censored_survival",
        "total_incident_episodes": len(df_incident),
        "entrenchment_across_duration_filters": entrenchment_tests,
        "runtime_s": round(time.time() - t0, 2)
    }
    (out_dir / "e02_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved E02 artifacts in {time.time() - t0:.2f}s!")

if __name__ == "__main__":
    run_e02()
