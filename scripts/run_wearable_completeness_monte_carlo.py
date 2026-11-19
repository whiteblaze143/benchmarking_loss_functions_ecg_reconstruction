#!/usr/bin/env python3
"""
Monte Carlo Wearable Completeness and Burden Uncertainty Bounds Analysis.
Decouples physical wear-time coverage from model classification errors.
Simulates 1,000 independent wear schedule realizations across post-ablation Holters.
Produces: wear_schedule_sensitivity_analysis.csv
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import wfdb

def run_monte_carlo_wearable_analysis(
    shdb_dir: str = "/data/mithunmanivannan/datasets/shdb_af",
    shdb_metadata: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/AdditionalData.csv",
    output_path: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/corrected_package/wear_schedule_sensitivity_analysis.csv",
    n_monte_carlo: int = 1000,
    seed: int = 42,
):
    print("=== [P6] Running Monte Carlo Wearable Completeness Analysis ===")
    rng = np.random.RandomState(seed)

    df_meta = pd.read_csv(shdb_metadata)
    ablation_annotated = df_meta[(df_meta["Previous_AF_Ablation"] == True) & (df_meta["Annotated"] == True)]
    print(f"Auditing {len(ablation_annotated)} annotated post-ablation Holter recordings...")

    # Load gold-standard episodes and second-level masks for each post-ablation Holter
    holter_records = []
    for idx, row in ablation_annotated.iterrows():
        did = int(row["Data_ID"])
        rec_path = os.path.join(shdb_dir, f"{did:03d}")
        if not os.path.exists(f"{rec_path}.atr"):
            continue

        header = wfdb.rdheader(rec_path)
        ann = wfdb.rdann(rec_path, "atr")
        fs = float(header.fs)
        sig_len = header.sig_len
        dur_sec = sig_len / fs

        # Parse ATR episodes
        episodes_30s = []
        episodes_6min = []
        sample_mask = np.zeros(int(dur_sec), dtype=bool)

        in_af = False
        start_s = 0
        for s, a in zip(ann.sample, ann.aux_note):
            a_clean = a.strip()
            if "(AFIB" in a_clean or "(AFL" in a_clean or "(AT" in a_clean:
                if not in_af:
                    in_af = True
                    start_s = s
            elif in_af and a_clean.startswith("("):
                dur_s = (s - start_s) / fs
                s_sec = int(start_s / fs)
                e_sec = min(int(s / fs), int(dur_sec))
                sample_mask[s_sec:e_sec] = True
                if dur_s >= 30.0:
                    episodes_30s.append((s_sec, e_sec, dur_s))
                if dur_s >= 360.0:
                    episodes_6min.append((s_sec, e_sec, dur_s))
                in_af = False
        if in_af:
            dur_s = (sig_len - start_s) / fs
            s_sec = int(start_s / fs)
            e_sec = int(dur_sec)
            sample_mask[s_sec:e_sec] = True
            if dur_s >= 30.0:
                episodes_30s.append((s_sec, e_sec, dur_s))
            if dur_s >= 360.0:
                episodes_6min.append((s_sec, e_sec, dur_s))

        true_burden_pct = (np.sum(sample_mask) / dur_sec) * 100.0 if dur_sec > 0 else 0.0

        holter_records.append({
            "did": did,
            "dur_sec": dur_sec,
            "sample_mask": sample_mask,
            "true_burden_pct": true_burden_pct,
            "episodes_30s": episodes_30s,
            "episodes_6min": episodes_6min,
        })

    print(f"Loaded {len(holter_records)} post-ablation Holters with exact second-level ground truth.")

    # Define Wear Schedules
    # Each schedule defined by target duration and circadian offset variability
    schedules = [
        {"name": "Continuous 24h Wear", "nominal_hours": 24.0, "is_spot_check": False},
        {"name": "Daytime 16h Wear (Active)", "nominal_hours": 16.0, "is_spot_check": False},
        {"name": "Workday 12h Wear", "nominal_hours": 12.0, "is_spot_check": False},
        {"name": "Sleep/Morning 8h Wear", "nominal_hours": 8.0, "is_spot_check": False},
        {"name": "Casual 4h Wear", "nominal_hours": 4.0, "is_spot_check": False},
        {"name": "Intermittent Spot-Checks (2x30s/d)", "nominal_hours": 60.0 / 3600.0, "is_spot_check": True},
    ]

    results = []

    for sched in schedules:
        s_name = sched["name"]
        nom_hrs = sched["nominal_hours"]
        is_spot = sched["is_spot_check"]

        # Track metrics across Monte Carlo iterations
        mc_coverage_pcts = []
        mc_sens_30s = []
        mc_sens_6min = []
        mc_physically_observable_30s = []
        mc_physically_observable_6min = []
        mc_detected_of_observable_30s = []
        mc_detected_of_observable_6min = []
        mc_bound_widths = []
        mc_bound_coverages = []
        mc_indeterminate_rates = []

        for mc_i in range(n_monte_carlo):
            tot_ref_30s = 0
            tot_obs_30s = 0
            tot_det_30s = 0

            tot_ref_6m = 0
            tot_obs_6m = 0
            tot_det_6m = 0

            bound_widths_rec = []
            bound_coverages_rec = []
            indeterminate_rec = []
            effective_wear_sec = []

            for rec in holter_records:
                dur_s = rec["dur_sec"]
                mask = rec["sample_mask"]
                tb = rec["true_burden_pct"]
                eps30 = rec["episodes_30s"]
                eps6m = rec["episodes_6min"]

                # Generate Wear Mask
                wear_mask = np.zeros(int(dur_s), dtype=bool)
                if nom_hrs >= 24.0:
                    wear_mask[:] = True
                elif is_spot:
                    # 2 spot checks of 30 seconds each, placed randomly in morning and evening
                    t1 = rng.randint(0, max(1, int(dur_s * 0.5) - 30))
                    t2 = rng.randint(int(dur_s * 0.5), max(int(dur_s * 0.5) + 1, int(dur_s) - 30))
                    wear_mask[t1 : t1 + 30] = True
                    wear_mask[t2 : t2 + 30] = True
                else:
                    wear_sec = int(nom_hrs * 3600)
                    # Random jitter on wear start time (circadian offset)
                    max_start = max(1, int(dur_s) - wear_sec)
                    start_t = rng.randint(0, max_start)
                    wear_mask[start_t : start_t + wear_sec] = True

                eff_sec = int(np.sum(wear_mask))
                effective_wear_sec.append(eff_sec / dur_s * 100.0)

                # Assume realistic classifier operating point on observed data:
                # 95% per-episode detection sensitivity on observable episodes
                det_prob = 0.95

                # Evaluate >= 30s episodes
                for s_e, e_e, d_e in eps30:
                    tot_ref_30s += 1
                    # Observable if at least 50% or 30s of episode is within wear_mask
                    ep_observed_sec = np.sum(wear_mask[s_e:e_e])
                    if ep_observed_sec >= min(d_e * 0.5, 30.0):
                        tot_obs_30s += 1
                        if rng.rand() < det_prob:
                            tot_det_30s += 1

                # Evaluate >= 6min episodes
                for s_e, e_e, d_e in eps6m:
                    tot_ref_6m += 1
                    ep_observed_sec = np.sum(wear_mask[s_e:e_e])
                    if ep_observed_sec >= 180.0:  # At least 3 minutes observed
                        tot_obs_6m += 1
                        if rng.rand() < det_prob:
                            tot_det_6m += 1

                # Partial Identification Bounds [B_min, B_max]
                # T_detected_af: AF seconds during wear_mask correctly detected
                af_observed_sec = np.sum(mask & wear_mask)
                # Model detection on observed AF (with slight classifier sensitivity factor)
                detected_af_sec = af_observed_sec * 0.95
                unobserved_sec = dur_s - eff_sec

                b_min = (detected_af_sec / dur_s) * 100.0
                b_max = ((detected_af_sec + unobserved_sec) / dur_s) * 100.0
                b_width = b_max - b_min

                bound_widths_rec.append(b_width)
                # Does true burden fall within [b_min, b_max]?
                bound_coverages_rec.append(b_min <= tb <= b_max)
                # Indeterminate: no AF detected, but unobserved time allows possible recurrence
                is_indet = (detected_af_sec == 0) and (unobserved_sec > 0) and (tb > 0)
                indeterminate_rec.append(is_indet)

            mc_coverage_pcts.append(np.mean(effective_wear_sec))
            mc_sens_30s.append((tot_det_30s / tot_ref_30s * 100.0) if tot_ref_30s > 0 else 100.0)
            mc_sens_6min.append((tot_det_6m / tot_ref_6m * 100.0) if tot_ref_6m > 0 else 100.0)
            mc_physically_observable_30s.append((tot_obs_30s / tot_ref_30s * 100.0) if tot_ref_30s > 0 else 100.0)
            mc_physically_observable_6min.append((tot_obs_6m / tot_ref_6m * 100.0) if tot_ref_6m > 0 else 100.0)
            mc_detected_of_observable_30s.append((tot_det_30s / tot_obs_30s * 100.0) if tot_obs_30s > 0 else 100.0)
            mc_detected_of_observable_6min.append((tot_det_6m / tot_obs_6m * 100.0) if tot_obs_6m > 0 else 100.0)
            mc_bound_widths.append(np.mean(bound_widths_rec))
            mc_bound_coverages.append(np.mean(bound_coverages_rec) * 100.0)
            mc_indeterminate_rates.append(np.mean(indeterminate_rec) * 100.0)

        results.append({
            "Wearable Monitoring Protocol": s_name,
            "Nominal Scheduled Hours": nom_hrs if nom_hrs >= 1.0 else round(nom_hrs, 4),
            "Mean Effective Completeness (%)": round(float(np.mean(mc_coverage_pcts)), 2),
            "Overall Recurrence Sens (>=30s) %": round(float(np.mean(mc_sens_30s)), 2),
            "Overall Recurrence Sens (>=30s) 95% CI": f"[{np.percentile(mc_sens_30s, 2.5):.1f}, {np.percentile(mc_sens_30s, 97.5):.1f}]",
            "Overall Recurrence Sens (>=6min) %": round(float(np.mean(mc_sens_6min)), 2),
            "Overall Recurrence Sens (>=6min) 95% CI": f"[{np.percentile(mc_sens_6min, 2.5):.1f}, {np.percentile(mc_sens_6min, 97.5):.1f}]",
            "Physically Observable (>=6min) %": round(float(np.mean(mc_physically_observable_6min)), 2),
            "Model Sens on Observable (>=6min) %": round(float(np.mean(mc_detected_of_observable_6min)), 2),
            "Mean Burden Bound Width [B_min, B_max] (% pts)": round(float(np.mean(mc_bound_widths)), 2),
            "Bound Empirical Coverage (%)": round(float(np.mean(mc_bound_coverages)), 2),
            "Indeterminate Recurrence Rate (%)": round(float(np.mean(mc_indeterminate_rates)), 2),
        })

    df_out = pd.DataFrame(results)
    print("\n=== MONTE CARLO WEARABLE COMPLETENESS RESULTS (1,000 ITERATIONS) ===")
    print(df_out[["Wearable Monitoring Protocol", "Mean Effective Completeness (%)", "Overall Recurrence Sens (>=30s) %", "Overall Recurrence Sens (>=6min) %", "Physically Observable (>=6min) %", "Mean Burden Bound Width [B_min, B_max] (% pts)", "Indeterminate Recurrence Rate (%)"]].to_string(index=False))

    df_out.to_csv(output_path, index=False)
    print(f"\nSaved Monte Carlo results to: {output_path}")

if __name__ == "__main__":
    run_monte_carlo_wearable_analysis()
