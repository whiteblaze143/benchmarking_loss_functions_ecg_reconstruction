#!/usr/bin/env python3
"""Script 22: Experiment B - Historical-Density Electrophysiological Feasibility.
Inventories consecutive AF episode pairs across LTAFDB, SHDB-AF, and IRIDIA-AF to determine
whether preceding AF density can be investigated prior to subsequent AF episodes.
Generates:
  - results/preceding_history_feasibility.csv
  - results/informative_patient_counts.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density

def main():
    manifests_dir = root_dir / "manifests"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT B: HISTORICAL-DENSITY ELECTROPHYSIOLOGICAL FEASIBILITY")
    print("=" * 80)

    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")
    df_intervals = pd.read_parquet(manifests_dir / "canonical_rhythm_intervals.parquet")

    # Get recording monitoring start times and end times
    rec_bounds = df_intervals.groupby(["dataset", "record_id", "patient_id"]).agg(
        rec_start_s=("start_time_s", "min"),
        rec_end_s=("end_time_s", "max")
    ).reset_index()

    # Sort runs chronologically per recording
    df_runs = df_runs.sort_values(["dataset", "record_id", "start_time_s"]).copy()

    # Preceding observation horizons to test
    horizons = {
        "30m": 1800.0,
        "1h": 3600.0,
        "3h": 10800.0,
        "6h": 21600.0,
        "12h": 43200.0
    }

    # Episode pair inventory
    pair_records = []
    
    # Group runs by record
    for (ds, r_id), r_runs in df_runs.groupby(["dataset", "record_id"]):
        r_meta = rec_bounds[(rec_bounds["dataset"] == ds) & (rec_bounds["record_id"] == r_id)].iloc[0]
        p_id = r_meta["patient_id"]
        rec_start = r_meta["rec_start_s"]
        rec_end = r_meta["rec_end_s"]

        runs_list = r_runs.to_dict("records")
        n_rec_runs = len(runs_list)
        if n_rec_runs < 2:
            continue

        for i in range(n_rec_runs - 1):
            e_prev = runs_list[i]
            e_sub = runs_list[i + 1]

            # 1. Verification of observation
            # Prev termination observed: not right-censored
            prev_term_obs = not bool(e_prev["is_right_censored"])
            # Sub onset observed: not left-censored
            sub_onset_obs = not bool(e_sub["is_left_censored"])

            # Durations
            prev_dur = float(e_prev["duration_s"])
            sub_dur = float(e_sub["duration_s"])

            # Gap (AF-free interval)
            gap_s = float(e_sub["start_time_s"] - e_prev["end_time_s"])

            # Flags
            has_prev_60s = (prev_dur >= 60.0)
            has_sub_60s = (sub_dur >= 60.0)
            has_sub_4min_petrutiu = (sub_dur >= 240.0)

            # Available monitoring before subsequent onset
            time_before_sub_s = float(e_sub["start_time_s"] - rec_start)

            pair_row = {
                "dataset": ds,
                "record_id": r_id,
                "patient_id": p_id,
                "pair_index": i,
                "prev_run_id": e_prev["run_id"],
                "sub_run_id": e_sub["run_id"],
                "prev_duration_s": prev_dur,
                "sub_duration_s": sub_dur,
                "interepisode_gap_s": gap_s,
                "prev_term_observed": prev_term_obs,
                "sub_onset_observed": sub_onset_obs,
                "has_prev_60s": has_prev_60s,
                "has_sub_60s": has_sub_60s,
                "has_sub_4min_petrutiu": has_sub_4min_petrutiu,
                "time_before_sub_s": time_before_sub_s
            }

            # Preceding history horizons
            for h_name, h_dur in horizons.items():
                has_horizon = (time_before_sub_s >= h_dur)
                pair_row[f"has_horizon_{h_name}"] = has_horizon

                if has_horizon:
                    h_start = float(e_sub["start_time_s"] - h_dur)
                    h_end = float(e_sub["start_time_s"])

                    # Fast backwards scan for preceding runs strictly within [h_start, h_end]
                    prec_af_s = 0.0
                    n_prec = 0
                    for j in range(i, -1, -1):
                        rj = runs_list[j]
                        re = float(rj["end_time_s"])
                        if re <= h_start:
                            break
                        rs = float(rj["start_time_s"])
                        rs_clip = max(h_start, rs)
                        re_clip = min(h_end, re)
                        if re_clip > rs_clip:
                            n_prec += 1
                            prec_af_s += (re_clip - rs_clip)

                    b_prec = prec_af_s / h_dur

                    pair_row[f"n_prec_runs_{h_name}"] = n_prec
                    pair_row[f"prec_burden_{h_name}"] = b_prec
                    pair_row[f"has_ge_2_prec_{h_name}"] = (n_prec >= 2)
                    pair_row[f"has_ge_3_prec_{h_name}"] = (n_prec >= 3)
                    pair_row[f"has_ge_4_prec_{h_name}"] = (n_prec >= 4)

                    # Continuous density is mathematically well-defined iff 0 < burden < 1 and at least 1 run
                    d_defined = (0.0 < b_prec < 1.0) and (n_prec >= 1)
                    pair_row[f"density_defined_{h_name}"] = d_defined
                else:
                    pair_row[f"n_prec_runs_{h_name}"] = 0
                    pair_row[f"prec_burden_{h_name}"] = np.nan
                    pair_row[f"has_ge_2_prec_{h_name}"] = False
                    pair_row[f"has_ge_3_prec_{h_name}"] = False
                    pair_row[f"has_ge_4_prec_{h_name}"] = False
                    pair_row[f"density_defined_{h_name}"] = False

            pair_records.append(pair_row)

    df_pairs = pd.DataFrame(pair_records)
    print(f"Total consecutive AF episode pairs inventoried: {len(df_pairs)} across {df_pairs['patient_id'].nunique()} unique patients.")

    # 2. Feasibility Table by Horizon and Complexity
    feasibility_rows = []
    cohorts = ["All_Cohorts", "LTAFDB", "SHDB-AF", "IRIDIA-AF"]

    for coh in cohorts:
        sub_c = df_pairs if coh == "All_Cohorts" else df_pairs[df_pairs["dataset"] == coh]
        if sub_c.empty:
            continue

        for h_name in horizons.keys():
            # Basic criteria: prev termination and sub onset observed, prev >= 60s, sub >= 60s
            c_basic = sub_c[
                sub_c["prev_term_observed"] &
                sub_c["sub_onset_observed"] &
                sub_c["has_prev_60s"] &
                sub_c["has_sub_60s"] &
                sub_c[f"has_horizon_{h_name}"]
            ]

            # With Petrutiu 4min requirement
            c_petrutiu = c_basic[c_basic["has_sub_4min_petrutiu"]]

            # Complexity tiers: >= 2, >= 3, >= 4 preceding runs with defined density
            c_2runs = c_basic[c_basic[f"has_ge_2_prec_{h_name}"] & c_basic[f"density_defined_{h_name}"]]
            c_3runs = c_basic[c_basic[f"has_ge_3_prec_{h_name}"] & c_basic[f"density_defined_{h_name}"]]
            c_4runs = c_basic[c_basic[f"has_ge_4_prec_{h_name}"] & c_basic[f"density_defined_{h_name}"]]

            c_petrutiu_2runs = c_petrutiu[c_petrutiu[f"has_ge_2_prec_{h_name}"] & c_petrutiu[f"density_defined_{h_name}"]]

            feasibility_rows.append({
                "cohort": coh,
                "horizon": h_name,
                "n_eligible_pairs_basic": len(c_basic),
                "n_patients_basic": c_basic["patient_id"].nunique(),
                "n_eligible_pairs_petrutiu_4min": len(c_petrutiu),
                "n_patients_petrutiu_4min": c_petrutiu["patient_id"].nunique(),
                "n_pairs_ge_2_runs_density": len(c_2runs),
                "n_patients_ge_2_runs_density": c_2runs["patient_id"].nunique(),
                "n_pairs_ge_3_runs_density": len(c_3runs),
                "n_patients_ge_3_runs_density": c_3runs["patient_id"].nunique(),
                "n_pairs_ge_4_runs_density": len(c_4runs),
                "n_patients_ge_4_runs_density": c_4runs["patient_id"].nunique(),
                "n_pairs_petrutiu_and_ge_2_runs": len(c_petrutiu_2runs),
                "n_patients_petrutiu_and_ge_2_runs": c_petrutiu_2runs["patient_id"].nunique()
            })

    df_feas = pd.DataFrame(feasibility_rows)
    out_feas = results_dir / "preceding_history_feasibility.csv"
    df_feas.to_csv(out_feas, index=False)
    print(f"Saved preceding history feasibility table to {out_feas}")

    # Summary of Informative Patients
    all_summary = df_feas[df_feas["cohort"] == "All_Cohorts"]
    print("\nInformative Patient Counts across All Cohorts:")
    print(all_summary[["horizon", "n_patients_basic", "n_patients_petrutiu_4min", "n_patients_ge_2_runs_density", "n_patients_ge_3_runs_density", "n_patients_petrutiu_and_ge_2_runs"]].to_string(index=False))

    # Save dedicated informative_patient_counts.csv
    out_pats = results_dir / "informative_patient_counts.csv"
    all_summary.to_csv(out_pats, index=False)
    print(f"Saved informative patient counts to {out_pats}")

    # Structural determination check for 2-run histories
    # When n_prec_runs == 2: density is mathematically determined by (burden, run1, run2, gap)
    # Check conditional variation for 3h horizon:
    h3_sub = df_pairs[df_pairs["has_ge_2_prec_3h"] & df_pairs["density_defined_3h"]].copy()
    if not h3_sub.empty:
        h3_2runs = h3_sub[h3_sub["n_prec_runs_3h"] == 2]
        h3_ge3 = h3_sub[h3_sub["n_prec_runs_3h"] >= 3]
        print(f"\nPreceding 3-Hour Horizon Complexity Distribution:")
        print(f"  Histories with exactly 2 runs (algebraically constrained): N={len(h3_2runs)} pairs ({h3_2runs['patient_id'].nunique()} patients)")
        print(f"  Histories with >= 3 runs (genuine organizational freedom): N={len(h3_ge3)} pairs ({h3_ge3['patient_id'].nunique()} patients)")

if __name__ == "__main__":
    main()
