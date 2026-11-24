#!/usr/bin/env python3
"""Script 09: Experiment E02 - Within-Patient Sensitivity Analysis.
Evaluates within-patient associations between continuity (S_max) and ventricular rate (Q120)
across nonoverlapping consecutive 12h windows in multi-window IRIDIA patients.
Generates:
  - results/E02_within_patient_results.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE E07: WITHIN-PATIENT SENSITIVITY ANALYSIS (MULTI-WINDOW IRIDIA)")
    print("=" * 80)

    pwin_parquet = features_dir / "patient_window_features.parquet"
    df_pwin = pd.read_parquet(pwin_parquet)

    # Filter to consecutive 12h windows in IRIDIA with density defined and valid Q120
    df_multi = df_pwin[
        (df_pwin["dataset"] == "IRIDIA-AF") &
        (df_pwin["horizon"] == "12h_nonoverlapping") &
        (df_pwin["is_density_defined"]) &
        (df_pwin["q_120"].notna())
    ].copy()

    # Identify informative patients with >= 2 windows and variation in S_max (Delta S_max >= 0.10)
    patient_counts = df_multi["patient_id"].value_counts()
    eligible_pats = patient_counts[patient_counts >= 2].index.tolist()

    df_eligible = df_multi[df_multi["patient_id"].isin(eligible_pats)].copy()
    print(f"IRIDIA multi-window candidates: {len(df_eligible)} windows across {len(eligible_pats)} patients.")

    informative_pats = []
    paired_contrasts = []

    for pat in eligible_pats:
        pat_wins = df_eligible[df_eligible["patient_id"] == pat].sort_values("s_max")
        smax_range = pat_wins["s_max"].max() - pat_wins["s_max"].min()
        if smax_range >= 0.10:
            informative_pats.append(pat)
            # Compare highest vs lowest continuity window for this patient
            hi_win = pat_wins.iloc[-1]
            lo_win = pat_wins.iloc[0]
            paired_contrasts.append({
                "patient_id": pat,
                "n_windows": len(pat_wins),
                "s_max_hi": float(hi_win["s_max"]),
                "s_max_lo": float(lo_win["s_max"]),
                "delta_s_max": float(hi_win["s_max"] - lo_win["s_max"]),
                "burden_hi": float(hi_win["af_burden"]),
                "burden_lo": float(lo_win["af_burden"]),
                "delta_burden": float(abs(hi_win["af_burden"] - lo_win["af_burden"])),
                "density_hi": float(hi_win["density_continuous"]),
                "density_lo": float(lo_win["density_continuous"]),
                "delta_density": float(abs(hi_win["density_continuous"] - lo_win["density_continuous"])),
                "q_120_hi": float(hi_win["q_120"]),
                "q_120_lo": float(lo_win["q_120"]),
                "delta_q_120": float(hi_win["q_120"] - lo_win["q_120"])
            })

    print(f"Informative patients with within-patient continuity variation (Delta S_max >= 0.10): {len(informative_pats)}")

    df_contrasts = pd.DataFrame(paired_contrasts)
    out_csv = results_dir / "E02_within_patient_results.csv"

    if not df_contrasts.empty:
        df_contrasts.to_csv(out_csv, index=False)
        print(f"Saved within-patient results to {out_csv}")
        print(f"Median within-patient Delta Q120: {df_contrasts['delta_q_120'].median()*100:.2f} percentage points")
        print(f"Mean within-patient Delta Q120: {df_contrasts['delta_q_120'].mean()*100:.2f} percentage points")
    else:
        pd.DataFrame([{
            "informative_patients": 0,
            "note": "Negligible within-patient continuity variation in multi-day windows"
        }]).to_csv(out_csv, index=False)
        print("Note: Negligible within-patient continuity variation detected; stated as limitation.")

if __name__ == "__main__":
    main()
