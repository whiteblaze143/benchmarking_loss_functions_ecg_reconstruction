"""Audit and report full 4-cell CAL0 calibration results.

Evaluates:
- D 1:1, D 1:3, S 1:1, S 1:3
- Rejection rates, Wilson 95% CIs, and independent verdicts
- Realized ratios: r_beat, r_micro, r_block
- S 1:3 Imbalance Qualification Criterion: r_beat <= 0.50
- Domain-level rejection breakdowns across all 9 source domains
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rep_stat_ecg.src.motifs.dependence_calibration import compute_wilson_interval

def audit_cal0(cal_dir: Path = Path("refine-logs/qvcg/m3h_calibration")):
    d_path = cal_dir / "CAL0_DISJOINT_RESULTS.parquet"
    s_path = cal_dir / "CAL0_SHARED_RESULTS.parquet"

    if not d_path.exists():
        print(f"Error: {d_path} not found.")
        return False

    df_d = pd.read_parquet(d_path)
    df_s = pd.read_parquet(s_path) if s_path.exists() else None

    print("=" * 100)
    print("M3H-CAL: FULL 4-CELL CAL0 AUDIT & VERIFICATION REPORT")
    print("=" * 100)

    cells = [
        ("D 1:1", df_d[df_d["balance_condition"] == "BALANCED"], "CAL0-D Balanced"),
        ("D 1:3", df_d[df_d["balance_condition"] == "IMBALANCED"], "CAL0-D Imbalanced"),
    ]
    if df_s is not None:
        cells.extend([
            ("S 1:1", df_s[df_s["balance_condition"] == "BALANCED"], "CAL0-S Balanced"),
            ("S 1:3", df_s[df_s["balance_condition"] == "IMBALANCED"], "CAL0-S Imbalanced"),
        ])

    table_rows = []
    for code, sub_df, desc in cells:
        n_tests = len(sub_df)
        if n_tests == 0:
            continue
        n_rej = int((sub_df["p_value"] <= 0.05).sum())
        alpha_hat = n_rej / n_tests
        low, high = compute_wilson_interval(n_rej, n_tests, conf=0.95)
        
        med_beat = float(sub_df["ratio_beats"].median())
        med_micro = float(sub_df["ratio_microstates"].median())
        med_block = float(sub_df["ratio_blocks"].median())

        p_vals = sub_df["p_value"].to_numpy()
        med_p = float(np.median(p_vals))
        p_eq_1 = float(np.mean(p_vals >= 0.9999))

        anti_cons = "PASS" if high <= 0.075 else "FAIL"
        if anti_cons == "PASS" and alpha_hat < 0.02:
            nom_cal = "STRONGLY_CONSERVATIVE"
        elif anti_cons == "PASS":
            nom_cal = "NOMINALLY_CALIBRATED"
        else:
            nom_cal = "ANTI_CONSERVATIVE"

        table_rows.append({
            "Cell": code,
            "Tests": n_tests,
            "Rejects": n_rej,
            "alpha_hat": f"{alpha_hat:.4f}",
            "Wilson_95_CI": f"[{low:.4f}, {high:.4f}]",
            "Median_p": f"{med_p:.4f}",
            "P(p=1)": f"{p_eq_1:.4f}",
            "Med_r_beat": f"{med_beat:.4f}",
            "Med_r_micro": f"{med_micro:.4f}",
            "Med_r_block": f"{med_block:.4f}",
            "AntiCons_Gate": anti_cons,
            "Nominal_Calib": nom_cal,
        })

    table_df = pd.DataFrame(table_rows)
    print("\n--- TABLE 1: 4-CELL CAL0 CALIBRATION SUMMARY (SAFETY VS NOMINAL CALIBRATION) ---")
    print(table_df[["Cell", "Tests", "Rejects", "alpha_hat", "Wilson_95_CI", 
                    "Median_p", "P(p=1)", "Med_r_beat", "Med_r_micro", "Med_r_block", 
                    "AntiCons_Gate", "Nominal_Calib"]].to_string(index=False))

    # Detailed Audit of S 1:3 Imbalance Qualification
    if df_s is not None:
        s13 = df_s[df_s["balance_condition"] == "IMBALANCED"]
        if len(s13) > 0:
            print("\n--- S 1:3 REALIZED IMBALANCE QUALIFICATION AUDIT ---")
            r_beats = s13["ratio_beats"].to_numpy()
            r_micro = s13["ratio_microstates"].to_numpy()
            r_block = s13["ratio_blocks"].to_numpy()

            # Qualification criterion: r_beat <= 0.50
            qualified = r_beats <= 0.50
            n_qual = int(np.sum(qualified))
            pct_qual = n_qual / len(s13) * 100.0

            print(f"Qualification Criterion: r_beat <= 0.50 (Nominal Target = 0.333)")
            print(f"Qualified replicates: {n_qual}/{len(s13)} ({pct_qual:.1f}%)")
            print(f"Realized Beat Ratio:       min={np.min(r_beats):.4f}, p25={np.percentile(r_beats, 25):.4f}, median={np.median(r_beats):.4f}, p75={np.percentile(r_beats, 75):.4f}, max={np.max(r_beats):.4f}")
            print(f"Realized Microstate Ratio: min={np.min(r_micro):.4f}, p25={np.percentile(r_micro, 25):.4f}, median={np.median(r_micro):.4f}, p75={np.percentile(r_micro, 75):.4f}, max={np.max(r_micro):.4f}")
            print(f"Realized Block Ratio:      min={np.min(r_block):.4f}, p25={np.percentile(r_block, 25):.4f}, median={np.median(r_block):.4f}, p75={np.percentile(r_block, 75):.4f}, max={np.max(r_block):.4f}")

            if n_qual == len(s13):
                qual_verdict = "QUALIFIED_STRESS_TEST (100% of replicates satisfy r_beat <= 0.50)"
            elif pct_qual >= 90.0:
                qual_verdict = f"QUALIFIED_STRESS_TEST ({pct_qual:.1f}% satisfy r_beat <= 0.50)"
            else:
                qual_verdict = f"IMBALANCE_FLOOR_IMPACTED ({pct_qual:.1f}% satisfy r_beat <= 0.50; structural floor above 0.5 in selected domains)"
            print(f"Imbalance Status: {qual_verdict}")

    # Domain Breakdown Table
    domains = sorted(df_d["domain_id"].unique())
    print("\n--- TABLE 2: REJECTIONS BY SOURCE DOMAIN ACROSS CELLS (Nominal alpha=0.05, 50 reps/cell) ---")
    dom_rows = []
    for d in domains:
        row = {"Domain": f"Domain {d:02d}"}
        sub_d = df_d[df_d["domain_id"] == d]
        row["D_1:1"] = f"{int((sub_d[sub_d['balance_condition'] == 'BALANCED']['p_value'] <= 0.05).sum())}/50"
        row["D_1:3"] = f"{int((sub_d[sub_d['balance_condition'] == 'IMBALANCED']['p_value'] <= 0.05).sum())}/50"
        
        if df_s is not None:
            sub_s = df_s[df_s["domain_id"] == d]
            row["S_1:1"] = f"{int((sub_s[sub_s['balance_condition'] == 'BALANCED']['p_value'] <= 0.05).sum())}/50"
            row["S_1:3"] = f"{int((sub_s[sub_s['balance_condition'] == 'IMBALANCED']['p_value'] <= 0.05).sum())}/50"
        dom_rows.append(row)

    print(pd.DataFrame(dom_rows).to_string(index=False))
    print("=" * 100)
    return True

if __name__ == "__main__":
    audit_cal0()
