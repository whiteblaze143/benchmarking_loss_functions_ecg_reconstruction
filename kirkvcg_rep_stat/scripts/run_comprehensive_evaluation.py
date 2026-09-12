"""Master Comprehensive Evaluation Orchestrator for KirkVCG repSpat on Actual Patient Records.

Evaluates KirkVCG repSpat strictly on real patient ECG recordings from the clinical cohort,
benchmarking against unconstrained HAC, temporal-only HAC, and standard K-Means across
diverse cardiac rhythms (Sinus Rhythm, Atrial Fibrillation, Sinus Bradycardia, Sinus Arrhythmia, etc.).
Logs final summary statistics to `kirkvcg_rep_stat/results/FINAL_BENCHMARK_REPORT.json`.
"""
from __future__ import annotations

import json
import os
import time
import pandas as pd

from kirkvcg_rep_stat.scripts.run_clinical_vcg_benchmark import run_clinical_benchmark


def main():
    start_time = time.time()
    print("=" * 80)
    print("STARTING COMPREHENSIVE KirkVCG repSpat EVALUATION ON ACTUAL PATIENT RECORDS")
    print("=" * 80)

    # Run Benchmark on Actual Clinical Patient Records
    clin_df = run_clinical_benchmark(split="test", max_records=12)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ACTUAL RECORDS EVALUATION COMPLETE IN {elapsed:.1f}s")
    print("=" * 80)

    # Build Master Report
    kirk_sub = clin_df[clin_df["method"] == "KirkVCG_repSpat"]
    temp_sub = clin_df[clin_df["method"] == "Temporal_Only_HAC"]
    uncon_sub = clin_df[clin_df["method"] == "Unconstrained_HAC"]
    km_sub = clin_df[clin_df["method"] == "Standard_KMeans"]

    report = {
        "status": "COMPLETED",
        "benchmark_type": "actual_clinical_records",
        "elapsed_seconds": round(elapsed, 2),
        "dataset": "RDB_clinical_cohort",
        "n_patient_records": len(clin_df["record_id"].unique()) if not clin_df.empty else 0,
        "rhythms_evaluated": sorted(list(clin_df["rhythm"].unique())) if not clin_df.empty else [],
        "metrics_summary": {
            "KirkVCG_repSpat": {
                "mean_ari": float(kirk_sub["ari_vs_wave_gt"].mean()) if not kirk_sub.empty else 0.0,
                "std_ari": float(kirk_sub["ari_vs_wave_gt"].std()) if not kirk_sub.empty else 0.0,
                "mean_nmi": float(kirk_sub["nmi_vs_wave_gt"].mean()) if not kirk_sub.empty else 0.0,
                "std_nmi": float(kirk_sub["nmi_vs_wave_gt"].std()) if not kirk_sub.empty else 0.0,
                "mean_cliques": float(kirk_sub["n_cliques"].mean()) if not kirk_sub.empty else 0.0,
                "mean_anomaly_fraction": float(kirk_sub["anomaly_fraction"].mean()) if not kirk_sub.empty else 0.0,
            },
            "Temporal_Only_HAC": {
                "mean_ari": float(temp_sub["ari_vs_wave_gt"].mean()) if not temp_sub.empty else 0.0,
                "std_ari": float(temp_sub["ari_vs_wave_gt"].std()) if not temp_sub.empty else 0.0,
                "mean_nmi": float(temp_sub["nmi_vs_wave_gt"].mean()) if not temp_sub.empty else 0.0,
                "std_nmi": float(temp_sub["nmi_vs_wave_gt"].std()) if not temp_sub.empty else 0.0,
            },
            "Unconstrained_HAC": {
                "mean_ari": float(uncon_sub["ari_vs_wave_gt"].mean()) if not uncon_sub.empty else 0.0,
                "std_ari": float(uncon_sub["ari_vs_wave_gt"].std()) if not uncon_sub.empty else 0.0,
                "mean_nmi": float(uncon_sub["nmi_vs_wave_gt"].mean()) if not uncon_sub.empty else 0.0,
                "std_nmi": float(uncon_sub["nmi_vs_wave_gt"].std()) if not uncon_sub.empty else 0.0,
            },
            "Standard_KMeans": {
                "mean_ari": float(km_sub["ari_vs_wave_gt"].mean()) if not km_sub.empty else 0.0,
                "std_ari": float(km_sub["ari_vs_wave_gt"].std()) if not km_sub.empty else 0.0,
                "mean_nmi": float(km_sub["nmi_vs_wave_gt"].mean()) if not km_sub.empty else 0.0,
                "std_nmi": float(km_sub["nmi_vs_wave_gt"].std()) if not km_sub.empty else 0.0,
            },
        },
        "rhythm_breakdown": {
            r: {
                "kirk_mean_ari": float(kirk_sub[kirk_sub["rhythm"] == r]["ari_vs_wave_gt"].mean()),
                "kirk_mean_anomaly_fraction": float(kirk_sub[kirk_sub["rhythm"] == r]["anomaly_fraction"].mean()),
                "kirk_mean_cliques": float(kirk_sub[kirk_sub["rhythm"] == r]["n_cliques"].mean()),
            }
            for r in sorted(clin_df["rhythm"].unique())
        } if not kirk_sub.empty else {},
    }

    report_path = "kirkvcg_rep_stat/results/FINAL_BENCHMARK_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Master actual records evaluation report saved to: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
