"""Master Comprehensive Evaluation Orchestrator for KirkVCG repSpat.

Chains both:
1. Benchmark 1: 3D VCG Trajectory Simulation (Section 4.1 analogue)
2. Benchmark 2: Real Clinical Multi-Lead ECG Benchmark (Section 4.2 analogue)
and logs final summary statistics to `kirkvcg_rep_stat/results/FINAL_BENCHMARK_REPORT.json`.
"""
from __future__ import annotations

import json
import os
import time
import pandas as pd

from kirkvcg_rep_stat.scripts.run_vcg_simulation_benchmark import run_simulation_benchmark
from kirkvcg_rep_stat.scripts.run_clinical_vcg_benchmark import run_clinical_benchmark


def main():
    start_time = time.time()
    print("=" * 80)
    print("STARTING COMPREHENSIVE KirkVCG repSpat EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Run Simulation Benchmark
    print("\n--- PHASE 1: 3D VCG SIMULATION BENCHMARK ---")
    sim_df = run_simulation_benchmark()

    # 2. Run Clinical Benchmark
    print("\n--- PHASE 2: CLINICAL MULTI-LEAD ECG BENCHMARK ---")
    clin_df = run_clinical_benchmark()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE EVALUATION COMPLETE IN {elapsed:.1f}s")
    print("=" * 80)

    # Build Master Report
    report = {
        "status": "COMPLETED",
        "elapsed_seconds": elapsed,
        "simulation_benchmark": {
            "n_experiments": len(sim_df),
            "kirkvcg_mean_ari": float(sim_df[sim_df["method"] == "KirkVCG_repSpat"]["ari"].mean()),
            "kirkvcg_mean_nmi": float(sim_df[sim_df["method"] == "KirkVCG_repSpat"]["nmi"].mean()),
            "temporal_only_mean_ari": float(sim_df[sim_df["method"] == "Temporal_Only_HAC"]["ari"].mean()),
            "unconstrained_mean_ari": float(sim_df[sim_df["method"] == "Unconstrained_HAC"]["ari"].mean()),
            "kmeans_mean_ari": float(sim_df[sim_df["method"] == "Standard_KMeans"]["ari"].mean()),
        },
        "clinical_benchmark": {
            "n_records": len(clin_df["record_id"].unique()) if not clin_df.empty else 0,
            "continuous_euclidean_mean_ari": float(clin_df[clin_df["metric"] == "Continuous_Euclidean"]["ari_vs_wave_gt"].mean()) if not clin_df.empty else 0.0,
            "binary_jaccard_mean_ari": float(clin_df[clin_df["metric"] == "Binary_Jaccard"]["ari_vs_wave_gt"].mean()) if not clin_df.empty else 0.0,
            "mean_cliques_discovered": float(clin_df["n_cliques"].mean()) if not clin_df.empty else 0.0,
        },
    }

    report_path = "kirkvcg_rep_stat/results/FINAL_BENCHMARK_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Master evaluation report saved to: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
