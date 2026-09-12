"""Master orchestrator for comprehensive evaluation of Temporal repSpat.

Runs:
1. Autoregressive Simulation Benchmark (Table 4 replication in 1D temporal domain)
2. Clinical ECG Benchmark on PTB-XL (Euclidean continuous vs. Jaccard binary markers)
3. Publication-quality summary figures and structured validation report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from temporal_rep_stat_ecg.scripts.run_ar_simulation_benchmark import run_full_simulation_benchmark
from temporal_rep_stat_ecg.scripts.run_clinical_ecg_benchmark import run_clinical_benchmark


def plot_simulation_comparison_figure(
    summary_df: pd.DataFrame,
    save_path: str = "temporal_rep_stat_ecg/figures/simulation_benchmark_comparison.png",
):
    """Plots comparative ARI performance across methods and configurations."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary_df))
    width = 0.20

    labels = [
        f"η={row['eta']}, p={int(row['p_features'])}, n={int(row['n_beats'])}"
        for _, row in summary_df.iterrows()
    ]

    rects1 = ax.bar(x - 1.5 * width, summary_df["cahc_ARI_median"], width, label="CAHC Alone (Over-segmentation)", color="#d62728")
    rects2 = ax.bar(x - 0.5 * width, summary_df["kmeans_ARI_median"], width, label="K-Means (Unconstrained)", color="#7f7f7f")
    rects3 = ax.bar(x + 0.5 * width, summary_df["unconstrained_ARI_median"], width, label="Unconstrained HAC", color="#bcbd22")
    rects4 = ax.bar(x + 1.5 * width, summary_df["temporal_repspat_ARI_median"], width, label="Temporal repSpat (Ours)", color="#2ca02c")

    ax.set_ylabel("Median Adjusted Rand Index (ARI)")
    ax.set_title("Recovery of Repeated Patterns under Temporal Autoregression")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved simulation comparison plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full Temporal repSpat evaluation suite.")
    parser.add_argument("--sim_seeds", type=int, default=5, help="Number of simulation seeds")
    parser.add_argument("--clinical_records", type=int, default=5, help="Number of clinical records")
    parser.add_argument("--output_dir", type=str, default="temporal_rep_stat_ecg/results")
    parser.add_argument("--figures_dir", type=str, default="temporal_rep_stat_ecg/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("EXECUTING FULL TEMPORAL repSpat EVALUATION PIPELINE")
    print("#" * 70 + "\n")

    # 1. Autoregressive Simulation Benchmark
    sim_summary_df = run_full_simulation_benchmark(
        n_seeds=args.sim_seeds,
        output_dir=args.output_dir,
    )
    plot_simulation_comparison_figure(
        sim_summary_df,
        save_path=os.path.join(args.figures_dir, "simulation_benchmark_comparison.png"),
    )

    # 2. Clinical ECG Benchmark (RDB Fiducial Ground Truth)
    clinical_results = run_clinical_benchmark(
        n_records=args.clinical_records,
        cache_root="data/rdb_wavelet_delineation_cache",
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )

    # Master Report
    master_report = {
        "framework": "temporal_rep_stat_ecg",
        "description": "Nonparametric discovery of Repeated Temporal Patterns (RTPs) in ECG",
        "simulation_summary": sim_summary_df.to_dict(orient="records"),
        "clinical_evaluated_records": len(clinical_results),
        "status": "COMPLETED",
    }
    master_report_path = os.path.join(args.output_dir, "master_evaluation_report.json")
    with open(master_report_path, "w") as f:
        json.dump(master_report, f, indent=2)

    print("\n" + "#" * 70)
    print(f"EVALUATION SUITE COMPLETE. Results saved to: {args.output_dir}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
