#!/usr/bin/env python3
"""Analysis engine for Theta-repSpat clinical benchmarks.

Computes paired contrasts with 95% bootstrap confidence intervals across the
4 representation arms:
  R1: Oracle 8-lead reference (ground-truth geometry)
  R0: Kors 3D VCG baseline
  R2: Nef 8-lead panoramic completion
  R3: Kors 3D VCG on Nef completion

Hierarchy of Evaluation:
  1. Recurrence Calibration & Graph Topology:
     - MMD pairwise rejection rate (alpha = 0.05)
     - Mean MMD test statistic
     - Maximal clique cluster count (Y_clique)
     - Graph density and edge retention
  2. External Interpretability Probe:
     - Wave concordance ARI and NMI against expert P, QRS, T delineations
     (strictly secondary probe, non-optimizing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    """Computes mean and 95% percentile bootstrap confidence interval."""
    valid = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(valid, size=len(valid), replace=True)) for _ in range(n_boot)]
    return float(np.mean(valid)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_phase_results(json_path: Path, output_dir: Path):
    with open(json_path) as f:
        records = json.load(f)

    if not records:
        print(f"No records found in {json_path}")
        return

    phase_name = json_path.stem
    print(f"\n{'='*75}")
    print(f"THETA-repSpat CLINICAL BENCHMARK ANALYSIS: {phase_name.upper()}")
    print(f"Total Patient Records: {len(records)}")
    print(f"{'='*75}\n")

    # Group by dataset
    datasets = sorted(list(set(r["dataset"] for r in records)))

    summary_rows = []
    for dset in datasets:
        dset_records = [r for r in records if r["dataset"] == dset]
        print(f"\n--- Dataset: {dset} (N={len(dset_records)}) ---")

        arms_data = {
            "R1_Oracle_8Lead": [],
            "R0_KorsVCG_3Lead": [],
            "R2_NefPanorama_8Lead": [],
            "R3_KorsNef_3Lead": [],
        }

        for r in dset_records:
            for arm_name, arm_metrics in r["arms"].items():
                arms_data[arm_name].append(arm_metrics)

        # 1. Absolute Arm Metrics Table
        print("\n[1. Recurrence Calibration & Graph Structure (Mean ± Std)]")
        print(f"{'Arm':25s} | {'Rejection Rate':16s} | {'Clique Clusters':16s} | {'Graph Density':14s} | {'Mean MMD':12s}")
        print("-" * 90)

        for arm_name, metrics_list in arms_data.items():
            rej = np.array([m["rejection_rate"] for m in metrics_list])
            cliques = np.array([m["n_clique_clusters"] for m in metrics_list])
            density = np.array([m["graph_density"] for m in metrics_list])
            mmd_stat = np.array([m["mean_mmd_stat"] for m in metrics_list])

            print(f"{arm_name:25s} | {np.mean(rej):.3f} ± {np.std(rej):.3f}    | {np.mean(cliques):.2f} ± {np.std(cliques):.2f}     | {np.mean(density):.3f} ± {np.std(density):.3f}   | {np.mean(mmd_stat):.4f}")

            summary_rows.append({
                "dataset": dset,
                "arm": arm_name,
                "rejection_rate_mean": float(np.mean(rej)),
                "rejection_rate_std": float(np.std(rej)),
                "clique_clusters_mean": float(np.mean(cliques)),
                "graph_density_mean": float(np.mean(density)),
                "mean_mmd_stat": float(np.mean(mmd_stat)),
            })

        # 2. Paired Contrasts with 95% Bootstrap CIs
        print("\n[2. Paired Contrasts (Bootstrap 95% CI)]")
        r1_list = arms_data["R1_Oracle_8Lead"]
        r0_list = arms_data["R0_KorsVCG_3Lead"]
        r2_list = arms_data["R2_NefPanorama_8Lead"]
        r3_list = arms_data["R3_KorsNef_3Lead"]

        # Delta R2 - R0
        d_rej_20 = np.array([m2["rejection_rate"] - m0["rejection_rate"] for m2, m0 in zip(r2_list, r0_list)])
        d_cli_20 = np.array([m2["n_clique_clusters"] - m0["n_clique_clusters"] for m2, m0 in zip(r2_list, r0_list)])
        d_den_20 = np.array([m2["graph_density"] - m0["graph_density"] for m2, m0 in zip(r2_list, r0_list)])

        # Delta R2 - R3
        d_rej_23 = np.array([m2["rejection_rate"] - m3["rejection_rate"] for m2, m3 in zip(r2_list, r3_list)])
        d_cli_23 = np.array([m2["n_clique_clusters"] - m3["n_clique_clusters"] for m2, m3 in zip(r2_list, r3_list)])

        # Delta R2 - R1 (gap to oracle)
        d_rej_21 = np.array([m2["rejection_rate"] - m1["rejection_rate"] for m2, m1 in zip(r2_list, r1_list)])

        for contrast_name, arr in [
            ("Delta Rejection (R2 - R0)", d_rej_20),
            ("Delta Rejection (R2 - R3)", d_rej_23),
            ("Delta Rejection (R2 - R1)", d_rej_21),
            ("Delta Cliques   (R2 - R0)", d_cli_20),
            ("Delta Cliques   (R2 - R3)", d_cli_23),
            ("Delta Density   (R2 - R0)", d_den_20),
        ]:
            mean, low, high = bootstrap_ci(arr)
            print(f"  {contrast_name:25s}: mean={mean:+.4f}, 95% CI=[{low:+.4f}, {high:+.4f}]")

        # 3. Secondary External Interpretability Probe: Wave Concordance
        has_wave = any(m["wave_concordance"] is not None for m in r1_list)
        if has_wave:
            print("\n[3. Secondary External Interpretability Probe: P/QRS/T Wave Concordance]")
            print(f"{'Arm':25s} | {'ARI (Clique)':16s} | {'NMI (Clique)':16s} | {'ARI (Initial)':16s}")
            print("-" * 80)
            for arm_name, metrics_list in arms_data.items():
                aris = [m["wave_concordance"]["ari_clique"] for m in metrics_list if m["wave_concordance"]]
                nmis = [m["wave_concordance"]["nmi_clique"] for m in metrics_list if m["wave_concordance"]]
                ari_inits = [m["wave_concordance"]["ari_initial"] for m in metrics_list if m["wave_concordance"]]
                print(f"{arm_name:25s} | {np.mean(aris):.4f} ± {np.std(aris):.4f}  | {np.mean(nmis):.4f} ± {np.std(nmis):.4f}  | {np.mean(ari_inits):.4f}")

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    csv_out = output_dir / f"{phase_name}_summary.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSaved analysis summary table to: {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Theta-repSpat Results")
    parser.add_argument("--json", type=str, default="results/theta_repspat_eval/theta_repspat_phase1_results.json")
    parser.add_argument("--output-dir", type=str, default="results/theta_repspat_eval")
    args = parser.parse_args()

    json_path = exp_root / args.json
    output_dir = exp_root / args.output_dir
    analyze_phase_results(json_path, output_dir)


if __name__ == "__main__":
    main()
