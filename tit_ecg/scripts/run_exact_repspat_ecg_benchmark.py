"""Clinical ECG/VCG Benchmark for Exact repSpat Adaptation.

Evaluates the exact repSpat methodology on real clinical multi-lead patient recordings
(Normal Sinus Rhythm, Atrial Fibrillation, Sinus Bradycardia, Sinus Tachycardia).

Outputs:
- Quantitative metrics: Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
  comparing Ground-Truth wave delineations (P, QRS, T) against:
  * Initial CAHC temporal clusters (C_g)
  * Algorithm 1 Connected Components (Y_CC)
  * Section 2.4 / Section 3.2 Maximal Cliques (Y_clique)
- Cluster similarity graph structure (node count, edge count, density)
- Maximal cliques and overlap audit
- Diagnostic summary figures in `tit_ecg/figures/`
- Full summary tables in `tit_ecg/results/benchmark_summary.csv` and `.json`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.data_loader import list_clinical_ecg_records, load_clinical_ecg_record
from tit_ecg.src.pipeline import TemporalRepSpatECG, VCGStateRepSpatResidual
from tit_ecg.src.visualization import plot_temporal_repspat_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Exact repSpat ECG/VCG Benchmark")
    parser.add_argument("--max-records", type=int, default=8, help="Number of clinical records to benchmark")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of time samples per record (1000 = 2s at 500Hz)")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of block permutations B")
    parser.add_argument("--m-grid", type=int, nargs="+", default=[4, 6, 8], help="Candidate m-NN grid")
    parser.add_argument("--G-grid", type=int, nargs="+", default=[4, 6, 8], help="Candidate CAHC cluster count G grid")
    parser.add_argument("--output-dir", type=str, default="tit_ecg/results", help="Directory to save metric tables")
    parser.add_argument("--figure-dir", type=str, default="tit_ecg/figures", help="Directory to save figures")
    parser.add_argument("--run-state-space-comparison", action="store_true", default=True, help="Also run secondary VCG state-space repSpat")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    print("=" * 80)
    print("EXACT repSpat CLINICAL ECG/VCG BENCHMARK")
    print(f"Paper Reference: Senanayake & Jeganathan (Spatial Statistics, 2026)")
    print(f"Minimal-Change Mapping: s_t = (tau_t, 0) in R^2, X(s_t) = v(t) in R^3 (Kors VCG)")
    print(f"Candidate Grid: m in {args.m_grid}, G in {args.G_grid}")
    print(f"Permutations: B={args.n_permutations}")
    print(f"Samples per record: N={args.n_samples} ({args.n_samples / 500.0:.2f} seconds)")
    print("=" * 80)

    # Discover diverse clinical records
    record_paths = list_clinical_ecg_records(max_records=args.max_records, balance_rhythms=True)
    print(f"Discovered {len(record_paths)} clinical patient records across rhythms:")
    for p in record_paths:
        print(f"  - {os.path.basename(p)}")

    cfg = RepSpatConfig(
        m_grid=args.m_grid,
        G_grid=args.G_grid,
        n_permutations=args.n_permutations,
        strict_block_exceed=True,
        p_value_correction=True,
        kmeans_n_init=15,
        random_state=42,
    )

    records_summary = []
    total_start = time.time()

    for idx, path in enumerate(record_paths):
        bname = os.path.basename(path).replace(".pt", "")
        print(f"\n[{idx + 1}/{len(record_paths)}] Processing {bname}...")
        t0 = time.time()

        try:
            rec = load_clinical_ecg_record(path)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            continue

        ecg = rec["ecg"][:args.n_samples]
        seg = rec["segmentation"][:args.n_samples]
        rhythm = rec["canonical_rhythm"]
        fs = rec["fs"]

        # 1. Fit Temporal repSpat (Reference Adaptation)
        model = TemporalRepSpatECG(config=cfg)
        model.fit(
            ecg_or_vcg=ecg,
            sampling_rate=fs,
            is_vcg=False,
            ground_truth_waves=seg,
        )
        res = model.results_
        elapsed = time.time() - t0

        m_star = res["m_star"]
        G_star = res["G_star"]
        concordance = res["concordance"]
        clique_audit = res["clique_audit"]
        G_sim = res["similarity_graph"]

        n_edges = G_sim.number_of_edges()
        density = nx.density(G_sim)
        n_clique_groups = res["n_clique_clusters"]
        n_cc_groups = res["n_cc_clusters"]

        print(f"  Done in {elapsed:.2f}s | Rhythm: {rhythm}")
        print(f"  Selected: m*={m_star}, G*={G_star} | Initial CAHC clusters: {res['n_initial_clusters']}")
        print(f"  Similarity Graph: {G_sim.number_of_nodes()} nodes, {n_edges} edges (density {density:.2f})")
        print(f"  Reassignment: CC groups={n_cc_groups} | Clique groups={n_clique_groups}")
        print(f"  Wave Concordance vs GT (P/QRS/T):")
        print(f"    - Initial CAHC:   ARI = {concordance['ari_initial']:.4f}, NMI = {concordance['nmi_initial']:.4f}")
        print(f"    - Clique (Prim.): ARI = {concordance['ari_clique']:.4f}, NMI = {concordance['nmi_clique']:.4f}")
        print(f"    - CC (Alg. 1):    ARI = {concordance['ari_cc']:.4f}, NMI = {concordance['nmi_cc']:.4f}")

        # Save diagnostic figure
        fig_path = os.path.join(args.figure_dir, f"{bname}_temporal_repspat.png")
        plot_temporal_repspat_summary(
            results=res,
            ecg_signal=ecg,
            ground_truth_waves=seg,
            title=f"Temporal repSpat on {bname} ({rhythm})",
            save_path=fig_path,
        )
        print(f"  Saved figure -> {fig_path}")

        records_summary.append({
            "record_id": bname,
            "rhythm": rhythm,
            "n_samples": args.n_samples,
            "duration_sec": args.n_samples / fs,
            "m_star": m_star,
            "G_star": G_star,
            "n_initial_clusters": res["n_initial_clusters"],
            "n_similarity_edges": n_edges,
            "graph_density": float(density),
            "n_cc_groups": n_cc_groups,
            "n_clique_groups": n_clique_groups,
            "ari_initial": concordance["ari_initial"],
            "nmi_initial": concordance["nmi_initial"],
            "ari_clique": concordance["ari_clique"],
            "nmi_clique": concordance["nmi_clique"],
            "ari_cc": concordance["ari_cc"],
            "nmi_cc": concordance["nmi_cc"],
            "has_clique_overlap": clique_audit["has_overlap"],
            "n_overlapping_nodes": len(clique_audit["overlapping_nodes"]),
            "elapsed_sec": elapsed,
        })

    # Optional: Run secondary state-space comparison on first record
    if args.run_state_space_comparison and record_paths:
        print("\n" + "-" * 60)
        print("RUNNING SECONDARY COMPARISON: VCG State-Space repSpat [s_t = v(t), X(s_t) = r(t)]")
        print("-" * 60)
        try:
            rec0 = load_clinical_ecg_record(record_paths[0])
            bname0 = os.path.basename(record_paths[0]).replace(".pt", "")
            ecg0 = rec0["ecg"][:args.n_samples]
            state_model = VCGStateRepSpatResidual(config=cfg)
            state_model.fit(ecg0, sampling_rate=rec0["fs"])
            s_res = state_model.results_
            print(f"  State-Space repSpat completed on {bname0}!")
            print(f"  Selected: m*={s_res['m_star']}, G*={s_res['G_star']}")
            print(f"  CC groups: {s_res['n_cc_clusters']} | Clique groups: {s_res['n_clique_clusters']}")

            # Save state-space summary to json
            with open(os.path.join(args.output_dir, f"{bname0}_state_space_comparison.json"), "w") as f:
                json.dump({
                    "record_id": bname0,
                    "m_star": s_res["m_star"],
                    "G_star": s_res["G_star"],
                    "n_cc_clusters": s_res["n_cc_clusters"],
                    "n_clique_clusters": s_res["n_clique_clusters"],
                    "pairwise_tests_count": len(s_res["pairwise_df"]),
                }, f, indent=2)
        except Exception as e:
            print(f"  Secondary state-space run encountered: {e}")

    # Save summary tables
    summary_df = pd.DataFrame(records_summary)
    csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    json_path = os.path.join(args.output_dir, "benchmark_summary.json")
    summary_df.to_csv(csv_path, index=False)

    summary_stats = {
        "n_records": len(summary_df),
        "mean_ari_initial": float(summary_df["ari_initial"].mean()) if len(summary_df) else 0.0,
        "mean_nmi_initial": float(summary_df["nmi_initial"].mean()) if len(summary_df) else 0.0,
        "mean_ari_clique": float(summary_df["ari_clique"].mean()) if len(summary_df) else 0.0,
        "mean_nmi_clique": float(summary_df["nmi_clique"].mean()) if len(summary_df) else 0.0,
        "mean_ari_cc": float(summary_df["ari_cc"].mean()) if len(summary_df) else 0.0,
        "mean_nmi_cc": float(summary_df["nmi_cc"].mean()) if len(summary_df) else 0.0,
        "mean_graph_density": float(summary_df["graph_density"].mean()) if len(summary_df) else 0.0,
        "records": records_summary,
        "total_elapsed_sec": time.time() - total_start,
    }

    with open(json_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print(f"Summary Table saved -> {csv_path}")
    print(f"Summary JSON saved  -> {json_path}")
    print(f"Mean Wave Concordance (ARI): Initial={summary_stats['mean_ari_initial']:.4f} | Clique={summary_stats['mean_ari_clique']:.4f} | CC={summary_stats['mean_ari_cc']:.4f}")
    print(f"Mean Wave Concordance (NMI): Initial={summary_stats['mean_nmi_initial']:.4f} | Clique={summary_stats['mean_nmi_clique']:.4f} | CC={summary_stats['mean_nmi_cc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
