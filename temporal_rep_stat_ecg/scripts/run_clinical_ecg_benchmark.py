"""Clinical ECG benchmark evaluating Temporal repSpat on real multi-lead records.

Implements Section 4.2 of the repSpat-to-ECG mapping:
- Analyzes patient recordings from the PTB-XL database (data/ptb_xl/records500).
- Compares Continuous Morphology Features (Euclidean distance) vs.
  Binary Clinical Markers (Jaccard distance).
- Discovers Repeated Temporal Patterns (RTPs) and extracts maximal cliques.
- Generates episode timelines and similarity network graphs.
"""
from __future__ import annotations

import argparse
import glob
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

from temporal_rep_stat_ecg.src.data.ecg_features import (
    load_ptbxl_record,
    extract_beat_features,
)
from temporal_rep_stat_ecg.src.pipeline import TemporalRepSpat
from temporal_rep_stat_ecg.src.visualization import (
    plot_temporal_clusters,
    plot_similarity_graph,
)


def run_clinical_record_evaluation(
    record_path: str,
    output_dir: str = "temporal_rep_stat_ecg/results",
    figures_dir: str = "temporal_rep_stat_ecg/figures",
) -> dict:
    """Evaluates Temporal repSpat on a single clinical record under Euclidean and Jaccard metrics."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    record_name = os.path.basename(record_path)
    signals, fs, lead_names = load_ptbxl_record(record_path)

    # Extract beat features
    feat_dict = extract_beat_features(signals, fs=fs, lead_idx=1)
    X_cont = feat_dict["continuous"]
    X_bin = feat_dict["binary"]
    timestamps = feat_dict["timestamps"]
    lead2_wave = signals[:, 1]

    n_beats = len(timestamps)
    print(f"Record {record_name}: extracted {n_beats} valid beats (fs={fs}Hz)")

    # 1. Continuous Features with Euclidean Distance
    pipeline_cont = TemporalRepSpat(
        metric="euclidean",
        m_neighbors=[2, 4],
        n_clusters=range(3, min(6, n_beats // 2 + 1)),
        kernel="IMQ",
        kernel_param=1.0,
        n_permutations=150,
        alpha=0.05,
        min_clique_size=3,
        random_state=42,
    )
    labels_cont = pipeline_cont.fit_predict(X_cont, timestamps)
    summary_cont = pipeline_cont.get_summary()

    # 2. Binary Clinical Markers with Jaccard Distance
    pipeline_bin = TemporalRepSpat(
        metric="jaccard",
        m_neighbors=4,
        n_clusters=summary_cont["best_G"],
        kernel="IMQ",
        kernel_param=1.0,
        n_permutations=150,
        alpha=0.05,
        min_clique_size=3,
        random_state=42,
    )
    labels_bin = pipeline_bin.fit_predict(X_bin, timestamps)
    summary_bin = pipeline_bin.get_summary()

    # Generate Visualizations
    fig_cont_timeline, _ = plot_temporal_clusters(
        timestamps=timestamps,
        labels=labels_cont,
        signal_wave=lead2_wave,
        title=f"Record {record_name}: Continuous Features (Euclidean) Discovered RTPs",
        save_path=os.path.join(figures_dir, f"{record_name}_continuous_timeline.png"),
    )
    plt.close(fig_cont_timeline)

    fig_cont_graph, _ = plot_similarity_graph(
        G=pipeline_cont.similarity_graph_,
        cliques=pipeline_cont.maximal_cliques_,
        title=f"Record {record_name}: Similarity Graph G_sim (Euclidean)",
        save_path=os.path.join(figures_dir, f"{record_name}_continuous_similarity_graph.png"),
    )
    plt.close(fig_cont_graph)

    fig_bin_timeline, _ = plot_temporal_clusters(
        timestamps=timestamps,
        labels=labels_bin,
        signal_wave=lead2_wave,
        title=f"Record {record_name}: Binary Markers (Jaccard) Discovered RTPs",
        save_path=os.path.join(figures_dir, f"{record_name}_binary_timeline.png"),
    )
    plt.close(fig_bin_timeline)

    return {
        "record_name": str(record_name),
        "n_beats": int(n_beats),
        "continuous_summary": summary_cont,
        "binary_summary": summary_bin,
        "continuous_labels": [int(x) for x in labels_cont],
        "binary_labels": [int(x) for x in labels_bin],
        "timestamps": [float(x) for x in timestamps],
    }


def run_clinical_benchmark(
    n_records: int = 5,
    base_dir: str = "data/ptb_xl/records500",
    output_dir: str = "temporal_rep_stat_ecg/results",
    figures_dir: str = "temporal_rep_stat_ecg/figures",
) -> list[dict]:
    """Runs clinical benchmark across a cohort of patient recordings."""
    print("=" * 70)
    print("STARTING TEMPORAL repSpat CLINICAL ECG BENCHMARK")
    print(f"Cohort size: {n_records} records | Database: PTB-XL")
    print("=" * 70)

    hea_files = glob.glob(os.path.join(base_dir, "**", "*.hea"), recursive=True)
    if not hea_files:
        raise FileNotFoundError(f"No .hea files found in {base_dir}")

    # Select records deterministically
    selected = sorted(hea_files)[:n_records]
    results = []

    for i, path in enumerate(selected):
        rec_id = path[:-4]
        print(f"\n[{i+1}/{n_records}] Processing: {rec_id} ...")
        res = run_clinical_record_evaluation(
            record_path=rec_id,
            output_dir=output_dir,
            figures_dir=figures_dir,
        )
        results.append(res)
        print(f"  Euclidean: {res['continuous_summary']['initial_episodes_count']} episodes -> {res['continuous_summary']['rtp_clusters_count']} RTPs (Cliques: {res['continuous_summary']['maximal_cliques_count']})")
        print(f"  Jaccard:   {res['binary_summary']['initial_episodes_count']} episodes -> {res['binary_summary']['rtp_clusters_count']} RTPs (Cliques: {res['binary_summary']['maximal_cliques_count']})")

    out_file = os.path.join(output_dir, "clinical_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved clinical benchmark results to: {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clinical ECG benchmark.")
    parser.add_argument("--records", type=int, default=5, help="Number of records to evaluate")
    parser.add_argument("--output_dir", type=str, default="temporal_rep_stat_ecg/results")
    parser.add_argument("--figures_dir", type=str, default="temporal_rep_stat_ecg/figures")
    args = parser.parse_args()

    run_clinical_benchmark(n_records=args.records, output_dir=args.output_dir, figures_dir=args.figures_dir)
