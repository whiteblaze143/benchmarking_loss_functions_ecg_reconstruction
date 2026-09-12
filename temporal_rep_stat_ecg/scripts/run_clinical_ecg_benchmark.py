"""Clinical ECG benchmark evaluating Temporal repSpat using ground truth RDB fiducial labels.

Analyzes patient recordings from the RDB dataset (`data/rdb_wavelet_delineation_cache/test/*.pt`)
with exact ground truth expert wave annotations.
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
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from temporal_rep_stat_ecg.src.data.ecg_features import (
    load_rdb_record,
    extract_beat_features,
    list_rdb_records,
)
from temporal_rep_stat_ecg.src.pipeline import TemporalRepSpat
from temporal_rep_stat_ecg.src.visualization import (
    plot_temporal_clusters,
    plot_similarity_graph,
)


def run_single_rdb_evaluation(
    pt_file: str,
    output_dir: str = "temporal_rep_stat_ecg/results",
    figures_dir: str = "temporal_rep_stat_ecg/figures",
) -> dict:
    """Evaluates Temporal repSpat on a single RDB patient record using ground truth labels."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    rec = load_rdb_record(pt_file)
    record_id = rec["record_id"]
    rhythm = rec["canonical_rhythm"]

    # Extract non-heuristic features from ground truth segmentation
    feat_dict = extract_beat_features(
        ecg_record=rec["waveform"],
        seg_record=rec["segmentation"],
        fs=rec["fs"],
        lead_idx=1,
    )
    X_cont = feat_dict["continuous"]
    X_bin = feat_dict["binary"]
    timestamps = feat_dict["timestamps"]
    lead2_wave = rec["waveform"][1] if rec["waveform"].shape[0] == 12 else rec["waveform"][:, 1]

    n_beats = len(timestamps)
    print(f"Record {record_id} ({rhythm}): {n_beats} beats extracted from ground truth labels")

    # 1. Continuous Features with Euclidean Distance
    pipeline_cont = TemporalRepSpat(
        metric="euclidean",
        m_neighbors=[2, 4],
        n_clusters=range(3, min(6, n_beats // 2 + 1)),
        kernel="IMQ",
        kernel_param=1.0,
        n_permutations=150,
        alpha=0.05,
        min_clique_size=2,
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
        min_clique_size=2,
        random_state=42,
    )
    labels_bin = pipeline_bin.fit_predict(X_bin, timestamps)
    summary_bin = pipeline_bin.get_summary()

    # Save visual figures
    fig_cont_tl, _ = plot_temporal_clusters(
        timestamps=timestamps,
        labels=labels_cont,
        signal_wave=lead2_wave,
        title=f"RDB {record_id} ({rhythm}) Continuous (Euclidean) RTPs",
        save_path=os.path.join(figures_dir, f"{record_id}_continuous_timeline.png"),
    )
    plt.close(fig_cont_tl)

    fig_cont_g, _ = plot_similarity_graph(
        G=pipeline_cont.similarity_graph_,
        cliques=pipeline_cont.maximal_cliques_,
        title=f"RDB {record_id} ({rhythm}) Similarity Graph G_sim (Euclidean)",
        save_path=os.path.join(figures_dir, f"{record_id}_continuous_similarity_graph.png"),
    )
    plt.close(fig_cont_g)

    fig_bin_tl, _ = plot_temporal_clusters(
        timestamps=timestamps,
        labels=labels_bin,
        signal_wave=lead2_wave,
        title=f"RDB {record_id} ({rhythm}) Binary (Jaccard) RTPs",
        save_path=os.path.join(figures_dir, f"{record_id}_binary_timeline.png"),
    )
    plt.close(fig_bin_tl)

    return {
        "record_id": str(record_id),
        "canonical_rhythm": str(rhythm),
        "n_beats": int(n_beats),
        "continuous_summary": summary_cont,
        "binary_summary": summary_bin,
        "continuous_labels": [int(x) for x in labels_cont],
        "binary_labels": [int(x) for x in labels_bin],
        "timestamps": [float(x) for x in timestamps],
    }


def run_clinical_benchmark(
    n_records: int = 10,
    cache_root: str = "data/rdb_wavelet_delineation_cache",
    output_dir: str = "temporal_rep_stat_ecg/results",
    figures_dir: str = "temporal_rep_stat_ecg/figures",
) -> list[dict]:
    """Runs clinical benchmark across RDB cohort with ground truth fiducials."""
    print("=" * 70)
    print("STARTING TEMPORAL repSpat CLINICAL ECG BENCHMARK (RDB FIDUCIAL GROUND TRUTH)")
    print(f"Cohort size: {n_records} records | Database: RDB")
    print("=" * 70)

    manifest_file = os.path.join(cache_root, "manifest.json")
    if os.path.isfile(manifest_file):
        all_recs = list_rdb_records(manifest_path=manifest_file, split="test", cache_root=cache_root)
        seen_rhythms: dict[str, int] = {}
        selected_paths: list[str] = []
        # Balance selection across distinct clinical rhythms
        for r in all_recs:
            rhythm = r["canonical_rhythm"]
            if seen_rhythms.get(rhythm, 0) < max(1, n_records // 5):
                selected_paths.append(r["pt_path"])
                seen_rhythms[rhythm] = seen_rhythms.get(rhythm, 0) + 1
            if len(selected_paths) == n_records:
                break
        if len(selected_paths) < n_records:
            for r in all_recs:
                if r["pt_path"] not in selected_paths:
                    selected_paths.append(r["pt_path"])
                if len(selected_paths) == n_records:
                    break
        selected = selected_paths
    else:
        pt_files = sorted(glob.glob(os.path.join(cache_root, "test", "*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {cache_root}/test")
        selected = pt_files[:n_records]

    results = []

    for i, path in enumerate(selected):
        rec_name = os.path.basename(path)
        print(f"\n[{i+1}/{n_records}] Processing: {rec_name} ...")
        res = run_single_rdb_evaluation(
            pt_file=path,
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
    parser = argparse.ArgumentParser(description="Run RDB Clinical ECG benchmark.")
    parser.add_argument("--records", type=int, default=10, help="Number of records to evaluate")
    parser.add_argument("--cache_root", type=str, default="data/rdb_wavelet_delineation_cache")
    parser.add_argument("--output_dir", type=str, default="temporal_rep_stat_ecg/results")
    parser.add_argument("--figures_dir", type=str, default="temporal_rep_stat_ecg/figures")
    args = parser.parse_args()

    run_clinical_benchmark(
        n_records=args.records,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
