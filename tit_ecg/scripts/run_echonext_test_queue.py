"""EchoNext External Test Set Queue for Exact repSpat ECG/VCG Adaptation.

Evaluates Temporal repSpat on the EchoNext external test cohort (5,442 records, 12 leads, 250 Hz).
Inverts dataset z-scoring to physical millivolts, computes 3D Kors VCG dipole trajectory,
and executes exact repSpat inference across external patient recordings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
import numpy as np
import pandas as pd

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.pipeline import TemporalRepSpatECG


def parse_args():
    parser = argparse.ArgumentParser(description="Run Exact repSpat on EchoNext Test Cohort")
    parser.add_argument("--max-records", type=int, default=30, help="Number of external test records to evaluate")
    parser.add_argument("--n-samples", type=int, default=750, help="Number of time samples per record (750 = 3.0s at 250Hz)")
    parser.add_argument("--n-permutations", type=int, default=500, help="Number of block permutations B")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--output-dir", type=str, default="tit_ecg/results/echonext_queue", help="Output directory")
    return parser.parse_args()


def process_echonext_worker(item: dict, n_samples: int, config_dict: dict) -> dict | None:
    """Worker function to process a single EchoNext recording."""
    try:
        t0 = time.time()
        idx = item["index"]
        raw_wave = item["waveform"]  # [2500, 12]
        mean = item["mean"]
        std = item["std"]

        # Invert dataset z-scoring to physical microvolts, then convert to millivolts
        wave_uv = raw_wave[:n_samples] * std + mean
        wave_mv = wave_uv / 1000.0  # [n_samples, 12]

        cfg = RepSpatConfig(**config_dict)
        model = TemporalRepSpatECG(config=cfg)
        model.fit(wave_mv, sampling_rate=250.0)

        res = model.results_
        elapsed = time.time() - t0

        G_sim = res["similarity_graph"]
        density = float(G_sim.number_of_edges() / (res["G_star"] * (res["G_star"] - 1) / 2)) if res["G_star"] > 1 else 0.0

        return {
            "record_index": int(idx),
            "m_star": int(res["m_star"]),
            "G_star": int(res["G_star"]),
            "n_similarity_edges": int(G_sim.number_of_edges()),
            "graph_density": density,
            "n_cc_groups": int(res["n_cc_clusters"]),
            "n_clique_groups": int(res["n_clique_clusters"]),
            "has_clique_overlap": bool(res["clique_audit"]["has_overlap"]),
            "elapsed_sec": float(elapsed),
        }
    except Exception as e:
        return {"record_index": int(item.get("index", -1)), "error": str(e)}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("ECHONEXT EXTERNAL TEST SET repSpat EVALUATION QUEUE")
    print(f"Total Cohort Available: 5,442 records")
    print(f"Evaluating: {args.max_records} records")
    print(f"Samples per record: N={args.n_samples} ({args.n_samples / 250.0:.2f} seconds at 250 Hz)")
    print(f"Permutations: B={args.n_permutations}")
    print(f"Workers: {args.num_workers}")
    print("=" * 80)

    # Load EchoNext waveforms and metadata
    waveforms_path = "data/echonext/EchoNext_test_waveforms.npy"
    provenance_path = "data/echonext/PROVENANCE.json"

    with open(provenance_path) as f:
        prov = json.load(f)
    mean = np.array(prov["normalization"]["mean"], dtype=float)
    std = np.array(prov["normalization"]["std"], dtype=float)

    waveforms = np.load(waveforms_path, mmap_mode="r")
    total_test = waveforms.shape[0]
    eval_n = min(args.max_records, total_test)

    # Deterministic sampling indices across the test cohort
    rng = np.random.RandomState(42)
    sample_indices = sorted(rng.choice(total_test, size=eval_n, replace=False))

    items = []
    for idx in sample_indices:
        # Shape: (1, 2500, 12) -> (2500, 12)
        w = waveforms[idx, 0]
        items.append({
            "index": int(idx),
            "waveform": np.asarray(w, dtype=float),
            "mean": mean,
            "std": std,
        })

    cfg_dict = {
        "m_grid": [4, 6, 8],
        "G_grid": [4, 6, 8],
        "n_permutations": args.n_permutations,
        "kmeans_n_init": 10,
        "strict_block_exceed": True,
        "p_value_correction": True,
        "random_state": 42,
    }

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_idx = {
            executor.submit(process_echonext_worker, item, args.n_samples, cfg_dict): item["index"]
            for item in items
        }

        for future in as_completed(future_to_idx):
            res = future.result()
            if res and "error" not in res:
                results.append(res)
                print(f"  EchoNext [{res['record_index']}]: m*={res['m_star']}, G*={res['G_star']}, "
                      f"CC={res['n_cc_groups']}, Clique={res['n_clique_groups']}, time={res['elapsed_sec']:.1f}s")
            elif res:
                print(f"  Error on index {res.get('record_index')}: {res.get('error')}")

    total_elapsed = time.time() - start_time
    print(f"\nCompleted {len(results)} records in {total_elapsed:.1f}s")

    summary_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "echonext_test_summary.csv")
    json_path = os.path.join(args.output_dir, "echonext_test_summary.json")
    summary_df.to_csv(csv_path, index=False)

    summary_stats = {
        "n_records_evaluated": len(summary_df),
        "total_test_cohort": total_test,
        "mean_graph_density": float(summary_df["graph_density"].mean()) if len(summary_df) else 0.0,
        "mean_cc_groups": float(summary_df["n_cc_groups"].mean()) if len(summary_df) else 0.0,
        "mean_clique_groups": float(summary_df["n_clique_groups"].mean()) if len(summary_df) else 0.0,
        "fraction_with_clique_overlap": float(summary_df["has_clique_overlap"].mean()) if len(summary_df) else 0.0,
        "total_elapsed_sec": total_elapsed,
    }

    with open(json_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("=" * 80)
    print("ECHONEXT QUEUE COMPLETED")
    print(f"CSV saved  -> {csv_path}")
    print(f"JSON saved -> {json_path}")
    print(f"Mean Graph Density: {summary_stats['mean_graph_density']:.3f}")
    print(f"Mean Clique Groups: {summary_stats['mean_clique_groups']:.2f} vs CC Groups: {summary_stats['mean_cc_groups']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
