"""Autoregressive simulation benchmark comparing Temporal repSpat vs baselines.

Translates Section 4.1 / CAR model simulation from repSpat:
- Evaluates:
  1. CAHC alone (enforces temporal contiguity but over-segments repeated episodes)
  2. Unconstrained Agglomerative Clustering (lacks temporal contiguity)
  3. K-Means (lacks temporal contiguity)
  4. Temporal repSpat (complete pipeline with CAHC + block-permutation MMD + maximal cliques)
- Computes Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and FMI
  across parameter grids: eta in {0.3, 0.8}, p in {5, 10}, n in {300, 600}.
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

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

from temporal_rep_stat_ecg.src.data.synthetic_ar import generate_autoregressive_ecg_simulation
from temporal_rep_stat_ecg.src.pipeline import TemporalRepSpat
from temporal_rep_stat_ecg.src.clustering.cahc import temporal_constrained_hac


def evaluate_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes ARI, NMI, and Fowlkes-Mallows index."""
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "FMI": float(fowlkes_mallows_score(y_true, y_pred)),
    }


def run_single_simulation_comparison(
    n_beats: int = 300,
    p_features: int = 5,
    eta: float = 0.5,
    noise_scale: float = 0.25,
    m_neighbors: int = 4,
    initial_G: int = 13,
    seed: int = 42,
) -> dict:
    """Runs one simulation comparison across all methods."""
    sim = generate_autoregressive_ecg_simulation(
        n_beats=n_beats,
        p_features=p_features,
        eta=eta,
        noise_scale=noise_scale,
        random_state=seed,
    )

    X = sim["X"]
    t = sim["timestamps"]
    y_true_rtp = sim["ground_truth_rtp"]  # Ground truth unified RTP labels (1=bg, 2=RTP-A, 3=RTP-B)

    # 1. Unconstrained K-Means (k = 3 true clusters)
    km = KMeans(n_clusters=3, n_init=10, random_state=seed)
    y_kmeans = km.fit_predict(X) + 1
    metrics_km = evaluate_clustering_metrics(y_true_rtp, y_kmeans)

    # 2. Unconstrained Agglomerative Clustering (Ward, k = 3)
    unconstrained_hac = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
    y_unconstrained = unconstrained_hac.fit_predict(X) + 1
    metrics_unconstrained = evaluate_clustering_metrics(y_true_rtp, y_unconstrained)

    # 3. CAHC alone (temporal contiguity, G = initial_G)
    y_cahc, _, _ = temporal_constrained_hac(
        X, t, n_clusters=initial_G, m_neighbors=m_neighbors, linkage="ward"
    )
    metrics_cahc = evaluate_clustering_metrics(y_true_rtp, y_cahc)

    # 4. Temporal repSpat (CAHC + IMQ MMD + Block Permutation + Maximal Cliques)
    pipeline = TemporalRepSpat(
        metric="euclidean",
        m_neighbors=m_neighbors,
        n_clusters=initial_G,
        kernel="IMQ",
        kernel_param=1.0,
        n_permutations=200,
        alpha=0.05,
        min_clique_size=3,
        random_state=seed,
    )
    y_repspat = pipeline.fit_predict(X, t)
    metrics_repspat = evaluate_clustering_metrics(y_true_rtp, y_repspat)
    summary = pipeline.get_summary()

    return {
        "seed": seed,
        "n_beats": n_beats,
        "p_features": p_features,
        "eta": eta,
        "metrics_kmeans": metrics_km,
        "metrics_unconstrained_hac": metrics_unconstrained,
        "metrics_cahc": metrics_cahc,
        "metrics_temporal_repspat": metrics_repspat,
        "cliques_found": summary["maximal_cliques_count"],
        "similarity_edges": summary["edges_in_similarity_graph"],
    }


def run_full_simulation_benchmark(
    n_seeds: int = 5,
    output_dir: str = "temporal_rep_stat_ecg/results",
) -> pd.DataFrame:
    """Executes systematic simulation grid across autocorrelation eta and dimensions p."""
    os.makedirs(output_dir, exist_ok=True)

    grid_configs = [
        {"eta": 0.3, "p_features": 5, "n_beats": 300},
        {"eta": 0.8, "p_features": 5, "n_beats": 300},
        {"eta": 0.3, "p_features": 10, "n_beats": 300},
        {"eta": 0.8, "p_features": 10, "n_beats": 300},
        {"eta": 0.3, "p_features": 5, "n_beats": 600},
        {"eta": 0.8, "p_features": 5, "n_beats": 600},
    ]

    all_records = []

    print("=" * 70)
    print("STARTING TEMPORAL repSpat AUTOREGRESSIVE SIMULATION BENCHMARK")
    print(f"Configs: {len(grid_configs)} | Seeds per config: {n_seeds}")
    print("=" * 70)

    for cfg in grid_configs:
        eta = cfg["eta"]
        p = cfg["p_features"]
        n = cfg["n_beats"]
        print(f"\n--- Running Config: eta={eta}, p={p}, n={n} ---")

        for s in range(n_seeds):
            seed = 42 + s * 101
            res = run_single_simulation_comparison(
                n_beats=n,
                p_features=p,
                eta=eta,
                seed=seed,
            )

            rec = {
                "eta": eta,
                "p_features": p,
                "n_beats": n,
                "seed": seed,
                "kmeans_ARI": res["metrics_kmeans"]["ARI"],
                "kmeans_NMI": res["metrics_kmeans"]["NMI"],
                "unconstrained_ARI": res["metrics_unconstrained_hac"]["ARI"],
                "unconstrained_NMI": res["metrics_unconstrained_hac"]["NMI"],
                "cahc_ARI": res["metrics_cahc"]["ARI"],
                "cahc_NMI": res["metrics_cahc"]["NMI"],
                "temporal_repspat_ARI": res["metrics_temporal_repspat"]["ARI"],
                "temporal_repspat_NMI": res["metrics_temporal_repspat"]["NMI"],
                "cliques_count": res["cliques_found"],
            }
            all_records.append(rec)
            print(f"  Seed {seed}: CAHC ARI={rec['cahc_ARI']:.3f} | Unconstrained ARI={rec['unconstrained_ARI']:.3f} | Temporal repSpat ARI={rec['temporal_repspat_ARI']:.3f}")

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "simulation_benchmark_raw.csv")
    df.to_csv(csv_path, index=False)

    # Aggregate by (eta, p_features, n_beats)
    agg_df = df.groupby(["eta", "p_features", "n_beats"]).agg(
        temporal_repspat_ARI_median=("temporal_repspat_ARI", "median"),
        temporal_repspat_ARI_mean=("temporal_repspat_ARI", "mean"),
        temporal_repspat_ARI_std=("temporal_repspat_ARI", "std"),
        cahc_ARI_median=("cahc_ARI", "median"),
        cahc_ARI_mean=("cahc_ARI", "mean"),
        unconstrained_ARI_median=("unconstrained_ARI", "median"),
        unconstrained_ARI_mean=("unconstrained_ARI", "mean"),
        kmeans_ARI_median=("kmeans_ARI", "median"),
        kmeans_ARI_mean=("kmeans_ARI", "mean"),
    ).reset_index()

    agg_csv_path = os.path.join(output_dir, "simulation_benchmark_summary.csv")
    agg_df.to_csv(agg_csv_path, index=False)

    print("\n" + "=" * 70)
    print("SIMULATION BENCHMARK SUMMARY (Adjusted Rand Index comparison):")
    print(agg_df.to_string(index=False))
    print("=" * 70)

    return agg_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Temporal repSpat simulation benchmark.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per config")
    parser.add_argument("--output_dir", type=str, default="temporal_rep_stat_ecg/results")
    args = parser.parse_args()

    run_full_simulation_benchmark(n_seeds=args.seeds, output_dir=args.output_dir)
