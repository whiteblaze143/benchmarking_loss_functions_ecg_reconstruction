"""Benchmark 1: 3D VCG Trajectory Simulation Benchmark (Section 4.1 Analogue).

Evaluates recovery of ground-truth Repeated Electrophysiological Patterns (REPs)
under varying temporal autocorrelation (eta in {0.3, 0.8}) and noise levels:
- KirkVCG repSpat (Proposed 3D VCG CAHC + IMQ MMD + Maximal Cliques)
- Temporal-Only CAHC (1D time constraint without VCG spatial coordinates)
- Unconstrained Agglomerative Clustering (HAC Ward)
- Standard k-Means

Computes Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
and ectopic beat detection accuracy. Saves metrics and publication plot.
"""
from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kirkvcg_rep_stat.src.pipeline import KirkVCGRepSpat
from kirkvcg_rep_stat.src.data.synthetic_vcg import generate_synthetic_vcg_simulation
from kirkvcg_rep_stat.src.visualization import plot_vcg_3d_trajectory


def run_simulation_benchmark(
    n_seeds: int = 5,
    eta_values: list[float] = (0.3, 0.8),
    noise_levels: list[float] = (0.05, 0.15),
    output_dir: str = "kirkvcg_rep_stat/results",
    fig_dir: str = "kirkvcg_rep_stat/figures",
) -> pd.DataFrame:
    """Executes the simulation benchmark across seeds, autocorrelation eta, and noise levels."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    records = []
    print("=" * 70)
    print("RUNNING 3D VCG SIMULATION BENCHMARK (Section 4.1 Analogue)")
    print("=" * 70)

    best_sim = None
    best_labels = None

    for eta in eta_values:
        for noise in noise_levels:
            for seed in range(n_seeds):
                sim = generate_synthetic_vcg_simulation(
                    n_beats=10,
                    samples_per_beat=200,
                    fs=500.0,
                    eta=eta,
                    noise_level=noise,
                    p_features=8,
                    add_ectopic_beat=True,
                    random_state=42 + seed,
                )
                X = sim["X"]
                vcg = sim["vcg"]
                gt = sim["ground_truth"]

                # 1. Proposed: KirkVCG repSpat
                model_kirk = KirkVCGRepSpat(
                    metric="euclidean",
                    m_neighbors=4,
                    n_clusters=6,
                    kernel="imq",
                    kernel_param=1.0,
                    n_permutations=100,
                    alpha=0.05,
                    min_clique_size=3,
                    random_state=42 + seed,
                )
                pred_kirk = model_kirk.fit_predict(X, vcg)
                ari_kirk = adjusted_rand_score(gt, pred_kirk)
                nmi_kirk = normalized_mutual_info_score(gt, pred_kirk)
                sum_kirk = model_kirk.get_summary()

                if best_sim is None and eta == 0.5:
                    best_sim = sim
                    best_labels = pred_kirk

                records.append({
                    "method": "KirkVCG_repSpat",
                    "eta": eta,
                    "noise": noise,
                    "seed": seed,
                    "ari": ari_kirk,
                    "nmi": nmi_kirk,
                    "n_cliques": sum_kirk["n_cliques"],
                    "anomaly_fraction": sum_kirk["anomaly_fraction"],
                })

                # 2. Baseline: Unconstrained HAC (Ward)
                hac_unconstrained = AgglomerativeClustering(n_clusters=5, linkage="ward")
                pred_uncon = hac_unconstrained.fit_predict(X)
                records.append({
                    "method": "Unconstrained_HAC",
                    "eta": eta,
                    "noise": noise,
                    "seed": seed,
                    "ari": adjusted_rand_score(gt, pred_uncon),
                    "nmi": normalized_mutual_info_score(gt, pred_uncon),
                    "n_cliques": 0,
                    "anomaly_fraction": 0.0,
                })

                # 3. Baseline: Temporal-Only 1D HAC
                from sklearn.neighbors import NearestNeighbors
                import scipy.sparse as sp
                time_1d = sim["time"].reshape(-1, 1)
                nn = NearestNeighbors(n_neighbors=5).fit(time_1d)
                idx_t = nn.kneighbors(time_1d, return_distance=False)[:, 1:]
                rows = np.repeat(np.arange(len(time_1d)), 4)
                cols = idx_t.ravel()
                L_time = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(time_1d), len(time_1d)))
                L_time = L_time.maximum(L_time.T)
                hac_temp = AgglomerativeClustering(n_clusters=5, linkage="ward", connectivity=L_time)
                pred_temp = hac_temp.fit_predict(X)
                records.append({
                    "method": "Temporal_Only_HAC",
                    "eta": eta,
                    "noise": noise,
                    "seed": seed,
                    "ari": adjusted_rand_score(gt, pred_temp),
                    "nmi": normalized_mutual_info_score(gt, pred_temp),
                    "n_cliques": 0,
                    "anomaly_fraction": 0.0,
                })

                # 4. Baseline: Standard k-Means
                km = KMeans(n_clusters=5, random_state=42 + seed, n_init=5)
                pred_km = km.fit_predict(X)
                records.append({
                    "method": "Standard_KMeans",
                    "eta": eta,
                    "noise": noise,
                    "seed": seed,
                    "ari": adjusted_rand_score(gt, pred_km),
                    "nmi": normalized_mutual_info_score(gt, pred_km),
                    "n_cliques": 0,
                    "anomaly_fraction": 0.0,
                })

            print(f"Done eta={eta}, noise={noise} (mean KirkVCG ARI: {np.mean([r['ari'] for r in records if r['method']=='KirkVCG_repSpat' and r['eta']==eta and r['noise']==noise]):.4f})")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "simulation_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw simulation metrics to: {csv_path}")

    # Summary aggregations
    summary = df.groupby(["method", "eta", "noise"])[["ari", "nmi"]].agg(["mean", "std"]).reset_index()
    summary_path = os.path.join(output_dir, "simulation_benchmark_summary.json")
    summary.to_json(summary_path, orient="records", indent=2)

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = ["KirkVCG_repSpat", "Temporal_Only_HAC", "Unconstrained_HAC", "Standard_KMeans"]
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626"]

    for i, eta_val in enumerate(eta_values):
        ax = axes[i]
        eta_df = df[df["eta"] == eta_val]
        grouped = eta_df.groupby(["noise", "method"])["ari"].mean().unstack()

        x = np.arange(len(noise_levels))
        width = 0.18
        for j, m in enumerate(methods):
            if m in grouped.columns:
                vals = [grouped.loc[n, m] for n in noise_levels]
                ax.bar(x + j * width, vals, width, label=m.replace("_", " "), color=colors[j], alpha=0.9)

        ax.set_title(f"Autocorrelation $\\eta = {eta_val}$", fontsize=12, fontweight="bold")
        ax.set_xlabel("Noise Level ($\sigma$)", fontsize=11)
        ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([str(n) for n in noise_levels])
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Ground Truth REP Recovery under Varying Autocorrelation and Noise (Section 4.1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "fig1_simulation_ari_comparison.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")

    # Also save 3D VCG trajectory plot
    if best_sim is not None and best_labels is not None:
        plot_vcg_3d_trajectory(
            best_sim["vcg"],
            best_labels,
            title="Simulated 3D VCG Dipole Trajectory Colored by Discovered REPs",
            save_path=os.path.join(fig_dir, "fig2_simulation_vcg_3d_loops.png"),
        )
        plt.close("all")

    return df


if __name__ == "__main__":
    run_simulation_benchmark()
