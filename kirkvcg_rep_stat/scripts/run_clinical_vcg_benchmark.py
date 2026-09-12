"""Benchmark 2: Real Clinical Multi-Lead ECG Benchmark (Section 4.2 Analogue).

Evaluates KirkVCG repSpat across real patient recordings from RDB / PTB-XL:
1. Compares Continuous (Euclidean) vs Binary (Jaccard) distance metrics
   (analogue to TNBC continuous intensities vs binary presence/absence).
2. Measures concordance with ground truth wave delineations (QRS, P, T waves).
3. Evaluates rhythm regularity, clique extraction, and arrhythmia isolation.
4. Generates publication-ready figures (12-lead timeline, similarity graph, 3D loops).
"""
from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kirkvcg_rep_stat.src.pipeline import KirkVCGRepSpat
from kirkvcg_rep_stat.src.features.ecg_features import extract_sample_features
from kirkvcg_rep_stat.src.data.rdb_loader import load_clinical_ecg_record, list_clinical_ecg_records
from kirkvcg_rep_stat.src.visualization import (
    plot_vcg_3d_trajectory,
    plot_ecg_timeline_with_reps,
    plot_similarity_graph,
    plot_transition_matrix,
)


def run_clinical_benchmark(
    split: str = "test",
    max_records: int = 12,
    output_dir: str = "kirkvcg_rep_stat/results",
    fig_dir: str = "kirkvcg_rep_stat/figures",
) -> pd.DataFrame:
    """Executes the clinical ECG benchmark across patient records."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("RUNNING REAL CLINICAL MULTI-LEAD ECG BENCHMARK (Section 4.2 Analogue)")
    print("=" * 70)

    record_paths = list_clinical_ecg_records(split=split, max_records=max_records)
    print(f"Discovered {len(record_paths)} clinical patient records.")

    records = []
    first_visualized = False

    for idx, rpath in enumerate(record_paths):
        try:
            data = load_clinical_ecg_record(rpath)
            ecg = data["ecg"]
            vcg = data["vcg"]
            gt_seg = data["segmentation"]
            rhythm = data["canonical_rhythm"]
            rec_id = data["record_id"]
            time = data["time"]

            # Limit to first 1500 samples (3.0 seconds = ~3-5 beats) for fast, responsive clustering
            n_eval = min(1500, len(ecg))
            ecg_sub = ecg[:n_eval]
            vcg_sub = vcg[:n_eval]
            gt_sub = gt_seg[:n_eval]
            time_sub = time[:n_eval]

            # Extract features
            feat_dict = extract_sample_features(ecg_sub, vcg=vcg_sub, fs=500.0)
            X_cont = feat_dict["continuous"]
            X_bin = feat_dict["binary"]

            # 1. Continuous Attributes (Euclidean Distance)
            kirk_cont = KirkVCGRepSpat(
                metric="euclidean",
                m_neighbors=4,
                n_clusters=6,
                kernel="imq",
                kernel_param=1.0,
                n_permutations=100,
                alpha=0.05,
                min_clique_size=3,
                random_state=42,
            )
            labels_cont = kirk_cont.fit_predict(X_cont, vcg_sub)
            sum_cont = kirk_cont.get_summary()

            ari_cont = adjusted_rand_score(gt_sub, labels_cont)
            nmi_cont = normalized_mutual_info_score(gt_sub, labels_cont)

            records.append({
                "record_id": rec_id,
                "rhythm": rhythm,
                "metric": "Continuous_Euclidean",
                "ari_vs_wave_gt": ari_cont,
                "nmi_vs_wave_gt": nmi_cont,
                "n_cliques": sum_cont["n_cliques"],
                "n_rep_states": sum_cont["final_rep_labels_count"],
                "anomaly_fraction": sum_cont["anomaly_fraction"],
            })

            # 2. Binary Markers (Jaccard Distance)
            kirk_bin = KirkVCGRepSpat(
                metric="jaccard",
                m_neighbors=4,
                n_clusters=6,
                kernel="imq",
                kernel_param=1.0,
                n_permutations=100,
                alpha=0.05,
                min_clique_size=3,
                random_state=42,
            )
            labels_bin = kirk_bin.fit_predict(X_bin, vcg_sub)
            sum_bin = kirk_bin.get_summary()

            ari_bin = adjusted_rand_score(gt_sub, labels_bin)
            nmi_bin = normalized_mutual_info_score(gt_sub, labels_bin)

            records.append({
                "record_id": rec_id,
                "rhythm": rhythm,
                "metric": "Binary_Jaccard",
                "ari_vs_wave_gt": ari_bin,
                "nmi_vs_wave_gt": nmi_bin,
                "n_cliques": sum_bin["n_cliques"],
                "n_rep_states": sum_bin["final_rep_labels_count"],
                "anomaly_fraction": sum_bin["anomaly_fraction"],
            })

            print(f"[{idx+1}/{len(record_paths)}] {rec_id} ({rhythm}): Continuous ARI={ari_cont:.3f}, Binary ARI={ari_bin:.3f}, Cliques={sum_cont['n_cliques']}")

            # Save detailed figures for representative patient record
            if not first_visualized and sum_cont["n_cliques"] >= 1:
                plot_ecg_timeline_with_reps(
                    time_sub,
                    ecg_sub[:, 1],
                    labels_cont,
                    lead_name="Lead II",
                    title=f"Discovered REPs along 12-Lead ECG ({rec_id}, {rhythm})",
                    save_path=os.path.join(fig_dir, "fig3_clinical_ecg_rep_timeline.png"),
                )

                plot_vcg_3d_trajectory(
                    vcg_sub,
                    labels_cont,
                    title=f"3D VCG Dipole Loops Colored by Discovered REPs ({rec_id})",
                    save_path=os.path.join(fig_dir, "fig4_clinical_vcg_3d_loops.png"),
                )

                plot_similarity_graph(
                    kirk_cont.similarity_graph_,
                    cliques=kirk_cont.maximal_cliques_,
                    title=f"Clinical Similarity Graph G_sim & Maximal Cliques ({rec_id})",
                    save_path=os.path.join(fig_dir, "fig5_clinical_similarity_graph_cliques.png"),
                )

                plot_transition_matrix(
                    kirk_cont.encoding_["transition_matrix"],
                    title=f"REP Markov Transition Matrix ({rec_id})",
                    save_path=os.path.join(fig_dir, "fig6_clinical_transition_dynamics.png"),
                )
                plt.close("all")
                first_visualized = True

        except Exception as exc:
            print(f"Error processing {rpath}: {exc}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "clinical_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved clinical results to: {csv_path}")

    # Aggregation: Continuous vs Binary
    summary = df.groupby("metric")[["ari_vs_wave_gt", "nmi_vs_wave_gt", "n_cliques", "anomaly_fraction"]].agg(["mean", "std"]).reset_index()
    summary_path = os.path.join(output_dir, "clinical_benchmark_summary.json")
    summary.to_json(summary_path, orient="records", indent=2)

    # Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["Continuous_Euclidean", "Binary_Jaccard"]
    means = [df[df["metric"] == m]["ari_vs_wave_gt"].mean() for m in metrics]
    stds = [df[df["metric"] == m]["ari_vs_wave_gt"].std() for m in metrics]

    bars = ax.bar(
        ["Continuous (Euclidean)", "Binary (Jaccard)"],
        means,
        yerr=stds,
        capsize=5,
        color=["#3b82f6", "#10b981"],
        width=0.45,
        alpha=0.85,
    )
    ax.set_ylabel("Concordance with Wave GT (ARI)", fontsize=11)
    ax.set_title("repSpat Distance Metric Comparison on Clinical ECG (Section 4.2)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m_val + 0.03, f"{m_val:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(fig_dir, "fig7_continuous_vs_binary_comparison.png")
    fig.savefig(comp_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison figure to: {comp_path}")

    return df


if __name__ == "__main__":
    run_clinical_benchmark()
