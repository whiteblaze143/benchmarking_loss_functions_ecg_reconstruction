"""Benchmark on Real Clinical Multi-Lead ECG Patient Records.

Evaluates KirkVCG repSpat strictly across real patient recordings from RDB / PTB-XL
with ground truth expert wave annotations (P-wave, QRS-complex, T-wave):
1. Runs KirkVCG repSpat with continuous physical electrophysiological attributes
   (lead voltages, dV/dt, d2V/dt2, Frenet-Serret curvature, torsion, velocity).
2. Compares against baselines:
   - Unconstrained Agglomerative Hierarchical Clustering (Ward.D2)
   - Temporal-Only Constrained Hierarchical Clustering (1D line graph)
   - Standard K-Means (Euclidean attribute space)
3. Measures concordance with ground truth wave delineations (ARI, NMI).
4. Evaluates rhythm regularity, clique discovery, and arrhythmia anomaly fractions.
5. Generates publication-ready figures for actual patient records.
"""
from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering, KMeans
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
    """Executes the benchmark on actual clinical patient recordings."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print("RUNNING BENCHMARK ON ACTUAL CLINICAL PATIENT ECG RECORDS")
    print("=" * 80)

    record_paths = list_clinical_ecg_records(split=split, max_records=max_records, balance_rhythms=True)
    print(f"Discovered {len(record_paths)} actual clinical patient records across diverse rhythms.")

    records = []
    first_visualized_sr = False
    first_visualized_af = False

    for idx, rpath in enumerate(record_paths):
        try:
            data = load_clinical_ecg_record(rpath)
            ecg = data["ecg"]
            vcg = data["vcg"]
            gt_seg = data["segmentation"]
            rhythm = data["canonical_rhythm"]
            rec_id = data["record_id"]
            time = data["time"]

            # Evaluate on first 1500 samples (3.0 seconds = ~3-5 beats)
            n_eval = min(1500, len(ecg))
            ecg_sub = ecg[:n_eval]
            vcg_sub = vcg[:n_eval]
            gt_sub = gt_seg[:n_eval]
            time_sub = time[:n_eval]

            # Extract continuous physical electrophysiological features (no synthetic binarization)
            feat_dict = extract_sample_features(ecg_sub, vcg=vcg_sub, fs=500.0)
            X_cont = feat_dict["continuous"]

            # 1. KirkVCG repSpat (Constrained CAHC + IMQ MMD + Block Permutation + Maximal Cliques)
            kirk = KirkVCGRepSpat(
                metric="euclidean",
                m_neighbors=4,
                n_clusters=6,
                use_hybrid=True,
                temporal_window=1,
                kernel="imq",
                kernel_param=1.0,
                n_permutations=100,
                alpha=0.05,
                min_clique_size=3,
                random_state=42,
            )
            labels_kirk = kirk.fit_predict(X_cont, vcg_sub)
            sum_kirk = kirk.get_summary()

            ari_kirk = float(adjusted_rand_score(gt_sub, labels_kirk))
            nmi_kirk = float(normalized_mutual_info_score(gt_sub, labels_kirk))

            # 2. Baseline: Unconstrained HAC (Ward.D2)
            hac_uncon = AgglomerativeClustering(n_clusters=6, metric="euclidean", linkage="ward")
            labels_uncon = hac_uncon.fit_predict(X_cont)
            ari_uncon = float(adjusted_rand_score(gt_sub, labels_uncon))
            nmi_uncon = float(normalized_mutual_info_score(gt_sub, labels_uncon))

            # 3. Baseline: Temporal-Only Constrained HAC (1D Line Graph)
            temp_conn = diags([1, 1], [-1, 1], shape=(n_eval, n_eval)).tocsr()
            hac_temp = AgglomerativeClustering(n_clusters=6, connectivity=temp_conn, metric="euclidean", linkage="ward")
            labels_temp = hac_temp.fit_predict(X_cont)
            ari_temp = float(adjusted_rand_score(gt_sub, labels_temp))
            nmi_temp = float(normalized_mutual_info_score(gt_sub, labels_temp))

            # 4. Baseline: Standard K-Means
            km = KMeans(n_clusters=6, random_state=42, n_init=10)
            labels_km = km.fit_predict(X_cont)
            ari_km = float(adjusted_rand_score(gt_sub, labels_km))
            nmi_km = float(normalized_mutual_info_score(gt_sub, labels_km))

            enc = sum_kirk.get("encoding") or {}
            planarity = float(enc.get("loop_planarity", 0.0))
            qrst_angle = float(enc.get("spatial_qrst_angle", 0.0))

            # Log record metrics
            records.append({
                "record_id": rec_id,
                "rhythm": rhythm,
                "method": "KirkVCG_repSpat",
                "ari_vs_wave_gt": ari_kirk,
                "nmi_vs_wave_gt": nmi_kirk,
                "n_cliques": sum_kirk["n_cliques"],
                "n_rep_states": sum_kirk["final_rep_labels_count"],
                "anomaly_fraction": sum_kirk["anomaly_fraction"],
                "loop_planarity": planarity,
                "spatial_qrst_angle": qrst_angle,
            })
            records.append({
                "record_id": rec_id,
                "rhythm": rhythm,
                "method": "Unconstrained_HAC",
                "ari_vs_wave_gt": ari_uncon,
                "nmi_vs_wave_gt": nmi_uncon,
                "n_cliques": 0,
                "n_rep_states": 6,
                "anomaly_fraction": 0.0,
                "loop_planarity": planarity,
                "spatial_qrst_angle": qrst_angle,
            })
            records.append({
                "record_id": rec_id,
                "rhythm": rhythm,
                "method": "Temporal_Only_HAC",
                "ari_vs_wave_gt": ari_temp,
                "nmi_vs_wave_gt": nmi_temp,
                "n_cliques": 0,
                "n_rep_states": 6,
                "anomaly_fraction": 0.0,
                "loop_planarity": planarity,
                "spatial_qrst_angle": qrst_angle,
            })
            records.append({
                "record_id": rec_id,
                "rhythm": rhythm,
                "method": "Standard_KMeans",
                "ari_vs_wave_gt": ari_km,
                "nmi_vs_wave_gt": nmi_km,
                "n_cliques": 0,
                "n_rep_states": 6,
                "anomaly_fraction": 0.0,
                "loop_planarity": planarity,
                "spatial_qrst_angle": qrst_angle,
            })

            print(
                f"[{idx+1}/{len(record_paths)}] {rec_id} ({rhythm}): "
                f"KirkVCG ARI={ari_kirk:.3f}, Temp ARI={ari_temp:.3f}, Uncon ARI={ari_uncon:.3f}, KMeans ARI={ari_km:.3f} | "
                f"Cliques={sum_kirk['n_cliques']}, Anomaly={sum_kirk['anomaly_fraction']:.2f}"
            )

            # Generate figures for representative normal rhythm (SR)
            if not first_visualized_sr and rhythm == "SR" and sum_kirk["n_cliques"] >= 1:
                plot_ecg_timeline_with_reps(
                    time_sub,
                    ecg_sub[:, 1],
                    labels_kirk,
                    lead_name="Lead II",
                    title=f"Discovered REPs on Actual Patient ECG ({rec_id}, Sinus Rhythm)",
                    save_path=os.path.join(fig_dir, "fig1_actual_ecg_rep_timeline.png"),
                )

                plot_vcg_3d_trajectory(
                    vcg_sub,
                    labels_kirk,
                    title=f"3D VCG Dipole Loops Colored by Discovered REPs ({rec_id}, SR)",
                    save_path=os.path.join(fig_dir, "fig2_actual_vcg_3d_loops.png"),
                )

                plot_similarity_graph(
                    kirk.similarity_graph_,
                    cliques=kirk.maximal_cliques_,
                    title=f"Clinical Similarity Graph G_sim & Maximal Cliques ({rec_id}, SR)",
                    save_path=os.path.join(fig_dir, "fig3_actual_similarity_graph_cliques.png"),
                )

                plot_transition_matrix(
                    kirk.encoding_["transition_matrix"],
                    title=f"REP Markov Transition Matrix ({rec_id}, SR)",
                    save_path=os.path.join(fig_dir, "fig4_actual_transition_matrix.png"),
                )
                plt.close("all")
                first_visualized_sr = True

            # Generate figures for representative arrhythmia (AF)
            if not first_visualized_af and rhythm == "AF":
                plot_ecg_timeline_with_reps(
                    time_sub,
                    ecg_sub[:, 1],
                    labels_kirk,
                    lead_name="Lead II",
                    title=f"Discovered REPs on Actual Arrhythmia ECG ({rec_id}, Atrial Fibrillation)",
                    save_path=os.path.join(fig_dir, "fig5_actual_af_ecg_timeline.png"),
                )
                plot_vcg_3d_trajectory(
                    vcg_sub,
                    labels_kirk,
                    title=f"3D VCG Irregular Loops ({rec_id}, Atrial Fibrillation)",
                    save_path=os.path.join(fig_dir, "fig6_actual_af_vcg_3d_loops.png"),
                )
                plt.close("all")
                first_visualized_af = True

        except Exception as exc:
            import traceback
            print(f"Error processing {rpath}: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "clinical_actual_records_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved clinical actual records results to: {csv_path}")

    # Summary per method
    summary = df.groupby("method")[["ari_vs_wave_gt", "nmi_vs_wave_gt", "n_cliques", "anomaly_fraction"]].agg(["mean", "std"]).reset_index()
    summary_path = os.path.join(output_dir, "clinical_actual_records_summary.json")
    summary.to_json(summary_path, orient="records", indent=2)

    # 1. Methods Comparison Figure (ARI & NMI on Actual Records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ["KirkVCG_repSpat", "Temporal_Only_HAC", "Unconstrained_HAC", "Standard_KMeans"]
    labels = ["KirkVCG repSpat\n(Ours)", "Temporal-Only\nHAC", "Unconstrained\nHAC", "Standard\nK-Means"]
    colors = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"]

    ari_means = [df[df["method"] == m]["ari_vs_wave_gt"].mean() for m in methods]
    ari_stds = [df[df["method"] == m]["ari_vs_wave_gt"].std() for m in methods]
    nmi_means = [df[df["method"] == m]["nmi_vs_wave_gt"].mean() for m in methods]
    nmi_stds = [df[df["method"] == m]["nmi_vs_wave_gt"].std() for m in methods]

    bars1 = axes[0].bar(labels, ari_means, yerr=ari_stds, capsize=5, color=colors, width=0.55, alpha=0.9)
    axes[0].set_ylabel("Adjusted Rand Index (ARI)", fontsize=11, fontweight="bold")
    axes[0].set_title("Concordance with Expert Wave Ground Truth (ARI)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, val in zip(bars1, ari_means):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold")

    bars2 = axes[1].bar(labels, nmi_means, yerr=nmi_stds, capsize=5, color=colors, width=0.55, alpha=0.9)
    axes[1].set_ylabel("Normalized Mutual Information (NMI)", fontsize=11, fontweight="bold")
    axes[1].set_title("Waveform Information Recovery (NMI)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, val in zip(bars2, nmi_means):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(fig_dir, "fig7_actual_records_methods_comparison.png")
    fig.savefig(comp_path, dpi=300)
    plt.close(fig)
    print(f"Saved methods comparison figure to: {comp_path}")

    # 2. Anomaly Fraction by Clinical Rhythm (KirkVCG repSpat)
    kirk_df = df[df["method"] == "KirkVCG_repSpat"]
    if not kirk_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rhythm_order = ["SR", "SB", "SA", "ST", "AF", "VT", "AT"]
        present_rhythms = [r for r in rhythm_order if r in kirk_df["rhythm"].unique()]
        means_anom = [kirk_df[kirk_df["rhythm"] == r]["anomaly_fraction"].mean() for r in present_rhythms]
        bars_anom = ax.bar(
            present_rhythms,
            means_anom,
            color=["#3b82f6" if r in ("SR", "SB") else "#ef4444" for r in present_rhythms],
            width=0.5,
            alpha=0.85,
        )
        ax.set_ylabel("Anomaly / Singleton Fraction", fontsize=11, fontweight="bold")
        ax.set_xlabel("Clinical Rhythm Diagnosis", fontsize=11, fontweight="bold")
        ax.set_title("repSpat Anomaly Fraction: Normal Sinus vs Arrhythmias", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        for bar, val in zip(bars_anom, means_anom):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01, f"{val:.2f}", ha="center", fontweight="bold")
        plt.tight_layout()
        anom_path = os.path.join(fig_dir, "fig8_actual_records_arrhythmia_anomaly.png")
        fig.savefig(anom_path, dpi=300)
        plt.close(fig)
        print(f"Saved arrhythmia anomaly figure to: {anom_path}")

    return df


if __name__ == "__main__":
    run_clinical_benchmark()
