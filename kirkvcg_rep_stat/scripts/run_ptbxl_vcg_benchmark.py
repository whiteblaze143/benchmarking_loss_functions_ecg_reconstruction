"""PTB-XL Clinical ECG Benchmark and Transform Sensitivity Analysis.

Implements Sections 9.2 and 10.1 of the Critical Review:
1. Validates KirkVCG repSpat directly on actual clinical ECGs from PTB-XL
   across 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).
2. Computes established VCG diagnostic parameters:
   - 3D VCG loop planarity index (Ghosal et al. 2024)
   - Spatial QRS-T angle (Kumar et al. 2023; Hughes et al. 2024)
   - Anomaly / singleton fraction and clique extraction (Bergfeldt et al. 2025)
3. Evaluates transform sensitivity comparing Kors regression vs Inverse Dower
   on pathological records (Vondrak & Penhaker 2022b Section 10.1).
4. Generates publication-ready figures for PTB-XL.
"""
from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kirkvcg_rep_stat.src.pipeline import KirkVCGRepSpat
from kirkvcg_rep_stat.src.features.ecg_features import extract_sample_features
from kirkvcg_rep_stat.src.data.ptbxl_loader import load_ptbxl_record, list_ptbxl_records
from kirkvcg_rep_stat.src.vcg.kinematics import compute_vcg_loop_planarity, compute_spatial_qrst_angle
from kirkvcg_rep_stat.src.visualization import (
    plot_vcg_3d_trajectory,
    plot_ecg_timeline_with_reps,
    plot_similarity_graph,
    plot_transition_matrix,
)


def run_ptbxl_benchmark(
    max_records: int = 15,
    output_dir: str = "kirkvcg_rep_stat/results",
    fig_dir: str = "kirkvcg_rep_stat/figures",
) -> pd.DataFrame:
    """Executes the PTB-XL benchmark and Kors vs Dower sensitivity analysis."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING PTB-XL CLINICAL BENCHMARK & VCG TRANSFORM SENSITIVITY ANALYSIS")
    print("=" * 80)

    record_ids = list_ptbxl_records(max_records=max_records, balance_superclasses=True)
    print(f"Selected {len(record_ids)} PTB-XL patient records across diagnostic superclasses.")

    records = []
    norm_rec_data = None
    mi_rec_data = None

    for idx, rec_id in enumerate(record_ids):
        try:
            data = load_ptbxl_record(rec_id)
            ecg = data["ecg"]
            vcg_kors = data["vcg_kors"]
            vcg_dower = data["vcg_dower"]
            primary_class = data["primary_class"]
            superclasses = data["superclasses"]
            report = data["report"]
            time = data["time"]

            # Evaluate on first 1500 samples (3.0 seconds = ~3-5 beats)
            n_eval = min(1500, len(ecg))
            ecg_sub = ecg[:n_eval]
            vcg_kors_sub = vcg_kors[:n_eval]
            vcg_dower_sub = vcg_dower[:n_eval]
            time_sub = time[:n_eval]

            # 1. Feature Extraction for Kors VCG
            feat_kors = extract_sample_features(ecg_sub, vcg=vcg_kors_sub, fs=500.0)
            X_kors = feat_kors["continuous"]

            # Fit KirkVCG repSpat on Kors VCG
            kirk_kors = KirkVCGRepSpat(
                metric="euclidean",
                m_neighbors=4,
                n_clusters=6,
                use_hybrid=True,
                temporal_window=1,
                kernel="imq",
                kernel_param=1.0,
                n_permutations=80,
                alpha=0.05,
                min_clique_size=3,
                random_state=42,
            )
            labels_kors = kirk_kors.fit_predict(X_kors, vcg_kors_sub)
            sum_kors = kirk_kors.get_summary()

            # 2. Feature Extraction & Fitting on Dower VCG (Sensitivity Analysis, Section 10.1)
            feat_dower = extract_sample_features(ecg_sub, vcg=vcg_dower_sub, fs=500.0)
            X_dower = feat_dower["continuous"]

            kirk_dower = KirkVCGRepSpat(
                metric="euclidean",
                m_neighbors=4,
                n_clusters=6,
                use_hybrid=True,
                temporal_window=1,
                kernel="imq",
                kernel_param=1.0,
                n_permutations=80,
                alpha=0.05,
                min_clique_size=3,
                random_state=42,
            )
            labels_dower = kirk_dower.fit_predict(X_dower, vcg_dower_sub)
            sum_dower = kirk_dower.get_summary()

            # Biomarkers
            planarity_kors = float(compute_vcg_loop_planarity(vcg_kors_sub))
            planarity_dower = float(compute_vcg_loop_planarity(vcg_dower_sub))

            enc_kors = sum_kors.get("encoding") or {}
            enc_dower = sum_dower.get("encoding") or {}
            qrst_kors = float(enc_kors.get("spatial_qrst_angle", 0.0))
            qrst_dower = float(enc_dower.get("spatial_qrst_angle", 0.0))

            # Transform difference (Frobenius distance between normalized trajectories)
            norm_diff = float(np.linalg.norm(vcg_kors_sub - vcg_dower_sub) / (np.linalg.norm(vcg_kors_sub) + 1e-8))

            records.append({
                "ecg_id": rec_id,
                "primary_class": primary_class,
                "superclasses": ",".join(superclasses),
                "cliques_kors": sum_kors["n_cliques"],
                "cliques_dower": sum_dower["n_cliques"],
                "anomaly_kors": sum_kors["anomaly_fraction"],
                "anomaly_dower": sum_dower["anomaly_fraction"],
                "planarity_kors": planarity_kors,
                "planarity_dower": planarity_dower,
                "qrst_kors": qrst_kors,
                "qrst_dower": qrst_dower,
                "kors_dower_rel_diff": norm_diff,
                "report": report,
            })

            print(
                f"[{idx+1}/{len(record_ids)}] PTB-XL {rec_id} ({primary_class}): "
                f"Cliques={sum_kors['n_cliques']}, Anomaly={sum_kors['anomaly_fraction']:.2f}, "
                f"Planarity={planarity_kors:.3f}, QRST_Angle={qrst_kors:.1f}° | Diff(Kors,Dower)={norm_diff:.3f}"
            )

            # Store for NORM vs MI comparison plot
            if primary_class == "NORM" and norm_rec_data is None:
                norm_rec_data = {
                    "ecg_id": rec_id,
                    "vcg": vcg_kors_sub,
                    "labels": labels_kors,
                    "time": time_sub,
                    "ecg": ecg_sub,
                    "planarity": planarity_kors,
                    "qrst": qrst_kors,
                }
            if primary_class == "MI" and mi_rec_data is None:
                mi_rec_data = {
                    "ecg_id": rec_id,
                    "vcg": vcg_kors_sub,
                    "labels": labels_kors,
                    "time": time_sub,
                    "ecg": ecg_sub,
                    "planarity": planarity_kors,
                    "qrst": qrst_kors,
                }

        except Exception as exc:
            import traceback
            print(f"Error processing PTB-XL record {rec_id}: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "ptbxl_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved PTB-XL results to: {csv_path}")

    # Build Master Report
    summary = df.groupby("primary_class")[["planarity_kors", "qrst_kors", "anomaly_kors", "cliques_kors"]].agg(["mean", "std"]).reset_index()
    report_dict = {
        "dataset": "PTB-XL",
        "n_evaluated": len(df),
        "classes_evaluated": list(df["primary_class"].unique()),
        "mean_kors_dower_rel_diff": float(df["kors_dower_rel_diff"].mean()),
        "class_summary": {
            sc: {
                "mean_planarity": float(df[df["primary_class"] == sc]["planarity_kors"].mean()),
                "std_planarity": float(df[df["primary_class"] == sc]["planarity_kors"].std()),
                "mean_qrst_angle": float(df[df["primary_class"] == sc]["qrst_kors"].mean()),
                "std_qrst_angle": float(df[df["primary_class"] == sc]["qrst_kors"].std()),
                "mean_anomaly_fraction": float(df[df["primary_class"] == sc]["anomaly_kors"].mean()),
                "mean_cliques": float(df[df["primary_class"] == sc]["cliques_kors"].mean()),
            }
            for sc in df["primary_class"].unique()
        },
        "transform_sensitivity_kors_vs_dower": {
            "planarity_correlation": float(df["planarity_kors"].corr(df["planarity_dower"])),
            "qrst_angle_correlation": float(df["qrst_kors"].corr(df["qrst_dower"])),
            "clique_agreement_rate": float((df["cliques_kors"] == df["cliques_dower"]).mean()),
        }
    }
    report_path = os.path.join(output_dir, "PTBXL_BENCHMARK_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved PTB-XL report to: {report_path}")

    # ==================== FIGURES ====================

    # Figure 9: 3D VCG Dipole Loops NORM vs MI (Ghosal et al. 2024 Planarity Hypothesis)
    if norm_rec_data is not None and mi_rec_data is not None:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        # NORM
        v_n = norm_rec_data["vcg"]
        l_n = norm_rec_data["labels"]
        ax1.scatter(v_n[:, 0], v_n[:, 1], v_n[:, 2], c=l_n, cmap="tab10", s=6, alpha=0.7)
        ax1.plot(v_n[:, 0], v_n[:, 1], v_n[:, 2], color="gray", alpha=0.3, linewidth=0.5)
        ax1.set_title(
            f"PTB-XL {norm_rec_data['ecg_id']} (NORM Control)\nPlanarity = {norm_rec_data['planarity']:.3f} | QRS-T Angle = {norm_rec_data['qrst']:.1f}°",
            fontsize=11, fontweight="bold",
        )
        ax1.set_xlabel("Vx (mV)")
        ax1.set_ylabel("Vy (mV)")
        ax1.set_zlabel("Vz (mV)")

        # MI
        v_m = mi_rec_data["vcg"]
        l_m = mi_rec_data["labels"]
        ax2.scatter(v_m[:, 0], v_m[:, 1], v_m[:, 2], c=l_m, cmap="tab10", s=6, alpha=0.7)
        ax2.plot(v_m[:, 0], v_m[:, 1], v_m[:, 2], color="gray", alpha=0.3, linewidth=0.5)
        ax2.set_title(
            f"PTB-XL {mi_rec_data['ecg_id']} (Myocardial Infarction)\nPlanarity = {mi_rec_data['planarity']:.3f} | QRS-T Angle = {mi_rec_data['qrst']:.1f}°",
            fontsize=11, fontweight="bold",
        )
        ax2.set_xlabel("Vx (mV)")
        ax2.set_ylabel("Vy (mV)")
        ax2.set_zlabel("Vz (mV)")

        plt.suptitle("3D VCG Dipole Loop Distortion in Acute Myocardial Infarction (PTB-XL)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        f9_path = os.path.join(fig_dir, "fig9_ptbxl_norm_vs_mi_loops.png")
        fig.savefig(f9_path, dpi=300)
        plt.close(fig)
        print(f"Saved Figure 9: {f9_path}")

    # Figure 10: Clinical Biomarkers Across PTB-XL Superclasses
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    present = [c for c in classes if c in df["primary_class"].unique()]

    # 1. Loop Planarity
    p_means = [df[df["primary_class"] == c]["planarity_kors"].mean() for c in present]
    p_stds = [df[df["primary_class"] == c]["planarity_kors"].std() for c in present]
    bars1 = axes[0].bar(present, p_means, yerr=p_stds, capsize=4, color="#3b82f6", alpha=0.85, width=0.5)
    axes[0].set_ylabel("Loop Planarity Index", fontsize=11, fontweight="bold")
    axes[0].set_title("VCG Loop Planarity (Ghosal et al. 2024)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].set_ylim(0.85, 1.0)
    for b, v in zip(bars1, p_means):
        axes[0].text(b.get_x() + b.get_width() / 2.0, v + 0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # 2. Spatial QRS-T Angle
    q_means = [df[df["primary_class"] == c]["qrst_kors"].mean() for c in present]
    q_stds = [df[df["primary_class"] == c]["qrst_kors"].std() for c in present]
    bars2 = axes[1].bar(present, q_means, yerr=q_stds, capsize=4, color="#ef4444", alpha=0.85, width=0.5)
    axes[1].set_ylabel("Spatial QRS-T Angle (°)", fontsize=11, fontweight="bold")
    axes[1].set_title("Spatial QRS-T Angle (Kumar/Hughes et al.)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[1].set_ylim(0, 180)
    for b, v in zip(bars2, q_means):
        axes[1].text(b.get_x() + b.get_width() / 2.0, v + 3.0, f"{v:.1f}°", ha="center", fontsize=9, fontweight="bold")

    # 3. Anomaly Fraction
    a_means = [df[df["primary_class"] == c]["anomaly_kors"].mean() for c in present]
    a_stds = [df[df["primary_class"] == c]["anomaly_kors"].std() for c in present]
    bars3 = axes[2].bar(present, a_means, yerr=a_stds, capsize=4, color="#10b981", alpha=0.85, width=0.5)
    axes[2].set_ylabel("Anomaly / Singleton Fraction", fontsize=11, fontweight="bold")
    axes[2].set_title("repSpat Anomaly Fraction", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[2].set_ylim(0, 1.0)
    for b, v in zip(bars3, a_means):
        axes[2].text(b.get_x() + b.get_width() / 2.0, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    f10_path = os.path.join(fig_dir, "fig10_ptbxl_pathology_biomarkers.png")
    fig.savefig(f10_path, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 10: {f10_path}")

    # Figure 11: Transform Sensitivity (Kors vs Dower)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(df["planarity_kors"], df["planarity_dower"], c="#6366f1", s=40, alpha=0.8)
    axes[0].plot([0.85, 1.0], [0.85, 1.0], "r--", alpha=0.6, label="Identity Line")
    axes[0].set_xlabel("Kors Loop Planarity", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Dower Loop Planarity", fontsize=11, fontweight="bold")
    axes[0].set_title("Planarity Consistency: Kors vs Dower", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].scatter(df["qrst_kors"], df["qrst_dower"], c="#ec4899", s=40, alpha=0.8)
    axes[1].plot([0, 180], [0, 180], "r--", alpha=0.6, label="Identity Line")
    axes[1].set_xlabel("Kors Spatial QRS-T Angle (°)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Dower Spatial QRS-T Angle (°)", fontsize=11, fontweight="bold")
    axes[1].set_title("QRS-T Angle Consistency: Kors vs Dower", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.suptitle("Transform Sensitivity Analysis (Section 10.1 of Critical Review)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    f11_path = os.path.join(fig_dir, "fig11_ptbxl_kors_vs_dower_sensitivity.png")
    fig.savefig(f11_path, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 11: {f11_path}")

    # Figure 12: Representative PTB-XL ECG Timeline
    if norm_rec_data is not None:
        plot_ecg_timeline_with_reps(
            norm_rec_data["time"],
            norm_rec_data["ecg"][:, 1],
            norm_rec_data["labels"],
            lead_name="Lead II",
            title=f"PTB-XL {norm_rec_data['ecg_id']} Discovered REPs along 12-Lead ECG",
            save_path=os.path.join(fig_dir, "fig12_ptbxl_rep_timeline.png"),
        )
        plt.close("all")
        print(f"Saved Figure 12: {os.path.join(fig_dir, 'fig12_ptbxl_rep_timeline.png')}")

    return df


if __name__ == "__main__":
    run_ptbxl_benchmark()
