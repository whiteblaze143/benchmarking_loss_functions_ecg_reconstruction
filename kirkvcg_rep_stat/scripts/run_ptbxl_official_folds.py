"""Large-Scale KirkVCG repSpat Evaluation on PTB-XL Official Stratified Folds.

Processes patient ECG recordings across official PTB-XL stratified folds:
- Folds 1–8: Training set (~17,418 records)
- Fold 9: Validation set (~2,183 records)
- Fold 10: Official Held-Out Test Set (~2,198 records)

Uses multi-core parallel processing (8 CPU workers) to compute:
1. 3D VCG dipole trajectory via Kors regression matrix
2. Continuous Frenet-Serret kinematics (speed, curvature, torsion, normal acceleration)
3. KirkVCG repSpat constrained hierarchical clustering & IMQ MMD block permutation
4. Discovered Repeated Electrophysiological Patterns (maximal cliques)
5. Clinical VCG biomarkers: Loop Planarity (Ghosal et al. 2024), Spatial QRS-T Angle (Kumar/Hughes et al.), Anomaly Fraction
6. Aggregates results across diagnostic superclasses (NORM, MI, STTC, CD, HYP)
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from kirkvcg_rep_stat.src.pipeline import KirkVCGRepSpat
from kirkvcg_rep_stat.src.features.ecg_features import extract_sample_features
from kirkvcg_rep_stat.src.vcg.transform import ecg_to_vcg_kors
from kirkvcg_rep_stat.src.vcg.kinematics import compute_vcg_loop_planarity, compute_spatial_qrst_angle


def process_single_ptbxl_record(
    record_info: dict,
) -> dict | None:
    """Worker function to process one PTB-XL patient record with KirkVCG repSpat."""
    try:
        tensor_path = record_info["tensor_path"]
        ecg_id = record_info["ecg_id"]
        fold = record_info["fold"]
        primary_class = record_info["primary_class"]
        superclasses = record_info["superclasses"]
        age = record_info["age"]
        sex = record_info["sex"]

        # Load tensor [12, 5000]
        t = torch.load(tensor_path, map_location="cpu", weights_only=False)
        if isinstance(t, torch.Tensor):
            ecg = t.numpy()
        elif isinstance(t, dict) and "waveform" in t:
            ecg = t["waveform"].numpy()
        else:
            ecg = np.asarray(t)

        if ecg.shape[0] == 12 and ecg.shape[1] > 12:
            ecg = ecg.T

        # Evaluate on first 1500 samples (3.0s window)
        n_eval = min(1500, len(ecg))
        ecg_sub = ecg[:n_eval]

        # 3D VCG trajectory via Kors regression
        vcg_sub = ecg_to_vcg_kors(ecg_sub)

        # Extract continuous physical attributes
        feat_dict = extract_sample_features(ecg_sub, vcg=vcg_sub, fs=500.0)
        X_cont = feat_dict["continuous"]

        # Fit KirkVCG repSpat (25 block permutations for high throughput)
        kirk = KirkVCGRepSpat(
            metric="euclidean",
            m_neighbors=4,
            n_clusters=6,
            use_hybrid=True,
            temporal_window=1,
            kernel="imq",
            kernel_param=1.0,
            n_permutations=25,
            alpha=0.05,
            min_clique_size=3,
            random_state=42,
        )
        labels = kirk.fit_predict(X_cont, vcg_sub)
        summary = kirk.get_summary()

        enc = summary.get("encoding") or {}
        planarity = float(enc.get("loop_planarity", 0.0))
        if planarity == 0.0:
            planarity = float(compute_vcg_loop_planarity(vcg_sub))
        qrst_angle = float(enc.get("spatial_qrst_angle", 0.0))

        return {
            "ecg_id": int(ecg_id),
            "fold": int(fold),
            "primary_class": str(primary_class),
            "superclasses": str(superclasses),
            "age": float(age),
            "sex": str(sex),
            "n_cliques": int(summary["n_cliques"]),
            "n_rep_states": int(summary["final_rep_labels_count"]),
            "anomaly_fraction": float(summary["anomaly_fraction"]),
            "loop_planarity": float(planarity),
            "spatial_qrst_angle": float(qrst_angle),
        }
    except Exception:
        return None


def run_official_folds_evaluation(
    folds: list[int] = (10,),
    max_per_fold: int | None = None,
    n_jobs: int = 8,
    data_dir: str = "data/ptb_xl",
    output_dir: str = "kirkvcg_rep_stat/results",
    fig_dir: str = "kirkvcg_rep_stat/figures",
) -> pd.DataFrame:
    """Runs parallel KirkVCG repSpat evaluation across specified official PTB-XL folds."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("KIRKVCG repSpat: LARGE-SCALE PTB-XL OFFICIAL FOLDS EVALUATION")
    print(f"Target Folds: {list(folds)} | Workers: {n_jobs} | Max per fold: {max_per_fold or 'ALL'}")
    print("=" * 80)

    # Load database metadata
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_csv = os.path.join(data_dir, "scp_statements.csv")
    df = pd.read_csv(csv_path)

    scp_df = pd.read_csv(scp_csv, index_col=0)
    diag_scp = scp_df[scp_df["diagnostic"] == 1.0]

    def get_classes(codes_str):
        d = ast.literal_eval(codes_str) if isinstance(codes_str, str) else {}
        classes = set()
        for k in d.keys():
            if k in diag_scp.index:
                classes.add(diag_scp.loc[k, "diagnostic_class"])
        return list(classes)

    def get_primary(classes):
        for sc in ["MI", "CD", "STTC", "HYP"]:
            if sc in classes:
                return sc
        if "NORM" in classes:
            return "NORM"
        return "OTHER"

    df["superclasses_list"] = df["scp_codes"].apply(get_classes)
    df["primary_class"] = df["superclasses_list"].apply(get_primary)

    # Build work items matching tensor files
    work_items = []
    split_names = {10: "test", 9: "val"}
    for f in range(1, 9):
        split_names[f] = "train"

    for fold in folds:
        fold_df = df[df["strat_fold"] == fold]
        split_name = split_names.get(fold, "train")
        tensor_dir = os.path.join(data_dir, "tensors", split_name)

        count = 0
        for _, row in fold_df.iterrows():
            ecg_id = int(row["ecg_id"])
            t_path = os.path.join(tensor_dir, f"{ecg_id}.pt")
            if os.path.isfile(t_path):
                work_items.append({
                    "ecg_id": ecg_id,
                    "fold": fold,
                    "tensor_path": t_path,
                    "primary_class": row["primary_class"],
                    "superclasses": ",".join(row["superclasses_list"]),
                    "age": float(row.get("age", 60.0)) if not pd.isna(row.get("age")) else 60.0,
                    "sex": str(row.get("sex", "M")),
                })
                count += 1
                if max_per_fold and count >= max_per_fold:
                    break

    print(f"Prepared {len(work_items)} patient records across folds {list(folds)}.")
    print(f"Launching parallel execution on {n_jobs} CPU worker processes...")

    results = []
    completed = 0
    total = len(work_items)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_single_ptbxl_record, item): item["ecg_id"] for item in work_items}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            completed += 1
            if res is not None:
                results.append(res)
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / max(0.1, elapsed)
                remaining = (total - completed) / max(0.01, rate)
                print(f"Progress: [{completed:4d}/{total:4d}] ({completed/total:5.1%}) | Rate: {rate:4.1f} rec/s | ETA: {remaining:4.0f}s | Valid: {len(results)}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s ({len(results)} valid records).")

    res_df = pd.DataFrame(results)
    csv_out = os.path.join(output_dir, f"ptbxl_folds_{'_'.join(map(str, folds))}_results.csv")
    parquet_out = os.path.join(output_dir, f"ptbxl_folds_{'_'.join(map(str, folds))}_results.parquet")
    res_df.to_csv(csv_out, index=False)
    res_df.to_parquet(parquet_out, index=False)
    print(f"Saved results to: {csv_out} and {parquet_out}")

    # Build Comprehensive Report
    class_summary = {}
    for sc in res_df["primary_class"].unique():
        sub = res_df[res_df["primary_class"] == sc]
        class_summary[sc] = {
            "n_records": len(sub),
            "mean_planarity": float(sub["loop_planarity"].mean()),
            "std_planarity": float(sub["loop_planarity"].std()),
            "mean_qrst_angle": float(sub["spatial_qrst_angle"].mean()),
            "std_qrst_angle": float(sub["spatial_qrst_angle"].std()),
            "mean_anomaly_fraction": float(sub["anomaly_fraction"].mean()),
            "std_anomaly_fraction": float(sub["anomaly_fraction"].std()),
            "mean_cliques": float(sub["n_cliques"].mean()),
        }

    fold_summary = {}
    for fld in res_df["fold"].unique():
        sub = res_df[res_df["fold"] == fld]
        fold_summary[str(fld)] = {
            "n_records": len(sub),
            "mean_planarity": float(sub["loop_planarity"].mean()),
            "mean_qrst_angle": float(sub["spatial_qrst_angle"].mean()),
            "mean_anomaly_fraction": float(sub["anomaly_fraction"].mean()),
            "mean_cliques": float(sub["n_cliques"].mean()),
        }

    report = {
        "status": "COMPLETED",
        "dataset": "PTB-XL",
        "official_folds_evaluated": list(folds),
        "total_records_evaluated": len(res_df),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_records_per_sec": round(len(res_df) / max(0.1, elapsed), 2),
        "diagnostic_class_summary": class_summary,
        "fold_summary": fold_summary,
    }

    report_path = os.path.join(output_dir, f"PTBXL_FOLDS_{'_'.join(map(str, folds))}_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Master folds report saved to: {report_path}")
    print(json.dumps(report, indent=2))

    # Figure 13: Clinical Biomarkers Distribution across PTB-XL Diagnostic Superclasses
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        classes_order = ["NORM", "MI", "STTC", "CD", "HYP"]
        present = [c for c in classes_order if c in res_df["primary_class"].unique()]

        data_p = [res_df[res_df["primary_class"] == c]["loop_planarity"].dropna().values for c in present]
        bp1 = axes[0].boxplot(data_p, tick_labels=present, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch in bp1['boxes']:
            patch.set_facecolor("#3b82f6")
            patch.set_alpha(0.7)
        axes[0].set_ylabel("Loop Planarity Index", fontsize=11, fontweight="bold")
        axes[0].set_title("3D VCG Loop Planarity (Ghosal et al. 2024)", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

        data_q = [res_df[res_df["primary_class"] == c]["spatial_qrst_angle"].dropna().values for c in present]
        bp2 = axes[1].boxplot(data_q, tick_labels=present, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch in bp2['boxes']:
            patch.set_facecolor("#ef4444")
            patch.set_alpha(0.7)
        axes[1].set_ylabel("Spatial QRS-T Angle (°)", fontsize=11, fontweight="bold")
        axes[1].set_title("Spatial QRS-T Angle (Kumar / Hughes et al.)", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

        data_a = [res_df[res_df["primary_class"] == c]["anomaly_fraction"].dropna().values for c in present]
        bp3 = axes[2].boxplot(data_a, tick_labels=present, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch in bp3['boxes']:
            patch.set_facecolor("#10b981")
            patch.set_alpha(0.7)
        axes[2].set_ylabel("Anomaly / Singleton Fraction", fontsize=11, fontweight="bold")
        axes[2].set_title("repSpat Anomaly Fraction", fontsize=11, fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")

        plt.suptitle(
            f"PTB-XL Official Benchmark ({'Folds ' + ','.join(map(str, folds))}, N={len(res_df)} Patients): Biomarkers by Superclass",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        f13_path = os.path.join(fig_dir, f"fig13_ptbxl_folds_{'_'.join(map(str, folds))}_distributions.png")
        fig.savefig(f13_path, dpi=300)
        plt.close(fig)
        print(f"Saved Figure 13: {f13_path}")
    except Exception as exc:
        print(f"Figure generation error: {exc}")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KirkVCG repSpat on PTB-XL official folds.")
    parser.add_argument("--folds", nargs="+", type=int, default=[10], help="Folds to evaluate (e.g. 10 or 1 2 3 ... 10)")
    parser.add_argument("--max_per_fold", type=int, default=None, help="Max records per fold (default: all)")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    args = parser.parse_args()

    run_official_folds_evaluation(
        folds=args.folds,
        max_per_fold=args.max_per_fold,
        n_jobs=args.n_jobs,
    )
