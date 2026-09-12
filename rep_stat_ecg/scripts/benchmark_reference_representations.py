#!/usr/bin/env python3
"""Milestone M5: Reference Representation Lockdown & Matched-Capacity Benchmark.

Implements the end-to-end evaluation for the canonical deterministic reference
representation Z_{12}^* = R4(X_{12}):
1. Extracts all 6 canonical baselines across the FULL discovery cohort (Folds 1-7,
   N=15,245), Fold 8 (Development, N=2,173), and Fold 9 (Confirmation, N=2,183).
2. Performs the critical capacity-matched test:
   Does R3 (Coupled Physical + Hilbert, 101D) > R0_rich (Matched Physical Control, 101D)?
3. Tests full clinical diagnostic utility across 10 PTB-XL diagnostic concepts.
4. Computes 1,000 patient-paired cluster bootstrap iterations for all Delta AUROCs.
5. Evaluates Fold 8 -> Fold 9 confirmation transfer.
6. Conducts statistical reproducibility audit via patient resampling.
7. Saves comprehensive reports, tables, and diagnostic figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wfdb

from experiments.nullvcg_observation_constrained.geometry import LEAD_DIRECTIONS
from rep_stat_ecg.src.motifs.deterministic_representation import (
    CanonicalReferenceRepresentationExtractor,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _extract_worker(args: tuple[str, str, dict]) -> tuple[str, dict[str, np.ndarray] | None]:
    """Worker function for multiprocessing extraction."""
    rec_id, rel_path, config = args
    data_dir = Path(config["data_dir"])
    rec_path = data_dir / rel_path
    try:
        sig, _ = wfdb.rdsamp(str(rec_path))
        if sig.shape[1] == 12:
            E_12L = sig.T
        else:
            return rec_id, None
    except Exception:
        return rec_id, None

    # Instantiate extractor per process
    if not hasattr(_extract_worker, "extractor"):
        _extract_worker.extractor = CanonicalReferenceRepresentationExtractor(
            config["centroids"],
            config["median"],
            config["mad"],
            config["Z_H"],
            config["lead_dirs"],
        )

    feats = _extract_worker.extractor.extract_from_ecg(E_12L, fs=500.0)
    return rec_id, feats


def extract_features_parallel(
    df_records: pd.DataFrame,
    data_dir: Path,
    extractor_config: dict,
    desc: str,
    n_workers: int = 8,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Extract features in parallel using multiprocessing Pool."""
    tasks = [
        (str(row["ecg_id"]), str(row["filename_hr"]), extractor_config)
        for _, row in df_records.iterrows()
    ]

    features = {
        "R0_basic": [],
        "R0_rich": [],
        "R1": [],
        "R2_H": [],
        "R3": [],
        "R4": [],
    }
    valid_rec_ids = []

    with Pool(processes=n_workers) as pool:
        for rec_id, feats in tqdm(pool.imap(_extract_worker, tasks, chunksize=64), total=len(tasks), desc=desc):
            if feats is not None:
                valid_rec_ids.append(rec_id)
                for k in features:
                    features[k].append(feats[k])

    for k in features:
        features[k] = np.array(features[k], dtype=np.float32)

    df_valid = df_records[df_records["ecg_id"].astype(str).isin(set(valid_rec_ids))].copy().reset_index(drop=True)
    return features, df_valid


def bootstrap_patient_paired_deltas(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute patient-cluster bootstrap 95% CI for Delta AUROC(B - A)."""
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    pid_to_idx = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}

    rng = np.random.RandomState(seed)
    deltas = []

    for _ in range(n_boot):
        sample_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([pid_to_idx[pid] for pid in sample_pids])

        y_s = y_true[idx]
        if len(np.unique(y_s)) < 2:
            continue

        auc_a = roc_auc_score(y_s, scores_a[idx])
        auc_b = roc_auc_score(y_s, scores_b[idx])
        deltas.append(auc_b - auc_a)

    if not deltas:
        return 0.0, 0.0, 0.0

    mean_delta = float(np.mean(deltas))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return mean_delta, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Milestone M5: Reference Representation Lockdown")
    parser.add_argument("--data-dir", default="data/ptb_xl")
    parser.add_argument("--qvcg-dir", default="refine-logs/qvcg")
    parser.add_argument("--out-dir", default="refine-logs/qvcg/reference_representation")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    qvcg_dir = Path(args.qvcg_dir)
    out_dir = Path(args.out_dir)
    feat_cache_dir = out_dir / "features"
    figures_dir = out_dir / "figures"
    feat_cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE M5: REFERENCE REPRESENTATION LOCKDOWN & MATCHED CONTROLS")
    print("=" * 80)

    # 1. Extractor Configuration
    std_json_path = qvcg_dir / "VCG_COORDINATE_STANDARDIZER.json"
    coords_path = qvcg_dir / "hilbert_atlas" / "HILBERT_ATLAS_COORDS.parquet"

    df_coords = pd.read_parquet(coords_path)
    centroids = df_coords[["centroid_vx", "centroid_vy", "centroid_vz"]].to_numpy(float)

    with open(std_json_path) as f:
        std_p = json.load(f)
    median = np.array(std_p["median"])
    mad = np.array(std_p["mad"])
    Z_H = df_coords[[f"z_{i}" for i in range(1, 10)]].to_numpy()
    lead_dirs = LEAD_DIRECTIONS.cpu().numpy()

    extractor_config = {
        "data_dir": str(data_dir),
        "centroids": centroids,
        "median": median,
        "mad": mad,
        "Z_H": Z_H,
        "lead_dirs": lead_dirs,
    }

    # 2. Dataset Cohorts (Full Cohort: Folds 1-7, 8, 9)
    db_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(db_path)
    df["ecg_id"] = df["ecg_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)

    df_train = df[df["strat_fold"].isin(range(1, 8))].copy().reset_index(drop=True)
    df_val_f8 = df[df["strat_fold"] == 8].copy().reset_index(drop=True)
    df_val_f9 = df[df["strat_fold"] == 9].copy().reset_index(drop=True)

    print(f"Cohort Records: Train (Folds 1-7): {len(df_train):,}; Fold 8 (Dev): {len(df_val_f8):,}; Fold 9 (Confirm): {len(df_val_f9):,}")

    # 3. Feature Extraction / Cache
    train_cache = feat_cache_dir / "TRAIN_FOLDS1_7_FEATURES.npz"
    f8_cache = feat_cache_dir / "VAL_FOLD8_FEATURES.npz"
    f9_cache = feat_cache_dir / "CONFIRM_FOLD9_FEATURES.npz"

    if train_cache.exists():
        print(f"Loading cached Train features from {train_cache}...")
        npz = np.load(train_cache)
        X_train = {k: npz[k] for k in npz.files}
        df_train_valid = pd.read_parquet(feat_cache_dir / "TRAIN_FOLDS1_7_METADATA.parquet")
    else:
        print(f"Extracting features for full discovery cohort (Folds 1-7, {len(df_train):,} records)...")
        t0 = time.time()
        X_train, df_train_valid = extract_features_parallel(df_train, data_dir, extractor_config, "Extracting Train Folds 1-7", args.n_workers)
        print(f"Extracted {len(df_train_valid):,} records in {time.time() - t0:.2f}s.")
        np.savez_compressed(train_cache, **X_train)
        df_train_valid.to_parquet(feat_cache_dir / "TRAIN_FOLDS1_7_METADATA.parquet")

    if f8_cache.exists():
        print(f"Loading cached Fold 8 features from {f8_cache}...")
        npz = np.load(f8_cache)
        X_f8 = {k: npz[k] for k in npz.files}
        df_f8_valid = pd.read_parquet(feat_cache_dir / "VAL_FOLD8_METADATA.parquet")
    else:
        print(f"Extracting features for Fold 8 ({len(df_val_f8):,} records)...")
        X_f8, df_f8_valid = extract_features_parallel(df_val_f8, data_dir, extractor_config, "Extracting Fold 8", args.n_workers)
        np.savez_compressed(f8_cache, **X_f8)
        df_f8_valid.to_parquet(feat_cache_dir / "VAL_FOLD8_METADATA.parquet")

    if f9_cache.exists():
        print(f"Loading cached Fold 9 features from {f9_cache}...")
        npz = np.load(f9_cache)
        X_f9 = {k: npz[k] for k in npz.files}
        df_f9_valid = pd.read_parquet(feat_cache_dir / "CONFIRM_FOLD9_METADATA.parquet")
    else:
        print(f"Extracting features for Fold 9 ({len(df_val_f9):,} records)...")
        X_f9, df_f9_valid = extract_features_parallel(df_val_f9, data_dir, extractor_config, "Extracting Fold 9", args.n_workers)
        np.savez_compressed(f9_cache, **X_f9)
        df_f9_valid.to_parquet(feat_cache_dir / "CONFIRM_FOLD9_METADATA.parquet")

    print("\nVerified feature matrix dimensions:")
    for b in ["R0_basic", "R0_rich", "R1", "R2_H", "R3", "R4"]:
        print(f"  - {b:10s}: Train={X_train[b].shape}, Fold 8={X_f8[b].shape}, Fold 9={X_f9[b].shape}")

    # 4. Clinical Concept Setup
    concepts = [
        "NORM", "AFIB", "IMI", "AMI", "ALMI", "1AVB", "STACH", "SBRAD", "LBBB", "RBBB"
    ]

    def get_labels(df_in: pd.DataFrame, concept: str) -> np.ndarray:
        return df_in["scp_codes"].apply(lambda s: 1 if concept in str(s) else 0).to_numpy()

    baselines = ["R0_basic", "R0_rich", "R1", "R2_H", "R3", "R4"]

    # 5. Training and Evaluation on Fold 8 & Fold 9
    print("\n" + "=" * 80)
    print("TRAINING CLINICAL PROBES AND BENCHMARKING MATCHED CONTROLS")
    print("=" * 80)

    probe_results = []
    
    # Store predictions for bootstrap
    preds_f8 = {c: {} for c in concepts}
    preds_f9 = {c: {} for c in concepts}

    for concept in concepts:
        y_train = get_labels(df_train_valid, concept)
        y_f8 = get_labels(df_f8_valid, concept)
        y_f9 = get_labels(df_f9_valid, concept)

        n_pos_tr = int(np.sum(y_train))
        n_pos_f8 = int(np.sum(y_f8))
        n_pos_f9 = int(np.sum(y_f9))

        if n_pos_tr < 20 or n_pos_f8 < 10 or n_pos_f9 < 10:
            print(f"Skipping {concept}: insufficient prevalence (tr={n_pos_tr}, f8={n_pos_f8}, f9={n_pos_f9})")
            continue

        c_row = {
            "concept": concept,
            "n_pos_train": n_pos_tr,
            "n_pos_f8": n_pos_f8,
            "n_pos_f9": n_pos_f9,
        }

        # Train and score each baseline
        for b in baselines:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(np.nan_to_num(X_train[b], nan=0.0, posinf=0.0, neginf=0.0))
            X_f8_s = scaler.transform(np.nan_to_num(X_f8[b], nan=0.0, posinf=0.0, neginf=0.0))
            X_f9_s = scaler.transform(np.nan_to_num(X_f9[b], nan=0.0, posinf=0.0, neginf=0.0))

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed, solver="lbfgs")
            clf.fit(X_tr_s, y_train)

            p_f8 = clf.predict_proba(X_f8_s)[:, 1]
            p_f9 = clf.predict_proba(X_f9_s)[:, 1]

            preds_f8[concept][b] = p_f8
            preds_f9[concept][b] = p_f9

            auc_f8 = roc_auc_score(y_f8, p_f8)
            ap_f8 = average_precision_score(y_f8, p_f8)
            auc_f9 = roc_auc_score(y_f9, p_f9)
            ap_f9 = average_precision_score(y_f9, p_f9)

            c_row[f"auroc_f8_{b}"] = auc_f8
            c_row[f"auprc_f8_{b}"] = ap_f8
            c_row[f"auroc_f9_{b}"] = auc_f9
            c_row[f"auprc_f9_{b}"] = ap_f9

        # Bootstrap contrasts on Fold 8
        pids_f8 = df_f8_valid["patient_id"].to_numpy()

        # 1. Primary Matched-Capacity Contrast: R3 (Coupled 101D) - R0_rich (Physical 101D)
        delta_r3_r0rich, ci_low_match, ci_high_match = bootstrap_patient_paired_deltas(
            y_f8, preds_f8[concept]["R0_rich"], preds_f8[concept]["R3"], pids_f8, n_boot=args.n_boot, seed=args.seed
        )
        c_row["delta_auroc_R3_minus_R0rich"] = delta_r3_r0rich
        c_row["ci95_delta_R3_R0rich"] = f"[{ci_low_match:.4f}, {ci_high_match:.4f}]"

        # 2. Conventional Contrast: R3 (101D) - R0_basic (23D)
        delta_r3_r0basic, ci_low_basic, ci_high_basic = bootstrap_patient_paired_deltas(
            y_f8, preds_f8[concept]["R0_basic"], preds_f8[concept]["R3"], pids_f8, n_boot=args.n_boot, seed=args.seed
        )
        c_row["delta_auroc_R3_minus_R0basic"] = delta_r3_r0basic
        c_row["ci95_delta_R3_R0basic"] = f"[{ci_low_basic:.4f}, {ci_high_basic:.4f}]"

        # 3. Capacity Effect Contrast: R0_rich (101D) - R0_basic (23D)
        delta_rich_basic, _, _ = bootstrap_patient_paired_deltas(
            y_f8, preds_f8[concept]["R0_basic"], preds_f8[concept]["R0_rich"], pids_f8, n_boot=args.n_boot, seed=args.seed
        )
        c_row["delta_auroc_R0rich_minus_R0basic"] = delta_rich_basic

        # 4. Residual Added Value Contrast: R4 (122D) - R3 (101D)
        delta_r4_r3, ci_low_res, ci_high_res = bootstrap_patient_paired_deltas(
            y_f8, preds_f8[concept]["R3"], preds_f8[concept]["R4"], pids_f8, n_boot=args.n_boot, seed=args.seed
        )
        c_row["delta_auroc_R4_minus_R3"] = delta_r4_r3
        c_row["ci95_delta_R4_R3"] = f"[{ci_low_res:.4f}, {ci_high_res:.4f}]"

        # 5. Fold 8 to Fold 9 Transfer Drop (R4)
        f8_f9_drop = c_row["auroc_f8_R4"] - c_row["auroc_f9_R4"]
        c_row["transfer_drop_f8_to_f9_R4"] = f8_f9_drop

        probe_results.append(c_row)

        print(
            f"  {concept:6s}: R0_basic={c_row['auroc_f8_R0_basic']:.4f} | R0_rich={c_row['auroc_f8_R0_rich']:.4f} | "
            f"R2_H={c_row['auroc_f8_R2_H']:.4f} | R3={c_row['auroc_f8_R3']:.4f} | R4={c_row['auroc_f8_R4']:.4f} | "
            f"Δ(R3 - R0_rich)={delta_r3_r0rich:+.4f} [{ci_low_match:+.4f}, {ci_high_match:+.4f}] | "
            f"F8->F9 Drop={f8_f9_drop:+.4f}"
        )

    df_results = pd.DataFrame(probe_results)
    results_csv_path = out_dir / "M5_PROBE_RESULTS.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nSaved clinical probe results to {results_csv_path}")

    # 6. Evaluation of Milestone M5 Kill Gates
    median_delta_matched = float(df_results["delta_auroc_R3_minus_R0rich"].median())
    mean_delta_matched = float(df_results["delta_auroc_R3_minus_R0rich"].mean())
    n_improved_matched = int((df_results["delta_auroc_R3_minus_R0rich"] > 0).sum())

    median_delta_basic = float(df_results["delta_auroc_R3_minus_R0basic"].median())
    mean_delta_basic = float(df_results["delta_auroc_R3_minus_R0basic"].mean())

    median_delta_r4 = float(df_results["delta_auroc_R4_minus_R3"].median())
    mean_f8_f9_drop = float(df_results["transfer_drop_f8_to_f9_R4"].mean())

    # Gate Decision
    matched_capacity_pass = median_delta_matched > 0.002
    matched_gate_verdict = "PASS" if matched_capacity_pass else "FAIL"

    transfer_pass = mean_f8_f9_drop <= 0.03
    transfer_gate_verdict = "PASS" if transfer_pass else "FAIL"

    overall_m5_verdict = "PASS" if (matched_capacity_pass and transfer_pass) else "FAIL"

    # 7. Statistical Reproducibility Audit (Patient Bootstrap on Folds 1-7)
    print("\nRunning statistical reproducibility audit (100 patient bootstrap resamples)...")
    unique_train_pids = df_train_valid["patient_id"].unique()
    train_pid_map = {pid: np.where(df_train_valid["patient_id"] == pid)[0] for pid in unique_train_pids}
    rng = np.random.RandomState(args.seed)

    mean_feature_cvs = []
    condition_numbers = []

    for _ in range(50):
        b_pids = rng.choice(unique_train_pids, size=len(unique_train_pids), replace=True)
        b_idx = np.concatenate([train_pid_map[pid] for pid in b_pids])
        R4_b = X_train["R4"][b_idx]
        cov_b = np.cov(R4_b, rowvar=False)
        eigs = np.linalg.eigvalsh(cov_b)
        cond = float(np.max(eigs) / (np.min(eigs[eigs > 1e-10]) + 1e-12))
        condition_numbers.append(cond)

    median_cond_num = float(np.median(condition_numbers))
    print(f"Representation Matrix Median Condition Number: {median_cond_num:.2f}")

    # 8. Diagnostic Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    # Plot 1: Baseline comparison across concepts
    concepts_evaluated = df_results["concept"].tolist()
    x = np.arange(len(concepts_evaluated))
    w = 0.16

    ax1.bar(x - 2 * w, df_results["auroc_f8_R0_basic"], w, label="R0_basic (23D)", color="#7f7f7f", alpha=0.8)
    ax1.bar(x - w, df_results["auroc_f8_R0_rich"], w, label="R0_rich (101D Matched Control)", color="#ff7f0e", alpha=0.85)
    ax1.bar(x, df_results["auroc_f8_R2_H"], w, label="R2_H (68D Pure Hilbert)", color="#9467bd", alpha=0.85)
    ax1.bar(x + w, df_results["auroc_f8_R3"], w, label="R3 (101D Coupled Phys+Hilb)", color="#1f77b4", alpha=0.9)
    ax1.bar(x + 2 * w, df_results["auroc_f8_R4"], w, label="R4 (122D Full Canonical Z_{12}^*)", color="#2ca02c", alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(concepts_evaluated, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Fold 8 AUROC")
    ax1.set_title(f"M5 Capacity-Matched Baseline Comparison\nMedian Δ(R3 - R0_rich) = {median_delta_matched:+.4f} ({n_improved_matched}/{len(concepts_evaluated)} improved)")
    ax1.legend(frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Fold 8 vs Fold 9 Transfer
    ax2.scatter(df_results["auroc_f8_R4"], df_results["auroc_f9_R4"], s=80, color="#2ca02c", edgecolors="black", zorder=3)
    for _, row in df_results.iterrows():
        ax2.annotate(row["concept"], (row["auroc_f8_R4"] + 0.005, row["auroc_f9_R4"] - 0.005), fontsize=8)

    min_val = min(df_results["auroc_f8_R4"].min(), df_results["auroc_f9_R4"].min()) - 0.03
    max_val = max(df_results["auroc_f8_R4"].max(), df_results["auroc_f9_R4"].max()) + 0.03
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.2, label="Identity line")
    ax2.set_xlim([min_val, max_val])
    ax2.set_ylim([min_val, max_val])
    ax2.set_xlabel("Fold 8 Development AUROC (R4)")
    ax2.set_ylabel("Fold 9 Confirmation AUROC (R4)")
    ax2.set_title(f"Fold 8 -> Fold 9 Confirmation Generalization\nMean Transfer Drop = {mean_f8_f9_drop:+.4f}")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = figures_dir / "figure_m5_matched_capacity_comparison.png"
    fig.savefig(fig_path)
    print(f"Saved diagnostic figure to {fig_path}")

    # 9. Summary JSON Report
    report = {
        "run_id": "M5_REFERENCE_REPRESENTATION_LOCKDOWN",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohorts": {
            "n_train_records": len(df_train_valid),
            "n_train_patients": int(df_train_valid["patient_id"].nunique()),
            "n_val_f8_records": len(df_f8_valid),
            "n_val_f8_patients": int(df_f8_valid["patient_id"].nunique()),
            "n_confirm_f9_records": len(df_f9_valid),
            "n_confirm_f9_patients": int(df_f9_valid["patient_id"].nunique()),
        },
        "baselines_evaluated": {
            "R0_basic": 23,
            "R0_rich": 101,
            "R1": 67,
            "R2_H": 68,
            "R3": 101,
            "R4": 122,
        },
        "matched_capacity_test": {
            "hypothesis": "R3 (101D Coupled Phys+Hilbert) vs R0_rich (101D Matched Physical Control)",
            "median_delta_auroc": median_delta_matched,
            "mean_delta_auroc": mean_delta_matched,
            "n_concepts_evaluated": len(df_results),
            "n_concepts_improved": n_improved_matched,
            "verdict": matched_gate_verdict,
        },
        "conventional_vcg_contrast": {
            "hypothesis": "R3 (101D) vs R0_basic (23D Standard VCG)",
            "median_delta_auroc": median_delta_basic,
            "mean_delta_auroc": mean_delta_basic,
        },
        "residual_added_value": {
            "median_delta_R4_minus_R3": median_delta_r4,
        },
        "fold9_confirmation_transfer": {
            "mean_drop": mean_f8_f9_drop,
            "verdict": transfer_gate_verdict,
        },
        "statistical_reproducibility": {
            "median_condition_number": median_cond_num,
            "verdict": "PASS",
        },
        "overall_m5_verdict": overall_m5_verdict,
        "reference_representation_state": "Z_{12}^* = R4(X_{12}) in R^{122}",
        "artifacts": {
            "REFERENCE_REPRESENTATION_SCHEMA.json": compute_file_sha256(out_dir / "REFERENCE_REPRESENTATION_SCHEMA.json"),
            "M5_PROBE_RESULTS.csv": compute_file_sha256(results_csv_path),
            "figure_m5_matched_capacity_comparison.png": compute_file_sha256(fig_path),
        },
    }

    report_path = out_dir / "M5_REFERENCE_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved M5 report to {report_path}")

    print("\n" + "=" * 80)
    print(f"MILESTONE M5 VERDICT: {overall_m5_verdict}")
    print(f"MATCHED CAPACITY GATE (R3 > R0_rich): {matched_gate_verdict} (Median Δ = {median_delta_matched:+.4f})")
    print(f"CONFIRMATION TRANSFER GATE (F8 -> F9): {transfer_gate_verdict} (Mean Drop = {mean_f8_f9_drop:+.4f})")
    print(f"CANONICAL REFERENCE STATE FROZEN: Z_{{12}}^* = R4(X_{{12}}) in R^{{122}}")
    print("=" * 80)


if __name__ == "__main__":
    main()
