#!/usr/bin/env python3
"""Milestone M1-C: Calibration-Efficiency Comparison Benchmark.

Head-to-head empirical comparison of calibration efficiency curves:
G_SO(K) = Performance_SO(K) - Performance_SO(0)
versus
G_CSFM(K) = Performance_CSFM(K) - Performance_CSFM(0)
for K in {0, 1, 2, 3, 5}.

Under identical personalization machinery:
- Reference construction:
    K=0:  X_0 = [z_t, 0]
    K=1:  X_1 = [z_t, z_t - z_{t-1}]
    K>=1: b_p^(K) = (1/K) * sum_{i=1}^K z_{p, t-i}
          X_K = [z_t, z_t - b_p^(K)]
- Evaluated on:
    1. SetOperator Native (d=256)
    2. CSFM Base Native (d=768)
    3. Matched Dimensionality d=256 (PCA on train patients)
    4. Matched Dimensionality d=128 (PCA on train patients)
- Clinical coverage vs K:
    Reports the proportion and exact count of clinical patients having K priors.
- Downstream Tasks:
    Task 1: State Change Detection (binary AUROC)
    Task 2: Diagnostic Superclass Classification (5-class Macro AUROC)
- Bootstrap 95% CIs:
    1,000 patient-clustered resamples for paired difference Delta G(K).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "prism_ecg_stage_m1_c_calibration_v1"
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("m1_c_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC computation."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


@dataclass
class TargetRecord:
    ecg_id: int
    patient_id: int
    pos: int
    strat_fold: int
    recording_date: pd.Timestamp
    y_multilabel: np.ndarray
    y_change: int
    prior_positions: List[int]
    prior_count: int


def build_target_records(
    df_multi: pd.DataFrame,
    labels: np.ndarray,
    logger: logging.Logger,
) -> List[TargetRecord]:
    """Construct strictly chronological target instances across repeated patients."""
    records: List[TargetRecord] = []
    for pid, group in df_multi.groupby("patient_id", sort=False):
        n_obs = len(group)
        if n_obs < 2:
            continue
        g_recs = group.to_dict("records")
        initial_label = labels[g_recs[0]["pos"]]

        for t_idx in range(1, n_obs):
            target = g_recs[t_idx]
            target_pos = target["pos"]
            y_t = labels[target_pos]
            target_date = target["recording_date"]

            priors = [r for r in g_recs[:t_idx] if r["recording_date"] < target_date]
            if len(priors) == 0:
                continue

            prior_pos_list = [r["pos"] for r in priors]
            y_change = int(not np.array_equal(y_t, initial_label))

            records.append(
                TargetRecord(
                    ecg_id=int(target["ecg_id"]),
                    patient_id=int(pid),
                    pos=target_pos,
                    strat_fold=int(target["strat_fold"]),
                    recording_date=target_date,
                    y_multilabel=y_t,
                    y_change=y_change,
                    prior_positions=prior_pos_list,
                    prior_count=len(prior_pos_list),
                )
            )

    logger.info(f"Constructed {len(records):,} chronological targets across {len(df_multi['patient_id'].unique()):,} patients.")
    return records


def extract_features_for_k(
    records: List[TargetRecord],
    z_matrix: np.ndarray,
    k: int,
    feature_mode: str = "concat",  # "concat" [z_t, z_t - b_p] or "diff" (z_t - b_p)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct X, y_change, y_multilabel for a specified calibration prior count K."""
    dim = z_matrix.shape[1]
    X_list = []
    y_change_list = []
    y_multi_list = []

    for r in records:
        z_t = z_matrix[r.pos]
        if k == 0:
            if feature_mode == "concat":
                feat = np.concatenate([z_t, np.zeros_like(z_t)])
            else:
                feat = z_t
        else:
            n_use = min(k, len(r.prior_positions))
            used_priors = r.prior_positions[-n_use:]
            b_p = np.mean(z_matrix[used_priors], axis=0)
            diff = z_t - b_p
            if feature_mode == "concat":
                feat = np.concatenate([z_t, diff])
            else:
                feat = diff

        X_list.append(feat)
        y_change_list.append(r.y_change)
        y_multi_list.append(r.y_multilabel)

    X = np.array(X_list, dtype=np.float32)
    y_change = np.array(y_change_list, dtype=np.int64)
    y_multi = np.array(y_multi_list, dtype=np.float32)
    return X, y_change, y_multi


def evaluate_probe_fold(
    X_train: np.ndarray,
    y_train_c: np.ndarray,
    y_train_m: np.ndarray,
    X_test: np.ndarray,
    y_test_c: np.ndarray,
    y_test_m: np.ndarray,
    c_val: float = 0.1,
) -> Dict[str, Any]:
    """Train standardized L2 linear probe on train split and evaluate on test split."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Task 1: Change Detection
    if len(np.unique(y_train_c)) >= 2 and len(np.unique(y_test_c)) >= 2:
        clf_c = LogisticRegression(C=c_val, max_iter=2000, random_state=42)
        clf_c.fit(X_train_s, y_train_c)
        probs_c = clf_c.predict_proba(X_test_s)[:, 1]
        auc_c = fast_auroc(y_test_c, probs_c)
        brier_c = float(np.mean((probs_c - y_test_c) ** 2))
    else:
        auc_c = float("nan")
        brier_c = float("nan")
        probs_c = np.zeros(len(y_test_c), dtype=np.float32)

    # Task 2: Multilabel Superclass
    aurocs_m = []
    probs_m = np.zeros_like(y_test_m, dtype=np.float32)
    for c in range(y_train_m.shape[1]):
        if len(np.unique(y_train_m[:, c])) >= 2 and len(np.unique(y_test_m[:, c])) >= 2:
            clf_m = LogisticRegression(C=c_val, max_iter=2000, random_state=42)
            clf_m.fit(X_train_s, y_train_m[:, c])
            p = clf_m.predict_proba(X_test_s)[:, 1]
            probs_m[:, c] = p
            auc_cls = fast_auroc(y_test_m[:, c], p)
            if not np.isnan(auc_cls):
                aurocs_m.append(auc_cls)
    macro_auc_m = float(np.mean(aurocs_m)) if aurocs_m else float("nan")

    return {
        "change_auroc": float(auc_c),
        "change_brier": brier_c,
        "change_probs": probs_c,
        "multi_macro_auroc": macro_auc_m,
        "multi_probs": probs_m,
    }


def run_bootstrap_delta(
    y_test_change: np.ndarray,
    so_probs: np.ndarray,
    csfm_probs: np.ndarray,
    test_patients: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Patient-clustered bootstrap for paired performance difference (SetOperator - CSFM)."""
    rng = np.random.RandomState(seed)
    unique_pats = np.unique(test_patients)
    pat_to_indices = {pid: np.where(test_patients == pid)[0] for pid in unique_pats}

    diffs = []
    for _ in range(n_bootstraps):
        sampled_pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
        idx_sample = np.concatenate([pat_to_indices[p] for p in sampled_pats])
        y_s = y_test_change[idx_sample]
        if len(np.unique(y_s)) < 2:
            continue
        auc_so = fast_auroc(y_s, so_probs[idx_sample])
        auc_csfm = fast_auroc(y_s, csfm_probs[idx_sample])
        diffs.append(auc_so - auc_csfm)

    diffs = np.array(diffs)
    return {
        "mean_delta": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0.0) if np.mean(diffs) > 0 else np.mean(diffs >= 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M1-C: Calibration-Efficiency Comparison Benchmark")
    parser.add_argument(
        "--so-cache",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"),
    )
    parser.add_argument(
        "--csfm-cache",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts"),
    )
    parser.add_argument(
        "--ptbxl-meta",
        type=Path,
        default=Path("data/ptbxl/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_m1_c_calibration_efficiency"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "run.log")

    logger.info("======================================================================")
    logger.info("=== Milestone M1-C: Calibration-Efficiency Comparison Benchmark ===")
    logger.info("======================================================================")
    logger.info(f"SetOperator Cache: {args.so_cache}")
    logger.info(f"CSFM Cache:        {args.csfm_cache}")
    logger.info(f"Contexts Dir:      {args.contexts}")
    logger.info(f"Metadata File:     {args.ptbxl_meta}")
    logger.info(f"Output Directory:  {args.output_dir}")

    # 1. Load Data & Metadata
    logger.info("\nLoading latents, labels, and metadata...")
    so_data = np.load(args.so_cache)
    all_z_so = so_data["all_z"]  # (17417, 256)
    all_ecg = so_data["all_ecg"]
    ecg_to_so_idx = {int(e): i for i, e in enumerate(all_ecg)}

    # Precomputed multilabel superclasses
    train_lab = np.load(args.contexts / "train_labels.npy")
    val_lab = np.load(args.contexts / "validation_labels.npy")
    all_lab = np.concatenate([train_lab, val_lab], axis=0)
    ecg_to_lab = {int(e): all_lab[i] for i, e in enumerate(all_ecg)}

    csfm_data = np.load(args.csfm_cache)
    z_csfm_native = csfm_data["csfm_z"]  # (4087, 768)
    csfm_ecg_ids = csfm_data["ecg_ids"]

    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    df_meta["ecg_id"] = df_meta.index.values

    # Restrict to repeated cohort
    df_multi = df_meta.loc[csfm_ecg_ids].copy().sort_values(["patient_id", "recording_date"])
    df_multi["pos"] = np.arange(len(df_multi))

    multi_labels = np.stack([ecg_to_lab[int(e)] for e in df_multi.index.values])

    # Slice SetOperator latents matching df_multi
    so_positions = [ecg_to_so_idx[e] for e in df_multi.index.values]
    z_so_native = all_z_so[so_positions].astype(np.float32)
    z_csfm_native = z_csfm_native[np.arange(len(df_multi))].astype(np.float32)

    logger.info(f"Cohort Size: {len(df_multi)} ECGs across {len(df_multi['patient_id'].unique())} patients.")
    logger.info(f"SetOperator Native: {z_so_native.shape}, CSFM Native: {z_csfm_native.shape}")

    # 2. Fit PCA on Training Patients for Matched Dimensionality
    logger.info("\nFitting PCA on Training Patients (strat_fold != 10)...")
    train_mask = (df_multi["strat_fold"] != 10).values
    pca_csfm_256 = PCA(n_components=256, random_state=args.seed).fit(z_csfm_native[train_mask])
    z_csfm_d256 = pca_csfm_256.transform(z_csfm_native).astype(np.float32)

    pca_csfm_128 = PCA(n_components=128, random_state=args.seed).fit(z_csfm_native[train_mask])
    z_csfm_d128 = pca_csfm_128.transform(z_csfm_native).astype(np.float32)

    pca_so_128 = PCA(n_components=128, random_state=args.seed).fit(z_so_native[train_mask])
    z_so_d128 = pca_so_128.transform(z_so_native).astype(np.float32)
    z_so_d256 = z_so_native

    # 3. Build Chronological Target Instances
    logger.info("\nBuilding chronological target instances...")
    targets = build_target_records(df_multi, multi_labels, logger)

    # Split targets: Train = Folds 1-7, Test = Fold 8
    train_targets = [r for r in targets if r.strat_fold in range(1, 8)]
    test_targets = [r for r in targets if r.strat_fold == 8]
    logger.info(f"Train Targets (Folds 1-7): {len(train_targets):,}")
    logger.info(f"Test Targets (Fold 8):     {len(test_targets):,}")

    # 4. Clinical Coverage vs K
    logger.info("\n--- Clinical Calibration Coverage vs K ---")
    k_values = [0, 1, 2, 3, 5]
    coverage_table = []
    for k in k_values:
        n_avail_train = sum(1 for r in train_targets if r.prior_count >= (1 if k > 0 else 0))
        n_avail_test = sum(1 for r in test_targets if r.prior_count >= (1 if k > 0 else 0))
        n_exact_test = sum(1 for r in test_targets if r.prior_count >= k)
        pct_test = (n_exact_test / len(test_targets)) * 100.0
        coverage_table.append({
            "K": k,
            "train_targets": n_avail_train,
            "test_targets": n_avail_test,
            "test_targets_with_k_priors": n_exact_test,
            "pct_coverage": pct_test,
        })
        logger.info(f"K={k}: {n_exact_test}/{len(test_targets)} test targets ({pct_test:.1f}% clinical coverage)")

    # 5. Evaluate Calibration Curves across Representations
    representations = [
        ("SetOperator (Native, d=256)", z_so_native),
        ("CSFM Base (Native, d=768)", z_csfm_native),
        ("SetOperator (Matched, d=256)", z_so_d256),
        ("CSFM Base (Matched, d=256)", z_csfm_d256),
        ("SetOperator (Matched, d=128)", z_so_d128),
        ("CSFM Base (Matched, d=128)", z_csfm_d128),
    ]

    curve_results = {}
    test_patient_ids = np.array([r.patient_id for r in test_targets])

    for rep_name, z_mat in representations:
        logger.info(f"\nEvaluating Calibration Curve for: {rep_name}...")
        k_eval = {}
        base_change_auc = None
        base_multi_auc = None

        for k in k_values:
            X_tr, y_tr_c, y_tr_m = extract_features_for_k(train_targets, z_mat, k, feature_mode="concat")
            X_te, y_te_c, y_te_m = extract_features_for_k(test_targets, z_mat, k, feature_mode="concat")

            probe_res = evaluate_probe_fold(X_tr, y_tr_c, y_tr_m, X_te, y_te_c, y_te_m, c_val=0.1)

            if k == 0:
                base_change_auc = probe_res["change_auroc"]
                base_multi_auc = probe_res["multi_macro_auroc"]
                gain_c = 0.0
                gain_m = 0.0
            else:
                gain_c = probe_res["change_auroc"] - base_change_auc
                gain_m = probe_res["multi_macro_auroc"] - base_multi_auc

            k_eval[f"K={k}"] = {
                "change_auroc": probe_res["change_auroc"],
                "change_gain_G(K)": gain_c,
                "change_brier": probe_res["change_brier"],
                "change_probs": probe_res["change_probs"].tolist(),
                "multi_macro_auroc": probe_res["multi_macro_auroc"],
                "multi_gain_G(K)": gain_m,
            }
            logger.info(
                f"   K={k} | Change AUROC: {probe_res['change_auroc']:.4f} (G(K): {gain_c:+.4f}) | "
                f"Multi AUROC: {probe_res['multi_macro_auroc']:.4f} (G(K): {gain_m:+.4f})"
            )

        curve_results[rep_name] = k_eval

    # 6. Paired Bootstrap Testing for Calibration Gain G(K) Difference
    logger.info("\nComputing Paired Bootstrap Tests for G_SO(K) - G_CSFM(K)...")
    bootstrap_gains = {}
    y_te_change = np.array([r.y_change for r in test_targets])

    for k in [1, 2, 3, 5]:
        so_p = np.array(curve_results["SetOperator (Matched, d=256)"][f"K={k}"]["change_probs"])
        csfm_p = np.array(curve_results["CSFM Base (Matched, d=256)"][f"K={k}"]["change_probs"])
        boot = run_bootstrap_delta(y_te_change, so_p, csfm_p, test_patient_ids, n_bootstraps=args.n_bootstrap, seed=args.seed)
        bootstrap_gains[f"K={k}"] = boot
        logger.info(f"   K={k} Matched d=256 Δ(SO - CSFM): {boot['mean_delta']:+.4f} [95% CI: {boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}] (p={boot['p_value']:.4f})")

    # 7. Generate Publication Table
    table_lines = [
        "# Milestone M1-C: Calibration-Efficiency Comparison Matrix",
        "",
        "### A. Change Detection Calibration Curve G(K) = AUROC(K) − AUROC(0)",
        "| Prior Count K | Clinical Coverage (%) | SetOperator (Native, d=256) | CSFM Base (Native, d=768) | SetOperator (Matched, d=256) | CSFM Base (Matched, d=256) | SetOperator (Matched, d=128) | CSFM Base (Matched, d=128) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for i, k in enumerate(k_values):
        cov = coverage_table[i]["pct_coverage"]
        row = [f"**K={k}**", f"{cov:.1f}%"]
        for rep_name, _ in representations:
            data_k = curve_results[rep_name][f"K={k}"]
            auc = data_k["change_auroc"]
            gain = data_k["change_gain_G(K)"]
            if k == 0:
                row.append(f"{auc:.4f} *(base)*")
            else:
                row.append(f"{auc:.4f} (**{gain:+.4f}**)")
        table_lines.append("| " + " | ".join(row) + " |")

    table_lines.append("")
    table_lines.append("### B. Multilabel Superclass Diagnostic Curve AUROC(K)",)
    table_lines.append("| Prior Count K | SetOperator (Native, d=256) | CSFM Base (Native, d=768) | SetOperator (Matched, d=256) | CSFM Base (Matched, d=256) | SetOperator (Matched, d=128) | CSFM Base (Matched, d=128) |")
    table_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for i, k in enumerate(k_values):
        row = [f"**K={k}**"]
        for rep_name, _ in representations:
            data_k = curve_results[rep_name][f"K={k}"]
            auc = data_k["multi_macro_auroc"]
            gain = data_k["multi_gain_G(K)"]
            if k == 0:
                row.append(f"{auc:.4f} *(base)*")
            else:
                row.append(f"{auc:.4f} (**{gain:+.4f}**)")
        table_lines.append("| " + " | ".join(row) + " |")

    table_lines.append("")
    table_lines.append("### C. Paired Patient-Level Difference Δ(SetOperator − CSFM Base) under Matched d=256")
    for k in [1, 2, 3, 5]:
        b = bootstrap_gains[f"K={k}"]
        table_lines.append(f"- **K={k} Difference**: {b['mean_delta']:+.4f} (95% CI: [{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}], p = {b['p_value']:.4f})")

    table_md = "\n".join(table_lines) + "\n"
    (args.output_dir / "calibration_efficiency_table.md").write_text(table_md)

    # 8. Generate Publication Visualization
    fig, (ax_main, ax_cov) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    k_plot = np.array(k_values)
    so_m256_gains = [curve_results["SetOperator (Matched, d=256)"][f"K={k}"]["change_gain_G(K)"] for k in k_values]
    csfm_m256_gains = [curve_results["CSFM Base (Matched, d=256)"][f"K={k}"]["change_gain_G(K)"] for k in k_values]
    so_nat_gains = [curve_results["SetOperator (Native, d=256)"][f"K={k}"]["change_gain_G(K)"] for k in k_values]
    csfm_nat_gains = [curve_results["CSFM Base (Native, d=768)"][f"K={k}"]["change_gain_G(K)"] for k in k_values]

    ax_main.plot(k_plot, so_m256_gains, marker="o", linewidth=2.8, markersize=8, color="#1f77b4", label="SetOperator (Matched d=256)")
    ax_main.plot(k_plot, csfm_m256_gains, marker="s", linewidth=2.8, markersize=8, color="#ff7f0e", label="CSFM Base (Matched d=256)")
    ax_main.plot(k_plot, so_nat_gains, linestyle="--", linewidth=1.8, color="#1f77b4", alpha=0.6, label="SetOperator (Native d=256)")
    ax_main.plot(k_plot, csfm_nat_gains, linestyle="--", linewidth=1.8, color="#ff7f0e", alpha=0.6, label="CSFM Base (Native d=768)")

    for i, k in enumerate(k_values):
        if k > 0:
            ax_main.text(k, so_m256_gains[i] + 0.005, f"{so_m256_gains[i]:+.3f}", ha="center", color="#1f77b4", fontweight="bold", fontsize=9)
            ax_main.text(k, csfm_m256_gains[i] - 0.012, f"{csfm_m256_gains[i]:+.3f}", ha="center", color="#ff7f0e", fontweight="bold", fontsize=9)

    ax_main.set_xlabel("Number of Historical Priors (K)")
    ax_main.set_ylabel("Calibration Gain G(K) = AUROC(K) - AUROC(0)")
    ax_main.set_title("A. Calibration Efficiency Scaling Curve", fontweight="bold")
    ax_main.set_xticks(k_values)
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="upper left")

    # Coverage Subplot
    cov_pcts = [c["pct_coverage"] for c in coverage_table]
    ax_cov.bar(k_values, cov_pcts, color="#4c72b0", width=0.6)
    for i, k in enumerate(k_values):
        ax_cov.text(k, cov_pcts[i] + 2, f"{cov_pcts[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_cov.set_xlabel("Required Priors (K)")
    ax_cov.set_ylabel("Clinical Patient Coverage (%)")
    ax_cov.set_title("B. Patient Population Coverage", fontweight="bold")
    ax_cov.set_xticks(k_values)
    ax_cov.set_ylim(0, 115)
    ax_cov.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = args.output_dir / "calibration_efficiency_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save full JSON results (without numpy arrays)
    clean_curve_results = {}
    for r_name, k_dict in curve_results.items():
        clean_curve_results[r_name] = {}
        for k_str, metric_dict in k_dict.items():
            clean_curve_results[r_name][k_str] = {
                "change_auroc": metric_dict["change_auroc"],
                "change_gain_G(K)": metric_dict["change_gain_G(K)"],
                "change_brier": metric_dict["change_brier"],
                "multi_macro_auroc": metric_dict["multi_macro_auroc"],
                "multi_gain_G(K)": metric_dict["multi_gain_G(K)"],
            }

    final_payload = {
        "schema_version": SCHEMA_VERSION,
        "n_train_targets": len(train_targets),
        "n_test_targets": len(test_targets),
        "coverage_table": coverage_table,
        "curve_results": clean_curve_results,
        "bootstrap_gains": bootstrap_gains,
    }

    results_file = args.output_dir / "results.json"
    results_file.write_text(json.dumps(final_payload, indent=2) + "\n")
    logger.info(f"\nSaved structured results to: {results_file}")
    logger.info(f"Saved comparison table to:    {args.output_dir / 'calibration_efficiency_table.md'}")
    logger.info(f"Saved calibration plot to:     {plot_path}")
    logger.info("\n" + table_md)
    logger.info("=== Milestone M1-C Execution COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
