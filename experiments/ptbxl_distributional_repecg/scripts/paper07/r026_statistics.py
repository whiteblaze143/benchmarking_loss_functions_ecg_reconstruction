#!/usr/bin/env python3
"""Milestone R026: Patient-Clustered Paired Bootstrap Statistics for CF-SetOperator.

Evaluates 2,000+ patient-clustered paired bootstrap resamples:
1. Gate A: Identity on Q8 full leads (Delta = AUROC_CF - AUROC_Orig).
2. Gate B: Redundancy immunity on All 16 leads (Delta_redundancy = AUROC_16 - AUROC_8).
3. Contrast C: Gain on Six-Limb leads (CF vs Orig SetOp).
4. Contrast D: Gain on Derived-Limb leads (CF vs Orig SetOp).
5. Contrast E: Sparse Aggregate comparison (CF-SetOp vs Learned-Q Control & OS-Only).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r026_statistics")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
BENCHMARK_DIR = CODEX_DIR / "paper07_operator/r026_benchmark_results"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r026_statistics"
R025_STATS_DIR = CODEX_DIR / "paper07_operator/r025_statistics"

SPARSE_CONFIGS = ["S3_reduced", "S2_reduced", "S1_smartwatch", "S_icm", "Derived_limb_only"]


def fast_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    class_aurocs = []
    for c in range(y_true.shape[1]):
        labels_c = y_true[:, c]
        if len(np.unique(labels_c)) > 1:
            class_aurocs.append(float(roc_auc_score(labels_c, y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
    return float(np.mean(class_aurocs))


def run_patient_cluster_bootstrap(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    preds_cf: Dict[str, np.ndarray],
    preds_orig: Dict[str, np.ndarray],
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    logger.info(f"Running {n_bootstraps:,} patient-clustered paired bootstrap iterations...")
    rng = np.random.default_rng(seed)

    unique_pts = np.unique(patient_ids)
    n_pts = len(unique_pts)

    pt_to_indices = {pt: np.where(patient_ids == pt)[0] for pt in unique_pts}

    # Tracking arrays
    delta_q8_identity = []
    delta_red_cf = []
    delta_red_orig = []
    delta_six_limb_gain = []
    delta_derived_limb_gain = []
    sparse_cf_samples = []
    sparse_orig_samples = []
    delta_sparse_gain = []

    valid_replicates = 0

    for b_idx in range(n_bootstraps):
        sampled_pts = rng.choice(unique_pts, size=n_pts, replace=True)
        idx_sample = np.concatenate([pt_to_indices[pt] for pt in sampled_pts])

        y_sample = labels[idx_sample]

        if (y_sample.sum(axis=0) == 0).any():
            continue

        valid_replicates += 1

        # 1. Gate A: Q8 Identity
        q8_cf = fast_macro_auroc(y_sample, preds_cf["Q8_full"][idx_sample])
        q8_orig = fast_macro_auroc(y_sample, preds_orig["Q8_full"][idx_sample])
        delta_q8_identity.append(q8_cf - q8_orig)

        # 2. Gate B: Redundancy (All 16 vs Q8)
        all16_cf = fast_macro_auroc(y_sample, preds_cf["All_16_leads"][idx_sample])
        all16_orig = fast_macro_auroc(y_sample, preds_orig["All_16_leads"][idx_sample])
        delta_red_cf.append(all16_cf - q8_cf)
        delta_red_orig.append(all16_orig - q8_orig)

        # 3. Six-Limb & Derived-Limb Gain
        six_cf = fast_macro_auroc(y_sample, preds_cf["S6_limb"][idx_sample])
        six_orig = fast_macro_auroc(y_sample, preds_orig["S6_limb"][idx_sample])
        delta_six_limb_gain.append(six_cf - six_orig)

        derived_cf = fast_macro_auroc(y_sample, preds_cf["Derived_limb_only"][idx_sample])
        derived_orig = fast_macro_auroc(y_sample, preds_orig["Derived_limb_only"][idx_sample])
        delta_derived_limb_gain.append(derived_cf - derived_orig)

        # 4. Sparse Aggregate
        s_cf = float(np.mean([fast_macro_auroc(y_sample, preds_cf[cfg][idx_sample]) for cfg in SPARSE_CONFIGS]))
        s_orig = float(np.mean([fast_macro_auroc(y_sample, preds_orig[cfg][idx_sample]) for cfg in SPARSE_CONFIGS]))
        sparse_cf_samples.append(s_cf)
        sparse_orig_samples.append(s_orig)
        delta_sparse_gain.append(s_cf - s_orig)

    def summarize_series(arr):
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "ci_95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        }

    stats = {
        "n_bootstraps_requested": n_bootstraps,
        "n_valid_replicates": valid_replicates,
        "valid_replicate_fraction": float(valid_replicates / n_bootstraps),
        "gate_a_q8_identity": {
            **summarize_series(delta_q8_identity),
            "status": "PASS" if abs(np.mean(delta_q8_identity)) < 1e-5 else "FAIL",
        },
        "gate_b_redundancy_immunity": {
            "cf_delta_16_vs_8": summarize_series(delta_red_cf),
            "orig_delta_16_vs_8": summarize_series(delta_red_orig),
            "status": "PASS" if abs(np.mean(delta_red_cf)) < 1e-5 else "FAIL",
        },
        "six_limb_gain": {
            **summarize_series(delta_six_limb_gain),
            "p_value_superiority": float(np.mean(np.array(delta_six_limb_gain) <= 0.0)),
        },
        "derived_limb_gain": {
            **summarize_series(delta_derived_limb_gain),
            "p_value_superiority": float(np.mean(np.array(delta_derived_limb_gain) <= 0.0)),
        },
        "sparse_aggregate": {
            "cf_sparse_aggregate": summarize_series(sparse_cf_samples),
            "orig_sparse_aggregate": summarize_series(sparse_orig_samples),
            "delta_sparse_gain": summarize_series(delta_sparse_gain),
            "p_value_superiority": float(np.mean(np.array(delta_sparse_gain) <= 0.0)),
        },
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute R026 Bootstrap Statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = BENCHMARK_DIR / f"r026_predictions_seed{args.seed}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    d = np.load(pred_path)
    labels = d["labels"]
    patient_ids = d["patient_ids"]

    # Convert logits to probabilities
    configs = [k.replace("cf__", "") for k in d.keys() if k.startswith("cf__")]
    preds_cf = {cfg: 1.0 / (1.0 + np.exp(-d[f"cf__{cfg}"])) for cfg in configs}
    preds_orig = {cfg: 1.0 / (1.0 + np.exp(-d[f"orig__{cfg}"])) for cfg in configs}

    stats = run_patient_cluster_bootstrap(
        labels=labels,
        patient_ids=patient_ids,
        preds_cf=preds_cf,
        preds_orig=preds_orig,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )

    out_file = args.out_dir / f"r026_bootstrap_statistics_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Bootstrap statistics saved to {out_file}")
    logger.info(f"Gate A (Identity): {stats['gate_a_q8_identity']['status']} (Mean Delta: {stats['gate_a_q8_identity']['mean']:.2e})")
    logger.info(f"Gate B (Redundancy): {stats['gate_b_redundancy_immunity']['status']} (CF Delta 16 vs 8: {stats['gate_b_redundancy_immunity']['cf_delta_16_vs_8']['mean']:.2e} vs Orig: {stats['gate_b_redundancy_immunity']['orig_delta_16_vs_8']['mean']:.4f})")
    logger.info(f"Six-Limb Gain: +{stats['six_limb_gain']['mean']:.4f} (95% CI: [{stats['six_limb_gain']['ci_95'][0]:.4f}, {stats['six_limb_gain']['ci_95'][1]:.4f}], p={stats['six_limb_gain']['p_value_superiority']:.4e})")
    logger.info(f"Derived-Limb Gain: +{stats['derived_limb_gain']['mean']:.4f} (95% CI: [{stats['derived_limb_gain']['ci_95'][0]:.4f}, {stats['derived_limb_gain']['ci_95'][1]:.4f}], p={stats['derived_limb_gain']['p_value_superiority']:.4e})")
    logger.info(f"Sparse Aggregate Gain: +{stats['sparse_aggregate']['delta_sparse_gain']['mean']:.4f} (95% CI: [{stats['sparse_aggregate']['delta_sparse_gain']['ci_95'][0]:.4f}, {stats['sparse_aggregate']['delta_sparse_gain']['ci_95'][1]:.4f}], p={stats['sparse_aggregate']['p_value_superiority']:.4e})")


if __name__ == "__main__":
    main()
