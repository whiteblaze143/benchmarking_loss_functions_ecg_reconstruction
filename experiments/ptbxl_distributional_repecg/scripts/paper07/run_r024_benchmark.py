#!/usr/bin/env python3
"""Milestone R024: Benchmark Basis-Only vs Operator-Span vs Canonical SetOperator.

Evaluates on real physical waveforms across all 13 test configurations:
1. Canonical controls: Q8 full, S6 precordial, S3, S2, S1 smartwatch
2. Limb lead geometry: S6 limb, Derived limb only (III, aVR, aVL, aVF), Lead III single, aVF single
3. Unseen oblique vectors: S_icm (V3-V2), Test obliques (3 random), S_icm + Lead I, All 16 leads

Performs 1,000-sample patient-clustered paired bootstrap testing with 95% CIs and p-values.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cudnn 1D conv bug on this environment
torch.backends.cudnn.enabled = False

from repecg.paper07_operator import OperatorSetModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_r024_benchmark")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
CANONICAL_CKPT = Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt")
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r024_benchmark_results"

CONFIGURATIONS = {
    "Q8_full": [0, 1, 2, 3, 4, 5, 6, 7],
    "S6_precordial": [2, 3, 4, 5, 6, 7],
    "S3_reduced": [0, 1, 3],
    "S2_reduced": [0, 1],
    "S1_smartwatch": [0],
    "S6_limb": [0, 1, 8, 9, 10, 11],
    "Derived_limb_only": [8, 9, 10, 11],
    "Lead_III_single": [8],
    "aVF_single": [11],
    "S_icm": [12],
    "S_icm_plus_leadI": [0, 12],
    "Test_obliques_3": [13, 14, 15],
    "All_16_leads": list(range(16)),
}

GRAPHECG_BENCHMARKS = {
    "Q8_full": 0.9128,
    "S6_precordial": 0.8784,
    "S6_limb": 0.8800,
    "S_icm": 0.7197,
    "S1_smartwatch": 0.7675,
}


def compute_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    class_aurocs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            class_aurocs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
    return float(np.mean(class_aurocs))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            class_aurocs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(y_true[:, c], y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auprc": float(np.mean(class_auprcs)),
    }


def predict_model_on_config(
    model: OperatorSetModel,
    operators: np.ndarray,  # [N, 16, 8]
    responses: np.ndarray,  # [N, 16, 16, 128]
    lead_indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    sub_ops = operators[:, lead_indices]
    sub_resps = responses[:, lead_indices]
    n_samples = len(operators)

    probs = []
    with torch.inference_mode():
        for i in range(0, n_samples, batch_size):
            o_b = torch.as_tensor(sub_ops[i : i + batch_size], dtype=torch.float32, device=device)
            r_b = torch.as_tensor(sub_resps[i : i + batch_size], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(o_b, r_b, return_reconstruction=False)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs, axis=0)


def run_patient_clustered_bootstrap(
    y_true: np.ndarray,
    patient_ids: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    patient_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    boot_scores = {name: [] for name in pred_dict}
    boot_deltas_span_vs_basis = []
    boot_deltas_span_vs_canon = []

    for _ in range(n_boot):
        sample_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([patient_to_indices[p] for p in sample_patients])

        y_boot = y_true[idx]
        scores = {}
        for name, probs in pred_dict.items():
            s = compute_macro_auroc(y_boot, probs[idx])
            boot_scores[name].append(s)
            scores[name] = s

        if "operator_span" in scores and "basis_only" in scores:
            boot_deltas_span_vs_basis.append(scores["operator_span"] - scores["basis_only"])
        if "operator_span" in scores and "canonical" in scores:
            boot_deltas_span_vs_canon.append(scores["operator_span"] - scores["canonical"])

    ci_results = {}
    for name, arr in boot_scores.items():
        arr = np.array(arr)
        ci_results[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }

    deltas_result = {}
    if boot_deltas_span_vs_basis:
        d = np.array(boot_deltas_span_vs_basis)
        p_val = float(2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        deltas_result["span_minus_basis"] = {
            "mean_delta": float(np.mean(d)),
            "ci_95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value": p_val,
        }

    if boot_deltas_span_vs_canon:
        d = np.array(boot_deltas_span_vs_canon)
        p_val = float(2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        deltas_result["span_minus_canonical"] = {
            "mean_delta": float(np.mean(d)),
            "ci_95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value": p_val,
        }

    return {"bootstrap_cis": ci_results, "bootstrap_deltas": deltas_result}


def main():
    parser = argparse.ArgumentParser(description="Run Milestone R024 Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_path = DATA_DIR / "representation_val_real_waveforms.npz"
    logger.info(f"Loading real physical validation split from {val_path}...")
    d_val = np.load(val_path)
    val_ops = d_val["operators"]        # [N, 16, 8]
    val_resps = d_val["responses"]      # [N, 16, 16, 128]
    val_labels = d_val["labels"]        # [N, 5]
    val_patients = d_val["patient_ids"] # [N]

    logger.info(f"Loaded validation set: {len(val_labels)} records across {len(np.unique(val_patients))} patients.")

    # Load Models
    models = {}

    # 1. Canonical Model
    logger.info(f"Loading Canonical Model from {CANONICAL_CKPT}...")
    ckpt_canon = torch.load(CANONICAL_CKPT, map_location="cpu")
    m_canon = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    m_canon.load_state_dict(ckpt_canon["state_dict"] if "state_dict" in ckpt_canon else ckpt_canon)
    m_canon.eval()
    models["canonical"] = m_canon

    # 2. Basis-Only Model
    basis_ckpt = MODELS_DIR / "basis_only/checkpoint.pt"
    if basis_ckpt.exists():
        logger.info(f"Loading Basis-Only Model from {basis_ckpt}...")
        ckpt_basis = torch.load(basis_ckpt, map_location="cpu")
        m_basis = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
        m_basis.load_state_dict(ckpt_basis["state_dict"])
        m_basis.eval()
        models["basis_only"] = m_basis
    else:
        logger.warning(f"Basis-Only checkpoint not found at {basis_ckpt}!")

    # 3. Operator-Span Model
    span_ckpt = MODELS_DIR / "operator_span/checkpoint.pt"
    if span_ckpt.exists():
        logger.info(f"Loading Operator-Span Model from {span_ckpt}...")
        ckpt_span = torch.load(span_ckpt, map_location="cpu")
        m_span = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
        m_span.load_state_dict(ckpt_span["state_dict"])
        m_span.eval()
        models["operator_span"] = m_span
    else:
        logger.warning(f"Operator-Span checkpoint not found at {span_ckpt}!")

    # Run Benchmark Across All Configurations
    results = {}
    logger.info("\n=== Running Matched Benchmark Across Configurations ===")

    for cfg_name, indices in CONFIGURATIONS.items():
        logger.info(f"\nEvaluating configuration: {cfg_name} (leads: {indices})...")
        cfg_preds = {}
        cfg_metrics = {}

        for m_name, model in models.items():
            probs = predict_model_on_config(model, val_ops, val_resps, indices, device)
            cfg_preds[m_name] = probs
            metrics = compute_metrics(val_labels, probs)
            cfg_metrics[m_name] = metrics
            logger.info(f"  [{m_name:14s}] Macro AUROC: {metrics['macro_auroc']:.4f} | Micro: {metrics['micro_auroc']:.4f} | AUPRC: {metrics['macro_auprc']:.4f}")

        # Run Bootstrap
        boot_res = run_patient_clustered_bootstrap(
            val_labels, val_patients, cfg_preds, n_boot=args.n_boot
        )

        cfg_result = {
            "lead_indices": indices,
            "metrics": cfg_metrics,
            "bootstrap": boot_res,
        }
        if cfg_name in GRAPHECG_BENCHMARKS:
            cfg_result["graphecg_benchmark"] = GRAPHECG_BENCHMARKS[cfg_name]

        results[cfg_name] = cfg_result

    # Save results
    out_json = args.output_dir / "r024_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved benchmark results to {out_json}.")


if __name__ == "__main__":
    main()
