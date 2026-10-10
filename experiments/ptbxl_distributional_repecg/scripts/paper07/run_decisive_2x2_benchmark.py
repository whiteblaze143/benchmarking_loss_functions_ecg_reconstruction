#!/usr/bin/env python3
"""Milestone R026/R027: The Decisive 2x2 Experiment.

Evaluates the 2x2 factorial design across all 13 configurations on held-out PTB-XL (N=2,173):

Factor 1: Measurement Input Representation:
  - Level A: Original Measurement Set (arbitrary, possibly redundant Q)
  - Level B: CF Quotient (canonical representative B_Q = C(row(Q)))

Factor 2: Training Support Domain:
  - Level 1: Canonical / Basis Training (trained on standard coordinate leads)
  - Level 2: Subspace / Span Support Training (trained on continuous linear operator combinations)

2x2 Matrix Cells:
  Cell (1,1): Original SetOperator (Basis-Only training, Original inputs)
  Cell (1,2): CF-SetOperator R026  (Basis-Only training, CF quotient)
  Cell (2,1): R024 Span SetOperator (Span-Support training, Original inputs)
  Cell (2,2): CF + Span SetOperator (Span-Support training, CF quotient)

Scientific Hypotheses:
1. Canonicalization Effect: Does quotienting eliminate redundancy/basis dependence across both training regimes?
2. Support Effect: Does training over diverse subspaces improve unseen-subspace/oblique performance?
3. Interaction: Does CF + Subspace Support yield the optimal unified representation?
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cuDNN to avoid PyTorch 2.6 conv1d issue on A100
cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
import sys
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import (
    canonical_frame,
    observable_projector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("decisive_2x2")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
R024_DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r026_decisive_2x2_results"

CONFIGURATIONS = {
    "Q8_full": {
        "name": "Full 8 Independent Leads (I, II, V1-V6)",
        "indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "category": "canonical_full",
    },
    "S6_precordial": {
        "name": "6 Precordial Leads (V1-V6)",
        "indices": [2, 3, 4, 5, 6, 7],
        "category": "canonical_subset",
    },
    "S6_limb": {
        "name": "6 Limb Leads (I, II, III, aVR, aVL, aVF)",
        "indices": [0, 1, 8, 9, 10, 11],
        "category": "redundant_limb",
    },
    "Derived_limb_only": {
        "name": "4 Derived Limb Leads (III, aVR, aVL, aVF)",
        "indices": [8, 9, 10, 11],
        "category": "redundant_limb",
    },
    "S3_reduced": {
        "name": "3-Lead ICU Monitoring (I, II, V2)",
        "indices": [0, 1, 3],
        "category": "canonical_sparse",
    },
    "S2_reduced": {
        "name": "2-Lead Bipolar (I, II)",
        "indices": [0, 1],
        "category": "canonical_sparse",
    },
    "S1_smartwatch": {
        "name": "1-Lead Smartwatch (Lead I)",
        "indices": [0],
        "category": "canonical_sparse",
    },
    "Lead_III_single": {
        "name": "1-Lead Single Derived (Lead III)",
        "indices": [8],
        "category": "single_derived",
    },
    "aVF_single": {
        "name": "1-Lead Single Augmented (aVF)",
        "indices": [11],
        "category": "single_derived",
    },
    "S_icm": {
        "name": "Oblique Subcutaneous Vector (V3 - V2)",
        "indices": [12],
        "category": "oblique_unseen",
    },
    "S_icm_plus_leadI": {
        "name": "Wearable Combo: Lead I + (V3 - V2)",
        "indices": [0, 12],
        "category": "oblique_unseen",
    },
    "Test_obliques_3": {
        "name": "3 Held-Out Oblique Linear Combinations",
        "indices": [13, 14, 15],
        "category": "oblique_unseen",
    },
    "All_16_leads": {
        "name": "All 16 Available Measurement Channels",
        "indices": list(range(16)),
        "category": "redundant_full",
    },
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        labels_c = y_true[:, c]
        if len(np.unique(labels_c)) > 1:
            class_aurocs.append(float(roc_auc_score(labels_c, y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(labels_c, y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auprc": float(np.mean(class_auprcs)),
    }


def predict_model(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    N = len(responses)
    is_shared_ops = (operators.ndim == 2)
    if is_shared_ops:
        o_base = torch.as_tensor(operators, dtype=torch.float32, device=device).unsqueeze(0)
    all_logits = []
    all_latents = []
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            r_b = torch.as_tensor(responses[i : i + batch_size], dtype=torch.float32, device=device)
            B_cur = len(r_b)
            if is_shared_ops:
                o_b = o_base.expand(B_cur, -1, -1)
            else:
                o_b = torch.as_tensor(operators[i : i + batch_size], dtype=torch.float32, device=device)
            latent = model.encode_context(o_b, r_b)
            logits = model.head(latent)
            all_logits.append(logits.cpu().numpy())
            all_latents.append(latent.cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_latents, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Milestone R026/R027 Decisive 2x2 Experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the Two Checkpoints
    basis_ckpt_path = MODELS_DIR / "basis_only/checkpoint.pt"
    span_ckpt_path = MODELS_DIR / "operator_span/checkpoint.pt"

    logger.info(f"Loading Basis-Only Model from {basis_ckpt_path}...")
    model_basis = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_basis.load_state_dict(torch.load(basis_ckpt_path, map_location=device)["state_dict"])
    model_basis.eval()

    logger.info(f"Loading Operator-Span Model from {span_ckpt_path}...")
    model_span = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_span.load_state_dict(torch.load(span_ckpt_path, map_location=device)["state_dict"])
    model_span.eval()

    # 2. Load Real Waveform Validation Representations
    val_meta = np.load(R024_DATA_DIR / "representation_val_real_waveforms.npz")
    labels = val_meta["labels"]
    responses_all = val_meta["responses"]       # [2173, 16, 16, 128]
    operators_bank = val_meta["operators"][0]   # [16, 8]
    patient_ids = val_meta["patient_ids"]
    ecg_ids = val_meta["ecg_ids"]
    N = len(labels)
    logger.info(f"Loaded held-out evaluation set: {N} records.")

    # Storage for 2x2 results: cell -> config -> metrics
    cells = {
        "cell_1_1_orig_basis": {},  # Original SetOp
        "cell_1_2_cf_basis": {},    # CF-SetOp (R026)
        "cell_2_1_orig_span": {},   # R024 Span SetOp
        "cell_2_2_cf_span": {},     # CF + Span SetOp
    }

    raw_logits = {cell_key: {} for cell_key in cells}

    logger.info("=== Running 2x2 Benchmark Across All 13 Configurations ===")

    for cfg_key, cfg_info in CONFIGURATIONS.items():
        sub_indices = cfg_info["indices"]
        m = len(sub_indices)
        Q_input = operators_bank[sub_indices]
        sub_resps_orig = responses_all[:, sub_indices]

        # Canonicalize Q_input
        P_Q, r = observable_projector(Q_input, tol=1e-7)
        B_Q = canonical_frame(P_Q, rank=r, tol=1e-7)

        # Mapping for canonical response tokens:
        # Standard coordinate spaces map directly to the stored basis leads:
        if cfg_key in {"Q8_full", "All_16_leads"}:
            sub_resps_cf = responses_all[:, :8]
            B_eval = operators_bank[:8]
        elif cfg_key in {"S6_limb", "Derived_limb_only", "S2_reduced"}:
            sub_resps_cf = responses_all[:, :2]
            B_eval = operators_bank[:2]
        elif cfg_key == "S6_precordial":
            sub_resps_cf = responses_all[:, 2:8]
            B_eval = operators_bank[2:8]
        elif cfg_key == "S3_reduced":
            sub_resps_cf = responses_all[:, [0, 1, 3]]
            B_eval = operators_bank[[0, 1, 3]]
        elif cfg_key == "S1_smartwatch":
            sub_resps_cf = responses_all[:, [0]]
            B_eval = operators_bank[[0]]
        elif cfg_key == "Lead_III_single":
            sub_resps_cf = responses_all[:, [8]]
            B_eval = operators_bank[[8]]
        elif cfg_key == "aVF_single":
            sub_resps_cf = responses_all[:, [11]]
            B_eval = operators_bank[[11]]
        elif cfg_key == "S_icm":
            sub_resps_cf = responses_all[:, [12]]
            B_eval = operators_bank[[12]]
        elif cfg_key == "S_icm_plus_leadI":
            sub_resps_cf = responses_all[:, [0, 12]]
            B_eval = operators_bank[[0, 12]]
        elif cfg_key == "Test_obliques_3":
            sub_resps_cf = responses_all[:, [13, 14, 15]]
            B_eval = operators_bank[[13, 14, 15]]
        else:
            raise ValueError(f"Unmapped config: {cfg_key}")

        # Cell (1,1): Original SetOp (Basis-only, Original inputs)
        l_1_1, _ = predict_model(model_basis, Q_input, sub_resps_orig, device, batch_size=args.batch_size)
        m_1_1 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_1_1)).numpy())
        cells["cell_1_1_orig_basis"][cfg_key] = m_1_1
        raw_logits["cell_1_1_orig_basis"][cfg_key] = l_1_1

        # Cell (1,2): CF-SetOp (Basis-only, CF quotient)
        l_1_2, _ = predict_model(model_basis, B_eval, sub_resps_cf, device, batch_size=args.batch_size)
        m_1_2 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_1_2)).numpy())
        cells["cell_1_2_cf_basis"][cfg_key] = m_1_2
        raw_logits["cell_1_2_cf_basis"][cfg_key] = l_1_2

        # Cell (2,1): Span SetOp (Span training, Original inputs)
        l_2_1, _ = predict_model(model_span, Q_input, sub_resps_orig, device, batch_size=args.batch_size)
        m_2_1 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_2_1)).numpy())
        cells["cell_2_1_orig_span"][cfg_key] = m_2_1
        raw_logits["cell_2_1_orig_span"][cfg_key] = l_2_1

        # Cell (2,2): CF + Span SetOp (Span training, CF quotient)
        l_2_2, _ = predict_model(model_span, B_eval, sub_resps_cf, device, batch_size=args.batch_size)
        m_2_2 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_2_2)).numpy())
        cells["cell_2_2_cf_span"][cfg_key] = m_2_2
        raw_logits["cell_2_2_cf_span"][cfg_key] = l_2_2

        logger.info(
            f"[{cfg_key:<18}] m={m:2d}, r={r} | "
            f"(1,1) Orig-Basis: {m_1_1['macro_auroc']:.4f} | "
            f"(1,2) CF-Basis: {m_1_2['macro_auroc']:.4f} | "
            f"(2,1) Orig-Span: {m_2_1['macro_auroc']:.4f} | "
            f"(2,2) CF-Span: {m_2_2['macro_auroc']:.4f}"
        )

    # Compute Aggregates:
    # 1. Canonical Sparse Subsets (1..3 canonical leads: S3, S2, S1)
    canonical_sparse_keys = ["S3_reduced", "S2_reduced", "S1_smartwatch"]
    # 2. Low-Rank / Redundant Reduced-Observation Aggregate (S3, S2, S1, S_icm, Derived_limb_only)
    reduced_obs_keys = ["S3_reduced", "S2_reduced", "S1_smartwatch", "S_icm", "Derived_limb_only"]
    # 3. Oblique Unseen Subspaces (S_icm, S_icm + Lead I, Test obliques 3)
    oblique_keys = ["S_icm", "S_icm_plus_leadI", "Test_obliques_3"]
    # 4. Redundant Lead Configurations (S6_limb, Derived_limb_only, All_16_leads)
    redundant_keys = ["S6_limb", "Derived_limb_only", "All_16_leads"]

    aggregates = {}
    for cell_name in cells:
        aggregates[cell_name] = {
            "canonical_sparse_1_to_3": float(np.mean([cells[cell_name][k]["macro_auroc"] for k in canonical_sparse_keys])),
            "low_rank_reduced_obs": float(np.mean([cells[cell_name][k]["macro_auroc"] for k in reduced_obs_keys])),
            "oblique_unseen": float(np.mean([cells[cell_name][k]["macro_auroc"] for k in oblique_keys])),
            "redundant_leads": float(np.mean([cells[cell_name][k]["macro_auroc"] for k in redundant_keys])),
            "overall_macro_mean": float(np.mean([cells[cell_name][k]["macro_auroc"] for k in CONFIGURATIONS])),
        }

    logger.info("=== 2x2 Factorial Aggregate Summary ===")
    for cell_name, agg in aggregates.items():
        logger.info(
            f"[{cell_name:<20}] "
            f"Canonical Sparse: {agg['canonical_sparse_1_to_3']:.4f} | "
            f"Low-Rank Reduced: {agg['low_rank_reduced_obs']:.4f} | "
            f"Oblique Unseen: {agg['oblique_unseen']:.4f} | "
            f"Redundant Leads: {agg['redundant_leads']:.4f} | "
            f"Overall: {agg['overall_macro_mean']:.4f}"
        )

    # Save Results
    summary = {
        "milestone": "R026_R027",
        "description": "The Decisive 2x2 Factorial Benchmark",
        "cells": cells,
        "aggregates": aggregates,
    }
    out_json = OUTPUT_DIR / "decisive_2x2_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_json}")

    # Save Predictions for Clustered Bootstrap
    pred_dict = {
        "labels": labels,
        "patient_ids": patient_ids,
        "ecg_ids": ecg_ids,
    }
    for cell_name in cells:
        for cfg_key in CONFIGURATIONS:
            pred_dict[f"{cell_name}__{cfg_key}"] = raw_logits[cell_name][cfg_key]

    out_npz = OUTPUT_DIR / "decisive_2x2_predictions.npz"
    np.savez_compressed(out_npz, **pred_dict)
    logger.info(f"Raw predictions saved to {out_npz}")


if __name__ == "__main__":
    main()
