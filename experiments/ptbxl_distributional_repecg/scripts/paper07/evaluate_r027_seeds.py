#!/usr/bin/env python3
"""Milestone R027: Multi-Seed 2x2 Factorial Evaluation across Optimizer Seeds.

Evaluates multiple seeds (e.g. seeds 42, 43, 44) across all 13 benchmark configurations:
Quantifies optimizer / checkpoint variability across all 4 cells:
  Cell (1,1): Orig-Basis
  Cell (1,2): CF-Basis
  Cell (2,1): Orig-Span
  Cell (2,2): CF-Span

Computes:
1. Mean AUROC and standard deviation across seeds for each cell.
2. Factorial interaction Delta_interaction = Delta_{CF|span} - Delta_{CF|basis} across seeds.
3. Quantifies whether sub-additive partial functional overlap is invariant across optimization runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cuDNN
cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
import sys
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import canonical_frame, observable_projector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_r027_seeds")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
R024_DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r027_heldout_grassmann_results"
ARTIFACT_DIR = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")

CONFIGURATIONS = {
    "Q8_full": {"name": "Full 8 Independent Leads (I, II, V1-V6)", "indices": [0, 1, 2, 3, 4, 5, 6, 7], "category": "canonical_full"},
    "S6_precordial": {"name": "6 Precordial Leads (V1-V6)", "indices": [2, 3, 4, 5, 6, 7], "category": "canonical_subset"},
    "S6_limb": {"name": "6 Limb Leads (I, II, III, aVR, aVL, aVF)", "indices": [0, 1, 8, 9, 10, 11], "category": "redundant_limb"},
    "Derived_limb_only": {"name": "4 Derived Limb Leads (III, aVR, aVL, aVF)", "indices": [8, 9, 10, 11], "category": "redundant_limb"},
    "S3_reduced": {"name": "3-Lead ICU Monitoring (I, II, V2)", "indices": [0, 1, 3], "category": "canonical_sparse"},
    "S2_reduced": {"name": "2-Lead Bipolar (I, II)", "indices": [0, 1], "category": "canonical_sparse"},
    "S1_smartwatch": {"name": "1-Lead Smartwatch (Lead I)", "indices": [0], "category": "canonical_sparse"},
    "Lead_III_single": {"name": "1-Lead Single Derived (Lead III)", "indices": [8], "category": "single_derived"},
    "aVF_single": {"name": "1-Lead Single Augmented (aVF)", "indices": [11], "category": "single_derived"},
    "S_icm": {"name": "Oblique Subcutaneous Vector (V3 - V2)", "indices": [12], "category": "oblique_unseen"},
    "S_icm_plus_leadI": {"name": "Wearable Combo: Lead I + (V3 - V2)", "indices": [0, 12], "category": "oblique_unseen"},
    "Test_obliques_3": {"name": "3 Held-Out Oblique Linear Combinations", "indices": [13, 14, 15], "category": "oblique_unseen"},
    "All_16_leads": {"name": "All 16 Available Measurement Channels", "indices": list(range(16)), "category": "redundant_full"},
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    class_aurocs = []
    for c in range(y_true.shape[1]):
        labels_c = y_true[:, c]
        if len(np.unique(labels_c)) > 1:
            class_aurocs.append(float(roc_auc_score(labels_c, y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
    }


def predict_model_batched(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    N = len(responses)
    is_shared_ops = (operators.ndim == 2)
    if is_shared_ops:
        o_base = torch.as_tensor(operators, dtype=torch.float32, device=device).unsqueeze(0)
    all_logits = []
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
    return np.concatenate(all_logits, axis=0)


def evaluate_single_seed(
    seed: int,
    responses_all: np.ndarray,
    operators_bank: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    logger.info(f"\n--- Evaluating Seed {seed} ---")
    if seed == 42:
        basis_ckpt_path = MODELS_DIR / "basis_only/checkpoint.pt"
        span_ckpt_path = MODELS_DIR / "operator_span/checkpoint.pt"
    else:
        basis_ckpt_path = MODELS_DIR / f"basis_only_seed{seed}/checkpoint.pt"
        span_ckpt_path = MODELS_DIR / f"operator_span_seed{seed}/checkpoint.pt"

    if not basis_ckpt_path.exists():
        raise FileNotFoundError(f"Missing basis checkpoint: {basis_ckpt_path}")
    if not span_ckpt_path.exists():
        raise FileNotFoundError(f"Missing span checkpoint: {span_ckpt_path}")

    model_basis = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_basis.load_state_dict(torch.load(basis_ckpt_path, map_location=device)["state_dict"])
    model_basis.eval()

    model_span = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_span.load_state_dict(torch.load(span_ckpt_path, map_location=device)["state_dict"])
    model_span.eval()

    seed_results = {
        "cell_1_1_orig_basis": {},
        "cell_1_2_cf_basis": {},
        "cell_2_1_orig_span": {},
        "cell_2_2_cf_span": {},
    }

    for cfg_key, cfg_info in CONFIGURATIONS.items():
        sub_indices = cfg_info["indices"]
        Q_input = operators_bank[sub_indices]
        sub_resps_orig = responses_all[:, sub_indices]

        # Canonicalize
        P_Q, r = observable_projector(Q_input, tol=1e-7)
        B_Q = canonical_frame(P_Q, rank=r, tol=1e-7)

        # Canonical responses mapping
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
            sub_resps_cf = sub_resps_orig
            B_eval = B_Q

        # Evaluate 4 cells
        l_1_1 = predict_model_batched(model_basis, Q_input, sub_resps_orig, device)
        m_1_1 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_1_1)).numpy())

        l_1_2 = predict_model_batched(model_basis, B_eval, sub_resps_cf, device)
        m_1_2 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_1_2)).numpy())

        l_2_1 = predict_model_batched(model_span, Q_input, sub_resps_orig, device)
        m_2_1 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_2_1)).numpy())

        l_2_2 = predict_model_batched(model_span, B_eval, sub_resps_cf, device)
        m_2_2 = compute_metrics(labels, torch.sigmoid(torch.tensor(l_2_2)).numpy())

        seed_results["cell_1_1_orig_basis"][cfg_key] = m_1_1["macro_auroc"]
        seed_results["cell_1_2_cf_basis"][cfg_key] = m_1_2["macro_auroc"]
        seed_results["cell_2_1_orig_span"][cfg_key] = m_2_1["macro_auroc"]
        seed_results["cell_2_2_cf_span"][cfg_key] = m_2_2["macro_auroc"]

    # Compute overall summary for this seed
    mean_1_1 = float(np.mean(list(seed_results["cell_1_1_orig_basis"].values())))
    mean_1_2 = float(np.mean(list(seed_results["cell_1_2_cf_basis"].values())))
    mean_2_1 = float(np.mean(list(seed_results["cell_2_1_orig_span"].values())))
    mean_2_2 = float(np.mean(list(seed_results["cell_2_2_cf_span"].values())))

    delta_cf_basis = mean_1_2 - mean_1_1
    delta_cf_span = mean_2_2 - mean_2_1
    delta_interaction = delta_cf_span - delta_cf_basis

    logger.info(
        f"[Seed {seed}] Overall AUROC: "
        f"Cell(1,1)={mean_1_1:.4f}, Cell(1,2)={mean_1_2:.4f}, "
        f"Cell(2,1)={mean_2_1:.4f}, Cell(2,2)={mean_2_2:.4f} | "
        f"Delta_CF|basis={delta_cf_basis:+.4f}, Delta_CF|span={delta_cf_span:+.4f} | "
        f"Interaction={delta_interaction:+.4f}"
    )

    return {
        "seed": seed,
        "mean_1_1_orig_basis": mean_1_1,
        "mean_1_2_cf_basis": mean_1_2,
        "mean_2_1_orig_span": mean_2_1,
        "mean_2_2_cf_span": mean_2_2,
        "delta_cf_basis": delta_cf_basis,
        "delta_cf_span": delta_cf_span,
        "delta_interaction": delta_interaction,
        "by_config": seed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Seed 2x2 Factorial")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_meta = np.load(R024_DATA_DIR / "representation_val_real_waveforms.npz")
    labels = val_meta["labels"]
    responses_all = val_meta["responses"]
    operators_bank = val_meta["operators"][0]

    all_seed_results = []
    for s in args.seeds:
        try:
            res = evaluate_single_seed(s, responses_all, operators_bank, labels, device)
            all_seed_results.append(res)
        except FileNotFoundError as e:
            logger.warning(f"Skipping seed {s}: {e}")

    if not all_seed_results:
        logger.error("No seed evaluations completed.")
        return

    # Aggregate across seeds
    n_seeds = len(all_seed_results)
    means_1_1 = [r["mean_1_1_orig_basis"] for r in all_seed_results]
    means_1_2 = [r["mean_1_2_cf_basis"] for r in all_seed_results]
    means_2_1 = [r["mean_2_1_orig_span"] for r in all_seed_results]
    means_2_2 = [r["mean_2_2_cf_span"] for r in all_seed_results]
    deltas_cf_basis = [r["delta_cf_basis"] for r in all_seed_results]
    deltas_cf_span = [r["delta_cf_span"] for r in all_seed_results]
    interactions = [r["delta_interaction"] for r in all_seed_results]

    summary = {
        "n_seeds": n_seeds,
        "seeds": [r["seed"] for r in all_seed_results],
        "cell_1_1_orig_basis": {"mean": float(np.mean(means_1_1)), "std": float(np.std(means_1_1, ddof=1)) if n_seeds > 1 else 0.0},
        "cell_1_2_cf_basis": {"mean": float(np.mean(means_1_2)), "std": float(np.std(means_1_2, ddof=1)) if n_seeds > 1 else 0.0},
        "cell_2_1_orig_span": {"mean": float(np.mean(means_2_1)), "std": float(np.std(means_2_1, ddof=1)) if n_seeds > 1 else 0.0},
        "cell_2_2_cf_span": {"mean": float(np.mean(means_2_2)), "std": float(np.std(means_2_2, ddof=1)) if n_seeds > 1 else 0.0},
        "delta_cf_basis": {"mean": float(np.mean(deltas_cf_basis)), "std": float(np.std(deltas_cf_basis, ddof=1)) if n_seeds > 1 else 0.0},
        "delta_cf_span": {"mean": float(np.mean(deltas_cf_span)), "std": float(np.std(deltas_cf_span, ddof=1)) if n_seeds > 1 else 0.0},
        "interaction": {"mean": float(np.mean(interactions)), "std": float(np.std(interactions, ddof=1)) if n_seeds > 1 else 0.0},
    }

    logger.info("\n=== MULTI-SEED 2x2 FACTORIAL SUMMARY ===")
    logger.info(f"Number of Seeds Evaluated: {n_seeds} ({summary['seeds']})")
    logger.info(f"Cell (1,1) Orig-Basis: {summary['cell_1_1_orig_basis']['mean']:.4f} +/- {summary['cell_1_1_orig_basis']['std']:.4f}")
    logger.info(f"Cell (1,2) CF-Basis:   {summary['cell_1_2_cf_basis']['mean']:.4f} +/- {summary['cell_1_2_cf_basis']['std']:.4f}")
    logger.info(f"Cell (2,1) Orig-Span:  {summary['cell_2_1_orig_span']['mean']:.4f} +/- {summary['cell_2_1_orig_span']['std']:.4f}")
    logger.info(f"Cell (2,2) CF-Span:    {summary['cell_2_2_cf_span']['mean']:.4f} +/- {summary['cell_2_2_cf_span']['std']:.4f}")
    logger.info(f"Interaction:           {summary['interaction']['mean']:+.4f} +/- {summary['interaction']['std']:.4f}")

    out_json = OUTPUT_DIR / "r027_multiseed_2x2_statistics.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_seed_results": all_seed_results}, f, indent=2)
    logger.info(f"Saved multi-seed statistics to {out_json}")


if __name__ == "__main__":
    main()
