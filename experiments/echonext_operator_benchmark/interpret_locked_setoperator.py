#!/usr/bin/env python3
"""Post-Hoc Diagnostic Interpretability Analysis on Locked SetOperator.

Analyzes the locked, publication-grade SetOperator model (PTB-XL Fold-9 Span Seed 42)
WITHOUT retraining or architectural modification:
1. Phase-Cell Sensitivity: Evaluates class-specific prediction shifts (Delta prob)
   under selective masking of each of the 16 cardiac phase cells.
2. Cross-Attention Query Inspection: Extracts attention weights assigned by the
   cross-attention pooling query across the 8 canonical measurement views.
3. Lead-Removal / Occlusion Sensitivity: Evaluates Macro AUROC and prediction drift
   under limb removal, precordial removal, and leave-one-lead-out ablations.
4. Operator Sensitivity: Evaluates response representation variance across measurement angles.

Strict Scientific Constraint:
Reports these as diagnostic and observational analyses of learned representations,
NOT automatically as causal physiological explanations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn

# Disable cudnn 1D conv bug on this environment
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("interpret_setoperator")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.paper07_operator.model import OperatorSetModel

DATA_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/representation_final_select_waveforms.npz")
CKPT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/ptbxl_three_way_v1/checkpoints/setoperator/span_s42.pt")
OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPER_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def compute_macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            scores.append(float(roc_auc_score(y_true[:, c], y_pred[:, c])))
        else:
            scores.append(0.5)
    return float(np.mean(scores)), scores


def extract_attention_weights(model: OperatorSetModel, ops: torch.Tensor, resps: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Extract cross-attention weights from the pooling layer."""
    batch, count = resps.shape[:2]
    q = model.encode_operator(ops, None)
    response = model.response(
        resps.reshape(batch * count, resps.shape[2], resps.shape[3])
    ).reshape(batch, count, 128)
    encoded = model.set_encoder(torch.cat((q, response), dim=-1), src_key_padding_mask=mask)
    query = model.pool_query.expand(batch, -1, -1)
    # PyTorch MultiheadAttention returns (attn_output, attn_output_weights)
    _, attn_weights = model.pool(query, encoded, encoded, key_padding_mask=mask, need_weights=True)
    return attn_weights.squeeze(1)  # (batch, count)


def main():
    parser = argparse.ArgumentParser(description="Interpretability Diagnostics for Locked SetOperator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Running interpretability diagnostics on device: %s", device)

    # 1. Load data
    logger.info("Loading Fold-9 evaluation data from %s", DATA_PATH)
    with np.load(DATA_PATH) as d:
        ops_all = d["operators"][:, :8].astype(np.float32)       # (2183, 8, 8)
        resps_all = d["responses"][:, :8].astype(np.float32)   # (2183, 8, 16, 128)
        labels_all = d["labels"].astype(np.float32)            # (2183, 5)

    n_records = len(labels_all)
    logger.info("Loaded %d records, ops: %s, resps: %s", n_records, ops_all.shape, resps_all.shape)

    # 2. Load model
    logger.info("Loading locked SetOperator checkpoint from %s", CKPT_PATH)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Baseline forward pass
    logger.info("Computing baseline predictions on full 8 canonical leads...")
    all_logits = []
    all_attns = []
    with torch.no_grad():
        for i in range(0, n_records, args.batch_size):
            b_ops = torch.from_numpy(ops_all[i : i + args.batch_size]).to(device)
            b_res = torch.from_numpy(resps_all[i : i + args.batch_size]).to(device)
            logits = model(b_ops, b_res)
            attns = extract_attention_weights(model, b_ops, b_res)
            all_logits.append(logits.cpu())
            all_attns.append(attns.cpu())

    baseline_logits = torch.cat(all_logits, dim=0).numpy()
    baseline_probs = torch.sigmoid(torch.from_numpy(baseline_logits)).numpy()
    baseline_attns = torch.cat(all_attns, dim=0).numpy()  # (N, 8)

    base_macro_auroc, base_class_aurocs = compute_macro_auroc(labels_all, baseline_probs)
    logger.info("Baseline Macro AUROC: %.4f (NORM: %.4f, MI: %.4f, STTC: %.4f, CD: %.4f, HYP: %.4f)",
                base_macro_auroc, *base_class_aurocs)

    # 4. Diagnostic Analysis 1: Cross-Attention Query Distribution
    mean_lead_attns = baseline_attns.mean(axis=0)  # (8,)
    lead_attns_by_class = {}
    for c_idx, c_name in enumerate(SUPER_CLASSES):
        pos_mask = labels_all[:, c_idx] == 1.0
        lead_attns_by_class[c_name] = baseline_attns[pos_mask].mean(axis=0).tolist() if pos_mask.sum() > 0 else mean_lead_attns.tolist()

    logger.info("--- Cross-Attention Weights across Canonical Leads ---")
    for l_idx, l_name in enumerate(LEAD_NAMES):
        logger.info("  Lead %-4s: Mean Weight = %.4f (MI: %.4f, STTC: %.4f, CD: %.4f)",
                    l_name, mean_lead_attns[l_idx],
                    lead_attns_by_class["MI"][l_idx],
                    lead_attns_by_class["STTC"][l_idx],
                    lead_attns_by_class["CD"][l_idx])

    # 5. Diagnostic Analysis 2: Phase-Cell Sensitivity (Masking 16 Phase Cells)
    logger.info("--- Computing Phase-Cell Sensitivity across 16 Phase Cells ---")
    phase_cell_impacts = []
    # 16 phase cells: 0..3 (PR/early QRS), 4..7 (QRS peak), 8..11 (ST segment), 12..15 (T wave)
    for p in range(16):
        masked_probs_list = []
        with torch.no_grad():
            for i in range(0, n_records, args.batch_size):
                b_ops = torch.from_numpy(ops_all[i : i + args.batch_size]).to(device)
                b_res = torch.from_numpy(resps_all[i : i + args.batch_size]).clone().to(device)
                # Mask phase cell p across all measured views
                b_res[:, :, p, :] = 0.0
                logits = model(b_ops, b_res)
                masked_probs_list.append(torch.sigmoid(logits).cpu())

        masked_probs = torch.cat(masked_probs_list, dim=0).numpy()
        prob_diff = np.abs(masked_probs - baseline_probs)  # (N, 5)
        mean_diff_per_class = prob_diff.mean(axis=0)       # (5,)
        macro_diff = float(mean_diff_per_class.mean())
        phase_cell_impacts.append({
            "phase_cell": p,
            "cardiac_phase_region": (
                "Early Ventricular / PR Interval" if p < 4 else
                "QRS Depolarization Complex" if p < 8 else
                "ST Segment / Early Repolarization" if p < 12 else
                "T Wave / Late Repolarization"
            ),
            "macro_delta_prob": macro_diff,
            "class_delta_probs": {c_name: float(mean_diff_per_class[c_idx]) for c_idx, c_name in enumerate(SUPER_CLASSES)},
        })
        logger.info("  Phase Cell %2d (%-32s): Macro Delta Prob = %.4f | MI: %.4f, STTC: %.4f, CD: %.4f",
                    p, phase_cell_impacts[-1]["cardiac_phase_region"], macro_diff,
                    mean_diff_per_class[1], mean_diff_per_class[2], mean_diff_per_class[3])

    # 6. Diagnostic Analysis 3: Lead-Removal / Occlusion Sensitivity
    logger.info("--- Computing Lead-Removal / Occlusion Sensitivity ---")
    lead_removal_results = {}

    subsets = {
        "Full 8 Leads (Canonical)": list(range(8)),
        "Precordial Only (V1-V6, Remove Limb)": list(range(2, 8)),
        "Limb Only (I, II, Remove Precordial)": [0, 1],
        "Lead II Single (Single Frontal Lead)": [1],
        "Lead I Single": [0],
        "Lead V1 Single": [2],
        "Lead V5 Single": [6],
    }
    # Add leave-one-out for all 8 leads
    for l_idx, l_name in enumerate(LEAD_NAMES):
        subsets[f"Leave-Out Lead {l_name}"] = [i for i in range(8) if i != l_idx]

    for name, indices in subsets.items():
        subset_probs_list = []
        with torch.no_grad():
            for i in range(0, n_records, args.batch_size):
                b_ops = torch.from_numpy(ops_all[i : i + args.batch_size, indices]).to(device)
                b_res = torch.from_numpy(resps_all[i : i + args.batch_size, indices]).to(device)
                logits = model(b_ops, b_res)
                subset_probs_list.append(torch.sigmoid(logits).cpu())

        sub_probs = torch.cat(subset_probs_list, dim=0).numpy()
        sub_auroc, sub_class_aurocs = compute_macro_auroc(labels_all, sub_probs)
        mean_prob_shift = float(np.abs(sub_probs - baseline_probs).mean())
        delta_auroc = sub_auroc - base_macro_auroc
        lead_removal_results[name] = {
            "num_leads": len(indices),
            "macro_auroc": sub_auroc,
            "delta_macro_auroc": delta_auroc,
            "mean_prob_shift": mean_prob_shift,
            "class_aurocs": {c_name: float(sub_class_aurocs[c_idx]) for c_idx, c_name in enumerate(SUPER_CLASSES)},
        }
        logger.info("  %-38s: AUROC = %.4f (Delta = %+.4f) | Mean Prob Shift = %.4f",
                    name, sub_auroc, delta_auroc, mean_prob_shift)

    # 7. Package and save diagnostic findings
    diagnostics = {
        "model_checkpoint": str(CKPT_PATH),
        "cohort": "PTB-XL Fold 9 (N=2,183 authentic patients)",
        "baseline_macro_auroc": base_macro_auroc,
        "baseline_class_aurocs": {c_name: float(base_class_aurocs[c_idx]) for c_idx, c_name in enumerate(SUPER_CLASSES)},
        "lead_cross_attention_weights": {
            "overall_mean": {LEAD_NAMES[i]: float(mean_lead_attns[i]) for i in range(8)},
            "by_diagnostic_class": lead_attns_by_class,
        },
        "phase_cell_sensitivity": phase_cell_impacts,
        "lead_removal_ablation": lead_removal_results,
    }

    out_file = OUT_DIR / "setoperator_fold9_interpretability_diagnostics.json"
    with open(out_file, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Saved complete diagnostic findings to %s", out_file)


if __name__ == "__main__":
    main()
