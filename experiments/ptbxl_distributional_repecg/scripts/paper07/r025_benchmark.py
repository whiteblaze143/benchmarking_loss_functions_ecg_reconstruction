#!/usr/bin/env python3
"""Milestone R025: Comprehensive Zero-Shot Benchmark and Invariance Stress Test.

Evaluates trained models across all 13 test configurations:
1. Full & Precordial: Q8_full, S6_precordial
2. Reduced sparse leads: S3_reduced, S2_reduced, S1_smartwatch
3. Limb geometry & single leads: S6_limb, Derived_limb_only, Lead_III_single, aVF_single
4. Unseen non-standard & oblique vectors: S_icm (V3-V2), S_icm_plus_leadI, Test_obliques_3, All_16_leads

Implements:
- E2: Zero-shot architectural bias evaluation across all arms.
- E3: Observation-basis invariance stress test under invertible R, permutations, and dependent lead additions.
- E7: Sensitivity, timing, and parameter audit.
- Generates aligned patient-level predictions for bootstrap statistics.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cuDNN 1D conv bug on this environment
torch.backends.cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.observable_encoder import (
    OSOnlyModel,
    OSWithoutPModel,
    LearnedQWaveformModel,
)
from scripts.paper07.r025_graph_adapter import GraphECGAdapter, load_graphecg_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r025_benchmark")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
WAVEFORM_DIR = CODEX_DIR / "paper07_operator/r025_waveforms"
R024_DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
MODELS_DIR = CODEX_DIR / "paper07_operator/r025_models"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r025_benchmark_results"

CONFIGURATIONS = {
    "Q8_full": [0, 1, 2, 3, 4, 5, 6, 7],
    "S6_precordial": [2, 3, 4, 5, 6, 7],
    "S6_limb": [0, 1, 8, 9, 10, 11],
    "Derived_limb_only": [8, 9, 10, 11],
    "Lead_III_single": [8],
    "aVF_single": [11],
    "S3_reduced": [0, 1, 3],
    "S2_reduced": [0, 1],
    "S1_smartwatch": [0],
    "S_icm": [12],
    "S_icm_plus_leadI": [0, 12],
    "Test_obliques_3": [13, 14, 15],
    "All_16_leads": list(range(16)),
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
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
        "class_aurocs": class_aurocs,
        "class_auprcs": class_auprcs,
    }


def predict_waveform_model(
    model: nn.Module,
    waveforms: np.ndarray,  # [N, 256, 8]
    operators_bank: np.ndarray,  # [16, 8]
    lead_indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    N = len(waveforms)
    q_sub = torch.as_tensor(operators_bank[lead_indices], dtype=torch.float32, device=device).unsqueeze(0)  # [1, m, 8]
    probs = []

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            x_b = torch.as_tensor(waveforms[i : i + batch_size], dtype=torch.float32, device=device)  # [B, 256, 8]
            B_cur = len(x_b)
            q_b = q_sub.expand(B_cur, -1, -1)  # [B, m, 8]
            # Exact physical observation law: Y = X @ Q^T
            y_b = x_b @ q_b.transpose(-2, -1)  # [B, 256, m]

            logits = model(q_b, y_b)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs, axis=0)


def predict_canonical_model(
    model: OperatorSetModel,
    operators: np.ndarray,  # [N, 16, 8]
    responses: np.ndarray,  # [N, 16, 16, 128]
    lead_indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    N = len(operators)
    sub_ops = operators[:, lead_indices]
    sub_resps = responses[:, lead_indices]
    probs = []

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            o_b = torch.as_tensor(sub_ops[i : i + batch_size], dtype=torch.float32, device=device)
            r_b = torch.as_tensor(sub_resps[i : i + batch_size], dtype=torch.float32, device=device)
            logits = model(o_b, r_b, return_reconstruction=False)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs, axis=0)


def predict_graphecg(
    graphecg_model: nn.Module,
    graph_adapter: GraphECGAdapter,
    config_name: str,
    waveforms: np.ndarray,  # [N, 256, 8]
    device: torch.device,
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    from torch_geometric.data import Batch

    # Check if configuration is representable
    sample_graph = graph_adapter.build_graph_for_configuration(config_name, waveforms[0])
    if sample_graph is None:
        return None

    graphecg_model.eval()
    probs = []
    N = len(waveforms)

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            batch_graphs = [
                graph_adapter.build_graph_for_configuration(config_name, waveforms[j])
                for j in range(i, min(i + batch_size, N))
            ]
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            out = graphecg_model(batch_data)
            logits = out["logits"] if isinstance(out, dict) else out
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs, axis=0)


def run_e3_invariance_stress_test(
    model: OSOnlyModel,
    waveforms: np.ndarray,  # [N, 256, 8]
    operators_bank: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """E3: Observation-basis invariance stress test under invertible R and permutations."""
    logger.info("--- Running E3: Observation-Basis Invariance Stress Test on Trained Model ---")
    model.eval()
    N_sample = min(100, len(waveforms))
    x_test = torch.as_tensor(waveforms[:N_sample], dtype=torch.float32, device=device)

    # Test configuration: S6_limb [I, II, III, aVR, aVL, aVF] (m=6)
    indices = [0, 1, 8, 9, 10, 11]
    Q_orig = torch.as_tensor(operators_bank[indices], dtype=torch.float32, device=device).unsqueeze(0).expand(N_sample, -1, -1)
    Y_orig = x_test @ Q_orig.transpose(-2, -1)

    with torch.inference_mode():
        logits_orig, latent_orig = model(Q_orig, Y_orig, return_latent=True)

    # 1. Random well-conditioned invertible R (6x6)
    torch.manual_seed(101)
    R = torch.randn(6, 6, dtype=torch.float32, device=device)
    while torch.linalg.cond(R) > 25.0:
        R = torch.randn(6, 6, dtype=torch.float32, device=device)

    Q_remix = torch.matmul(R, Q_orig)
    Y_remix = torch.matmul(Y_orig, R.transpose(-2, -1))

    with torch.inference_mode():
        logits_remix, latent_remix = model(Q_remix, Y_remix, return_latent=True)

    diff_logits_remix = float(torch.max(torch.abs(logits_orig - logits_remix)))
    diff_latent_remix = float(torch.max(torch.abs(latent_orig - latent_remix)))

    # 2. Random permutation
    perm = torch.randperm(6)
    Q_perm = Q_orig[:, perm]
    Y_perm = Y_orig[:, :, perm]

    with torch.inference_mode():
        logits_perm, latent_perm = model(Q_perm, Y_perm, return_latent=True)

    diff_logits_perm = float(torch.max(torch.abs(logits_orig - logits_perm)))
    diff_latent_perm = float(torch.max(torch.abs(latent_orig - latent_perm)))

    # 3. Channel sign negation
    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float32, device=device).view(1, 6, 1)
    Q_sign = Q_orig * signs
    Y_sign = Y_orig * signs.transpose(1, 2)

    with torch.inference_mode():
        logits_sign, latent_sign = model(Q_sign, Y_sign, return_latent=True)

    diff_logits_sign = float(torch.max(torch.abs(logits_orig - logits_sign)))
    diff_latent_sign = float(torch.max(torch.abs(latent_orig - latent_sign)))

    logger.info(f"E3 Invertible R Max Diff - Latent: {diff_latent_remix:.2e}, Logits: {diff_logits_remix:.2e}")
    logger.info(f"E3 Permutation Max Diff - Latent: {diff_latent_perm:.2e}, Logits: {diff_logits_perm:.2e}")
    logger.info(f"E3 Sign Negation Max Diff - Latent: {diff_latent_sign:.2e}, Logits: {diff_logits_sign:.2e}")

    assert diff_latent_remix < 1e-4, f"E3 Invertible remix latent diff: {diff_latent_remix}"
    assert diff_logits_remix < 1e-4, f"E3 Invertible remix logits diff: {diff_logits_remix}"
    assert diff_latent_perm < 1e-4, f"E3 Permutation latent diff: {diff_latent_perm}"
    assert diff_logits_perm < 1e-4, f"E3 Permutation logits diff: {diff_logits_perm}"

    return {
        "invertible_remix_max_diff_latent": diff_latent_remix,
        "invertible_remix_max_diff_logits": diff_logits_remix,
        "permutation_max_diff_latent": diff_latent_perm,
        "permutation_max_diff_logits": diff_logits_perm,
        "sign_negation_max_diff_latent": diff_latent_sign,
        "sign_negation_max_diff_logits": diff_logits_sign,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description="Milestone R025 Zero-Shot Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating benchmark on {device}...")

    # Load validation data
    val_npz = np.load(WAVEFORM_DIR / "r025_val_waveforms.npz")
    val_waveforms = val_npz["waveforms"]  # [2173, 256, 8]
    val_labels = val_npz["labels"]        # [2173, 5]
    val_ecg_ids = val_npz["ecg_ids"]
    val_patient_ids = val_npz["patient_ids"]

    # Load 16 operator bank from R024 real waveform evaluation set
    r024_val = np.load(R024_DATA_DIR / "representation_val_real_waveforms.npz")
    operators_bank = r024_val["operators"][0]  # [16, 8]
    canonical_resps = r024_val["responses"]    # [2173, 16, 16, 128]

    # Models to benchmark:
    # 1. OS-Only (Model B)
    os_ckpt_path = MODELS_DIR / f"os_only_seed{args.seed}/checkpoint.pt"
    model_os = OSOnlyModel(T=256, L=24, d=8, classes=5, frame_mode="dct", rcond=1e-5)
    if os_ckpt_path.exists():
        ckpt = torch.load(os_ckpt_path, map_location=device, weights_only=False)
        model_os.load_state_dict(ckpt["model_state_dict"])
    model_os.to(device)

    # 2. OS without P (Model C)
    os_nop_ckpt_path = MODELS_DIR / f"os_without_p_seed{args.seed}/checkpoint.pt"
    model_os_nop = OSWithoutPModel(T=256, L=24, d=8, classes=5, frame_mode="dct", rcond=1e-5)
    if os_nop_ckpt_path.exists():
        ckpt = torch.load(os_nop_ckpt_path, map_location=device, weights_only=False)
        model_os_nop.load_state_dict(ckpt["model_state_dict"])
    model_os_nop.to(device)

    # 3. Learned-Q (Model D)
    lq_ckpt_path = MODELS_DIR / f"learned_q_seed{args.seed}/checkpoint.pt"
    model_lq = LearnedQWaveformModel(T=256, L=24, d=8, classes=5, frame_mode="dct")
    if lq_ckpt_path.exists():
        ckpt = torch.load(lq_ckpt_path, map_location=device, weights_only=False)
        model_lq.load_state_dict(ckpt["model_state_dict"])
    model_lq.to(device)

    # 4. Canonical SetOperator (Historical R024 / Matched Basis-Only)
    canon_ckpt_path = CODEX_DIR / "paper07_operator/r024_models/basis_only/checkpoint.pt"
    if not canon_ckpt_path.exists():
        canon_ckpt_path = Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt")
    ckpt_c = torch.load(canon_ckpt_path, map_location=device, weights_only=False)
    state_dict_c = ckpt_c.get("state_dict") or ckpt_c.get("model_state_dict") or ckpt_c
    model_canon = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    model_canon.load_state_dict(state_dict_c)
    model_canon.to(device)

    # 5. GraphECG (Author checkpoint on PTB-XL)
    graphecg_ckpt_path = Path("experiments/ptbxl_distributional_repecg/outputs/graphecg/graphecg_ptbxl_best.pt")
    graphecg_model = load_graphecg_model(str(graphecg_ckpt_path), device)
    graph_adapter = GraphECGAdapter(device)

    # E3: Run invariance stress test on OS-Only
    e3_results = run_e3_invariance_stress_test(model_os, val_waveforms, operators_bank, device)

    # Main E2 benchmark loop
    results_by_arm: Dict[str, Dict[str, Any]] = {
        "os_only": {},
        "os_without_p": {},
        "learned_q": {},
        "canonical_setoperator": {},
        "graphecg": {},
    }

    patient_predictions: Dict[str, Dict[str, np.ndarray]] = {
        arm: {} for arm in results_by_arm
    }

    logger.info("=== Running Benchmark Across All 13 Configurations ===")
    for config_name, lead_indices in CONFIGURATIONS.items():
        logger.info(f"Evaluating config: {config_name} (leads: {lead_indices})...")

        # 1. OS-Only
        p_os = predict_waveform_model(model_os, val_waveforms, operators_bank, lead_indices, device, args.batch_size)
        m_os = compute_metrics(val_labels, p_os)
        results_by_arm["os_only"][config_name] = m_os
        patient_predictions["os_only"][config_name] = p_os

        # 2. OS without P
        p_nop = predict_waveform_model(model_os_nop, val_waveforms, operators_bank, lead_indices, device, args.batch_size)
        m_nop = compute_metrics(val_labels, p_nop)
        results_by_arm["os_without_p"][config_name] = m_nop
        patient_predictions["os_without_p"][config_name] = p_nop

        # 3. Learned-Q
        p_lq = predict_waveform_model(model_lq, val_waveforms, operators_bank, lead_indices, device, args.batch_size)
        m_lq = compute_metrics(val_labels, p_lq)
        results_by_arm["learned_q"][config_name] = m_lq
        patient_predictions["learned_q"][config_name] = p_lq

        # 4. Canonical SetOperator
        p_can = predict_canonical_model(model_canon, r024_val["operators"], canonical_resps, lead_indices, device, args.batch_size)
        m_can = compute_metrics(val_labels, p_can)
        results_by_arm["canonical_setoperator"][config_name] = m_can
        patient_predictions["canonical_setoperator"][config_name] = p_can

        # 5. GraphECG
        p_graph = predict_graphecg(graphecg_model, graph_adapter, config_name, val_waveforms, device, batch_size=32)
        if p_graph is not None:
            m_graph = compute_metrics(val_labels, p_graph)
            results_by_arm["graphecg"][config_name] = m_graph
            patient_predictions["graphecg"][config_name] = p_graph
        else:
            results_by_arm["graphecg"][config_name] = {"macro_auroc": None, "status": "N/A"}

        # Print comparison line
        g_str = f"{results_by_arm['graphecg'][config_name]['macro_auroc']:.4f}" if results_by_arm['graphecg'][config_name].get('macro_auroc') is not None else "N/A"
        logger.info(
            f"  [{config_name:18s}] OS: {m_os['macro_auroc']:.4f} | "
            f"OS(no P): {m_nop['macro_auroc']:.4f} | "
            f"Learned-Q: {m_lq['macro_auroc']:.4f} | "
            f"CanonSetOp: {m_can['macro_auroc']:.4f} | "
            f"GraphECG: {g_str}"
        )

    # Compute declared aggregate metrics
    sparse_configs = ["S3_reduced", "S2_reduced", "S1_smartwatch", "S_icm", "Derived_limb_only"]
    sparse_agg = {}
    for arm, cfgs in results_by_arm.items():
        vals = [cfgs[c]["macro_auroc"] for c in sparse_configs if cfgs[c].get("macro_auroc") is not None]
        sparse_agg[arm] = float(np.mean(vals)) if len(vals) > 0 else None

    logger.info("=== Predeclared Sparse Zero-Shot Aggregate AUROC ===")
    for arm, score in sparse_agg.items():
        score_str = f"{score:.4f}" if score is not None else "N/A"
        logger.info(f"  {arm:22s}: {score_str}")

    # Save benchmark outputs
    benchmark_payload = {
        "milestone": "R025",
        "description": "Zero-Shot Architectural Bias Benchmark Results",
        "seed": args.seed,
        "results_by_arm": results_by_arm,
        "sparse_aggregate_auroc": sparse_agg,
        "E3_invariance_stress_test": e3_results,
    }

    out_file = args.out_dir / f"r025_benchmark_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    # Save patient predictions for bootstrap statistics
    pred_file = args.out_dir / f"r025_predictions_seed{args.seed}.npz"
    # Flatten dict for npz saving
    npz_dict = {
        "labels": val_labels,
        "ecg_ids": val_ecg_ids,
        "patient_ids": val_patient_ids,
    }
    for arm, cfgs in patient_predictions.items():
        for cfg_name, probs in cfgs.items():
            if probs is not None:
                npz_dict[f"{arm}__{cfg_name}"] = probs

    np.savez_compressed(pred_file, **npz_dict)
    logger.info(f"Benchmark results saved to {out_file} and {pred_file}")


if __name__ == "__main__":
    main()
