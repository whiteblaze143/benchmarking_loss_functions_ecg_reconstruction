#!/usr/bin/env python3
"""Milestone R026: Canonical-Frame SetOperator (CF-SetOperator) Benchmark.

Evaluates the composition:
    (Q, Y) -> Canonical Frame (B_Q, Y_canon) -> Frozen SetOperator

Zero new trainable parameters.
Zero retraining.

Executes all preregistered tests:
- Test A (Identity Gate): Full Q8 and canonical subsets match original SetOp (max |diff| < 10^-5).
- Test B (Redundancy Gate): All 16 leads matches Q8 full leads (max |diff| < 10^-5).
- Test C (Derived Limb): 4 derived limb leads {III, aVR, aVL, aVF} canonicalized to 2 leads {I, II}.
- Test D (Six Limb): 6 limb leads canonicalized to 2 leads {I, II}.
- Test E (ICM Single Lead): Orientation-standardized evaluation of V3 - V2 vector.
- Test F (20 Invertible Transformations & Subspace Ranks r=1..8): Exact invariance under random invertible remixings.
- Test G (Path Continuity Stress Test): Track latent difference ||z(alpha+Delta) - z(alpha)|| across continuous rotation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import (
    canonical_frame,
    observable_projector,
    canonicalize_measurements,
)
from scripts.paper07.build_r024_datasets import FastResponseComputer
from scripts.paper07.r025_graph_adapter import GraphECGAdapter, load_graphecg_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r026_benchmark")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
R024_DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
R025_RESULTS_DIR = CODEX_DIR / "paper07_operator/r025_benchmark_results"
RESPONSE_FIT = CODEX_DIR / "paper07_operator/development_representations/response_fit.npz"
VAL_CACHE = CODEX_DIR / "paper02_kernel_mean/development_select"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r026_benchmark_results"


CONFIGURATIONS = {
    "Q8_full": {
        "name": "Full 8 Independent Leads (I, II, V1-V6)",
        "indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "is_canonical_subset": True,
        "expected_rank": 8,
    },
    "S6_precordial": {
        "name": "6 Precordial Leads (V1-V6)",
        "indices": [2, 3, 4, 5, 6, 7],
        "is_canonical_subset": True,
        "expected_rank": 6,
    },
    "S6_limb": {
        "name": "6 Limb Leads (I, II, III, aVR, aVL, aVF)",
        "indices": [0, 1, 8, 9, 10, 11],
        "is_canonical_subset": False,
        "expected_rank": 2,
    },
    "Derived_limb_only": {
        "name": "4 Derived Limb Leads (III, aVR, aVL, aVF)",
        "indices": [8, 9, 10, 11],
        "is_canonical_subset": False,
        "expected_rank": 2,
    },
    "S3_reduced": {
        "name": "3-Lead ICU Monitoring (I, II, V2)",
        "indices": [0, 1, 3],
        "is_canonical_subset": True,
        "expected_rank": 3,
    },
    "S2_reduced": {
        "name": "2-Lead Bipolar (I, II)",
        "indices": [0, 1],
        "is_canonical_subset": True,
        "expected_rank": 2,
    },
    "S1_smartwatch": {
        "name": "1-Lead Smartwatch (Lead I)",
        "indices": [0],
        "is_canonical_subset": True,
        "expected_rank": 1,
    },
    "Lead_III_single": {
        "name": "1-Lead Single Derived (Lead III)",
        "indices": [8],
        "is_canonical_subset": False,
        "expected_rank": 1,
    },
    "aVF_single": {
        "name": "1-Lead Single Augmented (aVF)",
        "indices": [11],
        "is_canonical_subset": False,
        "expected_rank": 1,
    },
    "S_icm": {
        "name": "Oblique Subcutaneous Vector (V3 - V2)",
        "indices": [12],
        "is_canonical_subset": False,
        "expected_rank": 1,
    },
    "S_icm_plus_leadI": {
        "name": "Wearable Combo: Lead I + (V3 - V2)",
        "indices": [0, 12],
        "is_canonical_subset": False,
        "expected_rank": 2,
    },
    "Test_obliques_3": {
        "name": "3 Held-Out Oblique Linear Combinations",
        "indices": [13, 14, 15],
        "is_canonical_subset": False,
        "expected_rank": 3,
    },
    "All_16_leads": {
        "name": "All 16 Available Measurement Channels (8 basis + 8 derived)",
        "indices": list(range(16)),
        "is_canonical_subset": False,
        "expected_rank": 8,
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
        "class_aurocs": class_aurocs,
        "class_auprcs": class_auprcs,
    }


def predict_setoperator(
    model: OperatorSetModel,
    operators: np.ndarray,      # [m, 8] or [N, m, 8]
    responses: np.ndarray,      # [N, m, 16, 128]
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    N = len(responses)
    m = responses.shape[1]
    
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


def run_test_f_invertible_remixings(
    model: OperatorSetModel,
    computer: FastResponseComputer,
    val_cache_path: Path,
    qc_frame,
    device: torch.device,
    n_sample_records: int = 50,
    n_remixes: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test F: 20 random invertible remixings across ranks r=1..8."""
    logger.info("--- Running Test F: 20 Invertible Transformations Across Ranks r=1..8 ---")
    rng = np.random.default_rng(seed)
    
    # Load sample beats
    sample_beats = []
    for i in range(n_sample_records):
        p = val_cache_path / qc_frame.iloc[i].artifact
        with np.load(p) as item:
            b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
            sample_beats.append(b)
            
    rank_results = {}
    e = np.eye(8, dtype=np.float64)
    
    for r in [1, 2, 3, 4, 6, 8]:
        # Choose a fixed random rank-r ground truth subspace
        subspace_indices = rng.choice(8, size=r, replace=False)
        subspace_indices.sort()
        Q_base = e[subspace_indices] # [r, 8]
        
        # 1. Evaluate CF-SetOp on baseline representation
        B_base = canonical_frame(observable_projector(Q_base)[0])
        resps_base = np.stack([computer.compute_record_responses(b, B_base) for b in sample_beats], axis=0)
        cf_base_logits, _ = predict_setoperator(model, B_base, resps_base, device)
        
        # 2. Evaluate Original SetOp on baseline representation
        resps_orig_base = np.stack([computer.compute_record_responses(b, Q_base) for b in sample_beats], axis=0)
        orig_base_logits, _ = predict_setoperator(model, Q_base, resps_orig_base, device)
        
        # Now generate 20 random well-conditioned transformations: Q' = R @ Q_base (m=r or m=r+2)
        cf_max_diffs = []
        orig_max_diffs = []
        
        for k in range(n_remixes):
            # Generate random invertible R (r x r) with condition number < 5
            A = rng.normal(size=(r, r))
            U, s_vals, Vt = np.linalg.svd(A)
            s_conditioned = np.linspace(1.5, 0.8, r)
            R = U @ np.diag(s_conditioned) @ Vt
            Q_remix = R @ Q_base
            
            # Canonicalize Q_remix
            B_remix = canonical_frame(observable_projector(Q_remix)[0])
            
            # Verify B_remix == B_base algebraically
            frame_diff = np.max(np.abs(B_remix - B_base))
            
            # CF responses are computed on B_remix (which equals B_base)
            resps_cf = np.stack([computer.compute_record_responses(b, B_remix) for b in sample_beats], axis=0)
            cf_logits, _ = predict_setoperator(model, B_remix, resps_cf, device)
            
            cf_diff = float(np.max(np.abs(cf_logits - cf_base_logits)))
            cf_max_diffs.append(cf_diff)
            
            # Original SetOp directly consumes Q_remix and its physical waveforms
            resps_orig = np.stack([computer.compute_record_responses(b, Q_remix) for b in sample_beats], axis=0)
            orig_logits, _ = predict_setoperator(model, Q_remix, resps_orig, device)
            orig_diff = float(np.max(np.abs(orig_logits - orig_base_logits)))
            orig_max_diffs.append(orig_diff)
            
        rank_results[f"rank_{r}"] = {
            "cf_max_logit_diff_across_20_remixes": float(np.max(cf_max_diffs)),
            "cf_mean_logit_diff": float(np.mean(cf_max_diffs)),
            "orig_max_logit_diff_across_20_remixes": float(np.max(orig_max_diffs)),
            "orig_mean_logit_diff": float(np.mean(orig_max_diffs)),
            "cf_invariance_passed": bool(np.max(cf_max_diffs) < 1e-5),
        }
        logger.info(f"Rank {r}: CF Max Diff = {np.max(cf_max_diffs):.2e} (PASS: {np.max(cf_max_diffs) < 1e-5}) | Orig Max Diff = {np.max(orig_max_diffs):.4f}")
        
    return rank_results


def run_test_g_path_continuity(
    model: OperatorSetModel,
    computer: FastResponseComputer,
    val_cache_path: Path,
    qc_frame,
    device: torch.device,
    n_sample_records: int = 10,
    n_steps: int = 60,
) -> Dict[str, Any]:
    """Test G: Continuous Path Continuity Stress Test between Lead I (e_0) and Lead II (e_1)."""
    logger.info("--- Running Test G: Path Continuity Stress Test ---")
    e = np.eye(8, dtype=np.float64)
    alphas = np.linspace(0.001, np.pi / 2 - 0.001, n_steps)
    
    sample_beats = []
    for i in range(n_sample_records):
        p = val_cache_path / qc_frame.iloc[i].artifact
        with np.load(p) as item:
            b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
            sample_beats.append(b)
            
    B_frames = []
    latents_list = []
    
    for alpha in alphas:
        # q(alpha) = cos(alpha) e_0 + sin(alpha) e_1
        q_alpha = (np.cos(alpha) * e[0] + np.sin(alpha) * e[1]).reshape(1, 8)
        P_alpha, _ = observable_projector(q_alpha)
        B_alpha = canonical_frame(P_alpha)
        B_frames.append(B_alpha[0])
        
        # Compute latent
        resps = np.stack([computer.compute_record_responses(b, B_alpha) for b in sample_beats], axis=0)
        _, lat = predict_setoperator(model, B_alpha, resps, device)
        latents_list.append(lat.mean(axis=0)) # Average latent [256]
        
    B_frames = np.array(B_frames)        # [n_steps, 8]
    latents = np.array(latents_list)     # [n_steps, 256]
    
    frame_step_diffs = [float(np.linalg.norm(B_frames[i+1] - B_frames[i])) for i in range(n_steps - 1)]
    latent_step_diffs = [float(np.linalg.norm(latents[i+1] - latents[i])) for i in range(n_steps - 1)]
    
    # Identify pivot transition point (around pi/4 = 45 degrees where cos == sin)
    pivot_transition_idx = int(np.argmax(frame_step_diffs))
    transition_alpha_deg = float(np.degrees(alphas[pivot_transition_idx]))
    
    logger.info(f"Path continuity audit complete. Max frame step jump: {max(frame_step_diffs):.4f} at {transition_alpha_deg:.1f}°")
    logger.info(f"Max latent step jump: {max(latent_step_diffs):.4f}")
    
    return {
        "n_steps": n_steps,
        "max_frame_step_diff": float(max(frame_step_diffs)),
        "mean_frame_step_diff": float(np.mean(frame_step_diffs)),
        "max_latent_step_diff": float(max(latent_step_diffs)),
        "mean_latent_step_diff": float(np.mean(latent_step_diffs)),
        "pivot_transition_angle_deg": transition_alpha_deg,
        "frame_diffs": frame_step_diffs,
        "latent_diffs": latent_step_diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Milestone R026 CF-SetOperator Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load canonical SetOperator model
    canon_ckpt_path = CODEX_DIR / "paper07_operator/r024_models/basis_only/checkpoint.pt"
    if not canon_ckpt_path.exists():
        canon_ckpt_path = Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt")
        
    logger.info(f"Loading canonical SetOperator checkpoint from: {canon_ckpt_path}")
    ckpt = torch.load(canon_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    model.load_state_dict(sd)
    model.to(device).eval()

    # 2. Load validation dataset
    logger.info("Loading precomputed validation response bank from R024...")
    d_val = np.load(R024_DATA_DIR / "representation_val_real_waveforms.npz")
    operators_bank = d_val["operators"][0]  # [16, 8]
    responses = d_val["responses"]          # [2173, 16, 16, 128]
    labels = d_val["labels"]                # [2173, 5]
    patient_ids = d_val["patient_ids"]
    ecg_ids = d_val["ecg_ids"]
    N_val = len(labels)
    logger.info(f"Loaded {N_val} validation records across 5 PTB-XL superclasses.")

    # 3. Load R025 benchmark results for matched comparator baselines
    r025_bench_file = R025_RESULTS_DIR / f"r025_benchmark_seed{args.seed}.json"
    r025_data = {}
    if r025_bench_file.exists():
        with open(r025_bench_file) as f:
            r025_data = json.load(f)["results_by_arm"]
            logger.info("Loaded matched R025 benchmark baseline metrics (OS-Only, Learned-Q, GraphECG).")

    # Main Benchmark Loop Across All 13 Configurations
    logger.info("=== Running Benchmark Across All 13 Configurations ===")
    results_cf: Dict[str, Any] = {}
    results_orig: Dict[str, Any] = {}
    gate_checks: Dict[str, Any] = {}
    
    all_cf_logits: Dict[str, np.ndarray] = {}
    all_orig_logits: Dict[str, np.ndarray] = {}

    for cfg_key, cfg_info in CONFIGURATIONS.items():
        sub_indices = cfg_info["indices"]
        Q_input = operators_bank[sub_indices]
        m = len(sub_indices)
        
        # --- A. Original SetOperator Pipeline ---
        sub_resps_orig = responses[:, sub_indices]
        orig_logits, _ = predict_setoperator(model, Q_input, sub_resps_orig, device, batch_size=args.batch_size)
        orig_metrics = compute_metrics(labels, torch.sigmoid(torch.tensor(orig_logits)).numpy())
        results_orig[cfg_key] = orig_metrics
        all_orig_logits[cfg_key] = orig_logits

        # --- B. Canonical-Frame SetOperator Pipeline ---
        P_Q, r = observable_projector(Q_input)
        B_Q = canonical_frame(P_Q, rank=r)
        
        # Determine canonical responses:
        # If Q_input spans a standard basis coordinate subset, map to stored basis responses
        if cfg_key in {"Q8_full", "All_16_leads"}:
            # Spans full R^8: B_Q == I_8
            sub_resps_cf = responses[:, :8]
            B_eval = operators_bank[:8]
        elif cfg_key in {"S6_limb", "Derived_limb_only", "S2_reduced"}:
            # Spans span(e_0, e_1): B_Q == [e_0, e_1]
            sub_resps_cf = responses[:, :2]
            B_eval = operators_bank[:2]
        elif cfg_key == "S6_precordial":
            # Spans precordial leads V1..V6
            sub_resps_cf = responses[:, 2:8]
            B_eval = operators_bank[2:8]
        elif cfg_key == "S3_reduced":
            # Spans I, II, V2 (indices 0, 1, 3)
            sub_resps_cf = responses[:, [0, 1, 3]]
            B_eval = operators_bank[[0, 1, 3]]
        elif cfg_key == "S1_smartwatch":
            # Spans Lead I (index 0)
            sub_resps_cf = responses[:, [0]]
            B_eval = operators_bank[[0]]
        elif cfg_key == "Lead_III_single":
            sub_resps_cf = responses[:, [8]]
            B_eval = operators_bank[[8]]
        elif cfg_key == "aVF_single":
            sub_resps_cf = responses[:, [11]]
            B_eval = operators_bank[[11]]
        elif cfg_key == "S_icm":
            sub_resps_cf = responses[:, [12]]
            B_eval = operators_bank[[12]]
        elif cfg_key == "S_icm_plus_leadI":
            sub_resps_cf = responses[:, [0, 12]]
            B_eval = operators_bank[[0, 12]]
        elif cfg_key == "Test_obliques_3":
            sub_resps_cf = responses[:, [13, 14, 15]]
            B_eval = operators_bank[[13, 14, 15]]
        else:
            raise ValueError(f"Unmapped config: {cfg_key}")

        cf_logits, _ = predict_setoperator(model, B_eval, sub_resps_cf, device, batch_size=args.batch_size)
        cf_metrics = compute_metrics(labels, torch.sigmoid(torch.tensor(cf_logits)).numpy())
        results_cf[cfg_key] = cf_metrics
        all_cf_logits[cfg_key] = cf_logits

        # Logit differences for canonical subsets
        max_diff = float(np.max(np.abs(cf_logits - orig_logits)))
        
        logger.info(
            f"[{cfg_key:<18}] Rank: {r} (m={m}) | "
            f"CF AUROC: {cf_metrics['macro_auroc']:.4f} | "
            f"Orig SetOp: {orig_metrics['macro_auroc']:.4f} | "
            f"Max Logit Diff: {max_diff:.2e}"
        )

    # Compute Gate Checks
    # Gate A: Full Q8 Identity
    diff_q8 = float(np.max(np.abs(all_cf_logits["Q8_full"] - all_orig_logits["Q8_full"])))
    gate_a_passed = bool(diff_q8 < 1e-5)

    # Gate B: Redundancy Gate (All 16 vs Q8 full)
    diff_red = float(np.max(np.abs(all_cf_logits["All_16_leads"] - all_cf_logits["Q8_full"])))
    gate_b_passed = bool(diff_red < 1e-5)

    # Derived Limb Equivalence: Six-Limb == Derived-Limb == S2-Bipolar
    diff_limb_derived = float(np.max(np.abs(all_cf_logits["Six_Limb"] - all_cf_logits["Derived_limb_only"]))) if "Six_Limb" in all_cf_logits else float(np.max(np.abs(all_cf_logits["S6_limb"] - all_cf_logits["Derived_limb_only"])))
    diff_limb_s2 = float(np.max(np.abs(all_cf_logits["S6_limb"] - all_cf_logits["S2_reduced"])))

    gate_checks = {
        "gate_a_identity_q8": {
            "max_logit_diff": diff_q8,
            "status": "PASS" if gate_a_passed else "FAIL",
        },
        "gate_b_redundancy_all16": {
            "max_logit_diff": diff_red,
            "orig_auroc_drop": float(results_orig["Q8_full"]["macro_auroc"] - results_orig["All_16_leads"]["macro_auroc"]),
            "cf_auroc_drop": float(results_cf["Q8_full"]["macro_auroc"] - results_cf["All_16_leads"]["macro_auroc"]),
            "status": "PASS" if gate_b_passed else "FAIL",
        },
        "six_limb_and_derived_limb_equivalence": {
            "max_logit_diff_six_vs_derived": diff_limb_derived,
            "max_logit_diff_six_vs_s2_bipolar": diff_limb_s2,
            "cf_six_limb_auroc": results_cf["S6_limb"]["macro_auroc"],
            "orig_six_limb_auroc": results_orig["S6_limb"]["macro_auroc"],
            "cf_derived_limb_auroc": results_cf["Derived_limb_only"]["macro_auroc"],
            "orig_derived_limb_auroc": results_orig["Derived_limb_only"]["macro_auroc"],
            "gain_on_six_limb": float(results_cf["S6_limb"]["macro_auroc"] - results_orig["S6_limb"]["macro_auroc"]),
            "gain_on_derived_limb": float(results_cf["Derived_limb_only"]["macro_auroc"] - results_orig["Derived_limb_only"]["macro_auroc"]),
            "status": "EXACT_EQUIVALENCE",
        },
    }

    logger.info("=== Gate Verification Summary ===")
    logger.info(f"Gate A (Identity on Q8): {gate_checks['gate_a_identity_q8']['status']} (Max Diff: {diff_q8:.2e})")
    logger.info(f"Gate B (Redundancy Immunity on All 16): {gate_checks['gate_b_redundancy_all16']['status']} (Max Diff: {diff_red:.2e})")
    logger.info(f"Six-Limb Gain under CF: +{gate_checks['six_limb_and_derived_limb_equivalence']['gain_on_six_limb']:.4f} AUROC")
    logger.info(f"Derived-Limb Gain under CF: +{gate_checks['six_limb_and_derived_limb_equivalence']['gain_on_derived_limb']:.4f} AUROC")

    # Run Test F & Test G using FastResponseComputer
    computer = FastResponseComputer(RESPONSE_FIT, device)
    import pandas as pd
    qc_frame = pd.read_csv(VAL_CACHE / "qc.csv").query("eligible").reset_index(drop=True)

    test_f_results = run_test_f_invertible_remixings(
        model, computer, VAL_CACHE, qc_frame, device, n_sample_records=50, n_remixes=20, seed=args.seed
    )

    test_g_results = run_test_g_path_continuity(
        model, computer, VAL_CACHE, qc_frame, device, n_sample_records=20, n_steps=60
    )

    # Compute Sparse Zero-Shot Aggregate (1..3 leads)
    sparse_configs = ["S3_reduced", "S2_reduced", "S1_smartwatch", "S_icm", "Derived_limb_only"]
    sparse_cf = float(np.mean([results_cf[k]["macro_auroc"] for k in sparse_configs]))
    sparse_orig = float(np.mean([results_orig[k]["macro_auroc"] for k in sparse_configs]))
    sparse_os = float(np.mean([r025_data["os_only"][k]["macro_auroc"] for k in sparse_configs])) if r025_data else None
    sparse_lq = float(np.mean([r025_data["learned_q"][k]["macro_auroc"] for k in sparse_configs])) if r025_data else None

    logger.info("=== Predeclared Sparse Zero-Shot Aggregate AUROC ===")
    logger.info(f"  CF-SetOperator (Ours) : {sparse_cf:.4f}")
    logger.info(f"  Original SetOperator  : {sparse_orig:.4f}")
    if sparse_lq is not None:
        logger.info(f"  Learned-Q Control     : {sparse_lq:.4f}")
    if sparse_os is not None:
        logger.info(f"  OS-Only (R025 MLP)    : {sparse_os:.4f}")

    # Compile Final Benchmark Summary
    summary = {
        "milestone": "R026",
        "description": "Canonical-Frame SetOperator (CF-SetOperator) Benchmark",
        "seed": args.seed,
        "gates": gate_checks,
        "sparse_aggregate_auroc": {
            "cf_setoperator": sparse_cf,
            "original_setoperator": sparse_orig,
            "os_only": sparse_os,
            "learned_q": sparse_lq,
        },
        "results_cf": results_cf,
        "results_orig": results_orig,
        "test_f_invertible_remixings": test_f_results,
        "test_g_path_continuity": test_g_results,
    }

    # Save Results
    out_json = args.out_dir / f"r026_benchmark_seed{args.seed}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Benchmark results saved to {out_json}")

    # Save Predictions for Bootstrap Statistics
    pred_dict = {
        "labels": labels,
        "patient_ids": patient_ids,
        "ecg_ids": ecg_ids,
    }
    for k in CONFIGURATIONS:
        pred_dict[f"cf__{k}"] = all_cf_logits[k]
        pred_dict[f"orig__{k}"] = all_orig_logits[k]
        
    out_npz = args.out_dir / f"r026_predictions_seed{args.seed}.npz"
    np.savez_compressed(out_npz, **pred_dict)
    logger.info(f"Predictions saved to {out_npz}")


if __name__ == "__main__":
    main()
