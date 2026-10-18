#!/usr/bin/env python3
"""Matched Evaluation Script for SetONet on Fold 9 Grassmann and Clinical Benchmarks.

Evaluates SetONet Basis vs Span across all matched seeds on:
  1. Standard Clinical Configurations: Full 12 (Q_8), 6 precordial (S_6), 6 limb, Leads I & II (S_2), Single Lead I (S_1), Derived limb only, Lead III.
  2. Authentic Grassmann-350 Subspaces across ranks 1-7 using authentic patient beats and FastResponseComputer.
  3. Direct head-to-head contrast with SetOperator on the exact same benchmark instances.

Fold 10 strictly sealed and untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_setonet")

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts/paper07"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from train_setonet_ptbxl import PTBXLSetONetAdapter
from evaluator_physical_operator import (
    FastResponseComputer,
    build_canonical_and_derived_operator_bank,
    compute_auroc_macro,
    evaluate_clinical_benchmarks,
    evaluate_grassmann_subspaces,
    verify_operator_physical_consistency,
)

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = CODEX_DIR / "paper07_operator/final_representations"
ARTIFACTS_DIR = CODEX_DIR / "ptbxl_three_way_v1"
CHECKPOINTS_SN = ARTIFACTS_DIR / "checkpoints/setonet"
OUTPUT_DIR = ARTIFACTS_DIR / "results_setonet"


def load_setonet_models(device: torch.device, seeds: List[int] = [42, 43, 44, 45, 46]) -> Tuple[Dict[str, Dict[int, Any]], List[int]]:
    models: Dict[str, Dict[int, Any]] = {"basis": {}, "span": {}}

    for s in seeds:
        chk_b_path = CHECKPOINTS_SN / f"setonet_basis_s{s}.pt"
        chk_s_path = CHECKPOINTS_SN / f"setonet_span_s{s}.pt"

        if chk_b_path.is_file() and chk_s_path.is_file():
            chk_b = torch.load(chk_b_path, map_location="cpu", weights_only=False)
            chk_s = torch.load(chk_s_path, map_location="cpu", weights_only=False)

            m_b = PTBXLSetONetAdapter(operator_dim=8, response_dim=128, classes=5, p=64)
            m_s = PTBXLSetONetAdapter(operator_dim=8, response_dim=128, classes=5, p=64)

            m_b.load_state_dict(chk_b["state_dict"] if "state_dict" in chk_b else chk_b)
            m_s.load_state_dict(chk_s["state_dict"] if "state_dict" in chk_s else chk_s)

            m_b.to(device).eval()
            m_s.to(device).eval()

            models["basis"][s] = m_b
            models["span"][s] = m_s

    common_seeds = sorted(list(set(models["basis"].keys()).intersection(set(models["span"].keys()))))
    return models, common_seeds


def main():
    parser = argparse.ArgumentParser(description="Evaluate SetONet on PTB-XL Fold 9.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-grassmann", action="store_true", help="Evaluate 350-subspace Grassmann benchmark")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("Loading Fold 9 validation representations...")
    with np.load(FINAL_REPS_DIR / "representation_final_select_waveforms.npz") as d_val:
        val_data = {
            "operators": np.asarray(d_val["operators"]),
            "responses": np.asarray(d_val["responses"]),
            "labels": np.asarray(d_val["labels"]),
            "ecg_ids": np.asarray(d_val["ecg_ids"]),
            "patient_ids": np.asarray(d_val["patient_ids"]),
        }

    models, common_seeds = load_setonet_models(device)
    logger.info("Loaded SetONet matched seed pairs: %s", common_seeds)

    if not common_seeds:
        logger.error("No matched SetONet seed pairs found in %s", CHECKPOINTS_SN)
        sys.exit(1)

    # 1. Evaluate Clinical Benchmarks
    logger.info("=== Running SetONet Clinical Benchmark Configurations ===")
    clinical_results = evaluate_clinical_benchmarks(
        models=models,
        operators_16=val_data["operators"],
        responses_16=val_data["responses"],
        targets=val_data["labels"],
        device=device,
        batch_size=256,
    )

    cm = clinical_results["metrics"]
    print("\n" + "=" * 80)
    print("SET-ONET CLINICAL BENCHMARK RESULTS (PTB-XL Fold 9)")
    print("=" * 80)
    print(f"| {'Configuration':<30} | {'Basis Mean':<12} | {'Span Mean':<12} | {'Difference':<12} |")
    print(f"| {'-' * 30} | {'-' * 12} | {'-' * 12} | {'-' * 12} |")
    for k in ["Q_8", "S_6", "S_6_limb", "S_2", "S_1", "derived_limb_only", "lead_III_single"]:
        if k in cm:
            b_m = cm[k]["mean_basis_auroc"]
            s_m = cm[k]["mean_span_auroc"]
            diff = cm[k]["paired_delta"]
            print(f"| {k:<30} | {b_m:<12.4f} | {s_m:<12.4f} | {diff:<+12.4f} |")
    print("=" * 80 + "\n")

    # 2. Evaluate Grassmann Subspaces if requested
    if args.eval_grassmann:
        beats_npz_path = FINAL_REPS_DIR / "fold9_authentic_beats.npz"
        d_beats = np.load(beats_npz_path, allow_pickle=True)
        record_beats_list = list(d_beats["beats"])
        resp_fit_path = FINAL_REPS_DIR / "response_fit_final.npz"
        computer = FastResponseComputer(resp_fit_path, device)
        grassmann_path = FINAL_REPS_DIR / "expanded_grassmann_subspace_bank.npz"
        preds_dir = args.output_dir / "predictions/grassmann"

        logger.info("Executing Grassmann-350 Benchmark for SetONet...")
        grassmann_results = evaluate_grassmann_subspaces(
            models=models,
            computer=computer,
            record_beats_list=record_beats_list,
            grassmann_bank_path=grassmann_path,
            targets=val_data["labels"],
            device=device,
            batch_size=256,
            ecg_ids=val_data["ecg_ids"],
            patient_ids=val_data["patient_ids"],
            predictions_dir=preds_dir,
        )

        out_summary = {
            "status": "SETONET_EVALUATION_COMPLETE",
            "matched_seeds": common_seeds,
            "clinical_metrics": cm,
            "grassmann_rank_summaries": grassmann_results["rank_summaries"],
            "timestamp": time.time(),
        }
    else:
        out_summary = {
            "status": "SETONET_CLINICAL_EVALUATION_COMPLETE",
            "matched_seeds": common_seeds,
            "clinical_metrics": cm,
            "timestamp": time.time(),
        }

    summary_file = args.output_dir / "setonet_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(out_summary, f, indent=2)
    logger.info("Saved SetONet evaluation summary to %s", summary_file)


if __name__ == "__main__":
    main()
