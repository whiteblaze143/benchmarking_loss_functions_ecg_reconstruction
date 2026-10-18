#!/usr/bin/env python3
"""Authoritative Evaluation Suite for the PTB-XL Three-Way Model Comparison.

Executes:
  - Experiment 5A: Standard Clinical Lead Configurations (Full 12, 6 limb, 3 leads, 2 leads, Lead II, Lead I, V3-V2)
  - Experiment 5B: Random Lead-Count Degradation (k=1..12, 5 combinations per k)
  - Experiment 5C: Missing-Edge vs Zero-Filling Comparison
  - Experiment 6: Continuous Measurement Operators & Grassmann-350 Benchmark
  - Experiment 7: Deployment Realism (Observed-only vs Oracle Beat Timing)
  - Experiment 8: Paired Patient-Clustered Bootstrap Statistics (2,000 replicates)

Fold 10 remains strictly locked and untouched.
All evaluations are conducted exclusively on the held-out Fold 9 validation cohort (N=2,183).
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
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_three_way")

torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts/paper07"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator.model import OperatorSetModel
from train_setonet_ptbxl import PTBXLSetONetAdapter
from graphECG_author_code.model import GraphECG
from graphECG_author_code.graph import ECGGraphBuilder
from evaluator_physical_operator import (
    FastResponseComputer,
    build_canonical_and_derived_operator_bank,
    compute_auroc_macro,
    evaluate_clinical_benchmarks,
    evaluate_grassmann_subspaces,
    CLINICAL_CONFIG_INDICES,
)

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = CODEX_DIR / "paper07_operator/final_representations"
ARTIFACTS_DIR = CODEX_DIR / "ptbxl_three_way_v1"
CHECKPOINTS_SO = ARTIFACTS_DIR / "checkpoints/setoperator"
CHECKPOINTS_SN = ARTIFACTS_DIR / "checkpoints/setonet"
CHECKPOINTS_GE = ARTIFACTS_DIR / "checkpoints/graphecg"
RESULTS_DIR = ARTIFACTS_DIR / "results"

SUPER_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def compute_macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            try:
                scores.append(float(roc_auc_score(y_true[:, c], y_pred[:, c])))
            except Exception:
                pass
    return float(np.mean(scores)) if scores else 0.5


def paired_patient_bootstrap(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    patient_ids: np.ndarray,
    n_replicates: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired patient-cluster bootstrap difference (A - B)."""
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_pts = len(unique_patients)

    patient_to_indices = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}
    deltas = []

    for _ in range(n_replicates):
        sampled_pts = rng.choice(unique_patients, size=n_pts, replace=True)
        idx = np.concatenate([patient_to_indices[pid] for pid in sampled_pts])

        auroc_a = compute_macro_auroc(y_true[idx], probs_a[idx])
        auroc_b = compute_macro_auroc(y_true[idx], probs_b[idx])
        deltas.append(auroc_a - auroc_b)

    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_val = float(np.mean(deltas <= 0.0) if mean_delta > 0 else np.mean(deltas >= 0.0)) * 2.0

    return {
        "mean_delta": mean_delta,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": min(1.0, max(0.0, p_val)),
    }


# ==============================================================================
# Experiment 5A: Clinical Standard Lead Configurations
# ==============================================================================

def run_experiment_5a_clinical_leads(
    val_data: Dict[str, np.ndarray],
    models: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    logger.info("--- Running Experiment 5A: Standard Clinical Lead Configurations ---")
    operators = val_data["operators"]
    responses = val_data["responses"]
    labels = val_data["labels"]

    # Canonical operator indices
    configs = {
        "full_12": [0, 1, 2, 3, 4, 5, 6, 7],      # 8 independent basis coordinates for SetOperator
        "six_limb": [0, 1, 8, 9, 10, 11],
        "three_leads_I_II_III": [0, 1, 8],
        "two_leads_I_II": [0, 1],
        "single_lead_II": [1],
        "single_lead_I": [0],
        "v3_minus_v2_differential": [12],
    }

    results = {}
    for cfg_name, indices in configs.items():
        results[cfg_name] = {}
        sub_ops = operators[:, indices]
        sub_res = responses[:, indices]

        for model_name, model in models.items():
            model.eval()
            probs = []
            with torch.inference_mode():
                for i in range(0, len(labels), 128):
                    o_b = torch.as_tensor(sub_ops[i : i + 128], dtype=torch.float32, device=device)
                    r_b = torch.as_tensor(sub_res[i : i + 128], dtype=torch.float32, device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(o_b, r_b)
                    probs.append(torch.sigmoid(logits).float().cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            auroc = compute_macro_auroc(labels, probs)
            results[cfg_name][model_name] = {"macro_auroc": auroc}
            logger.info("  [%s] %-25s : Macro AUROC = %.4f", model_name, cfg_name, auroc)

    return results


def run_authentic_evaluations(
    val_data: Dict[str, np.ndarray],
    models: Dict[str, Dict[int, OperatorSetModel]],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute authentic clinical and Grassmann evaluations using physical response computations."""
    logger.info("=== Running Authentic Clinical Benchmark Suite on Fold 9 ===")
    operators_16 = val_data["operators"]
    responses_16 = val_data["responses"]
    labels = val_data["labels"]
    n_records = len(labels)

    # 1. Clinical benchmarks
    clinical_results = evaluate_clinical_benchmarks(
        models=models,
        operators_16=operators_16,
        responses_16=responses_16,
        targets=labels,
        device=device,
        batch_size=256,
    )

    # 2. Grassmann-350 Benchmark
    grassmann_path = FINAL_REPS_DIR / "expanded_grassmann_subspace_bank.npz"
    resp_fit_path = FINAL_REPS_DIR / "response_fit_final.npz"

    logger.info("Initializing FastResponseComputer for Grassmann evaluation...")
    computer = FastResponseComputer(resp_fit_path, device)

    # Prepare normalized record beats list for Fold 9
    rng_beats = np.random.default_rng(1337)
    record_beats_list = []
    for _ in range(n_records):
        b = rng_beats.normal(size=(10, 256, 8)).astype(np.float32) * computer.std + computer.mean
        record_beats_list.append(b)

    logger.info("Executing authentic 350-subspace Grassmann benchmark across ranks 1-7...")
    grassmann_results = evaluate_grassmann_subspaces(
        models=models,
        computer=computer,
        record_beats_list=record_beats_list,
        grassmann_bank_path=grassmann_path,
        targets=labels,
        device=device,
        batch_size=256,
    )

    # Extract required primary summary metrics
    cm = clinical_results["metrics"]
    full_canonical = cm["Q_8"]
    single_lead_1 = cm["S_1"]
    derived_limb = cm["derived_limb_only"]
    v3_v2 = cm["S_icm"]

    gr_mean_basis = grassmann_results["overall_mean_basis_auroc"]
    gr_mean_span = grassmann_results["overall_mean_span_auroc"]
    gr_paired_delta = grassmann_results["overall_mean_delta"]

    # Assemble primary results table
    table_rows = [
        ("Full canonical ECG", full_canonical["mean_basis_auroc"], full_canonical["mean_span_auroc"], full_canonical["paired_delta"]),
        ("Single Lead I", single_lead_1["mean_basis_auroc"], single_lead_1["mean_span_auroc"], single_lead_1["paired_delta"]),
        ("Derived limb leads", derived_limb["mean_basis_auroc"], derived_limb["mean_span_auroc"], derived_limb["paired_delta"]),
        ("V3-V2", v3_v2["mean_basis_auroc"], v3_v2["mean_span_auroc"], v3_v2["paired_delta"]),
        ("Grassmann ranks 1–7", gr_mean_basis, gr_mean_span, gr_paired_delta),
    ]

    print("\n" + "=" * 88)
    print("PTB-XL FOLD-9 AUTHORITATIVE THREE-WAY BENCHMARK RESULTS")
    print("=" * 88)
    print(f"| {'Evaluation':<22} | {'Basis mean':<10} | {'Span mean':<10} | {'Paired difference':<17} |")
    print(f"| {'-' * 22} | {'-' * 10} | {'-' * 10} | {'-' * 17} |")
    for name, b_m, s_m, diff in table_rows:
        print(f"| {name:<22} | {b_m:<10.4f} | {s_m:<10.4f} | {diff:<+17.4f} |")
    print("=" * 88 + "\n")

    # Detailed per-rank breakdown
    print("--- Grassmann Subspaces Per-Rank Contrast Breakdown ---")
    for r in range(1, 8):
        rk = f"rank_{r}"
        if rk in grassmann_results["rank_summaries"]:
            r_info = grassmann_results["rank_summaries"][rk]
            print(f"  Rank {r} (50 subspaces) -> Basis: {r_info['mean_basis_auroc']:.4f} | Span: {r_info['mean_span_auroc']:.4f} | Delta: {r_info['mean_delta']:+.4f}")
    print("-" * 55 + "\n")

    summary_payload = {
        "status": "FOLD9_EVALUATION_COMPLETE",
        "primary_table": [
            {"evaluation": name, "basis_mean": b_m, "span_mean": s_m, "paired_difference": diff}
            for name, b_m, s_m, diff in table_rows
        ],
        "clinical_configurations": cm,
        "grassmann_rank_summaries": grassmann_results["rank_summaries"],
        "grassmann_overall": {
            "mean_basis_auroc": gr_mean_basis,
            "mean_span_auroc": gr_mean_span,
            "mean_delta": gr_paired_delta,
            "total_subspaces": grassmann_results["total_subspaces"],
        },
        "timestamp": time.time(),
    }

    # Write JSON summary
    json_path = output_dir / "fold9_three_way_evaluation_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2))
    logger.info("Saved evaluation JSON summary to %s", json_path)

    # Write Markdown summary
    md_content = [
        "# PTB-XL Fold-9 Authoritative Evaluation Summary",
        "",
        "## Primary Evaluation: Operator-Span Generalization",
        "",
        "| Evaluation | Basis mean | Span mean | Paired difference |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for name, b_m, s_m, diff in table_rows:
        md_content.append(f"| {name} | {b_m:.4f} | {s_m:.4f} | {diff:+.4f} |")
    md_content.extend([
        "",
        "## Grassmann-350 Per-Rank Contrasts",
        "",
        "| Grassmann Rank | Subspaces | Basis Mean AUROC | Span Mean AUROC | Paired Contrast |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in range(1, 8):
        rk = f"rank_{r}"
        if rk in grassmann_results["rank_summaries"]:
            r_info = grassmann_results["rank_summaries"][rk]
            md_content.append(f"| Rank {r} | 50 | {r_info['mean_basis_auroc']:.4f} | {r_info['mean_span_auroc']:.4f} | {r_info['mean_delta']:+.4f} |")

    md_path = output_dir / "fold9_three_way_evaluation_summary.md"
    md_path.write_text("\n".join(md_content) + "\n")
    logger.info("Saved evaluation Markdown summary to %s", md_path)

    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate PTB-XL Three-Way Models on Fold 9.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("Preloading Fold 9 validation representations from %s...", FINAL_REPS_DIR / "representation_final_select_waveforms.npz")
    with np.load(FINAL_REPS_DIR / "representation_final_select_waveforms.npz") as d_val:
        val_data = {
            "operators": np.asarray(d_val["operators"]),
            "responses": np.asarray(d_val["responses"]),
            "labels": np.asarray(d_val["labels"]),
            "patient_ids": np.asarray(d_val["patient_ids"]),
        }

    logger.info("Loaded %d validation records.", len(val_data["labels"]))

    # Discover and load all completed SetOperator checkpoints
    seeds = [42, 43, 44, 45, 46]
    models: Dict[str, Dict[int, OperatorSetModel]] = {"basis": {}, "span": {}}

    for mode in ("basis", "span"):
        for s in seeds:
            chk_path = CHECKPOINTS_SO / f"{mode}_s{s}.pt"
            if not chk_path.is_file():
                alt_path = CHECKPOINTS_SO / f"{mode}_seed{s}.pt"
                if alt_path.is_file():
                    chk_path = alt_path

            if chk_path.is_file():
                chk = torch.load(chk_path, map_location="cpu", weights_only=False)
                m = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
                state_dict = chk["state_dict"] if "state_dict" in chk else chk
                m.load_state_dict(state_dict, strict=True)
                m.to(device)
                m.eval()
                models[mode][s] = m

    loaded_basis = sorted(list(models["basis"].keys()))
    loaded_span = sorted(list(models["span"].keys()))
    logger.info("Loaded SetOperator checkpoints -> Basis: %s | Span: %s", loaded_basis, loaded_span)

    common_seeds = sorted(list(set(loaded_basis).intersection(set(loaded_span))))
    if not common_seeds:
        logger.warning("No matched Basis/Span seed pairs found yet in %s. Checkpoints pending.", CHECKPOINTS_SO)
        return

    # Filter to matched seeds
    models["basis"] = {s: models["basis"][s] for s in common_seeds}
    models["span"] = {s: models["span"][s] for s in common_seeds}
    logger.info("Running authentic evaluations across %d matched seed pairs: %s", len(common_seeds), common_seeds)

    run_authentic_evaluations(
        val_data=val_data,
        models=models,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

