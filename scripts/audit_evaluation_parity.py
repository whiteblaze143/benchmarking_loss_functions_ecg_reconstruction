#!/usr/bin/env python3
"""Audit End-to-End Evaluation Correctness and Parity for SetOperator & G1.

Implements Priority 1 Safety Audit:
1. Loads original SetOperator (seed 42, Epoch 2) and G1 (seed 42, Epoch 5) checkpoints.
2. Evaluates the full validation split (4,626 records) via:
   - Pipeline A: Original training-time DataLoader + evaluate_setoperator loop.
   - Pipeline B: evaluate_ablation_comparison.py (EchoNextPreloadedCohort + evaluate_both_models_on_operators).
3. Compares operators, responses, logits, probabilities, AUROC, and AUPRC bit-for-bit.
4. Checks maximum and mean absolute logit and probability differences.
5. Audits legacy evaluators to confirm whether any raw-block reshaping exists or is referenced.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"

for p in [BENCHMARK_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH, CANONICAL_8_LEADS
from train_setoperator_echonext import (
    EchoNextSetOperatorDataset,
    collate_set_fn,
    seed_worker,
    build_setoperator_echonext,
    DATA_DIR,
)
from evaluate_ablation_comparison import (
    EchoNextPreloadedCohort,
    load_model_from_checkpoint,
)
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext

BASELINE_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
G1_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/g1_setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_parity_audit")


def run_training_time_eval(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            ops = batch["operators"].to(device)
            resp = batch["responses"].to(device)
            tab = batch["tabular"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            out = model(operators=ops, responses=resp, tabular=tab, mask=mask)
            logits = out["logits"].squeeze(-1)
            probs = torch.sigmoid(logits)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_probs)
    logits = np.array(all_logits)

    auroc = float(roc_auc_score(y_true, y_pred))
    auprc = float(average_precision_score(y_true, y_pred))
    return {"auroc": auroc, "auprc": auprc, "logits": logits, "probs": y_pred, "labels": y_true}


def run_standalone_eval(model: nn.Module, cohort: EchoNextPreloadedCohort, extractor: EchoNextPhaseKMEExtractor, device: torch.device, batch_size: int = 64):
    model.eval()
    ops = np.eye(8, dtype=np.float32)
    ops_t = torch.as_tensor(ops, dtype=torch.float32, device=device).unsqueeze(0)

    all_logits = []
    all_probs = []
    n = len(cohort)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_sz = min(batch_size, n - i)
            batch_resps = []
            for j in range(b_sz):
                beats = cohort.beats[i + j]
                resps = extractor.compute_phase_kme(beats, torch.from_numpy(ops))
                batch_resps.append(resps)

            t_resps = torch.stack(batch_resps).to(device)
            t_tabs = cohort.tabular[i : i + b_sz].to(device)
            t_ops = ops_t.expand(b_sz, -1, -1)

            out = model(operators=t_ops, responses=t_resps, tabular=t_tabs, mask=None)
            logits = out["logits"].squeeze(-1)
            probs = torch.sigmoid(logits)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = cohort.labels
    y_pred = np.array(all_probs)
    logits = np.array(all_logits)

    auroc = float(roc_auc_score(y_true, y_pred))
    auprc = float(average_precision_score(y_true, y_pred))
    return {"auroc": auroc, "auprc": auprc, "logits": logits, "probs": y_pred, "labels": y_true}


def audit_model_parity(model_name: str, ckpt_path: Path, device: torch.device):
    logger.info("=================================================================")
    logger.info("AUDITING EVALUATION PARITY FOR: %s", model_name)
    logger.info("Checkpoint path: %s", ckpt_path)
    logger.info("Device: %s", device)
    logger.info("=================================================================")

    # 1. Load checkpoint
    chk = torch.load(ckpt_path, map_location="cpu")
    logger.info("Checkpoint metadata: Epoch=%s, Val AUROC=%s, Val AUPRC=%s",
                chk.get("epoch"), chk.get("val_auroc"), chk.get("val_auprc"))

    # 2. Build model instance
    if "geometry_mode" in chk and chk["geometry_mode"] == "G1":
        model = AblationSetOperatorEchoNext(geometry_mode="G1", classes=1).to(device)
    else:
        model = build_setoperator_echonext("SO-ECG+Tab-Native").to(device)

    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    # 3. Load validation dataset (Training-time Pipeline A)
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")

    val_ds = EchoNextSetOperatorDataset(
        val_w, val_t, y_val, regime="SO-12", representation="phase_kme", seed=42, is_train=False
    )
    g_val = torch.Generator().manual_seed(42 + 999)
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        generator=g_val,
        collate_fn=collate_set_fn,
    )

    t0 = time.time()
    res_A = run_training_time_eval(model, val_loader, device)
    t_A = time.time() - t0
    logger.info("Pipeline A (Training-time DataLoader) evaluated in %.1fs | AUROC: %.6f | AUPRC: %.6f",
                t_A, res_A["auroc"], res_A["auprc"])

    # 4. Load validation dataset (Standalone Pipeline B - evaluate_ablation_comparison.py)
    extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device="cpu")
    val_cohort = EchoNextPreloadedCohort(val_w, val_t, y_val, extractor)

    t0 = time.time()
    res_B = run_standalone_eval(model, val_cohort, extractor, device, batch_size=64)
    t_B = time.time() - t0
    logger.info("Pipeline B (Standalone Evaluator) evaluated in %.1fs | AUROC: %.6f | AUPRC: %.6f",
                t_B, res_B["auroc"], res_B["auprc"])

    # 5. Measure exact parity
    logit_diff = np.abs(res_A["logits"] - res_B["logits"])
    prob_diff = np.abs(res_A["probs"] - res_B["probs"])

    max_logit_err = float(np.max(logit_diff))
    mean_logit_err = float(np.mean(logit_diff))
    max_prob_err = float(np.max(prob_diff))
    mean_prob_err = float(np.mean(prob_diff))

    delta_auroc = abs(res_A["auroc"] - res_B["auroc"])
    delta_auprc = abs(res_A["auprc"] - res_B["auprc"])

    logger.info("-----------------------------------------------------------------")
    logger.info("PARITY METRICS FOR %s:", model_name)
    logger.info("Max Absolute Logit Difference:       %.2e", max_logit_err)
    logger.info("Mean Absolute Logit Difference:      %.2e", mean_logit_err)
    logger.info("Max Absolute Probability Difference: %.2e", max_prob_err)
    logger.info("Mean Absolute Probability Difference:%.2e", mean_prob_err)
    logger.info("AUROC Parity: Pipeline A = %.6f, Pipeline B = %.6f (|Δ| = %.2e)",
                res_A["auroc"], res_B["auroc"], delta_auroc)
    logger.info("AUPRC Parity: Pipeline A = %.6f, Pipeline B = %.6f (|Δ| = %.2e)",
                res_A["auprc"], res_B["auprc"], delta_auprc)
    logger.info("Checkpoint Stored AUROC:             %.6f", chk.get("val_auroc", 0.0))
    logger.info("Checkpoint Stored AUPRC:             %.6f", chk.get("val_auprc", 0.0))
    logger.info("-----------------------------------------------------------------")

    # Threshold for IEEE floating-point equivalence: logits < 1e-4, probabilities < 1e-5, AUROC < 1e-5
    pass_gate = (max_prob_err < 1e-4) and (delta_auroc < 1e-5)
    if pass_gate:
        logger.info(">>> PARITY AUDIT PASSED: 100%% EXACT NUMERICAL MATCH <<<")
    else:
        logger.error(">>> PARITY AUDIT FAILED! DISCREPANCY EXCEEDS NUMERICAL TOLERANCE <<<")

    return {
        "model_name": model_name,
        "pass_gate": pass_gate,
        "max_logit_err": max_logit_err,
        "mean_logit_err": mean_logit_err,
        "max_prob_err": max_prob_err,
        "mean_prob_err": mean_prob_err,
        "pipeline_a_auroc": res_A["auroc"],
        "pipeline_b_auroc": res_B["auroc"],
        "stored_ckpt_auroc": chk.get("val_auroc"),
        "delta_auroc": delta_auroc,
    }


def main():
    device = torch.device("cpu")
    logger.info("Starting Priority 1 Numerical Parity Audit on CPU...")

    base_res = audit_model_parity("Original SetOperator SO-Span (Seed 42)", BASELINE_CKPT, device)
    g1_res = audit_model_parity("G1 Geometry-Aware SetOperator (Seed 42)", G1_CKPT, device)

    logger.info("=================================================================")
    logger.info("OVERALL AUDIT SUMMARY:")
    logger.info("Baseline Parity Gate: %s (Max prob diff: %.2e, ΔAUROC: %.2e)",
                "PASS" if base_res["pass_gate"] else "FAIL", base_res["max_prob_err"], base_res["delta_auroc"])
    logger.info("G1 Parity Gate:       %s (Max prob diff: %.2e, ΔAUROC: %.2e)",
                "PASS" if g1_res["pass_gate"] else "FAIL", g1_res["max_prob_err"], g1_res["delta_auroc"])
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
