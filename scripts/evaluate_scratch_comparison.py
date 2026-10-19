#!/usr/bin/env python3
"""Definitive Matched Evaluation: Original SetOperator vs G1 Geometry-Aware SetOperator on EchoNext.

Evaluates both models strictly on identical validation records:
1. Canonical clinical configurations (Full 12, Six limb, 3 leads, 2 leads, Lead II, Lead I, V3-V2)
2. Development Grassmann operator bank (21 continuous random subspaces across ranks 1..7)
3. Calibration (ECE, Brier score) and patient-paired prediction drift
4. Basis-change stability under random orthogonal rotations within subspace
5. Redundant-lead sensitivity (R02 6-limb vs R04 2-lead spanning same 2D plane)
6. Inference latency benchmarking (ms/patient)
7. Parameter verification (active classification vs total instantiated)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"
MODELS_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"

for p in [SRC_DIR, BENCHMARK_DIR, MODELS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from setoperator_echonext import build_setoperator_echonext
from train_g1_setoperator_echonext import G1SetOperatorEchoNext
from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH
from repecg.paper07_operator.measurements import normalize_operator
from eval_echonext_clinical_benchmarks import OPERATORS_BY_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_comparison")

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]

BASELINE_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
G1_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/g1_setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
DEV_BANK_PATH = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/g1_checkpoint_adapted_s42/dev_bank/ptbxl_dev_operator_bank.npz")
OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/evaluations/scratch_comparison_s42")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b + 1])
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


class EchoNextPreloadedCohort:
    """Pre-extracts and caches beats for the validation records in memory to eliminate redundant CPU re-extraction."""

    def __init__(self, val_waveforms: np.ndarray, val_tabular: np.ndarray, val_labels: np.ndarray, extractor: EchoNextPhaseKMEExtractor, limit: Optional[int] = None):
        n = len(val_labels) if limit is None else min(limit, len(val_labels))
        self.tabular = torch.as_tensor(val_tabular[:n], dtype=torch.float32)
        self.labels = np.array(val_labels[:n], dtype=np.float32)

        cache_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/cache/val_beats_cached.pt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and limit is None:
            logger.info("Loading pre-extracted validation beats from %s...", cache_path)
            self.beats = torch.load(cache_path, map_location="cpu")
            logger.info("Loaded %d beat tensors from cache.", len(self.beats))
        else:
            logger.info("Pre-extracting beats for %d validation records on CPU...", n)
            t0 = time.time()
            self.beats = []
            for i in range(n):
                wf_8 = val_waveforms[i, 0][:, CANONICAL_8_LEADS]
                b = extractor.extract_beats(wf_8)
                self.beats.append(torch.from_numpy(b))
            dur = time.time() - t0
            logger.info("Pre-extracted beats for %d records in %.1fs (%.1f rec/s).", n, dur, n / max(dur, 1e-4))
            if limit is None:
                torch.save(self.beats, cache_path)
                logger.info("Saved validation beats cache to %s", cache_path)

    def __len__(self) -> int:
        return len(self.labels)


def evaluate_both_models_on_operators(
    base_model: nn.Module,
    g1_model: nn.Module,
    operators: np.ndarray,
    extractor: EchoNextPhaseKMEExtractor,
    val_cohort: EchoNextPreloadedCohort,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    base_model.eval()
    g1_model.eval()

    k = len(operators)
    ops_t = torch.as_tensor(operators, dtype=torch.float32, device=device)
    ops_batch = ops_t.unsqueeze(0)

    base_probs = []
    g1_probs = []
    n = len(val_cohort)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_sz = min(batch_size, n - i)
            batch_resps = []
            for j in range(b_sz):
                beats = val_cohort.beats[i + j]
                resps = extractor.compute_phase_kme(beats, torch.from_numpy(operators))
                batch_resps.append(resps)

            t_resps = torch.stack(batch_resps).to(device)
            t_tabs = val_cohort.tabular[i : i + b_sz].to(device)
            t_ops = ops_batch.expand(b_sz, -1, -1)

            out_b = base_model(operators=t_ops, responses=t_resps, tabular=t_tabs, mask=None)
            p_b = torch.sigmoid(out_b["logits"].squeeze(-1)).cpu().numpy()
            base_probs.extend(p_b.tolist())

            out_g = g1_model(operators=t_ops, responses=t_resps, tabular=t_tabs, mask=None)
            p_g = torch.sigmoid(out_g["logits"].squeeze(-1)).cpu().numpy()
            g1_probs.extend(p_g.tolist())

    total_time = time.time() - t0
    latency_ms_per_rec = (total_time / (2 * n)) * 1000.0  # per model per record

    y_true = val_cohort.labels
    y_b = np.array(base_probs)
    y_g = np.array(g1_probs)

    auc_b = float(roc_auc_score(y_true, y_b))
    auc_g = float(roc_auc_score(y_true, y_g))
    prc_b = float(average_precision_score(y_true, y_b))
    prc_g = float(average_precision_score(y_true, y_g))
    ece_b = compute_ece(y_true, y_b)
    ece_g = compute_ece(y_true, y_g)
    brier_b = float(brier_score_loss(y_true, y_b))
    brier_g = float(brier_score_loss(y_true, y_g))

    mean_abs_diff = float(np.mean(np.abs(y_g - y_b)))
    cos_sim = float(np.dot(y_g, y_b) / (np.linalg.norm(y_g) * np.linalg.norm(y_b) + 1e-8))

    return {
        "baseline_auroc": auc_b,
        "g1_auroc": auc_g,
        "delta_auroc": auc_g - auc_b,
        "baseline_auprc": prc_b,
        "g1_auprc": prc_g,
        "delta_auprc": prc_g - prc_b,
        "baseline_ece": ece_b,
        "g1_ece": ece_g,
        "baseline_brier": brier_b,
        "g1_brier": brier_g,
        "mean_abs_prob_diff": mean_abs_diff,
        "cosine_similarity": cos_sim,
        "latency_ms_per_record": latency_ms_per_rec,
        "base_probs": y_b,
        "g1_probs": y_g,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--baseline-ckpt", type=str, default=str(BASELINE_CKPT))
    parser.add_argument("--g1-ckpt", type=str, default=str(G1_CKPT))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_ckpt_path = Path(args.baseline_ckpt)
    g1_ckpt_path = Path(args.g1_ckpt)

    logger.info("==================================================")
    logger.info("MATCHED COMPARISON: ORIGINAL SETOPERATOR VS G1")
    logger.info("Baseline Checkpoint: %s", base_ckpt_path)
    logger.info("G1 Checkpoint:       %s", g1_ckpt_path)
    logger.info("Device:              %s", device)
    logger.info("==================================================")

    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at {base_ckpt_path}")
    if not g1_ckpt_path.exists():
        raise FileNotFoundError(f"G1 checkpoint not found at {g1_ckpt_path}")

    # Load Baseline Model
    logger.info("Loading original SetOperator baseline...")
    base_model = build_setoperator_echonext("SO-ECG+Tab-Native").to(device)
    base_chk = torch.load(base_ckpt_path, map_location="cpu")
    base_model.load_state_dict(base_chk["model_state_dict"])
    base_model.eval()

    # Load G1 Model
    logger.info("Loading G1 Geometry-Aware SetOperator...")
    g1_model = G1SetOperatorEchoNext(tabular_dim=7, geometry_mode="G1").to(device)
    g1_chk = torch.load(g1_ckpt_path, map_location="cpu")
    g1_model.load_state_dict(g1_chk["model_state_dict"])
    g1_model.eval()

    # Parameter count verification
    base_clf_params = sum(p.numel() for n, p in base_model.named_parameters() if not n.startswith("decoder"))
    g1_clf_params = sum(p.numel() for n, p in g1_model.named_parameters() if not n.startswith("core.decoder"))
    logger.info("Parameters: Baseline Active Clf=%s | G1 Active Clf=%s (Delta=%+d)",
                f"{base_clf_params:,}", f"{g1_clf_params:,}", g1_clf_params - base_clf_params)

    # Load Validation Data
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")

    extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device="cpu")
    val_cohort = EchoNextPreloadedCohort(val_w, val_t, y_val, extractor, limit=args.limit_val)
    logger.info("Cohort loaded with %d patients.", len(val_cohort))

    # 1. Clinical Lead Configurations Evaluation
    clinical_results = {}
    clinical_raw_probs = {}
    logger.info("--- 1. Evaluating Canonical Clinical Configurations ---")
    for cfg_name, ops in OPERATORS_BY_CONFIG.items():
        res = evaluate_both_models_on_operators(
            base_model, g1_model, ops, extractor, val_cohort, device
        )
        clinical_raw_probs[cfg_name] = {
            "base_probs": res.pop("base_probs"),
            "g1_probs": res.pop("g1_probs"),
        }
        clinical_results[cfg_name] = res
        logger.info("  %-22s: Baseline=%.4f | G1=%.4f | Δ=%+.4f | ΔAUPRC=%+.4f | Diff=%.4f",
                    cfg_name, res["baseline_auroc"], res["g1_auroc"], res["delta_auroc"],
                    res["delta_auprc"], res["mean_abs_prob_diff"])

    # Redundant-lead sensitivity: compare R02 (6 limb leads) vs R04 (2 limb leads)
    # Both span the identical 2D Einthoven frontal plane
    r02_base_p = clinical_raw_probs["R02_six_limb"]["base_probs"]
    r02_g1_p = clinical_raw_probs["R02_six_limb"]["g1_probs"]
    r04_base_p = clinical_raw_probs["R04_leads_I_II"]["base_probs"]
    r04_g1_p = clinical_raw_probs["R04_leads_I_II"]["g1_probs"]

    base_redundant_diff = float(np.mean(np.abs(r02_base_p - r04_base_p)))
    g1_redundant_diff = float(np.mean(np.abs(r02_g1_p - r04_g1_p)))
    redundant_lead_analysis = {
        "description": "Comparison between R02 (6 limb leads) and R04 (Leads I, II), spanning identical 2D Einthoven subspace",
        "baseline_auroc_r02": clinical_results["R02_six_limb"]["baseline_auroc"],
        "baseline_auroc_r04": clinical_results["R04_leads_I_II"]["baseline_auroc"],
        "baseline_delta_r02_minus_r04": clinical_results["R02_six_limb"]["baseline_auroc"] - clinical_results["R04_leads_I_II"]["baseline_auroc"],
        "baseline_mean_abs_prediction_shift": base_redundant_diff,
        "g1_auroc_r02": clinical_results["R02_six_limb"]["g1_auroc"],
        "g1_auroc_r04": clinical_results["R04_leads_I_II"]["g1_auroc"],
        "g1_delta_r02_minus_r04": clinical_results["R02_six_limb"]["g1_auroc"] - clinical_results["R04_leads_I_II"]["g1_auroc"],
        "g1_mean_abs_prediction_shift": g1_redundant_diff,
    }
    logger.info("Redundant Lead Shift (R02 vs R04): Baseline=%.4f | G1=%.4f", base_redundant_diff, g1_redundant_diff)

    # 2. Development Grassmann Subspace Bank Evaluation
    logger.info("--- 2. Evaluating Development Grassmann Operator Bank (21 Subspaces) ---")
    dev_bank_data = np.load(DEV_BANK_PATH)
    grassmann_results = {}
    all_dev_b_aurocs = []
    all_dev_g_aurocs = []
    all_dev_b_auprcs = []
    all_dev_g_auprcs = []
    all_basis_stability_base = []
    all_basis_stability_g1 = []

    rng = np.random.default_rng(2026)

    for r in range(1, 8):
        rank_key = f"rank_{r}"
        subspaces = dev_bank_data[rank_key]
        n_subs = len(subspaces)
        r_b_aurocs = []
        r_g_aurocs = []
        r_b_auprcs = []
        r_g_auprcs = []
        r_rot_shifts_base = []
        r_rot_shifts_g1 = []

        for s_idx in range(n_subs):
            sub_mat = subspaces[s_idx].astype(np.float32)  # (r, 8)
            res = evaluate_both_models_on_operators(
                base_model, g1_model, sub_mat, extractor, val_cohort, device
            )
            r_b_aurocs.append(res["baseline_auroc"])
            r_g_aurocs.append(res["g1_auroc"])
            r_b_auprcs.append(res["baseline_auprc"])
            r_g_auprcs.append(res["g1_auprc"])

            # Basis change stability test: rotate within subspace if r >= 2
            if r >= 2:
                # Sample random orthogonal matrix in O(r)
                U = ortho_group.rvs(dim=r, random_state=rng).astype(np.float32)
                sub_rot = U @ sub_mat  # (r, 8)
                res_rot = evaluate_both_models_on_operators(
                    base_model, g1_model, sub_rot, extractor, val_cohort, device
                )
                shift_b = float(np.mean(np.abs(res["base_probs"] - res_rot["base_probs"])))
                shift_g = float(np.mean(np.abs(res["g1_probs"] - res_rot["g1_probs"])))
                r_rot_shifts_base.append(shift_b)
                r_rot_shifts_g1.append(shift_g)

        mean_b_auc = float(np.mean(r_b_aurocs))
        mean_g_auc = float(np.mean(r_g_aurocs))
        mean_b_prc = float(np.mean(r_b_auprcs))
        mean_g_prc = float(np.mean(r_g_auprcs))

        all_dev_b_aurocs.extend(r_b_aurocs)
        all_dev_g_aurocs.extend(r_g_aurocs)
        all_dev_b_auprcs.extend(r_b_auprcs)
        all_dev_g_auprcs.extend(r_g_auprcs)

        if r_rot_shifts_base:
            all_basis_stability_base.extend(r_rot_shifts_base)
            all_basis_stability_g1.extend(r_rot_shifts_g1)

        grassmann_results[rank_key] = {
            "baseline_auroc": mean_b_auc,
            "g1_auroc": mean_g_auc,
            "delta_auroc": mean_g_auc - mean_b_auc,
            "baseline_auprc": mean_b_prc,
            "g1_auprc": mean_g_prc,
            "delta_auprc": mean_g_prc - mean_b_prc,
            "subspace_aurocs_baseline": r_b_aurocs,
            "subspace_aurocs_g1": r_g_aurocs,
            "basis_change_mean_shift_baseline": float(np.mean(r_rot_shifts_base)) if r_rot_shifts_base else 0.0,
            "basis_change_mean_shift_g1": float(np.mean(r_rot_shifts_g1)) if r_rot_shifts_g1 else 0.0,
        }
        logger.info("  Rank %d (%d subs): Baseline=%.4f | G1=%.4f | Δ=%+.4f",
                    r, n_subs, mean_b_auc, mean_g_auc, mean_g_auc - mean_b_auc)

    overall_dev_b_auc = float(np.mean(all_dev_b_aurocs))
    overall_dev_g_auc = float(np.mean(all_dev_g_aurocs))
    overall_dev_b_prc = float(np.mean(all_dev_b_auprcs))
    overall_dev_g_prc = float(np.mean(all_dev_g_auprcs))

    grassmann_results["overall_development_bank"] = {
        "n_subspaces": len(all_dev_b_aurocs),
        "baseline_auroc": overall_dev_b_auc,
        "g1_auroc": overall_dev_g_auc,
        "delta_auroc": overall_dev_g_auc - overall_dev_b_auc,
        "baseline_auprc": overall_dev_b_prc,
        "g1_auprc": overall_dev_g_prc,
        "delta_auprc": overall_dev_g_prc - overall_dev_b_prc,
        "basis_change_shift_baseline": float(np.mean(all_basis_stability_base)),
        "basis_change_shift_g1": float(np.mean(all_basis_stability_g1)),
    }
    logger.info("  OVERALL DEV BANK: Baseline=%.4f | G1=%.4f | Δ=%+.4f | ΔAUPRC=%+.4f",
                overall_dev_b_auc, overall_dev_g_auc, overall_dev_g_auc - overall_dev_b_auc,
                overall_dev_g_prc - overall_dev_b_prc)

    # 3. Compile Master Report
    report = {
        "status": "COMPLETED",
        "scientific_question": "Does explicit measurement geometry (G1) trained from scratch improve out-of-distribution ECG operator generalization?",
        "baseline_experiment": {
            "model": "Original SetOperator SO-Span",
            "checkpoint": str(base_ckpt_path),
            "best_epoch": base_chk.get("epoch"),
            "val_auroc_at_ckpt": base_chk.get("val_auroc"),
            "active_classification_parameters": base_clf_params,
        },
        "g1_experiment": {
            "model": "G1 Geometry-Aware SetOperator SO-Span",
            "checkpoint": str(g1_ckpt_path),
            "best_epoch": g1_chk.get("epoch"),
            "val_auroc_at_ckpt": g1_chk.get("val_auroc"),
            "active_classification_parameters": g1_clf_params,
            "added_parameters": g1_clf_params - base_clf_params,
        },
        "validation_records": len(val_cohort),
        "canonical_clinical_configurations": clinical_results,
        "redundant_lead_analysis": redundant_lead_analysis,
        "development_grassmann_bank": grassmann_results,
        "timestamp": time.time(),
    }

    report_path = OUT_DIR / "matched_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved complete matched comparison report to %s", report_path)


if __name__ == "__main__":
    main()
