#!/usr/bin/env python3
"""Unified Matched Evaluation: Original SetOperator vs Any Ablation Variant on EchoNext.

Evaluates baseline and ablation strictly on identical validation records:
1. Canonical clinical configurations (Full 12, Six limb, 3 leads, 2 leads, Lead II, Lead I, V3-V2)
2. Development Grassmann operator bank (21 continuous random subspaces across ranks 1..7)
3. Calibration (ECE, Brier score) and patient-paired prediction drift
4. Redundant-lead sensitivity (R02 6-limb vs R04 2-lead spanning same 2D plane)
5. Inference latency benchmarking (ms/patient)
6. Parameter verification (active classification vs total instantiated)
7. Generates ablation_eval_report.json and result.json per P9 standard.
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
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext, TabularMLPEchoNext
from train_g1_setoperator_echonext import G1SetOperatorEchoNext
from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH
from eval_echonext_clinical_benchmarks import OPERATORS_BY_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_ablation")

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]
BASELINE_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
DEV_BANK_PATH = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/echonext_dev_operator_bank.npz")


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
    """Pre-extracts and caches beats for the validation records in memory."""

    def __init__(self, val_waveforms: np.ndarray, val_tabular: np.ndarray, val_labels: np.ndarray, extractor: EchoNextPhaseKMEExtractor, limit: Optional[int] = None):
        n = len(val_labels) if limit is None else min(limit, len(val_labels))
        self.tabular = torch.as_tensor(val_tabular[:n], dtype=torch.float32)
        self.labels = np.array(val_labels[:n], dtype=np.float32)

        local_cache = Path("/tmp/val_beats_cached.pt")
        cache_path = local_cache if local_cache.exists() else Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/cache/val_beats_cached.pt")
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
            logger.info("Pre-extracted beats for %d records in %.1fs.", n, dur)
            if limit is None:
                torch.save(self.beats, cache_path)

    def __len__(self) -> int:
        return len(self.labels)


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any], int]:
    chk = torch.load(ckpt_path, map_location="cpu")
    tab_mode = chk.get("tabular_mode")

    if tab_mode == "tabular_only":
        model = TabularMLPEchoNext(tabular_dim=7, classes=1).to(device)
    elif "geometry_mode" in chk or "set_encoder_mode" in chk or "phase_mode" in chk:
        model = AblationSetOperatorEchoNext(
            operator_mode=chk.get("operator_mode", "continuous"),
            geometry_mode=chk.get("geometry_mode", "baseline"),
            phase_mode=chk.get("phase_mode", "standard_circular"),
            set_encoder_mode=chk.get("set_encoder_mode", "standard_transformer"),
            pooling_mode=chk.get("pooling_mode", "learned_query"),
            tabular_mode=chk.get("tabular_mode", "native_additive"),
            classes=1,
        ).to(device)
    else:
        # Check if legacy G1 or baseline
        if "W_g1" in chk.get("model_state_dict", {}):
            model = G1SetOperatorEchoNext(tabular_dim=7, geometry_mode="G1").to(device)
        else:
            model = build_setoperator_echonext("SO-ECG+Tab-Native").to(device)

    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    # Active classification parameter count
    clf_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("decoder") and not n.startswith("core.decoder"))
    return model, chk, clf_params


def evaluate_both_models_on_operators(
    base_model: nn.Module,
    abl_model: nn.Module,
    operators: np.ndarray,
    extractor: EchoNextPhaseKMEExtractor,
    val_cohort: EchoNextPreloadedCohort,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    base_model.eval()
    abl_model.eval()

    k = len(operators)
    ops_t = torch.as_tensor(operators, dtype=torch.float32, device=device)
    ops_batch = ops_t.unsqueeze(0)

    base_probs = []
    abl_probs = []
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

            out_a = abl_model(operators=t_ops, responses=t_resps, tabular=t_tabs, mask=None)
            p_a = torch.sigmoid(out_a["logits"].squeeze(-1)).cpu().numpy()
            abl_probs.extend(p_a.tolist())

    total_time = time.time() - t0
    latency_ms_per_rec = (total_time / (2 * n)) * 1000.0

    y_true = val_cohort.labels
    y_b = np.array(base_probs)
    y_a = np.array(abl_probs)

    auc_b = float(roc_auc_score(y_true, y_b))
    auc_a = float(roc_auc_score(y_true, y_a))
    prc_b = float(average_precision_score(y_true, y_b))
    prc_a = float(average_precision_score(y_true, y_a))
    ece_b = compute_ece(y_true, y_b)
    ece_a = compute_ece(y_true, y_a)
    brier_b = float(brier_score_loss(y_true, y_b))
    brier_a = float(brier_score_loss(y_true, y_a))

    mean_abs_diff = float(np.mean(np.abs(y_a - y_b)))
    cos_sim = float(np.dot(y_a, y_b) / (np.linalg.norm(y_a) * np.linalg.norm(y_b) + 1e-8))

    return {
        "baseline_auroc": auc_b,
        "ablation_auroc": auc_a,
        "delta_auroc": auc_a - auc_b,
        "baseline_auprc": prc_b,
        "ablation_auprc": prc_a,
        "delta_auprc": prc_a - prc_b,
        "baseline_ece": ece_b,
        "ablation_ece": ece_a,
        "baseline_brier": brier_b,
        "ablation_brier": brier_a,
        "mean_abs_prob_diff": mean_abs_diff,
        "cosine_similarity": cos_sim,
        "latency_ms_per_record": latency_ms_per_rec,
        "base_probs": y_b,
        "abl_probs": y_a,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ablation Variant vs Original Baseline")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--baseline-ckpt", type=str, default=str(BASELINE_CKPT))
    parser.add_argument("--ablation-ckpt", type=str, required=True)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--hypothesis", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_ckpt_path = Path(args.baseline_ckpt)
    abl_ckpt_path = Path(args.ablation_ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else abl_ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("MATCHED COMPARISON: BASELINE VS ABLATION: %s", args.experiment_id)
    logger.info("Baseline Checkpoint: %s", base_ckpt_path)
    logger.info("Ablation Checkpoint: %s", abl_ckpt_path)
    logger.info("Device:              %s", device)
    logger.info("==================================================")

    base_model, base_chk, base_clf_params = load_model_from_checkpoint(base_ckpt_path, device)
    abl_model, abl_chk, abl_clf_params = load_model_from_checkpoint(abl_ckpt_path, device)

    logger.info("Parameters: Baseline Active Clf=%s | Ablation Active Clf=%s (Delta=%+d)",
                f"{base_clf_params:,}", f"{abl_clf_params:,}", abl_clf_params - base_clf_params)

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
            base_model, abl_model, ops, extractor, val_cohort, device
        )
        clinical_raw_probs[cfg_name] = {
            "base_probs": res.pop("base_probs"),
            "abl_probs": res.pop("abl_probs"),
        }
        clinical_results[cfg_name] = res
        logger.info("  %-22s: Baseline=%.4f | Ablation=%.4f | Δ=%+.4f | ΔAUPRC=%+.4f | Diff=%.4f",
                    cfg_name, res["baseline_auroc"], res["ablation_auroc"], res["delta_auroc"],
                    res["delta_auprc"], res["mean_abs_prob_diff"])

    # Redundant-lead sensitivity (R02 vs R04)
    r02_base_p = clinical_raw_probs["R02_six_limb"]["base_probs"]
    r02_abl_p = clinical_raw_probs["R02_six_limb"]["abl_probs"]
    r04_base_p = clinical_raw_probs["R04_leads_I_II"]["base_probs"]
    r04_abl_p = clinical_raw_probs["R04_leads_I_II"]["abl_probs"]

    base_redundant_diff = float(np.mean(np.abs(r02_base_p - r04_base_p)))
    abl_redundant_diff = float(np.mean(np.abs(r02_abl_p - r04_abl_p)))
    redundant_lead_analysis = {
        "description": "Comparison between R02 (6 limb leads) and R04 (Leads I, II), spanning identical 2D Einthoven subspace",
        "baseline_auroc_r02": clinical_results["R02_six_limb"]["baseline_auroc"],
        "baseline_auroc_r04": clinical_results["R04_leads_I_II"]["baseline_auroc"],
        "baseline_delta_r02_minus_r04": clinical_results["R02_six_limb"]["baseline_auroc"] - clinical_results["R04_leads_I_II"]["baseline_auroc"],
        "baseline_mean_abs_prediction_shift": base_redundant_diff,
        "ablation_auroc_r02": clinical_results["R02_six_limb"]["ablation_auroc"],
        "ablation_auroc_r04": clinical_results["R04_leads_I_II"]["ablation_auroc"],
        "ablation_delta_r02_minus_r04": clinical_results["R02_six_limb"]["ablation_auroc"] - clinical_results["R04_leads_I_II"]["ablation_auroc"],
        "ablation_mean_abs_prediction_shift": abl_redundant_diff,
    }

    dev_bank_data = np.load(DEV_BANK_PATH)
    total_subspaces = sum(len(dev_bank_data[f"rank_{r}"]) for r in range(1, 8))
    logger.info("--- 2. Evaluating Frozen EchoNext Development Grassmann Bank (%d Subspaces) ---", total_subspaces)
    grassmann_results = {}
    all_dev_b_aurocs = []
    all_dev_a_aurocs = []
    all_dev_b_auprcs = []
    all_dev_a_auprcs = []

    for r in range(1, 8):
        rank_key = f"rank_{r}"
        subspaces = dev_bank_data[rank_key]
        n_subs = len(subspaces)
        r_b_aurocs = []
        r_a_aurocs = []
        r_b_auprcs = []
        r_a_auprcs = []

        for s_idx in range(n_subs):
            sub_mat = subspaces[s_idx].astype(np.float32)
            res = evaluate_both_models_on_operators(
                base_model, abl_model, sub_mat, extractor, val_cohort, device
            )
            r_b_aurocs.append(res["baseline_auroc"])
            r_a_aurocs.append(res["ablation_auroc"])
            r_b_auprcs.append(res["baseline_auprc"])
            r_a_auprcs.append(res["ablation_auprc"])

        mean_b_auc = float(np.mean(r_b_aurocs))
        mean_a_auc = float(np.mean(r_a_aurocs))
        mean_b_prc = float(np.mean(r_b_auprcs))
        mean_a_prc = float(np.mean(r_a_auprcs))

        all_dev_b_aurocs.extend(r_b_aurocs)
        all_dev_a_aurocs.extend(r_a_aurocs)
        all_dev_b_auprcs.extend(r_b_auprcs)
        all_dev_a_auprcs.extend(r_a_auprcs)

        grassmann_results[rank_key] = {
            "baseline_auroc": mean_b_auc,
            "ablation_auroc": mean_a_auc,
            "delta_auroc": mean_a_auc - mean_b_auc,
            "baseline_auprc": mean_b_prc,
            "ablation_auprc": mean_a_prc,
            "delta_auprc": mean_a_prc - mean_b_prc,
            "subspace_aurocs_baseline": r_b_aurocs,
            "subspace_aurocs_ablation": r_a_aurocs,
        }
        logger.info("  Rank %d (%d subs): Baseline=%.4f | Ablation=%.4f | Δ=%+.4f",
                    r, n_subs, mean_b_auc, mean_a_auc, mean_a_auc - mean_b_auc)

    overall_dev_b_auc = float(np.mean(all_dev_b_aurocs))
    overall_dev_a_auc = float(np.mean(all_dev_a_aurocs))
    overall_dev_b_prc = float(np.mean(all_dev_b_auprcs))
    overall_dev_a_prc = float(np.mean(all_dev_a_auprcs))

    grassmann_results["overall_development_bank"] = {
        "n_subspaces": len(all_dev_b_aurocs),
        "baseline_auroc": overall_dev_b_auc,
        "ablation_auroc": overall_dev_a_auc,
        "delta_auroc": overall_dev_a_auc - overall_dev_b_auc,
        "baseline_auprc": overall_dev_b_prc,
        "ablation_auprc": overall_dev_a_prc,
        "delta_auprc": overall_dev_a_prc - overall_dev_b_prc,
    }
    logger.info("  OVERALL DEV BANK: Baseline=%.4f | Ablation=%.4f | Δ=%+.4f | ΔAUPRC=%+.4f",
                overall_dev_b_auc, overall_dev_a_auc, overall_dev_a_auc - overall_dev_b_auc,
                overall_dev_a_prc - overall_dev_b_prc)

    # 2b. Rotated-Basis Stability (Within-subspace orthonormal basis changes: Q_rot = U @ Q, U in O(r))
    logger.info("--- 2b. Evaluating Rotated-Basis Invariance across Multi-Lead Subspaces ---")
    basis_rotation_results = {}
    rng_rot = np.random.default_rng(2026)
    all_shifts_b = []
    all_shifts_a = []
    for r in range(2, 8):
        rank_key = f"rank_{r}"
        subspaces = dev_bank_data[rank_key]
        n_test = min(3, len(subspaces))
        r_shifts_b = []
        r_shifts_a = []
        for s_idx in range(n_test):
            Q = subspaces[s_idx].astype(np.float32)
            A = rng_rot.normal(size=(r, r))
            U, _ = np.linalg.qr(A)
            Q_rot = (U @ Q).astype(np.float32)

            res_orig = evaluate_both_models_on_operators(base_model, abl_model, Q, extractor, val_cohort, device)
            res_rot = evaluate_both_models_on_operators(base_model, abl_model, Q_rot, extractor, val_cohort, device)

            p_b_orig = res_orig["base_probs"]
            p_b_rot = res_rot["base_probs"]
            p_a_orig = res_orig["abl_probs"]
            p_a_rot = res_rot["abl_probs"]

            r_shifts_b.append(float(np.mean(np.abs(p_b_orig - p_b_rot))))
            r_shifts_a.append(float(np.mean(np.abs(p_a_orig - p_a_rot))))

        m_shift_b = float(np.mean(r_shifts_b))
        m_shift_a = float(np.mean(r_shifts_a))
        all_shifts_b.extend(r_shifts_b)
        all_shifts_a.extend(r_shifts_a)
        basis_rotation_results[rank_key] = {
            "baseline_mean_abs_shift": m_shift_b,
            "ablation_mean_abs_shift": m_shift_a,
        }
        logger.info("  Rank %d Rotated-Basis Shift: Baseline=%.4f | Ablation=%.4f", r, m_shift_b, m_shift_a)

    basis_rotation_results["overall"] = {
        "baseline_mean_abs_shift": float(np.mean(all_shifts_b)),
        "ablation_mean_abs_shift": float(np.mean(all_shifts_a)),
    }

    # 3. Read train summary if exists
    train_sum_path = out_dir / "train_summary.json"
    train_history = []
    train_duration_min = 0.0
    if train_sum_path.exists():
        with open(train_sum_path, "r") as f:
            ts = json.load(f)
            train_history = ts.get("history", [])
            train_duration_min = ts.get("train_duration_minutes", 0.0)

    # 4. Compile P9 result.json
    result_data = {
        "experiment_id": args.experiment_id,
        "hypothesis": args.hypothesis,
        "model_variant": {
            "geometry_mode": abl_chk.get("geometry_mode", "baseline"),
            "operator_mode": abl_chk.get("operator_mode", "continuous"),
            "phase_mode": abl_chk.get("phase_mode", "standard_circular"),
            "set_encoder_mode": abl_chk.get("set_encoder_mode", "standard_transformer"),
            "pooling_mode": abl_chk.get("pooling_mode", "learned_query"),
            "tabular_mode": abl_chk.get("tabular_mode", "native_additive"),
            "representation": abl_chk.get("representation", "phase_kme"),
        },
        "active_parameters": {
            "baseline": base_clf_params,
            "ablation": abl_clf_params,
            "delta": abl_clf_params - base_clf_params,
        },
        "best_epoch": abl_chk.get("epoch"),
        "validation_metric_at_best_epoch": {
            "val_auroc": abl_chk.get("val_auroc"),
            "val_auprc": abl_chk.get("val_auprc"),
        },
        "canonical_evaluation": clinical_results["R01_full12"],
        "development_grassmann_bank": grassmann_results,
        "rotated_basis_stability": basis_rotation_results,
        "calibration_and_reliability": {
            "canonical_ece": clinical_results["R01_full12"]["ablation_ece"],
            "canonical_brier": clinical_results["R01_full12"]["ablation_brier"],
            "redundant_lead_shift": abl_redundant_diff,
        },
        "inference_throughput": {
            "latency_ms_per_record": clinical_results["R01_full12"]["latency_ms_per_record"],
        },
        "actual_gpu_hours": train_duration_min / 60.0,
        "paired_difference_vs_baseline": {
            "canonical_auroc_delta": clinical_results["R01_full12"]["delta_auroc"],
            "canonical_auprc_delta": clinical_results["R01_full12"]["delta_auprc"],
            "overall_dev_bank_auroc_delta": overall_dev_a_auc - overall_dev_b_auc,
            "overall_dev_bank_auprc_delta": overall_dev_a_prc - overall_dev_b_prc,
        },
        "training_history": train_history,
        "timestamp": time.time(),
    }

    result_json_path = out_dir / "result.json"
    with open(result_json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info("Saved P9 result summary to %s", result_json_path)

    # Also save comprehensive evaluation report
    report_path = out_dir / "ablation_evaluation_report.json"
    full_report = {
        "status": "COMPLETED",
        "experiment_id": args.experiment_id,
        "canonical_clinical_configurations": clinical_results,
        "redundant_lead_analysis": redundant_lead_analysis,
        "development_grassmann_bank": grassmann_results,
        "rotated_basis_stability": basis_rotation_results,
        "timestamp": time.time(),
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info("Saved comprehensive evaluation report to %s", report_path)


if __name__ == "__main__":
    main()
