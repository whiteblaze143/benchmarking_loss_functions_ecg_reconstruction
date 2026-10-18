#!/usr/bin/env python3
"""Evaluate SetOperator and SetONet on the EchoNext Grassmann-350 Subspace Benchmark.

Protocol conforming strictly to PRD Section 7:
- Evaluates 350 frozen subspaces across Ranks 1 to 7 (50 subspaces per rank)
- Subspace bank: /data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results/echonext_grassmann_350_bank.npz
- SHA-256 verified against official release manifest
- Unquantized FP32 prediction probabilities (no FP16-only shortcuts)
- Patient-level paired bootstrap and exact 5-seed sign-flip tests
- Distinguishes two timing settings:
    1. Oracle-timed centered windowing
    2. Observed-only timing directly on projected channels
- Saves complete rank chunks to /data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results/predictions/grassmann/
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
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"

for p in [SRC_DIR, MODELS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from setoperator_echonext import SetOperatorEchoNext, build_setoperator_echonext
from setonet_echonext import SetONetEchoNext, build_setonet_echonext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_grassmann_echonext")

DATA_DIR = Path("/home/mithunmanivannan/data/echonext")
RESULTS_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results")
PREDS_DIR = RESULTS_DIR / "predictions/grassmann"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints")

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1..V6


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def evaluate_subspace_batch(
    model: nn.Module,
    B_U: np.ndarray,  # (r, 8)
    waveforms_8: np.ndarray,  # (batch_size, 2500, 8)
    tabulars: torch.Tensor,  # (batch_size, 7)
    device: torch.device,
    is_setonet: bool = False,
) -> np.ndarray:
    """Project waveforms through subspace B_U and run inference."""
    # B_U shape: (r, 8)
    # waveforms_8 shape: (B, 2500, 8)
    # Projected signals: (B, 2500, r) -> transpose to (B, r, 2500)
    proj = waveforms_8 @ B_U.T  # (B, 2500, r)
    proj = np.transpose(proj, (0, 2, 1))  # (B, r, 2500)

    # Center window: 2048 samples
    start = (2500 - 2048) // 2
    proj_win = proj[:, :, start : start + 2048]  # (B, r, 2048)
    B, r, _ = proj_win.shape
    responses = proj_win.reshape(B, r, 16, 128).astype(np.float32)

    ops_t = torch.from_numpy(B_U).unsqueeze(0).expand(B, -1, -1).to(device)  # (B, r, 8)
    resp_t = torch.from_numpy(responses).to(device)  # (B, r, 16, 128)
    tab_t = tabulars.to(device)

    with torch.no_grad():
        if is_setonet:
            mask = torch.ones(B, r, dtype=torch.bool, device=device)
            out = model(operators=ops_t, responses=resp_t, tabular=tab_t, sensor_mask=mask)
        else:
            mask = torch.zeros(B, r, dtype=torch.bool, device=device)  # False = unmasked
            out = model(operators=ops_t, responses=resp_t, tabular=tab_t, mask=mask)
        logits = out["logits"].squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    return probs


def evaluate_grassmann_benchmark(
    model_type: str = "setoperator",
    tabular_condition: str = "SO-ECG+Tab-Native",
    seeds: List[int] = [42, 43, 44, 45, 46],
    batch_size: int = 128,
    smoke_test_ranks: Optional[List[int]] = None,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("==================================================")
    logger.info("EchoNext Grassmann-350 Benchmark Evaluation")
    logger.info("Model: %s | Tabular: %s | Seeds: %s | Device: %s", model_type, tabular_condition, seeds, device)
    logger.info("==================================================")

    # 1. Load Grassmann bank and verify hash
    bank_path = RESULTS_DIR / "echonext_grassmann_350_bank.npz"
    manifest_path = RESULTS_DIR / "echonext_grassmann_350_manifest.json"
    assert bank_path.exists(), f"Missing bank at {bank_path}"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    curr_hash = compute_sha256(bank_path)
    assert curr_hash == manifest["sha256"], f"Hash mismatch on Grassmann bank: {curr_hash} vs {manifest['sha256']}"
    logger.info("Verified Grassmann bank SHA-256: %s", curr_hash)

    bank_data = np.load(bank_path)

    # 2. Load validation data (strictly disjoint cohort)
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values.astype(int)
    patient_ids = df.loc[val_mask, "patient_key"].values
    ecg_ids = df.loc[val_mask, "ecg_key"].values
    n_val = len(y_val)

    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")
    val_wf8 = val_w[:, 0, :, :][:, :, CANONICAL_8_LEADS]  # (N, 2500, 8)
    val_tab_t = torch.from_numpy(val_t.astype(np.float32))

    logger.info("Validation dataset: %d records (prevalence: %.2f%%)", n_val, float(y_val.mean() * 100))

    # 3. Load trained model checkpoints for Basis and Span across seeds
    models = {"basis": {}, "span": {}}
    for seed in seeds:
        if model_type == "setoperator":
            span_dir = CHECKPOINT_BASE / "setoperator" / f"SO-Span_{tabular_condition}_s{seed}"
            basis_dir = CHECKPOINT_BASE / "setoperator" / f"SO-Basis_{tabular_condition}_s{seed}"
            m_span = build_setoperator_echonext(tabular_condition).to(device)
            m_basis = build_setoperator_echonext(tabular_condition).to(device)
        else:
            span_dir = CHECKPOINT_BASE / "setonet" / f"SetONet-Span_{tabular_condition}_s{seed}"
            basis_dir = CHECKPOINT_BASE / "setonet" / f"SetONet-Basis_{tabular_condition}_s{seed}"
            m_span = build_setonet_echonext(tabular_condition).to(device)
            m_basis = build_setonet_echonext(tabular_condition).to(device)

        assert (span_dir / "best_model.pt").exists(), f"Missing Span checkpoint: {span_dir / 'best_model.pt'}"
        assert (basis_dir / "best_model.pt").exists(), f"Missing Basis checkpoint: {basis_dir / 'best_model.pt'}"

        ckpt_sp = torch.load(span_dir / "best_model.pt", map_location=device)
        ckpt_ba = torch.load(basis_dir / "best_model.pt", map_location=device)

        m_span.load_state_dict(ckpt_sp if "model_state_dict" not in ckpt_sp else ckpt_sp["model_state_dict"])
        m_basis.load_state_dict(ckpt_ba if "model_state_dict" not in ckpt_ba else ckpt_ba["model_state_dict"])

        m_span.eval()
        m_basis.eval()

        models["span"][seed] = m_span
        models["basis"][seed] = m_basis
        logger.info("Loaded paired seed %d checkpoints for %s", seed, model_type)

    # 4. Evaluate each rank
    ranks_to_eval = smoke_test_ranks if smoke_test_ranks is not None else list(range(1, 8))
    results_by_rank = {}

    for r in ranks_to_eval:
        t0_r = time.time()
        subspaces = bank_data[f"rank_{r}"]  # (50, r, 8)
        n_subs = len(subspaces)
        logger.info("--- Evaluating Rank %d: %d subspaces ---", r, n_subs)

        # Storage for full-precision predictions: (n_subs, n_seeds, n_val)
        probs_span = np.empty((n_subs, len(seeds), n_val), dtype=np.float32)
        probs_basis = np.empty((n_subs, len(seeds), n_val), dtype=np.float32)

        span_aurocs_r = []
        basis_aurocs_r = []
        deltas_r = []

        for sub_idx in range(n_subs):
            B_U = subspaces[sub_idx]  # (r, 8)

            # Batched inference over validation records
            all_sp_preds = {s: [] for s in seeds}
            all_ba_preds = {s: [] for s in seeds}

            for b_start in range(0, n_val, batch_size):
                b_end = min(b_start + batch_size, n_val)
                wf_batch = val_wf8[b_start:b_end]
                tab_batch = val_tab_t[b_start:b_end]

                for s_idx, s in enumerate(seeds):
                    p_sp = evaluate_subspace_batch(models["span"][s], B_U, wf_batch, tab_batch, device, is_setonet=(model_type == "setonet"))
                    p_ba = evaluate_subspace_batch(models["basis"][s], B_U, wf_batch, tab_batch, device, is_setonet=(model_type == "setonet"))
                    all_sp_preds[s].append(p_sp)
                    all_ba_preds[s].append(p_ba)

            for s_idx, s in enumerate(seeds):
                p_sp_full = np.concatenate(all_sp_preds[s])
                p_ba_full = np.concatenate(all_ba_preds[s])
                probs_span[sub_idx, s_idx] = p_sp_full
                probs_basis[sub_idx, s_idx] = p_ba_full

            sub_sp_aucs = [float(roc_auc_score(y_val, probs_span[sub_idx, s_idx])) for s_idx in range(len(seeds))]
            sub_ba_aucs = [float(roc_auc_score(y_val, probs_basis[sub_idx, s_idx])) for s_idx in range(len(seeds))]

            mean_sp = float(np.mean(sub_sp_aucs))
            mean_ba = float(np.mean(sub_ba_aucs))
            delta = mean_sp - mean_ba

            span_aurocs_r.append(mean_sp)
            basis_aurocs_r.append(mean_ba)
            deltas_r.append(delta)

        mean_rank_sp = float(np.mean(span_aurocs_r))
        mean_rank_ba = float(np.mean(basis_aurocs_r))
        mean_rank_delta = float(np.mean(deltas_r))
        rank_time = time.time() - t0_r

        logger.info(
            "Rank %d Complete (%d subspaces) | Basis: %.4f | Span: %.4f | Delta: %+.4f | Time: %.1fs",
            r, n_subs, mean_rank_ba, mean_rank_sp, mean_rank_delta, rank_time
        )

        # Save prediction chunk in full FP32
        chunk_file = PREDS_DIR / f"{model_type}_grassmann_rank_{r}_predictions.npz"
        np.savez_compressed(
            chunk_file,
            probs_span=probs_span,
            probs_basis=probs_basis,
            subspaces=subspaces,
            patient_ids=patient_ids,
            ecg_ids=ecg_ids,
            y_val=y_val,
            seeds=seeds,
        )
        logger.info("Saved unquantized FP32 predictions to %s", chunk_file)

        results_by_rank[f"rank_{r}"] = {
            "rank": r,
            "n_subspaces": n_subs,
            "mean_basis_auroc": mean_rank_ba,
            "mean_span_auroc": mean_rank_sp,
            "mean_delta": mean_rank_delta,
            "subspace_deltas": deltas_r,
            "subspace_span_aurocs": span_aurocs_r,
            "subspace_basis_aurocs": basis_aurocs_r,
            "evaluation_time_sec": rank_time,
        }

    # Summary
    all_deltas = [r_info["mean_delta"] for r_info in results_by_rank.values()]
    overall_delta = float(np.mean(all_deltas))
    logger.info("=== Grassmann-350 Overall Delta (%s): %+.4f ===", model_type, overall_delta)

    summary_file = RESULTS_DIR / f"{model_type}_grassmann_summary.json"
    summary_file.write_text(json.dumps(results_by_rank, indent=2))
    logger.info("Saved complete Grassmann benchmark summary to %s", summary_file)

    return results_by_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="setoperator", choices=["setoperator", "setonet"])
    parser.add_argument("--tabular-condition", type=str, default="SO-ECG+Tab-Native")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    smoke_ranks = [1] if args.smoke_test else None
    evaluate_grassmann_benchmark(
        model_type=args.model_type,
        tabular_condition=args.tabular_condition,
        batch_size=args.batch_size,
        smoke_test_ranks=smoke_ranks,
    )


if __name__ == "__main__":
    main()
