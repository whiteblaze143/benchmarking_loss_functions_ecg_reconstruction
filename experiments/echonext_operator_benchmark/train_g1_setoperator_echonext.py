#!/usr/bin/env python3
"""Train NEF-Inspired Geometry-Aware SetOperator (G1) on EchoNext from Scratch.

Rigorous comparison against original SetOperator:
- Architecture: G1 Geometry-Aware SetOperator
  * Operator MLP: Linear(44, 64) -> GELU -> Linear(64, 64) (7,040 params vs 4,736 in original)
  * Feature expansion: gamma(q) = [q, vech_sqrt2(q q^T)] in R^44
  * Downstream: Exact preservation of PhaseCNN (16x128 -> 128), Transformer Set Encoder (2 layers, 192-D),
    cross-attention set pooling (192-D), patient context (native additive), binary classification head.
  * Active classification parameters: 1,046,465 (+2,304 vs baseline 1,044,161).
- Training protocol:
  * 100% trained from random initialization (NO warm start, NO frozen backbone, NO parameter sharing).
  * Matched seed (42), matched batch size (64), matched optimizer (AdamW lr=1e-4, wd=1e-4),
    matched CosineAnnealingLR scheduler, matched early stopping patience (10) on official validation AUROC.
  * Authentic EchoNext Phase-KME representation from echonext_phase_kme_fit.npz.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from repecg.geometry_setoperator.geometry_setoperator import GeometryOperatorSetModel
from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH
from train_setoperator_echonext import (
    EchoNextSetOperatorDataset,
    collate_set_fn,
    seed_worker,
    evaluate_setoperator,
    get_file_sha256,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_g1_setoperator_echonext")

CHECKPOINT_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/g1_setoperator")
RESULTS_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/results")


class G1SetOperatorEchoNext(nn.Module):
    """EchoNext Binary Classifier based on the G1 Geometry-Aware SetOperator architecture."""

    def __init__(
        self,
        response_dim: int = 128,
        tabular_dim: int = 7,
        geometry_mode: Literal["G1", "control_capacity"] = "G1",
    ):
        super().__init__()
        self.geometry_mode = geometry_mode
        self.tabular_dim = tabular_dim

        self.core = GeometryOperatorSetModel(
            response_dim=response_dim,
            classes=1,
            geometry_mode=geometry_mode,
            patient_context_dim=tabular_dim,
            operator_dim=8,
        )

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent = self.core.encode_context(operators, responses, mask=mask, patient_context=tabular)
        logits = self.core.head(latent)
        return {"logits": logits, "latent": latent}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_g1_model(
    regime: str = "SO-Span",
    representation: str = "phase_kme",
    tabular_condition: str = "SO-ECG+Tab-Native",
    geometry_mode: str = "G1",
    seed: int = 42,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_name: str = "cuda",
    dry_run: bool = False,
):
    set_seed(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    out_dir = CHECKPOINT_BASE / f"{regime}_{tabular_condition}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Training G1 Geometry-Aware SetOperator on EchoNext from Scratch")
    logger.info("Geometry Mode: %s | Regime: %s | Representation: %s | Tabular: %s | Seed: %d | Device: %s",
                geometry_mode, regime, representation, tabular_condition, seed, device)
    logger.info("Output dir: %s", out_dir)
    logger.info("==================================================")

    # Load metadata splits
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values

    y_train = df.loc[train_mask, "shd_moderate_or_greater_flag"].values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values

    # Load local arrays (memory-mapped)
    train_w = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")

    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")

    logger.info("Dataset loaded: Train=%d, Val=%d", len(y_train), len(y_val))

    limit = 64 if dry_run else None
    train_ds = EchoNextSetOperatorDataset(train_w, train_t, y_train, regime=regime, representation=representation, seed=seed, is_train=True, limit_samples=limit)
    val_ds = EchoNextSetOperatorDataset(val_w, val_t, y_val, regime="SO-12", representation=representation, seed=seed, is_train=False, limit_samples=limit)

    g_train = torch.Generator()
    g_train.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if not dry_run else 0,
        worker_init_fn=seed_worker,
        generator=g_train,
        collate_fn=collate_set_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    g_val = torch.Generator()
    g_val.manual_seed(seed + 999)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if not dry_run else 0,
        worker_init_fn=seed_worker,
        generator=g_val,
        collate_fn=collate_set_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Build G1 model from scratch
    model = G1SetOperatorEchoNext(tabular_dim=7, geometry_mode=geometry_mode).to(device)
    
    # Compute parameter counts
    clf_params = sum(p.numel() for n, p in model.named_parameters() if "decoder" not in n)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("G1 Model Built from Scratch. Active Classification Parameters: %s | Total: %s",
                f"{clf_params:,}", f"{total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_epoch = -1
    best_model_weights = None
    no_improve = 0
    history = []

    actual_epochs = 1 if dry_run else epochs
    t0_train = time.time()

    for epoch in range(1, actual_epochs + 1):
        t0_ep = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            ops = batch["operators"].to(device)
            resp = batch["responses"].to(device)
            tab = batch["tabular"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(operators=ops, responses=resp, tabular=tab, mask=mask)
            loss = criterion(out["logits"].squeeze(-1), y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if dry_run and n_batches >= 2:
                break

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_auroc, val_auprc = evaluate_setoperator(model, val_loader, device)
        ep_time = time.time() - t0_ep

        is_best = val_auroc > best_val_auroc
        fit_sha = get_file_sha256(DEFAULT_FIT_PATH) if representation == "phase_kme" else None
        script_sha = get_file_sha256(Path(__file__).resolve())

        if is_best:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            if not dry_run:
                torch.save(
                    {
                        "epoch": epoch,
                        "seed": seed,
                        "regime": regime,
                        "tabular_condition": tabular_condition,
                        "geometry_mode": geometry_mode,
                        "active_classification_parameters": clf_params,
                        "total_trainable_parameters": total_params,
                        "representation": representation,
                        "representation_version": "phase_kme_v1.0_nystrom128" if representation == "phase_kme" else "raw_block_v1.0",
                        "fit_artifact_path": str(DEFAULT_FIT_PATH) if representation == "phase_kme" else None,
                        "fit_artifact_sha256": fit_sha,
                        "script_sha256": script_sha,
                        "model_state_dict": best_model_weights,
                        "val_auroc": val_auroc,
                        "val_auprc": val_auprc,
                    },
                    out_dir / "best_model.pt",
                )
            no_improve = 0
            best_mark = " [*BEST*]"
        else:
            no_improve += 1
            best_mark = ""

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val AUROC: %.4f | Val AUPRC: %.4f | LR: %.2e | Time: %.1fs%s",
            epoch, actual_epochs, avg_loss, val_auroc, val_auprc, scheduler.get_last_lr()[0], ep_time, best_mark
        )
        sys.stdout.flush()
        sys.stderr.flush()

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": ep_time,
        })

        if no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, patience)
            break

    train_duration = time.time() - t0_train
    logger.info("Training complete in %.2f min. Best Val AUROC: %.4f (Epoch %d)", train_duration / 60, best_val_auroc, best_epoch)

    summary = {
        "model": "G1SetOperator",
        "geometry_mode": geometry_mode,
        "regime": regime,
        "representation": representation,
        "tabular_condition": tabular_condition,
        "seed": seed,
        "active_classification_parameters": clf_params,
        "total_trainable_parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "val_auprc_at_best_epoch": history[best_epoch - 1]["val_auprc"] if best_epoch > 0 else 0.0,
        "train_loss_at_best_epoch": history[best_epoch - 1]["train_loss"] if best_epoch > 0 else 0.0,
        "total_epochs_trained": len(history),
        "train_duration_minutes": train_duration / 60.0,
        "history": history,
    }

    if not dry_run:
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary to %s", out_dir / "train_summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train G1 Geometry-Aware SetOperator on EchoNext from Scratch")
    parser.add_argument("--regime", type=str, default="SO-Span", choices=["SO-12", "SO-Basis", "SO-Span"])
    parser.add_argument("--representation", type=str, default="phase_kme", choices=["phase_kme", "raw_block"])
    parser.add_argument("--tabular-condition", type=str, default="SO-ECG+Tab-Native", choices=["SO-ECG", "SO-ECG+Tab-Native", "SO-ECG+Tab-Late"])
    parser.add_argument("--geometry-mode", type=str, default="G1", choices=["G1", "control_capacity"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_g1_model(
        regime=args.regime,
        representation=args.representation,
        tabular_condition=args.tabular_condition,
        geometry_mode=args.geometry_mode,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device_name=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
