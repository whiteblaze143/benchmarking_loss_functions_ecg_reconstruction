#!/usr/bin/env python3
"""Train any SetOperator Architectural Ablation on EchoNext.

Configurable runner for Paper 07 168-Hour Ablation Campaign.
Supports all 22 experimental variants in ablation_week_manifest.json.
Enforces 100% random initialization from scratch.
Saves checkpoints and summary to isolated directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"

for p in [SRC_DIR, BENCHMARK_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from repecg.ablation_setoperator.ablation_models import (
    AblationSetOperatorEchoNext,
    TabularMLPEchoNext,
)
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
logger = logging.getLogger("train_ablation")

ABLATION_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/ablations")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ablation_experiment(
    exp_id: str,
    geometry_mode: str = "baseline",
    operator_mode: str = "continuous",
    phase_mode: str = "standard_circular",
    set_encoder_mode: str = "standard_transformer",
    pooling_mode: str = "learned_query",
    tabular_mode: str = "native_additive",
    representation: str = "phase_kme",
    regime: str = "SO-Span",
    seed: int = 42,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_name: str = "cuda",
    dry_run: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    out_dir = ABLATION_BASE / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Training SetOperator Ablation: %s", exp_id)
    logger.info("Geometry: %s | Phase: %s | Set: %s | Pooling: %s | Tabular: %s",
                geometry_mode, phase_mode, set_encoder_mode, pooling_mode, tabular_mode)
    logger.info("Regime: %s | Rep: %s | Seed: %d | Device: %s", regime, representation, seed, device)
    logger.info("Output dir: %s", out_dir)
    logger.info("==================================================")

    # Load metadata splits
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values

    y_train = df.loc[train_mask, "shd_moderate_or_greater_flag"].values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values

    train_w = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")

    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")

    limit = 64 if dry_run else None
    is_tab_only = (tabular_mode == "tabular_only")
    train_ds = EchoNextSetOperatorDataset(
        train_w, train_t, y_train, regime=regime, representation=representation, seed=seed, is_train=True, limit_samples=limit, tabular_only=is_tab_only
    )
    val_ds = EchoNextSetOperatorDataset(
        val_w, val_t, y_val, regime="SO-12", representation=representation, seed=seed, is_train=False, limit_samples=limit, tabular_only=is_tab_only
    )

    g_train = torch.Generator().manual_seed(seed)
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
    g_val = torch.Generator().manual_seed(seed + 999)
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

    # Build Model
    if tabular_mode == "tabular_only":
        model = TabularMLPEchoNext(tabular_dim=7, classes=1).to(device)
    else:
        model = AblationSetOperatorEchoNext(
            operator_mode=operator_mode,
            geometry_mode=geometry_mode,
            phase_mode=phase_mode,
            set_encoder_mode=set_encoder_mode,
            pooling_mode=pooling_mode,
            tabular_mode=tabular_mode,
            classes=1,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Ablation Model Built. Active Trainable Parameters: %s", f"{total_params:,}")

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
            out = model(operators=ops, responses=resp, tabular=tab if tabular_mode != "none" else None, mask=mask)
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

        if is_best:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            if not dry_run:
                torch.save(
                    {
                        "experiment_id": exp_id,
                        "epoch": epoch,
                        "seed": seed,
                        "geometry_mode": geometry_mode,
                        "phase_mode": phase_mode,
                        "set_encoder_mode": set_encoder_mode,
                        "pooling_mode": pooling_mode,
                        "tabular_mode": tabular_mode,
                        "active_parameters": total_params,
                        "representation": representation,
                        "fit_artifact_path": str(DEFAULT_FIT_PATH) if representation == "phase_kme" else None,
                        "fit_artifact_sha256": fit_sha,
                        "model_state_dict": best_model_weights,
                        "val_auroc": val_auroc,
                        "val_auprc": val_auprc,
                    },
                    out_dir / "best_model.pt"
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

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_seconds": ep_time,
            "is_best": is_best,
        })

        if not dry_run and no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d).", epoch, patience)
            break

    train_duration = time.time() - t0_train
    summary = {
        "status": "COMPLETED",
        "experiment_id": exp_id,
        "geometry_mode": geometry_mode,
        "phase_mode": phase_mode,
        "set_encoder_mode": set_encoder_mode,
        "pooling_mode": pooling_mode,
        "tabular_mode": tabular_mode,
        "representation": representation,
        "seed": seed,
        "active_parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "val_auprc_at_best_epoch": history[best_epoch - 1]["val_auprc"] if best_epoch > 0 else 0.0,
        "train_loss_at_best_epoch": history[best_epoch - 1]["train_loss"] if best_epoch > 0 else 0.0,
        "total_epochs_trained": len(history),
        "train_duration_minutes": train_duration / 60.0,
        "history": history,
    }

    if not dry_run:
        torch.save(
            {
                "experiment_id": exp_id,
                "epoch": len(history),
                "seed": seed,
                "geometry_mode": geometry_mode,
                "phase_mode": phase_mode,
                "set_encoder_mode": set_encoder_mode,
                "pooling_mode": pooling_mode,
                "tabular_mode": tabular_mode,
                "active_parameters": total_params,
                "representation": representation,
                "fit_artifact_path": str(DEFAULT_FIT_PATH) if representation == "phase_kme" else None,
                "fit_artifact_sha256": fit_sha,
                "model_state_dict": model.state_dict(),
                "val_auroc": history[-1]["val_auroc"] if history else 0.0,
                "val_auprc": history[-1]["val_auprc"] if history else 0.0,
            },
            out_dir / "final_model.pt"
        )
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary and final_model.pt to %s", out_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train SetOperator Ablation Variant")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--geometry-mode", type=str, default="baseline")
    parser.add_argument("--operator-mode", type=str, default="continuous")
    parser.add_argument("--phase-mode", type=str, default="standard_circular")
    parser.add_argument("--set-encoder-mode", type=str, default="standard_transformer")
    parser.add_argument("--pooling-mode", type=str, default="learned_query")
    parser.add_argument("--tabular-mode", type=str, default="native_additive")
    parser.add_argument("--representation", type=str, default="phase_kme")
    parser.add_argument("--regime", type=str, default="SO-Span")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_ablation_experiment(
        exp_id=args.experiment_id,
        geometry_mode=args.geometry_mode,
        operator_mode=args.operator_mode,
        phase_mode=args.phase_mode,
        set_encoder_mode=args.set_encoder_mode,
        pooling_mode=args.pooling_mode,
        tabular_mode=args.tabular_mode,
        representation=args.representation,
        regime=args.regime,
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
