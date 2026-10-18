#!/usr/bin/env python3
"""Train SetOperator on the official EchoNext benchmark for Binary Structural Heart Disease (SHD).

Conforms strictly to the established SetOperator architecture:
- OperatorSetModel with PhaseCNN response encoder (16x128 -> 128-D)
- Continuous operator encoder (8-D cardiac coordinate basis -> 64-D)
- Transformer Set Encoder (d_model=192, 2 layers) + Cross-Attention Set Pooling (192-D) -> Latent (256-D)
- Binary classification head (Linear(256, 1) -> BCEWithLogitsLoss)
- Supports multimodal (SO-ECG+Tab-Native) and ECG-only (SO-ECG) tracks.
- Regimes:
    * SO-12: Full canonical 8-lead basis
    * SO-Basis: Variable canonical measurement subsets (k in [1, 8])
    * SO-Span: Continuous operator augmentation (50% canonical, 25% dense, 25% sparse)
- Official test split remains strictly sealed and protected during training.
"""

from __future__ import annotations

import argparse
import copy
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

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from setoperator_echonext import SetOperatorEchoNext, build_setoperator_echonext
from repecg.paper07_operator.measurements import (
    canonical_operators,
    sample_sparse_operators,
    normalize_operator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setoperator_echonext")

DATA_DIR = Path("/home/mithunmanivannan/data/echonext")
CHECKPOINT_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator")
RESULTS_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results")

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1..V6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EchoNextSetOperatorDataset(Dataset):
    """Dataset for SetOperator on EchoNext waveforms and tabular features."""

    def __init__(
        self,
        waveforms: np.ndarray,
        tabular: np.ndarray,
        labels: np.ndarray,
        regime: str = "SO-Span",
        seed: int = 42,
        is_train: bool = True,
        limit_samples: Optional[int] = None,
    ):
        self.waveforms = waveforms  # (N, 1, 2500, 12)
        self.tabular = tabular.astype(np.float32)  # (N, 7)
        self.labels = labels.astype(np.float32)  # (N,)
        self.regime = regime
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        self.basis_operators = np.eye(8, dtype=np.float32)
        self.length = len(labels) if limit_samples is None else min(limit_samples, len(labels))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wf_12 = self.waveforms[idx, 0]  # (2500, 12)
        tab = self.tabular[idx]
        y = self.labels[idx]

        # Extract the 8 canonical leads: I, II, V1-V6
        wf_8 = wf_12[:, CANONICAL_8_LEADS]  # (2500, 8)

        # Sample operator configuration based on regime
        if self.is_train:
            if self.regime == "SO-12":
                ops = self.basis_operators
            elif self.regime == "SO-Basis":
                k = self.rng.integers(1, 9)
                perm = self.rng.permutation(8)[:k]
                ops = self.basis_operators[perm]
            elif self.regime == "SO-Span":
                prob = self.rng.uniform(0, 1)
                k = self.rng.integers(1, 9)
                if prob < 0.50:
                    perm = self.rng.permutation(8)[:k]
                    ops = self.basis_operators[perm]
                elif prob < 0.75:
                    ops = np.stack([normalize_operator(self.rng.normal(size=8).astype(np.float32)) for _ in range(k)], axis=0)
                else:
                    ops = sample_sparse_operators(k, self.rng).astype(np.float32)
            else:
                raise ValueError(f"Unknown regime: {self.regime}")
        else:
            # Default evaluation: full canonical basis
            ops = self.basis_operators

        # Compute projected lead signals: (set_size, 2500)
        lead_signals = wf_8 @ ops.T  # (2500, set_size)
        lead_signals = lead_signals.T  # (set_size, 2500)

        # Segment into 16 slices of length 128 (16 * 128 = 2048 samples, center window)
        start_samp = (2500 - 2048) // 2
        signals_windowed = lead_signals[:, start_samp : start_samp + 2048]
        responses = signals_windowed.reshape(len(ops), 16, 128).astype(np.float32)

        return {
            "operators": torch.from_numpy(ops.astype(np.float32)),
            "responses": torch.from_numpy(responses),
            "tabular": torch.from_numpy(tab),
            "label": torch.tensor(y, dtype=torch.float32),
        }


def collate_set_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate variable-sized sets with padding mask."""
    max_set_size = max(item["operators"].shape[0] for item in batch)
    b_size = len(batch)

    padded_ops = torch.zeros(b_size, max_set_size, 8, dtype=torch.float32)
    padded_resp = torch.zeros(b_size, max_set_size, 16, 128, dtype=torch.float32)
    mask = torch.ones(b_size, max_set_size, dtype=torch.bool)  # True = padded

    tabulars = torch.stack([item["tabular"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    for i, item in enumerate(batch):
        sz = item["operators"].shape[0]
        padded_ops[i, :sz] = item["operators"]
        padded_resp[i, :sz] = item["responses"]
        mask[i, :sz] = False

    return {
        "operators": padded_ops,
        "responses": padded_resp,
        "tabular": tabulars,
        "mask": mask,
        "labels": labels,
    }


def evaluate_setoperator(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            ops = batch["operators"].to(device)
            resp = batch["responses"].to(device)
            tab = batch["tabular"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].numpy()

            out = model(operators=ops, responses=resp, tabular=tab, mask=mask)
            prob = torch.sigmoid(out["logits"]).squeeze(-1).cpu().numpy()

            all_preds.extend(prob)
            all_labels.extend(y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auroc = float(roc_auc_score(all_labels, all_preds))
    auprc = float(average_precision_score(all_labels, all_preds))
    return auroc, auprc


def train_setoperator_seed(
    regime: str = "SO-Span",
    tabular_condition: str = "SO-ECG+Tab-Native",
    seed: int = 42,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 10,
    dry_run: bool = False,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    out_dir = CHECKPOINT_BASE / f"{regime}_{tabular_condition}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Training SetOperator on EchoNext (SHD Binary Detection)")
    logger.info("Regime: %s | Tabular: %s | Seed: %d | Device: %s", regime, tabular_condition, seed, device)
    logger.info("Output dir: %s", out_dir)
    logger.info("==================================================")

    # Load metadata
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
    train_ds = EchoNextSetOperatorDataset(train_w, train_t, y_train, regime=regime, seed=seed, is_train=True, limit_samples=limit)
    val_ds = EchoNextSetOperatorDataset(val_w, val_t, y_val, regime="SO-12", seed=seed, is_train=False, limit_samples=limit)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if not dry_run else 0,
        collate_fn=collate_set_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if not dry_run else 0,
        collate_fn=collate_set_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = build_setoperator_echonext(tabular_condition).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SetOperator model built. Trainable parameters: %s", f"{total_params:,}")

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
        "model": "SetOperator",
        "regime": regime,
        "tabular_condition": tabular_condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auroc),
        "total_train_time_sec": float(train_duration),
        "total_parameters": total_params,
        "dry_run": dry_run,
        "history": history,
    }

    if not dry_run:
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved training summary to %s", out_dir / "train_summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="SO-Span", choices=["SO-12", "SO-Basis", "SO-Span"])
    parser.add_argument("--tabular-condition", type=str, default="SO-ECG+Tab-Native", choices=["SO-ECG", "SO-ECG+Tab-Native", "SO-ECG+Tab-Concat"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Quick GPU smoke test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_setoperator_seed(
        regime=args.regime,
        tabular_condition=args.tabular_condition,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dry_run=args.dry_run,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
