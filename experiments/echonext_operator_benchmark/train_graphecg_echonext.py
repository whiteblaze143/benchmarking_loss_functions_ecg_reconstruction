#!/usr/bin/env python3
"""Train GraphECG on the official EchoNext benchmark for Binary Structural Heart Disease (SHD).

Conforms strictly to the author's architecture:
- 13 nodes (3D spatial electrode positions + WCT + augmented limb centers)
- 24 directed edges for standard 12-lead ECG
- Spherical harmonic degree-4 node encoding + Conv1D edge encoding
- 7 tabular clinical features fusion
- AdamW (lr=1e-4, wd=0.01) + CosineAnnealingLR + BCEWithLogitsLoss
- Early stopping on Validation AUROC (strictly disjoint patient validation cohort)
- Official test split remains strictly sealed and protected during training.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
GRAPH_SRC = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src/graphECG_author_code"
if str(GRAPH_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPH_SRC))

from graph import ECGGraphBuilder
from model import GraphECG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graphecg_echonext")

DATA_DIR = Path("/home/mithunmanivannan/data/echonext")
CHECKPOINT_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/graphecg")
RESULTS_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EchoNextGraphDataset(Dataset):
    """Memory-mapped EchoNext PyG Graph Dataset."""

    def __init__(
        self,
        waveforms_mmap: np.ndarray,
        tabular_mmap: np.ndarray,
        labels: np.ndarray,
        builder: ECGGraphBuilder,
        limit_samples: Optional[int] = None,
    ):
        self.waveforms = waveforms_mmap
        self.tabular = tabular_mmap
        self.labels = labels
        self.builder = builder
        self.length = len(labels) if limit_samples is None else min(limit_samples, len(labels))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        # Waveform shape: (1, 2500, 12) -> transpose to (12, 2500)
        raw_wf = np.array(self.waveforms[idx, 0], dtype=np.float32).T  # (12, 2500)
        tab = np.array(self.tabular[idx], dtype=np.float32)
        label = float(self.labels[idx])

        # Standard 12-lead graph: 13 nodes, 24 directed edges
        graph = self.builder.build_from_array(raw_wf, lead_indices=None, bidirectional=True)
        return graph, torch.from_numpy(tab), torch.tensor(label, dtype=torch.float32)


def collate_graphs(batch):
    graphs = [item[0] for item in batch]
    tabs = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    batched_graphs = Batch.from_data_list(graphs)
    return batched_graphs, tabs, labels


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_graphs, tabs, labels in loader:
            batch_graphs = batch_graphs.to(device)
            tabs = tabs.to(device)
            out = model(batch_graphs, tabular=tabs)
            logits = out["logits"].squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    auroc = float(roc_auc_score(y_true, y_pred))
    auprc = float(average_precision_score(y_true, y_pred))
    return auroc, auprc


def train_graphecg_seed(
    seed: int,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    dry_run: bool = False,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    run_dir = CHECKPOINT_BASE / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("GraphECG EchoNext SHD Training | Seed %d | Device: %s", seed, device)
    logger.info("Destination directory: %s", run_dir)
    logger.info("==================================================")

    # Load metadata
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values

    train_labels = df[train_mask]["shd_moderate_or_greater_flag"].values.astype(int)
    val_labels = df[val_mask]["shd_moderate_or_greater_flag"].values.astype(int)

    logger.info("Cohort: Train=%d (prevalence: %.2f%%), Val=%d (prevalence: %.2f%%)",
                len(train_labels), float(train_labels.mean() * 100),
                len(val_labels), float(val_labels.mean() * 100))

    # Memory map arrays
    train_w = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy", mmap_mode="r")
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy", mmap_mode="r")

    builder = ECGGraphBuilder()

    limit = 64 if dry_run else None
    train_dataset = EchoNextGraphDataset(train_w, train_t, train_labels, builder, limit_samples=limit)
    val_dataset = EchoNextGraphDataset(val_w, val_t, val_labels, builder, limit_samples=limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=4 if not dry_run else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=2 if not dry_run else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = GraphECG(
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_layers=3,
        tabular_dim=7,
        num_classes=1,
        dropout=0.5,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("GraphECG initialized with %s trainable parameters.", f"{total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auroc = -1.0
    best_epoch = -1
    best_weights = None
    epochs_no_improve = 0
    history = []

    actual_epochs = 1 if dry_run else epochs
    t0_train = time.time()

    for epoch in range(1, actual_epochs + 1):
        t0_ep = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_graphs, tabs, labels in train_loader:
            batch_graphs = batch_graphs.to(device)
            tabs = tabs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(batch_graphs, tabular=tabs)
            logits = out["logits"].squeeze(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if dry_run and n_batches >= 2:
                break

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        val_auroc, val_auprc = evaluate_validation(model, val_loader, device)
        ep_time = time.time() - t0_ep

        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            if not dry_run:
                torch.save(
                    {
                        "epoch": epoch,
                        "seed": seed,
                        "model_state_dict": best_weights,
                        "val_auroc": val_auroc,
                        "val_auprc": val_auprc,
                        "total_params": total_params,
                    },
                    run_dir / "graphecg_best.pt",
                )
            epochs_no_improve = 0
            best_mark = " [*BEST*]"
        else:
            epochs_no_improve += 1
            best_mark = ""

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val AUROC: %.4f | Val AUPRC: %.4f | LR: %.2e | Time: %.1fs%s",
            epoch, actual_epochs, train_loss, val_auroc, val_auprc, scheduler.get_last_lr()[0], ep_time, best_mark
        )
        sys.stdout.flush()
        sys.stderr.flush()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": ep_time,
        })

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d (patience %d reached).", epoch, patience)
            break

    total_train_time = time.time() - t0_train
    logger.info("Training complete in %.2f min. Best Val AUROC: %.4f at Epoch %d", total_train_time / 60, best_val_auroc, best_epoch)

    summary = {
        "model": "GraphECG",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auroc),
        "total_train_time_min": float(total_train_time / 60),
        "total_params": total_params,
        "dry_run": dry_run,
        "history": history,
    }

    if not dry_run:
        (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Saved summary to %s", run_dir / "train_summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true", help="Quick GPU smoke test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_graphecg_seed(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        dry_run=args.dry_run,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
