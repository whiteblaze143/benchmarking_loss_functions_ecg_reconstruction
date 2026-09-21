#!/usr/bin/env python3
"""
train_graphecg_ptbxl.py

Trains GraphECG (Ansari et al., Stanford 2026) on the PTB-XL benchmark
using the exact standardized split:
  - Folds 1-7: Training
  - Folds 9-10: Validation / Model Selection
  - Fold 8: Held-out Test Evaluation

Multilabel 5-class superclass diagnosis: NORM, MI, STTC, CD, HYP.
Outputs checkpoints and standardized metrics compatible with cross-paper evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb

# CRITICAL for NVIDIA A100 / PyTorch 2.6: disable cuDNN to prevent ptrDesc->finalize() bug
torch.backends.cudnn.enabled = False

# Add repo to sys.path
REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from graphECG_author_code.graph import ECGGraphBuilder
from graphECG_author_code.model import GraphECG
from torch_geometric.data import Batch, Data
from repecg.common.metrics import multilabel_metrics
from repecg.common.labels import load_superdiagnostic_map, encode_superdiagnostic

SUPER_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


class PTBXLGraphDataset(Dataset):
    """PTB-XL Dataset yielding PyG graph representations for GraphECG."""

    def __init__(
        self,
        records: pd.DataFrame,
        ptb_root: Path,
        label_map: Dict[str, str],
        builder: ECGGraphBuilder,
        sampling_rate: int = 100,  # 100 Hz = 1000 samples, 500 Hz = 5000 samples
        bidirectional: bool = True,
        lead_indices: List[int] | None = None,
    ):
        self.records = records.reset_index()
        self.ptb_root = ptb_root
        self.label_map = label_map
        self.builder = builder
        self.sampling_rate = sampling_rate
        self.bidirectional = bidirectional
        self.lead_indices = lead_indices
        self.fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Data, np.ndarray, int]:
        row = self.records.iloc[idx]
        ecg_id = int(row["ecg_id"])
        fn = self.ptb_root / str(row[self.fn_col])
        signal, _ = wfdb.rdsamp(str(fn))  # (T, 12)
        
        # Ensure finite and standardize per lead
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        # Build PyG Graph: shape needs to be (12, T)
        graph = self.builder.build_from_array(
            signal.T,
            lead_indices=self.lead_indices,
            bidirectional=self.bidirectional,
        )

        label = encode_superdiagnostic(row["scp_codes"], self.label_map, SUPER_CLASSES)
        return graph, label, ecg_id


def pyg_collate(batch):
    graphs, labels, ecg_ids = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    batch_labels = torch.tensor(np.stack(labels), dtype=torch.float32)
    batch_ids = torch.tensor(ecg_ids, dtype=torch.long)
    return batch_graph, batch_labels, batch_ids


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_graph, labels, _ in loader:
        batch_graph = batch_graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(batch_graph)
        loss = criterion(out["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    all_ids = []

    for batch_graph, labels, ecg_ids in loader:
        batch_graph = batch_graph.to(device)
        out = model(batch_graph)
        all_logits.append(out["logits"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_ids.append(ecg_ids.numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = 1.0 / (1.0 + np.exp(-logits))

    metrics = multilabel_metrics(y_true, y_prob)
    return metrics, y_prob, y_true


def main():
    parser = argparse.ArgumentParser(description="Train GraphECG on PTB-XL")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500], help="Sampling rate")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs/graphecg", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 batches to verify wiring")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GraphECG PTB-XL Training | Device: {device} | Seed: {args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load PTB-XL metadata and statements
    ptb_root = REPO_DIR / "data/ptb_xl"
    df = pd.read_csv(ptb_root / "ptbxl_database.csv", index_col="ecg_id")
    label_map = load_superdiagnostic_map(ptb_root / "scp_statements.csv")

    # Splits: 1-7 train, 9-10 val, 8 held-out test
    train_df = df[df["strat_fold"].isin(range(1, 8))]
    val_df = df[df["strat_fold"].isin([9, 10])]
    test_df = df[df["strat_fold"] == 8]

    if args.dry_run:
        print("Dry run mode: using 128 samples per split")
        train_df = train_df.iloc[:128]
        val_df = val_df.iloc[:64]
        test_df = test_df.iloc[:64]
        args.epochs = 2

    print(f"Cohort splits: Train={len(train_df)} | Val={len(val_df)} | Test(Fold 8)={len(test_df)}")

    builder = ECGGraphBuilder()
    train_ds = PTBXLGraphDataset(train_df, ptb_root, label_map, builder, sampling_rate=args.sampling_rate)
    val_ds = PTBXLGraphDataset(val_df, ptb_root, label_map, builder, sampling_rate=args.sampling_rate)
    test_ds = PTBXLGraphDataset(test_df, ptb_root, label_map, builder, sampling_rate=args.sampling_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pyg_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pyg_collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pyg_collate, num_workers=2)

    # Initialize GraphECG
    model = GraphECG(
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_layers=3,
        tabular_dim=0,
        num_classes=len(SUPER_CLASSES),
        dropout=0.3,
    ).to(device)

    print(f"GraphECG Architecture instantiated | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auroc = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_model_state = None
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_metrics, _, _ = evaluate_dataset(model, val_loader, device)
        val_auroc = float(val_metrics.get("macro_auroc", 0.0))
        val_auprc = float(val_metrics.get("macro_auprc", 0.0))
        ep_duration = time.time() - ep_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_macro_auroc": round(val_auroc, 5),
            "val_macro_auprc": round(val_auprc, 5),
            "lr": round(scheduler.get_last_lr()[0], 7),
            "duration_s": round(ep_duration, 1),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

        if val_auroc > best_val_auroc + 1e-5:
            best_val_auroc = val_auroc
            best_epoch = epoch
            stale_epochs = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save intermediate best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                args.output_dir / "graphecg_ptbxl_best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nTraining completed in {(time.time() - t0)/60:.1f} min. Best Val Macro AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    # Load best checkpoint and evaluate on Fold 8 held-out test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics, y_prob, y_true = evaluate_dataset(model, test_loader, device)
    print("\n" + "="*50)
    print("PTB-XL FOLD 8 HELD-OUT TEST RESULTS (GraphECG):")
    print(f"  Macro AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"  Micro AUROC: {test_metrics['micro_auroc']:.4f}")
    print(f"  Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"  Classwise AUROC: {[round(x, 4) for x in test_metrics['classwise_auroc']]}")
    print("="*50)

    # Save standardized Fold 8 JSON for cross-paper comparison
    cross_eval_dir = EXPERIMENT_DIR / "outputs/cross_paper_evaluation"
    cross_eval_dir.mkdir(parents=True, exist_ok=True)
    
    fold8_payload = {
        "paper_id": "graphecg",
        "model_name": "GraphECG (Ansari et al., 2026)",
        "best_variant": "electrode_centric_gnn",
        "best_macro_auroc": float(test_metrics["macro_auroc"]),
        "micro_auroc": float(test_metrics["micro_auroc"]),
        "macro_auprc": float(test_metrics["macro_auprc"]),
        "classwise_auroc": [float(x) for x in test_metrics["classwise_auroc"]],
        "n_samples": len(test_df),
        "best_epoch": best_epoch,
        "checkpoint": str(args.output_dir / "graphecg_ptbxl_best.pt"),
    }
    
    with open(cross_eval_dir / "graphecg_fold8.json", "w") as f:
        json.dump(fold8_payload, f, indent=2)
    
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved evaluation to {cross_eval_dir / 'graphecg_fold8.json'}")


if __name__ == "__main__":
    main()
