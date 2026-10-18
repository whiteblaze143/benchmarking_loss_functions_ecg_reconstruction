#!/usr/bin/env python3
"""Authoritative Five-Seed GraphECG Training Script for PTB-XL (Folds 1-8).

Corrects the historical split defect:
  - Folds 1-8: Training (17,418 raw / 17,417 eligible)
  - Fold 9: Model Selection / Validation (2,183 records)
  - Fold 10: Locked Confirmation (STRICTLY SEALED during development)

Architecture:
  - Electrode-centric GNN (Ansari et al., Stanford 2026)
  - 13 electrode/reference nodes, directed waveform edges, CNN edge encoding
  - Spherical harmonic positional encodings, 3 message-passing layers
  - tabular_dim = 0 (ECG-only comparison)
  - 5 superclass diagnostic logits: NORM, MI, STTC, CD, HYP

Seeds: 42, 43, 44, 45, 46
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wfdb

torch.backends.cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from graphECG_author_code.graph import ECGGraphBuilder
from graphECG_author_code.model import GraphECG
from torch_geometric.data import Batch, Data
from repecg.common.metrics import multilabel_metrics
from repecg.common.labels import load_superdiagnostic_map, encode_superdiagnostic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_graphecg_ptbxl_three_way")

SUPER_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


class PTBXLGraphDataset(Dataset):
    """PTB-XL Dataset yielding PyG graph representations for GraphECG."""

    def __init__(
        self,
        records: pd.DataFrame,
        ptb_root: Path,
        label_map: Dict[str, str],
        builder: ECGGraphBuilder,
        sampling_rate: int = 100,
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
        signal, _ = wfdb.rdsamp(str(fn))

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

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
    labels = torch.from_numpy(np.stack(labels, axis=0)).float()
    ecg_ids = torch.tensor(ecg_ids, dtype=torch.long)
    return batch_graph, labels, ecg_ids


def evaluate_dataset(model: GraphECG, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_graph, labels, _ in loader:
            batch_graph = batch_graph.to(device)
            out = model(batch_graph)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = multilabel_metrics(y_true, y_prob)
    return metrics, y_prob, y_true


def train_single_seed(
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ptb_root: Path,
    label_map: Dict[str, str],
    output_dir: Path,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    wd: float = 1e-4,
    sampling_rate: int = 100,
    device_str: str = "cuda",
) -> Path:
    logger.info("=== Starting GraphECG Training: Seed %d ===", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    builder = ECGGraphBuilder()

    train_ds = PTBXLGraphDataset(train_df, ptb_root, label_map, builder, sampling_rate=sampling_rate)
    val_ds = PTBXLGraphDataset(val_df, ptb_root, label_map, builder, sampling_rate=sampling_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pyg_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate, num_workers=2)

    model = GraphECG(
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_layers=3,
        tabular_dim=0,
        num_classes=len(SUPER_CLASSES),
        dropout=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auroc = -1.0
    best_epoch = 0
    stale = 0
    best_state = None
    history = []

    best_checkpoint_path = output_dir / f"graphecg_ge12_s{seed}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_graph, labels, _ in train_loader:
            batch_graph = batch_graph.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(batch_graph)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))

        val_metrics, _, _ = evaluate_dataset(model, val_loader, device)
        val_auroc = float(val_metrics.get("macro_auroc", 0.0))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_auroc": val_auroc,
            "val_macro_auprc": float(val_metrics.get("macro_auprc", 0.0)),
            "lr": float(scheduler.get_last_lr()[0]),
        })

        if val_auroc > best_val_auroc + 1e-5:
            best_val_auroc = val_auroc
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            logger.info("[GE12_s%d] Epoch %2d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f (*) BEST",
                        seed, epoch, epochs, train_loss, val_auroc)
        else:
            stale += 1
            if epoch % 5 == 0 or stale >= patience:
                logger.info("[GE12_s%d] Epoch %2d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f (stale %d/%d)",
                            seed, epoch, epochs, train_loss, val_auroc, stale, patience)

        if stale >= patience:
            logger.info("[GE12_s%d] Early stopping reached at epoch %d.", seed, epoch)
            break

    assert best_state is not None, "Training failed to produce valid state"
    checkpoint_payload = {
        "state_dict": best_state,
        "seed": seed,
        "intervention": "ge12",
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "history": history,
        "hyperparameters": {
            "lr": lr,
            "wd": wd,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "sampling_rate": sampling_rate,
        },
        "model": "GraphECG (electrode-centric GNN, tabular_dim=0)",
    }
    torch.save(checkpoint_payload, best_checkpoint_path)

    # Strict reload verification
    chk = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    v_model = GraphECG(
        node_dim=128, edge_dim=192, hidden_dim=192, num_layers=3,
        tabular_dim=0, num_classes=len(SUPER_CLASSES), dropout=0.3,
    )
    v_model.load_state_dict(chk["state_dict"], strict=True)
    logger.info("Saved and strictly verified checkpoint %s (Best Val AUROC: %.4f at epoch %d)",
                best_checkpoint_path, best_val_auroc, best_epoch)

    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train 5-seed GraphECG models on PTB-XL (Corrected Folds 1-8 / 9 split).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--sampling-rate", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ptb_root = REPO_DIR / "data/ptb_xl"
    df = pd.read_csv(ptb_root / "ptbxl_database.csv", index_col="ecg_id")
    label_map = load_superdiagnostic_map(ptb_root / "scp_statements.csv")

    # CORRECTED SPLIT: Train Folds 1-8, Val Fold 9, Fold 10 sealed
    train_df = df[df["strat_fold"].isin(range(1, 9))]
    val_df = df[df["strat_fold"] == 9]

    logger.info("Cohort splits verified: Train (Folds 1-8) = %d | Val (Fold 9) = %d | Fold 10 SEALED",
                len(train_df), len(val_df))

    checkpoints = {}
    for seed in args.seeds:
        chk_path = train_single_seed(
            seed=seed,
            train_df=train_df,
            val_df=val_df,
            ptb_root=ptb_root,
            label_map=label_map,
            output_dir=args.output_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            sampling_rate=args.sampling_rate,
            device_str=args.device,
        )
        import hashlib
        sha = hashlib.sha256(chk_path.read_bytes()).hexdigest()
        checkpoints[f"ge12_s{seed}"] = {"path": str(chk_path), "sha256": sha}

    ledger_path = args.output_dir / "checkpoints_selection_ledger.json"
    ledger_path.write_text(json.dumps(checkpoints, indent=2))
    logger.info("GraphECG training complete. Ledger saved to %s", ledger_path)


if __name__ == "__main__":
    main()
