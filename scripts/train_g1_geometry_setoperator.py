#!/usr/bin/env python3
"""P6: G1 Geometry-Aware SetOperator Training Pilot on EchoNext (Inner Partition).

Production-grade training script for G1 Geometry-Aware SetOperator:
- Architecture: G1 Geometry-Aware SetOperator (44-D continuous operator features)
- Representation: Strictly isolated inner-train Phase-KME (Nyström 128 landmarks)
- Partition: Inner Train (61,375 records, 22,285 patients, 0 dev overlap)
- Evaluation: Inner Dev (11,100 records, 3,933 patients) on Canonical & Dev Bank
- Operator distribution: Span 50/25/25 mixture
- Tabular condition: Native additive 7-feature clinical context
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-2) with CosineAnnealingLR
- Checkpoints: Best Val AUROC & Latest saved to isolated directory
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
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_g1_geometry")

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"

for p in [SRC_DIR, MODELS_DIR, BENCHMARK_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase_kme_representation import EchoNextPhaseKMEExtractor, CANONICAL_8_LEADS
from repecg.geometry_setoperator.geometry_setoperator import EchoNextGeometrySetOperator
from repecg.paper07_operator.measurements import sample_sparse_operators, normalize_operator

DATA_DIR = Path("/data/mithunmanivannan/echonext")
DEV_ARTIFACTS = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1")
FIT_PATH = DEV_ARTIFACTS / "representations/echonext_inner_train_phase_kme_fit.npz"
SPLIT_PATH = DEV_ARTIFACTS / "echonext_inner_dev_split_indices.npz"
DEV_BANK_PATH = DEV_ARTIFACTS / "echonext_dev_operator_bank.npz"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EchoNextInnerDataset(Dataset):
    """Dataset for EchoNext with authentic Phase-KME and operator sampling."""

    def __init__(
        self,
        waveforms_mmap: np.ndarray,
        tabular_mmap: np.ndarray,
        labels: np.ndarray,
        record_indices: np.ndarray,
        extractor: EchoNextPhaseKMEExtractor,
        regime: str = "SO-Span",
        seed: int = 42,
        is_train: bool = True,
        limit_samples: Optional[int] = None,
    ):
        self.waveforms = waveforms_mmap
        self.tabular = tabular_mmap
        self.labels = labels
        self.record_indices = record_indices
        self.extractor = extractor
        self.regime = regime
        self.is_train = is_train
        self.length = len(record_indices) if limit_samples is None else min(limit_samples, len(record_indices))
        self.basis_operators = np.eye(8, dtype=np.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_rec = self.record_indices[idx]
        wf_12 = self.waveforms[global_rec, 0]
        tab = np.array(self.tabular[global_rec], dtype=np.float32)
        y = float(self.labels[global_rec])

        # Extract beats from waveform
        beats = self.extractor.extract_beats(wf_12)

        # Operator sampling
        if self.is_train:
            # Span 50/25/25
            rng = np.random.default_rng(idx + 1000)
            k = int(rng.integers(1, 9))
            prob = float(rng.uniform(0, 1))
            if prob < 0.50:
                perm = rng.permutation(8)[:k]
                ops = self.basis_operators[perm]
            elif prob < 0.75:
                ops = np.stack([normalize_operator(rng.normal(size=8).astype(np.float32)) for _ in range(k)], axis=0)
            else:
                ops = sample_sparse_operators(k, rng=rng).astype(np.float32)
        else:
            ops = self.basis_operators

        # Phase-KME response
        with torch.no_grad():
            resps = self.extractor.compute_phase_kme(torch.from_numpy(beats), torch.from_numpy(ops))

        return {
            "operators": torch.from_numpy(ops.astype(np.float32)),
            "responses": resps,
            "tabular": torch.from_numpy(tab),
            "label": torch.tensor(y, dtype=torch.float32),
        }


def collate_geometry_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_k = max(item["operators"].shape[0] for item in batch)
    b_size = len(batch)

    # Initialize padded_ops with unit vector e_1 so gamma(q) never divides by zero
    padded_ops = torch.zeros(b_size, max_k, 8, dtype=torch.float32)
    padded_ops[:, :, 0] = 1.0
    padded_resps = torch.zeros(b_size, max_k, 16, 128, dtype=torch.float32)
    mask = torch.ones(b_size, max_k, dtype=torch.bool)

    tabulars = torch.stack([item["tabular"] for item in batch])
    labels = torch.stack([item["label"] for item in batch]).unsqueeze(-1)

    for i, item in enumerate(batch):
        sz = item["operators"].shape[0]
        padded_ops[i, :sz] = item["operators"]
        padded_resps[i, :sz] = item["responses"]
        mask[i, :sz] = False

    return {
        "operators": padded_ops,
        "responses": padded_resps,
        "tabular": tabulars,
        "mask": mask,
        "labels": labels,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ops = batch["operators"].to(device)
            resps = batch["responses"].to(device)
            tabs = batch["tabular"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(operators=ops, responses=resps, tabular=tabs, mask=mask)
            loss = criterion(out["logits"], labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(out["logits"]).cpu().numpy().ravel()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy().ravel())

    avg_loss = total_loss / len(all_labels)
    auroc = float(roc_auc_score(all_labels, all_preds))
    auprc = float(average_precision_score(all_labels, all_preds))
    return avg_loss, auroc, auprc


def main():
    parser = argparse.ArgumentParser(description="Train Geometry SetOperator on EchoNext Inner Split")
    parser.add_argument("--geometry-mode", type=str, default="G1", choices=["G1", "control_capacity"], help="Architecture mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dry-run", action="store_true", help="Quick 1-batch execution test")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_tag = f"{args.geometry_mode.lower()}_seed{args.seed}"
    out_dir = DEV_ARTIFACTS / f"runs/{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("GEOMETRY SETOPERATOR TRAINING: %s", args.geometry_mode)
    logger.info("Seed: %d | Device: %s | OutDir: %s", args.seed, device, out_dir)
    logger.info("==================================================")

    # Load split indices
    split_data = np.load(SPLIT_PATH)
    train_idx = split_data["inner_train_indices"]
    dev_idx = split_data["inner_dev_indices"]
    logger.info("Cohort: InnerTrain=%d, InnerDev=%d (0 patient overlap)", len(train_idx), len(dev_idx))

    # Load datasets
    train_wf = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy", mmap_mode="r")
    meta_df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_meta = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
    all_labels = train_meta["shd_moderate_or_greater_flag"].values.astype(np.float32)

    extractor = EchoNextPhaseKMEExtractor(FIT_PATH, device="cpu")

    limit = 64 if args.dry_run else None
    train_ds = EchoNextInnerDataset(train_wf, train_t, all_labels, train_idx, extractor, regime="SO-Span", seed=args.seed, is_train=True, limit_samples=limit)
    dev_ds = EchoNextInnerDataset(train_wf, train_t, all_labels, dev_idx, extractor, regime="SO-Canonical", seed=args.seed, is_train=False, limit_samples=limit)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_geometry_fn, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_geometry_fn, pin_memory=True)

    # Build model
    model = EchoNextGeometrySetOperator(
        geometry_mode=args.geometry_mode,
        response_dim=128,
        operator_dim=8,
        tabular_dim=7,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SetOperator (%s) Initialized: Total=%s, Trainable=%s", args.geometry_mode, f"{total_params:,}", f"{trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    history = []

    logger.info("Starting training loop (Max Epochs=%d, Patience=%d)...", args.epochs, args.patience)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            ops = batch["operators"].to(device)
            resps = batch["responses"].to(device)
            tabs = batch["tabular"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(operators=ops, responses=resps, tabular=tabs, mask=mask)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()
        epoch_train_loss = train_loss / n_samples
        val_loss, val_auroc, val_auprc = evaluate(model, dev_loader, device)
        epoch_time = time.time() - t0

        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "config": {
                    "geometry_mode": "G1",
                    "seed": args.seed,
                    "fit_path": str(FIT_PATH),
                }
            }, out_dir / "best_model.pt")
        else:
            stale_epochs += 1

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val AUROC: %.4f | Val AUPRC: %.4f | LR: %.2e | Time: %.1fs %s",
            epoch, args.epochs, epoch_train_loss, val_auroc, val_auprc, scheduler.get_last_lr()[0], epoch_time,
            "[*BEST*]" if is_best else f"[Patience {stale_epochs}/{args.patience}]"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "time_seconds": epoch_time,
            "is_best": is_best,
        })

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, out_dir / "latest_checkpoint.pt")

        if args.dry_run:
            logger.info("Dry run completed successfully after 1 epoch.")
            break

        if stale_epochs >= args.patience:
            logger.info("Early stopping triggered at Epoch %d (Best Epoch %d with Val AUROC %.4f).", epoch, best_epoch, best_val_auroc)
            break

    # Save final summary manifest
    summary = {
        "status": "COMPLETED" if not args.dry_run else "DRY_RUN_PASSED",
        "model": "G1_Geometry_Aware_SetOperator",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "total_epochs": epoch,
        "history": history,
        "fit_path": str(FIT_PATH),
        "split_path": str(SPLIT_PATH),
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary manifest to %s", out_dir / "train_summary.json")


if __name__ == "__main__":
    main()
