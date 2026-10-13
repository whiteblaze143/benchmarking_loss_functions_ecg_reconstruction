#!/usr/bin/env python3
"""Milestone R024: Train Basis-Only and Operator-Span SetOperator Models.

Trains matched OperatorSetModel (unchanged architecture):
- Arm 1: basis_only (trained strictly on canonical 8 basis leads {e_1, ..., e_8})
- Arm 2: operator_span (trained on 8 basis leads + 8 continuous linear combinations on S^7)

Validation evaluated on real physical waveform validation set.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cudnn 1D conv bug on this environment
torch.backends.cudnn.enabled = False

from repecg.paper07_operator import OperatorSetModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_r024_models")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            class_aurocs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(y_true[:, c], y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auprc": float(np.mean(class_auprcs)),
    }


def evaluate_on_validation(
    model: OperatorSetModel,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    # Evaluate on the 8 basis leads (indices 0..7)
    ops = val_data["operators"][:, :8]
    resps = val_data["responses"][:, :8]
    labels = val_data["labels"]

    probs = []
    with torch.inference_mode():
        for i in range(0, len(labels), batch_size):
            o_b = torch.as_tensor(ops[i : i + batch_size], dtype=torch.float32, device=device)
            r_b = torch.as_tensor(resps[i : i + batch_size], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(o_b, r_b, return_reconstruction=False)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return compute_metrics(labels, probs)


def train_single_arm(
    arm: str,
    train_npz_path: Path,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    logger.info(f"--- Starting training for Arm: {arm} ---")
    logger.info(f"Loading training data from {train_npz_path}...")
    d_train = np.load(train_npz_path, mmap_mode="r")
    train_ops = d_train["operators"]      # [N, K, 8]
    train_resps = d_train["responses"]    # [N, K, 16, 128]
    train_labels = d_train["labels"]      # [N, 5]
    n_records, n_candidates, _ = train_ops.shape
    logger.info(f"Train dataset: {n_records} records, {n_candidates} candidate operators per record.")

    # Instantiate unchanged SetOperator
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    arm_dir = MODELS_DIR / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = rng.permutation(n_records)
        epoch_losses = []

        for start in range(0, n_records, args.batch):
            idx = perm[start : start + args.batch]
            batch_ops = train_ops[idx]      # [B, K, 8]
            batch_resps = train_resps[idx]  # [B, K, 16, 128]
            batch_y = torch.as_tensor(train_labels[idx], dtype=torch.float32, device=device)

            # Sample subset cardinality m ~ U(1, min(7, n_candidates - 1))
            m = rng.integers(1, min(8, n_candidates))
            # Pick m random distinct operators for each item in batch
            # For vectorization: choose m random indices per item
            sub_indices = np.stack([rng.choice(n_candidates, size=m, replace=False) for _ in range(len(idx))])

            # Gather context
            b_idx = np.arange(len(idx))[:, None]
            c_ops = torch.as_tensor(batch_ops[b_idx, sub_indices], dtype=torch.float32, device=device)
            c_resps = torch.as_tensor(batch_resps[b_idx, sub_indices], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(c_ops, c_resps, return_reconstruction=False)
                loss = loss_fn(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))

        # Validation evaluation
        val_metrics = evaluate_on_validation(model, val_data, device, batch_size=args.batch)
        val_auroc = val_metrics["macro_auroc"]

        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_macro_auroc": val_auroc,
            "val_micro_auroc": val_metrics["micro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
        })

        if val_auroc > best_score + 1e-5:
            best_score = val_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            logger.info(f"[{arm}] Epoch {epoch:2d}/{args.max_epochs}: Loss={mean_loss:.4f} | Val AUROC={val_auroc:.4f} (*) BEST")
        else:
            stale += 1
            logger.info(f"[{arm}] Epoch {epoch:2d}/{args.max_epochs}: Loss={mean_loss:.4f} | Val AUROC={val_auroc:.4f} (stale {stale}/{args.patience})")

        if stale >= args.patience:
            logger.info(f"[{arm}] Reached early stopping patience {args.patience} at epoch {epoch}.")
            break

    # Save best checkpoint
    assert best_state is not None
    ckpt_path = arm_dir / "checkpoint.pt"
    torch.save({
        "arm": arm,
        "best_epoch": best_epoch,
        "best_macro_auroc": best_score,
        "state_dict": best_state,
        "history": history,
        "hyperparameters": {
            "lr": args.lr,
            "wd": args.wd,
            "batch": args.batch,
            "seed": args.seed,
        },
    }, ckpt_path)

    summary = {
        "arm": arm,
        "best_epoch": best_epoch,
        "best_macro_auroc": best_score,
        "history": history,
    }
    with open(arm_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[{arm}] Training complete. Best checkpoint saved to {ckpt_path} (Val AUROC = {best_score:.4f} at epoch {best_epoch}).")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Milestone R024 Models")
    parser.add_argument("--arm", choices=["basis_only", "operator_span", "both"], default="both")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    val_path = DATA_DIR / "representation_val_real_waveforms.npz"
    logger.info(f"Loading real physical validation split from {val_path}...")
    d_val = np.load(val_path)
    val_data = {k: d_val[k] for k in d_val.files}

    arms_to_run = ["basis_only", "operator_span"] if args.arm == "both" else [args.arm]

    for arm in arms_to_run:
        train_path = DATA_DIR / f"representation_train_{arm}.npz"
        train_single_arm(arm, train_path, val_data, device, args)


if __name__ == "__main__":
    main()
