#!/usr/bin/env python3
"""Milestone R025: Train Observable-Subspace SetOperator (OS-SetOp) and Matched Controls.

Trains matched architectures strictly on full 8 independent leads only:
- Arm B: os_only (OSOnlyModel with quotient projection P and Z = H @ X_obs)
- Arm C: os_without_p (OSWithoutPModel: isolates the observability mask)
- Arm D: learned_q (LearnedQWaveformModel: capacity-matched control on raw (Q, Y))
- Arm A: canonical_setoperator (Matched retrain of OperatorSetModel on full-lead basis only)

Evaluation during training is strictly full-lead validation (fold 8).
Zero-shot evaluation on sparse and unseen measurement configurations is performed in r025_benchmark.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

# Critical for PyTorch 2.6 on A100: disable cuDNN to prevent 1D conv bug
torch.backends.cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.observable_encoder import (
    OSOnlyModel,
    OSWithoutPModel,
    LearnedQWaveformModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r025_train")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
WAVEFORM_DIR = CODEX_DIR / "paper07_operator/r025_waveforms"
R024_DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
MODELS_DIR = CODEX_DIR / "paper07_operator/r025_models"


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


def evaluate_val_full_lead(
    arm: str,
    model: nn.Module,
    val_waveforms: np.ndarray,  # [N_val, 256, 8]
    val_labels: np.ndarray,     # [N_val, 5]
    val_resps: np.ndarray | None,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    N = len(val_labels)
    # Basis operators for 8 independent leads: eye(8)
    e8 = torch.eye(8, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 8, 8]

    probs = []
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            x_b = torch.as_tensor(val_waveforms[i : i + batch_size], dtype=torch.float32, device=device)
            B_cur = len(x_b)
            q_b = e8.expand(B_cur, -1, -1)  # [B, 8, 8]
            y_b = x_b  # in full lead: Y = X @ I = X

            if arm == "canonical_setoperator":
                r_b = torch.as_tensor(val_resps[i : i + batch_size, :8], dtype=torch.float32, device=device)
                logits = model(q_b, r_b, return_reconstruction=False)
            else:
                logits = model(q_b, y_b)

            probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    return compute_metrics(val_labels, probs)


def train_single_arm(
    arm: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    logger.info(f"=== Starting Training for Arm: {arm} (Seed: {seed}) ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load waveform data
    train_npz = np.load(WAVEFORM_DIR / "r025_train_waveforms.npz", mmap_mode="r")
    val_npz = np.load(WAVEFORM_DIR / "r025_val_waveforms.npz", mmap_mode="r")

    train_waveforms = train_npz["waveforms"]  # [15244, 256, 8]
    train_labels = train_npz["labels"]        # [15244, 5]
    val_waveforms = val_npz["waveforms"]      # [2173, 256, 8]
    val_labels = val_npz["labels"]            # [2173, 5]

    train_resps = None
    val_resps = None
    if arm == "canonical_setoperator":
        # Load Nyström responses
        d_tr_resp = np.load(R024_DATA_DIR / "representation_train_basis_only.npz", mmap_mode="r")
        d_val_resp = np.load(R024_DATA_DIR / "representation_val_real_waveforms.npz", mmap_mode="r")
        train_resps = d_tr_resp["responses"][:, :8]  # [15244, 8, 16, 128]
        val_resps = d_val_resp["responses"][:, :8]

    # Instantiate model
    T = 256
    L = 24
    d = 8
    classes = 5

    if arm == "os_only":
        model = OSOnlyModel(T=T, L=L, d=d, classes=classes, frame_mode="dct", rcond=1e-5)
    elif arm == "os_without_p":
        model = OSWithoutPModel(T=T, L=L, d=d, classes=classes, frame_mode="dct", rcond=1e-5)
    elif arm == "learned_q":
        model = LearnedQWaveformModel(T=T, L=L, d=d, classes=classes, frame_mode="dct")
    elif arm == "canonical_setoperator":
        model = OperatorSetModel(response_dim=128, classes=classes, operator_mode="continuous")
    else:
        raise ValueError(f"Unknown arm: {arm}")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model {arm} initialized with {n_params:,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = args.out_dir / f"{arm}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(train_labels)
    batch_size = args.batch_size
    best_val_auroc = -1.0
    best_epoch = -1
    best_state_dict = None
    best_metrics = {}
    patience_counter = 0

    e8 = torch.eye(8, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 8, 8]

    t0_train = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_losses = []

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            B_cur = len(idx)

            lbl_b = torch.as_tensor(train_labels[idx], dtype=torch.float32, device=device)
            x_b = torch.as_tensor(train_waveforms[idx], dtype=torch.float32, device=device)
            q_b = e8.expand(B_cur, -1, -1)
            y_b = x_b  # full-lead training: Y = X @ I = X

            optimizer.zero_grad(set_to_none=True)

            if arm == "canonical_setoperator":
                r_b = torch.as_tensor(train_resps[idx], dtype=torch.float32, device=device)
                logits = model(q_b, r_b, return_reconstruction=False)
            else:
                logits = model(q_b, y_b)

            loss = loss_fn(logits, lbl_b)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))

        # Evaluate on validation
        val_metrics = evaluate_val_full_lead(
            arm, model, val_waveforms, val_labels, val_resps, device, batch_size=batch_size
        )
        val_auroc = val_metrics["macro_auroc"]
        val_auprc = val_metrics["macro_auprc"]

        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(val_metrics)
            patience_counter = 0
            star = "(*)"
        else:
            patience_counter += 1
            star = "   "

        logger.info(
            f"Epoch {epoch:02d}/{args.max_epochs:02d} | Loss: {mean_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f} {star}"
        )

        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    total_time = time.time() - t0_train
    logger.info(f"Training completed in {total_time:.1f}s. Best Epoch: {best_epoch} with Val AUROC: {best_val_auroc:.4f}")

    # Save checkpoint and summary
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "arm": arm,
            "seed": seed,
            "model_state_dict": best_state_dict,
            "best_epoch": best_epoch,
            "val_metrics": best_metrics,
            "num_parameters": n_params,
            "args": vars(args),
        },
        ckpt_path,
    )

    summary = {
        "arm": arm,
        "seed": seed,
        "num_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_val_auroc,
        "best_val_macro_auprc": best_metrics.get("macro_auprc", 0.0),
        "total_train_time_sec": total_time,
        "checkpoint_path": str(ckpt_path),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train R025 Models")
    parser.add_argument("--arm", type=str, default="all", choices=["all", "os_only", "os_without_p", "learned_q", "canonical_setoperator"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    arms = ["os_only", "os_without_p", "learned_q"] if args.arm == "all" else [args.arm]

    results = []
    for arm in arms:
        for seed in args.seeds:
            res = train_single_arm(arm, seed, args, device)
            results.append(res)

    print("\n=== Training Summary ===")
    for r in results:
        print(f"[{r['arm']} | seed {r['seed']}] Best Val AUROC: {r['best_val_macro_auroc']:.4f} (epoch {r['best_epoch']})")


if __name__ == "__main__":
    main()
