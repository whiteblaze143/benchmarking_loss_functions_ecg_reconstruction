#!/usr/bin/env python3
"""Authoritative Five-Seed SetONet Training Script for PTB-XL (Folds 1-8).

Implements the official SetONet branch-trunk architecture adapted for PTB-XL:
- Branch Network:
    Encodes sensor coordinates (8-dim operator coordinates q_i) concatenated
    with ECG response representations (Phase-KME response u_i).
    Applies Phi MLP -> Set Aggregation (Attention/Mean) -> Rho MLP.
    Outputs branch latent vectors: B in R^{batch x p x 1}.
- Trunk Network:
    Evaluates target query locations (5-class diagnostic queries y_c for NORM, MI, STTC, CD, HYP).
    Outputs trunk latent vectors: T in R^{batch x 5 x p x 1}.
- Prediction:
    Inner product over latent dimension p: logit_c = sum_{k=1}^p B_k * T_{c, k} + bias.

Experimental Protocol (Aligned with SetOperator):
- Training on Folds 1-8, Validation on Fold 9, Fold 10 strictly SEALED.
- Class-weighted BCEWithLogitsLoss.
- AdamW optimizer (lr=3e-4, wd=1e-3), gradient norm clipping 1.0.
- CosineAnnealingLR scheduler.
- Context cardinality m in {1..7} uniformly sampled.
- Seeds 42, 43, 44, 45, 46 (matched paired initializations).
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_setonet_ptbxl")

torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
SETONET_DIR = REPO_DIR / "external/SetONet"

if str(SETONET_DIR) not in sys.path:
    sys.path.insert(0, str(SETONET_DIR))

from Models.SetONet import SetONet


class PTBXLSetONetAdapter(nn.Module):
    """Clean adapter connecting SetONet branch-trunk architecture to PTB-XL."""

    def __init__(
        self,
        operator_dim: int = 8,
        response_dim: int = 128,  # pooled 128 or flattened 2048
        classes: int = 5,
        p: int = 64,
        phi_hidden: int = 256,
        rho_hidden: int = 256,
        trunk_hidden: int = 256,
        aggregation: str = "attention",
        variant: str = "standard",  # "standard" or "petrov_attention" (for SN-Key)
    ):
        super().__init__()
        self.operator_dim = operator_dim
        self.response_dim = response_dim
        self.classes = classes
        self.p = p

        # 5 distinct diagnostic class query vectors (fixed orthonormal or learned embeddings)
        self.register_buffer("class_queries", torch.eye(classes, dtype=torch.float32))

        # Official SetONet model instantiation
        self.setonet = SetONet(
            input_size_src=operator_dim,
            output_size_src=response_dim,
            input_size_tgt=classes,
            output_size_tgt=1,
            p=p,
            phi_hidden_size=phi_hidden,
            phi_output_size=128,
            rho_hidden_size=rho_hidden,
            trunk_hidden_size=trunk_hidden,
            n_trunk_layers=4,
            use_deeponet_bias=True,
            pos_encoding_type="skip",  # continuous operator coordinates are already physical
            use_positional_encoding=False,
            aggregation_type=aggregation,
            branch_head_type="petrov_attention" if variant == "key" else "standard",
        )

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        operators: (B, m, 8)
        responses: (B, m, 16, 128) -> pooled to (B, m, 128)
        Returns: logits (B, 5)
        """
        B, m = operators.shape[:2]

        # Pool response over 16 phase cells to match response_dim=128
        if responses.dim() == 4:
            res_feat = responses.mean(dim=2)  # (B, m, 128)
        else:
            res_feat = responses

        # Target class queries for trunk: (B, 5, 5)
        ys = self.class_queries.unsqueeze(0).expand(B, -1, -1)

        # Forward through official SetONet
        out = self.setonet(operators, res_feat, ys)  # (B, 5, 1)
        return out.squeeze(-1)


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


def evaluate_on_fold9(
    model: PTBXLSetONetAdapter,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    ops = val_data["operators"][:, :8]
    resps = val_data["responses"][:, :8]
    labels = val_data["labels"]

    probs = []
    with torch.inference_mode():
        for i in range(0, len(labels), batch_size):
            o_b = torch.as_tensor(ops[i : i + batch_size], dtype=torch.float32, device=device)
            r_b = torch.as_tensor(resps[i : i + batch_size], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(o_b, r_b)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return compute_metrics(labels, probs)


def train_single_seed(
    seed: int,
    intervention: str,
    initial_state: Dict[str, torch.Tensor],
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    output_dir: Path,
    epochs: int = 80,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    wd: float = 1e-3,
    variant: str = "standard",
    device_str: str = "cuda",
) -> Path:
    logger.info("=== Starting SetONet Training: Seed %d | Intervention: %s ===", seed, intervention)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    train_ops = train_data["operators"]
    train_resps = train_data["responses"]
    train_labels = train_data["labels"]
    n_records, n_candidates, _ = train_ops.shape

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    model = PTBXLSetONetAdapter(variant=variant).to(device)
    model.load_state_dict(initial_state)

    prevalence = torch.as_tensor(train_labels.mean(axis=0), dtype=torch.float32, device=device)
    positive_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history: List[Dict[str, Any]] = []

    best_checkpoint_path = output_dir / f"{intervention}_s{seed}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_records)
        epoch_losses: List[float] = []

        for start in range(0, n_records, batch_size):
            idx = perm[start : start + batch_size]
            b_len = len(idx)

            # Sample m strictly in [1, 7]
            m = int(rng.integers(1, min(8, n_candidates)))
            sub_indices = np.stack([rng.choice(n_candidates, size=m, replace=False) for _ in range(b_len)])
            c_ops = torch.as_tensor(train_ops[idx[:, None], sub_indices], dtype=torch.float32, device=device)
            c_resps = torch.as_tensor(train_resps[idx[:, None], sub_indices], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(train_labels[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(c_ops, c_resps)
                loss = criterion(logits, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        val_metrics = evaluate_on_fold9(model, val_data, device, batch_size=batch_size)
        val_auroc = val_metrics["macro_auroc"]

        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_macro_auroc": val_auroc,
            "val_micro_auroc": val_metrics["micro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_auroc > best_val_auroc + 1e-5:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            logger.info(
                "[%s_s%d] Epoch %2d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f (*) BEST",
                intervention, seed, epoch, epochs, mean_loss, val_auroc,
            )
        else:
            stale += 1
            if epoch % 5 == 0 or stale >= patience:
                logger.info(
                    "[%s_s%d] Epoch %2d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f (stale %d/%d)",
                    intervention, seed, epoch, epochs, mean_loss, val_auroc, stale, patience,
                )

        if stale >= patience:
            logger.info("[%s_s%d] Early stopping patience reached at epoch %d.", intervention, seed, epoch)
            break

    assert best_state is not None, "Training failed to produce valid state"
    checkpoint_payload = {
        "state_dict": best_state,
        "seed": seed,
        "intervention": intervention,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auroc),
        "history": history,
        "hyperparameters": {
            "lr": lr,
            "wd": wd,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "positive_weights": positive_weight.cpu().tolist(),
        },
        "variant": variant,
        "constructor": "PTBXLSetONetAdapter(operator_dim=8, response_dim=128, classes=5)",
    }
    torch.save(checkpoint_payload, best_checkpoint_path)

    # Verify reload
    chk = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    v_model = PTBXLSetONetAdapter(variant=variant)
    v_model.load_state_dict(chk["state_dict"], strict=True)
    logger.info("Saved and verified checkpoint %s (Best Val AUROC: %.4f)", best_checkpoint_path, best_val_auroc)

    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train 5-seed SetONet models on PTB-XL.")
    parser.add_argument("--mode", choices=["basis", "span", "both"], default="both")
    parser.add_argument("--variant", choices=["standard", "key"], default="standard")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--train-basis-npz", type=Path, required=True)
    parser.add_argument("--train-span-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    logger.info("Loading Fold-9 validation representations...")
    with np.load(args.val_npz) as d_val:
        val_data = {
            "operators": np.asarray(d_val["operators"]),
            "responses": np.asarray(d_val["responses"]),
            "labels": np.asarray(d_val["labels"]),
        }

    modes = ["basis", "span"] if args.mode == "both" else [args.mode]
    train_datasets = {}
    for mode in modes:
        p = args.train_basis_npz if mode == "basis" else args.train_span_npz
        logger.info("Loading %s training representations from %s...", mode.upper(), p)
        with np.load(p) as d_train:
            train_datasets[mode] = {
                "operators": np.asarray(d_train["operators"]),
                "responses": np.asarray(d_train["responses"]),
                "labels": np.asarray(d_train["labels"]),
            }

    checkpoints = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        template = PTBXLSetONetAdapter(variant=args.variant).to(device)
        initial_state = copy.deepcopy(template.state_dict())
        del template

        for mode in modes:
            chk_path = train_single_seed(
                seed=seed,
                intervention=f"setonet_{mode}" if args.variant == "standard" else f"setonet_key_{mode}",
                initial_state=initial_state,
                train_data=train_datasets[mode],
                val_data=val_data,
                output_dir=args.output_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                wd=args.wd,
                variant=args.variant,
                device_str=args.device,
            )
            sha = hashlib.sha256(chk_path.read_bytes()).hexdigest()
            checkpoints[f"{mode}_s{seed}"] = {"path": str(chk_path), "sha256": sha}

    ledger_path = args.output_dir / "checkpoints_selection_ledger.json"
    ledger_path.write_text(json.dumps(checkpoints, indent=2))
    logger.info("SetONet training complete. Ledger saved to %s", ledger_path)


if __name__ == "__main__":
    main()
