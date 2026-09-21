#!/usr/bin/env python3
"""Train Paper 07 OperatorSetModel strictly on Full Leads (P07_FULLLEAD_ONLY).

Isolates representation inductive bias from subset data augmentation:
- Context is strictly the 8 canonical independent basis leads during training (m = 8).
- No random subset omission during optimization.
- Enables fair zero-shot comparison against full-lead models.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from repecg.common.metrics import multilabel_metrics
from repecg.common.variants import get_variants_for_paper
from repecg.paper07_operator import OperatorSetModel, symmetric_bernoulli_kl

LEARNING_RATES = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)
RECONSTRUCTION_WEIGHT = 0.1
ORIENTATION_WEIGHT = 0.05
UNKNOWN_ID = -1


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load(path: Path, *, auxiliary: bool) -> dict[str, np.ndarray]:
    required = ["operators", "responses", "labels", "ecg_ids", "patient_ids"]
    if auxiliary:
        required.append("negative_responses")
    with np.load(path) as item:
        missing = set(required).difference(item.files)
        if missing:
            raise ValueError(f"missing Paper 7 arrays: {sorted(missing)}")
        return {name: np.asarray(item[name]) for name in required}


def _forward_batch_fulllead(
    model: OperatorSetModel,
    operators: torch.Tensor,
    responses: torch.Tensor,
    target_idx: torch.Tensor | None = None,
    *,
    auxiliary: bool,
    continuous: bool,
    negative_responses: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Pass all 8 canonical independent leads as context during training."""
    context_operators = operators[:, :8]
    context_responses = responses[:, :8]

    if not auxiliary:
        logits = model(
            context_operators if continuous else None,
            context_responses,
            operator_ids=None,
            return_reconstruction=False,
        )
        return logits, None, None

    rows = torch.arange(len(operators), device=operators.device)
    assert target_idx is not None
    target_operator = operators[rows, target_idx]

    logits, reconstruction = model(
        context_operators if continuous else None,
        context_responses,
        operator_ids=None,
        target_operator=target_operator if continuous else None,
        target_operator_ids=None,
        return_reconstruction=True,
    )
    target_response = responses[rows, target_idx]
    reconstruction_loss = nn.functional.mse_loss(reconstruction, target_response)
    orientation_loss = None
    if continuous:
        if negative_responses is None:
            raise ValueError("continuous auxiliary requires explicit F(-q)")
        negative_context = negative_responses[:, :8]
        negative_logits, negative_reconstruction = model(
            -context_operators,
            negative_context,
            target_operator=-target_operator,
            return_reconstruction=True,
        )
        negative_target = negative_responses[rows, target_idx]
        reconstruction_loss = 0.5 * (
            reconstruction_loss + nn.functional.mse_loss(negative_reconstruction, negative_target)
        )
        orientation_loss = symmetric_bernoulli_kl(logits, negative_logits)

    return logits, reconstruction_loss, orientation_loss


def _train_step(
    model: OperatorSetModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    labels: torch.Tensor,
    operators: torch.Tensor,
    responses: torch.Tensor,
    target_idx: torch.Tensor | None,
    *,
    auxiliary: bool,
    continuous: bool,
    negative_responses: torch.Tensor | None,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, reconstruction, orientation = _forward_batch_fulllead(
            model, operators, responses, target_idx,
            auxiliary=auxiliary, continuous=continuous, negative_responses=negative_responses,
        )
        loss = loss_fn(logits, labels)
        if reconstruction is not None:
            loss = loss + RECONSTRUCTION_WEIGHT * reconstruction
        if orientation is not None:
            loss = loss + ORIENTATION_WEIGHT * orientation
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(float("nan"), device=loss.device)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser(description="Train P07_FULLLEAD_ONLY")
    parser.add_argument("--representations", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper07_fulllead_only"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="continuous_primary", choices=["continuous_primary", "continuous_auxiliary"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    _seed(args.seed)

    auxiliary = args.variant.endswith("auxiliary")
    continuous = True

    train = _load(args.representations / "representation_train.npz", auxiliary=auxiliary)
    validation = _load(args.representations / "representation_val.npz", auxiliary=auxiliary)

    model = OperatorSetModel(
        response_dim=128,
        classes=train["labels"].shape[1],
        operator_mode="continuous",
    ).cuda()

    prevalence = torch.as_tensor(train["labels"].mean(axis=0), device="cuda")
    positive_weight = ((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0

    print(f"=== Training P07_FULLLEAD_ONLY ({args.variant}) ===", flush=True)
    print(f"Context: Strictly 8 independent basis leads (Zero subset augmentation)", flush=True)
    print(f"Train records: {len(train['labels'])}, Val records: {len(validation['labels'])}", flush=True)

    train_ops = torch.from_numpy(train["operators"]).cuda()
    train_resps = torch.from_numpy(train["responses"]).cuda()
    train_labels = torch.from_numpy(train["labels"]).cuda()
    train_neg = torch.from_numpy(train["negative_responses"]).cuda() if auxiliary else None

    val_ops = torch.from_numpy(validation["operators"][:, :8]).cuda()
    val_resps = torch.from_numpy(validation["responses"][:, :8]).cuda()

    n_samples = len(train_labels)
    n_batches = int(np.ceil(n_samples / args.batch))

    for epoch in range(1, args.max_epochs + 1):
        perm = torch.randperm(n_samples, device="cuda")
        total_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * args.batch : (b + 1) * args.batch]
            ops_b = train_ops[idx]
            resps_b = train_resps[idx]
            labels_b = train_labels[idx]
            neg_b = train_neg[idx] if auxiliary else None
            target_idx = torch.randint(0, 8, (len(idx),), device="cuda") if auxiliary else None

            loss = _train_step(
                model, optimizer, loss_fn, labels_b, ops_b, resps_b, target_idx,
                auxiliary=auxiliary, continuous=continuous, negative_responses=neg_b,
            )
            total_loss += float(loss)

        avg_loss = total_loss / n_batches

        # Evaluate on validation (using full 8 independent leads)
        model.eval()
        val_probs = []
        with torch.inference_mode():
            for i in range(0, len(validation["labels"]), 256):
                o_v = val_ops[i : i + 256]
                r_v = val_resps[i : i + 256]
                logits = model(o_v, r_v, return_reconstruction=False)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_metrics = multilabel_metrics(validation["labels"], val_probs)
        val_auroc = val_metrics["macro_auroc"]

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(
                {"state_dict": best_state, "val_auroc": best_val_auroc, "epoch": epoch, "variant": args.variant},
                args.output / f"{args.variant}_best.pt",
            )
            status = " [BEST SAVED]"
        else:
            patience_counter += 1
            status = ""

        if epoch % 5 == 0 or status:
            print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val Macro AUROC: {val_auroc:.4f}{status}", flush=True)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best val AUROC: {best_val_auroc:.4f})", flush=True)
            break

    print(f"\n[Finished] Best checkpoint saved to {args.output / f'{args.variant}_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
