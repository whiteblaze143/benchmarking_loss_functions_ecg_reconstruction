#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from repecg.braid_field import (
    BraidFieldClassifier,
    BraidFieldConfig,
    braid_invariance_loss,
    embedding_variance_floor_loss,
    kl_standard_normal,
)
from repecg.common.metrics import multilabel_metrics


VARIANTS = (
    "field",
    "field_braid",
    "field_braid_event",
    "field_braid_inv",
    "field_braid_prob",
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as item:
        return {key: np.asarray(item[key]) for key in item.files}


def _subset_view(
    operators: torch.Tensor,
    responses: torch.Tensor,
    *,
    generator: torch.Generator,
    min_count: int,
    max_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, count = responses.shape[:2]
    chosen = int(torch.randint(min_count, max_count + 1, (), generator=generator, device=responses.device))
    score = torch.rand(batch, count, generator=generator, device=responses.device)
    index = score.argsort(dim=1)[:, :chosen]
    row = torch.arange(batch, device=responses.device).unsqueeze(1)
    return operators[row, index], responses[row, index], index


def _expand_operators(base: torch.Tensor, batch: int) -> torch.Tensor:
    if base.ndim == 2:
        return base.unsqueeze(0).expand(batch, -1, -1)
    return base


def _forward_loss(
    model: BraidFieldClassifier,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    *,
    variant: str,
    training_regime: str,
    generator: torch.Generator,
    field_weight: float,
    inv_weight: float,
    collapse_weight: float,
    separation_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    responses = batch["context_responses"]
    operators = _expand_operators(batch["context_operators"], len(labels))
    query_ops = batch["query_operators"]
    query_coords = batch["query_coords"]
    target_field = batch["target_field"]

    if training_regime == "subset_aug" or variant == "field_braid_inv":
        op_a, resp_a, _ = _subset_view(
            operators, responses, generator=generator, min_count=1, max_count=operators.shape[1]
        )
    else:
        op_a, resp_a = operators, responses

    out_a = model(op_a, resp_a, query_ops, query_coords)
    diagnosis = loss_fn(out_a["logits"], labels)
    field_loss = nn.functional.mse_loss(out_a["field"], target_field)
    total = diagnosis + field_weight * field_loss
    inv = out_a["field"].new_zeros(())
    collapse = out_a["field"].new_zeros(())

    if variant == "field_braid_inv":
        op_b, resp_b, _ = _subset_view(
            operators, responses, generator=generator, min_count=1, max_count=operators.shape[1]
        )
        out_b = model(op_b, resp_b, query_ops, query_coords)
        total = total + loss_fn(out_b["logits"], labels)
        total = total + field_weight * nn.functional.mse_loss(out_b["field"], target_field)
        inv = braid_invariance_loss(out_a["braid_embedding"], out_b["braid_embedding"])
        collapse = 0.5 * (
            embedding_variance_floor_loss(out_a["braid_embedding"])
            + embedding_variance_floor_loss(out_b["braid_embedding"])
        )
        total = total + inv_weight * inv + collapse_weight * collapse

    separation = out_a["separation_loss"]
    total = total + separation_weight * separation

    kl = out_a["field"].new_zeros(())
    if variant == "field_braid_prob":
        kl = kl_standard_normal(out_a["posterior_mu"], out_a["posterior_logvar"])
        total = total + kl_weight * kl

    return total, {
        "diagnosis": float(diagnosis.detach()),
        "field": float(field_loss.detach()),
        "invariance": float(inv.detach()),
        "collapse": float(collapse.detach()),
        "separation": float(separation.detach()),
        "kl": float(kl.detach()),
    }


def _tensor_batch(
    data: dict[str, np.ndarray], index: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    result = {
        "labels": torch.as_tensor(data["labels"][index], device=device),
        "context_responses": torch.as_tensor(data["context_responses"][index], device=device),
        "target_field": torch.as_tensor(data["target_field"][index], device=device),
        "context_operators": torch.as_tensor(data["context_operators"], device=device),
        "query_operators": torch.as_tensor(data["query_operators"], device=device),
        "query_coords": torch.as_tensor(data["query_coords"], device=device),
    }
    return result


def _evaluate(
    model: BraidFieldClassifier,
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    probability, field_losses = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch_size):
            index = np.arange(start, min(start + batch_size, len(data["labels"])))
            batch = _tensor_batch(data, index, device)
            ops = _expand_operators(batch["context_operators"], len(index))
            out = model(
                ops, batch["context_responses"], batch["query_operators"], batch["query_coords"]
            )
            probability.append(torch.sigmoid(out["logits"]).float().cpu())
            field_losses.append(float(nn.functional.mse_loss(out["field"], batch["target_field"]).cpu()))
    return torch.cat(probability).numpy(), float(np.mean(field_losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--training-regime", choices=("full_only", "subset_aug"), default="full_only")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-weight", type=float, default=0.25)
    parser.add_argument("--inv-weight", type=float, default=0.1)
    parser.add_argument("--collapse-weight", type=float, default=0.05)
    parser.add_argument("--separation-weight", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # cuDNN 9.2 fails to finalize Conv1d descriptors on this CUDA 12.4 host.
        torch.backends.cudnn.enabled = False
    train = _load(args.dataset / "train.npz")
    val = _load(args.dataset / "val.npz")
    config = BraidFieldConfig(
        response_dim=train["context_responses"].shape[-1],
        classes=train["labels"].shape[-1],
        variant=args.variant,
    )
    model = BraidFieldClassifier(config).to(device)
    prevalence = torch.as_tensor(train["labels"].mean(axis=0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        order = rng.permutation(len(train["labels"]))
        epoch_terms = []
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            batch = _tensor_batch(train, index, device)
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed * 1_000_003 + epoch * 10_007 + start)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, terms = _forward_loss(
                    model, batch, loss_fn, variant=args.variant,
                    training_regime=args.training_regime, generator=generator,
                    field_weight=args.field_weight, inv_weight=args.inv_weight,
                    collapse_weight=args.collapse_weight, separation_weight=args.separation_weight,
                    kl_weight=args.kl_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_terms.append({"loss": float(loss.detach()), **terms})
        scheduler.step()

        probability, val_field = _evaluate(model, val, batch_size=args.batch, device=device)
        metrics = multilabel_metrics(val["labels"], probability)
        score = float(metrics["macro_auroc"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean([x["loss"] for x in epoch_terms])),
            "val_macro_auroc": score,
            "val_macro_auprc": float(metrics["macro_auprc"]),
            "val_field_mse": val_field,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "config": config.to_dict(),
        "training_regime": args.training_regime,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_score,
        "seed": args.seed,
        "history": history,
    }
    torch.save(payload, args.output / f"{args.variant}_{args.training_regime}_best.pt")
    (args.output / f"{args.variant}_{args.training_regime}_summary.json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "state_dict"}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
