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


def build_training_vocabulary(operators: np.ndarray) -> np.ndarray:
    """Return a stable vocabulary containing only exact training operators."""
    if operators.ndim != 3 or operators.shape[-1] != 8:
        raise ValueError("operators must have shape (records,set,8)")
    return np.unique(np.ascontiguousarray(operators).reshape(-1, 8), axis=0)


def map_operator_ids(operators: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """Map exact rows to training IDs and every unseen row to one UNK ID."""
    lookup = {row.tobytes(): index for index, row in enumerate(np.ascontiguousarray(vocabulary))}
    flat = np.ascontiguousarray(operators).reshape(-1, 8)
    ids = np.fromiter((lookup.get(row.tobytes(), UNKNOWN_ID) for row in flat), dtype=np.int64)
    return ids.reshape(operators.shape[:-1])


def sample_context_target_indices(
    batch_size: int, candidates: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample m in [1,6], with a distinct held-out target per row."""
    if candidates < 7:
        raise ValueError("Paper 7 requires at least seven candidate pairs")
    context_size = int(torch.randint(1, 7, (), generator=generator, device=device))
    order = torch.rand((batch_size, candidates), generator=generator, device=device).argsort(dim=1)
    return order[:, :context_size], order[:, context_size]


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(values), device=values.device).unsqueeze(1)
    return values[rows, indices]


def _load(path: Path, *, auxiliary: bool) -> dict[str, np.ndarray]:
    required = ["operators", "responses", "labels", "ecg_ids", "patient_ids"]
    if auxiliary:
        required.append("negative_responses")
    with np.load(path) as item:
        missing = set(required).difference(item.files)
        if missing:
            raise ValueError(f"missing Paper 7 arrays: {sorted(missing)}")
        return {name: np.asarray(item[name]) for name in required}


def _forward_batch(
    model: OperatorSetModel,
    operators: torch.Tensor,
    responses: torch.Tensor,
    operator_ids: torch.Tensor | None,
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    auxiliary: bool,
    continuous: bool,
    negative_responses: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    rows = torch.arange(len(operators), device=operators.device)
    context_operators = _gather(operators, context)
    context_responses = _gather(responses, context)
    context_ids = _gather(operator_ids, context) if operator_ids is not None else None
    target_operator = operators[rows, target]
    target_ids = operator_ids[rows, target] if operator_ids is not None else None
    result = model(
        context_operators if continuous else None,
        context_responses,
        operator_ids=context_ids,
        target_operator=target_operator if continuous else None,
        target_operator_ids=target_ids,
        return_reconstruction=auxiliary,
    )
    if not auxiliary:
        assert isinstance(result, torch.Tensor)
        return result, None, None
    logits, reconstruction = result
    target_response = responses[rows, target]
    reconstruction_loss = nn.functional.mse_loss(reconstruction, target_response)
    orientation_loss = None
    if continuous:
        if negative_responses is None:
            raise ValueError("continuous auxiliary requires explicit F(-q)")
        negative_context = _gather(negative_responses, context)
        negative_logits, negative_reconstruction = model(
            -context_operators,
            negative_context,
            target_operator=-target_operator,
            return_reconstruction=True,
        )
        negative_target = negative_responses[rows, target]
        reconstruction_loss = 0.5 * (
            reconstruction_loss + nn.functional.mse_loss(negative_reconstruction, negative_target)
        )
        orientation_loss = symmetric_bernoulli_kl(logits, negative_logits)
    return logits, reconstruction_loss, orientation_loss


def _train_variant(
    model: OperatorSetModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    labels: torch.Tensor,
    operators: torch.Tensor,
    responses: torch.Tensor,
    operator_ids: torch.Tensor | None,
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    auxiliary: bool,
    continuous: bool,
    negative_responses: torch.Tensor | None,
) -> torch.Tensor:
    """Optimize one grid cell on one shared mini-batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, reconstruction, orientation = _forward_batch(
            model, operators, responses, operator_ids, context, target,
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
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
        optimizer.step()
    return loss.detach()


def _evaluate(
    model: OperatorSetModel,
    data: dict[str, np.ndarray],
    operator_ids: np.ndarray | None,
    *,
    batch_size: int,
    auxiliary: bool,
    continuous: bool,
) -> tuple[np.ndarray, float | None, float | None]:
    model.eval()
    probabilities = []
    reconstruction_losses = []
    orientation_losses = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch_size):
            stop = min(start + batch_size, len(data["labels"]))
            operators = torch.as_tensor(data["operators"][start:stop], device="cuda")
            responses = torch.as_tensor(data["responses"][start:stop], device="cuda")
            ids = (
                torch.as_tensor(operator_ids[start:stop], device="cuda")
                if operator_ids is not None else None
            )
            negative = (
                torch.as_tensor(data["negative_responses"][start:stop], device="cuda")
                if auxiliary and continuous else None
            )
            context = torch.arange(6, device="cuda").expand(stop - start, -1)
            target = torch.full((stop - start,), 6, device="cuda", dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, reconstruction, orientation = _forward_batch(
                    model, operators, responses, ids, context, target,
                    auxiliary=auxiliary, continuous=continuous, negative_responses=negative,
                )
            probabilities.append(torch.sigmoid(logits).float().cpu())
            if reconstruction is not None:
                reconstruction_losses.append(float(reconstruction.float().cpu()))
            if orientation is not None:
                orientation_losses.append(float(orientation.float().cpu()))
    return (
        torch.cat(probabilities).numpy(),
        float(np.mean(reconstruction_losses)) if reconstruction_losses else None,
        float(np.mean(orientation_losses)) if orientation_losses else None,
    )


def _cell_path(root: Path, variant: str, learning_rate: float, weight_decay: float) -> Path:
    return root / f"{variant}_lr{learning_rate:g}_wd{weight_decay:g}"


def _save_cell(
    cell: dict[str, object], variant: str, seed: int, validation: dict[str, np.ndarray],
    vocabulary_hash: str | None, peak_vram_bytes: int,
) -> None:
    path = cell["path"]
    assert isinstance(path, Path)
    path.mkdir(parents=True, exist_ok=True)
    probability = cell["best_probability"]
    state = cell["best_state"]
    if probability is None or state is None:
        probability = np.full_like(validation["labels"], 0.5, dtype=np.float32)
        model = cell["model"]
        assert isinstance(model, nn.Module)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        cell["best_score"] = 0.5
        cell["best_epoch"] = 0
        cell["best_mechanism"] = {"response_mse": None, "orientation_symmetric_kl": None}
    assert isinstance(probability, np.ndarray) and isinstance(state, dict)
    summary = {
        "variant": variant,
        "learning_rate": cell["learning_rate"],
        "weight_decay": cell["weight_decay"],
        "best_epoch": cell["best_epoch"],
        "epochs_run": len(cell["history"]),
        "metrics": multilabel_metrics(validation["labels"], probability),
        "mechanism": cell["best_mechanism"],
        "loss_contract": cell["loss_contract"],
        "vocabulary_sha256": vocabulary_hash,
        "peak_vram_bytes": peak_vram_bytes,
        "execution": "shared_host_arrays_minibatched_to_cuda",
        "history": cell["history"],
    }
    torch.save({"variant": variant, "seed": seed, "summary": summary, "state_dict": state}, path / "checkpoint.pt")
    np.savez_compressed(
        path / "predictions.npz", probability=probability, labels=validation["labels"],
        ecg_ids=validation["ecg_ids"], patient_ids=validation["patient_ids"],
    )
    _atomic_json(path / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=tuple(get_variants_for_paper(7)), required=True)
    parser.add_argument("--training-regime", default="full_only", choices=["full_only"])
    parser.add_argument("--learning-rate", type=float, action="append")
    parser.add_argument("--weight-decay", type=float, action="append")
    args = parser.parse_args()

    manifest = json.loads((args.representations / "manifest.json").read_text())
    if manifest.get("kind") != "paper07_measurement_operator_response_sets" or manifest.get("status") != "complete":
        raise ValueError("Paper 7 requires a complete measurement-operator response-set artifact")
    auxiliary = args.variant.endswith("auxiliary")
    continuous = args.variant.startswith("continuous")
    train = _load(args.representations / "representation_train.npz", auxiliary=auxiliary)
    validation = _load(args.representations / "representation_val.npz", auxiliary=auxiliary)

    vocabulary = build_training_vocabulary(train["operators"])
    vocabulary_hash = hashlib.sha256(np.ascontiguousarray(vocabulary).tobytes()).hexdigest()
    train_ids = None if continuous else map_operator_ids(train["operators"], vocabulary)
    validation_ids = None if continuous else map_operator_ids(validation["operators"], vocabulary)
    args.output.mkdir(parents=True, exist_ok=True)
    cells_root = args.output / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    vocabulary_path = args.output / "categorical_training_vocabulary.npz"
    if not vocabulary_path.exists():
        np.savez_compressed(vocabulary_path, operators=vocabulary, unknown_id=np.asarray(UNKNOWN_ID))

    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")
    _seed(args.seed)
    template = OperatorSetModel(
        classes=train["labels"].shape[1], operator_mode="continuous" if continuous else "categorical",
        vocabulary_size=len(vocabulary) if not continuous else 0,
    ).cuda()
    initial_state = copy.deepcopy(template.state_dict())
    del template
    prevalence = torch.as_tensor(train["labels"].mean(axis=0), device="cuda")
    positive_weight = ((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    loss_contract = {
        "diagnosis_bce": 1.0,
        "response_mse": RECONSTRUCTION_WEIGHT if auxiliary else 0.0,
        "orientation_symmetric_kl": ORIENTATION_WEIGHT if auxiliary and continuous else 0.0,
        "explicit_negative_response": bool(auxiliary and continuous),
    }
    learning_rates = tuple(args.learning_rate) if args.learning_rate else LEARNING_RATES
    weight_decays = tuple(args.weight_decay) if args.weight_decay else WEIGHT_DECAYS
    if any(value <= 0 for value in (*learning_rates, *weight_decays)):
        raise ValueError("learning rates and weight decays must be positive")
    cells = []
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            path = _cell_path(cells_root, args.variant, learning_rate, weight_decay)
            if (path / "summary.json").exists():
                continue
            model = OperatorSetModel(
                classes=train["labels"].shape[1], operator_mode="continuous" if continuous else "categorical",
                vocabulary_size=len(vocabulary) if not continuous else 0,
            ).cuda()
            model.load_state_dict(initial_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            cells.append({
                "path": path, "learning_rate": learning_rate, "weight_decay": weight_decay,
                "model": model, "optimizer": optimizer,
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs),
                "best_score": -np.inf, "best_epoch": 0, "best_state": None,
                "best_probability": None, "best_mechanism": None, "stale": 0,
                "history": [], "active": True, "saved": False, "loss_contract": loss_contract,
            })
    if not cells:
        print(json.dumps({"variant": args.variant, "status": "already_complete"}))
        return

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.max_epochs + 1):
        active = [cell for cell in cells if cell["active"]]
        if not active:
            break
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        permutation = rng.permutation(len(train["labels"]))
        epoch_loss = {id(cell): [] for cell in active}
        for start in range(0, len(permutation), args.batch):
            index = permutation[start:start + args.batch]
            operators = torch.as_tensor(train["operators"][index], device="cuda")
            responses = torch.as_tensor(train["responses"][index], device="cuda")
            labels = torch.as_tensor(train["labels"][index], device="cuda")
            ids = torch.as_tensor(train_ids[index], device="cuda") if train_ids is not None else None
            negative = (
                torch.as_tensor(train["negative_responses"][index], device="cuda")
                if auxiliary and continuous else None
            )
            generator = torch.Generator(device="cuda")
            generator.manual_seed(args.seed * 1_000_000 + epoch * 100_000 + start)
            context, target = sample_context_target_indices(len(index), operators.shape[1], generator, operators.device)
            for cell in active:
                model = cell["model"]
                optimizer = cell["optimizer"]
                assert isinstance(model, OperatorSetModel) and isinstance(optimizer, torch.optim.Optimizer)
                loss = _train_variant(
                    model, optimizer, loss_fn, labels, operators, responses, ids, context, target,
                    auxiliary=auxiliary, continuous=continuous, negative_responses=negative,
                )
                epoch_loss[id(cell)].append(float(loss.detach().cpu()))

        for cell in active:
            model = cell["model"]
            scheduler = cell["scheduler"]
            assert isinstance(model, OperatorSetModel)
            scheduler.step()
            probability, reconstruction, orientation = _evaluate(
                model, validation, validation_ids, batch_size=args.batch,
                auxiliary=auxiliary, continuous=continuous,
            )
            has_invalid = np.isnan(probability).any() or np.isinf(probability).any()
            if has_invalid:
                score = -1.0
            else:
                try:
                    score = float(multilabel_metrics(validation["labels"], probability)["macro_auroc"])
                except Exception:
                    score = -1.0
            history = cell["history"]
            assert isinstance(history, list)
            losses = [l for l in epoch_loss[id(cell)] if not (np.isnan(l) or np.isinf(l))]
            history.append({
                "epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0,
                "val_macro_auroc": max(score, 0.0), "val_response_mse": reconstruction,
                "val_orientation_symmetric_kl": orientation,
            })
            if score > float(cell["best_score"]) + 1e-6:
                cell.update({
                    "best_score": score, "best_epoch": epoch,
                    "best_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    "best_probability": probability,
                    "best_mechanism": {"response_mse": reconstruction, "orientation_symmetric_kl": orientation},
                    "stale": 0,
                })
            else:
                cell["stale"] = int(cell["stale"]) + 1
                if int(cell["stale"]) >= args.patience or has_invalid:
                    cell["active"] = False
                    _save_cell(cell, args.variant, args.seed, validation, vocabulary_hash, int(torch.cuda.max_memory_allocated()))
                    cell["saved"] = True
        print(json.dumps({
            "variant": args.variant, "epoch": epoch,
            "active_cells": sum(bool(cell["active"]) for cell in cells),
            "scores": [round(float(cell["best_score"]), 6) for cell in cells],
            "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }), flush=True)

    peak = int(torch.cuda.max_memory_allocated())
    for cell in cells:
        if not cell["saved"]:
            _save_cell(cell, args.variant, args.seed, validation, vocabulary_hash, peak)


if __name__ == "__main__":
    main()
