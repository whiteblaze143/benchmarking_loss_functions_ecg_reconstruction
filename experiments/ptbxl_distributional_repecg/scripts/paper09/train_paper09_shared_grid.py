from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from repecg.common.metrics import multilabel_metrics
from repecg.common.variants import get_variants_for_paper
from repecg.paper09_counterfactual import CounterfactualOperatorSetModel


LEARNING_RATES = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)
COUNTERFACTUAL_WEIGHT = 0.1
INVARIANCE_WEIGHT = 0.05


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, np.ndarray]:
    required = ("operators", "responses", "labels", "ecg_ids", "patient_ids")
    with np.load(path) as item:
        missing = set(required).difference(item.files)
        if missing:
            raise ValueError(f"missing Paper 9 arrays: {sorted(missing)}")
        return {name: np.asarray(item[name]) for name in required}


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(values), device=values.device).unsqueeze(1)
    return values[rows, indices]


def sample_disjoint_contexts(
    batch_size: int,
    candidates: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample two disjoint 1--3-pair contexts and one held-out target."""
    if candidates < 7:
        raise ValueError("Paper 9 requires at least seven operator-response pairs")
    size = int(torch.randint(1, 4, (), generator=generator, device=device))
    order = torch.rand((batch_size, candidates), generator=generator, device=device).argsort(1)
    return order[:, :size], order[:, size:2 * size], order[:, 2 * size]


def normalized_state_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = nn.functional.normalize(left.float(), dim=-1)
    right = nn.functional.normalize(right.float(), dim=-1)
    return nn.functional.mse_loss(left, right)


def _mismatch_across_records(values: torch.Tensor) -> torch.Tensor:
    if len(values) < 2:
        raise ValueError("mismatch destroyer requires at least two records")
    return values.roll(1, dims=0)


def _objectives(
    model: CounterfactualOperatorSetModel,
    operators: torch.Tensor,
    responses: torch.Tensor,
    labels: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module,
    *,
    diagnosis_only: bool,
    mismatch: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    rows = torch.arange(len(operators), device=operators.device)
    first_q, first_f = _gather(operators, first), _gather(responses, first)
    if mismatch:
        first_q = _mismatch_across_records(first_q)
    if diagnosis_only:
        logits = model(first_q, first_f)
        return loss_fn(logits, labels), logits, None, None

    second_q, second_f = _gather(operators, second), _gather(responses, second)
    target_q, target_f = operators[rows, target], responses[rows, target]
    if mismatch:
        second_q = _mismatch_across_records(second_q)
        target_q = _mismatch_across_records(target_q)
    logits, prediction, first_state = model(
        first_q, first_f, target_operator=target_q,
        return_counterfactual=True, return_state=True,
    )
    _, second_state = model(second_q, second_f, return_state=True)
    counterfactual = nn.functional.mse_loss(prediction, target_f)
    invariance = normalized_state_mse(first_state, second_state)
    loss = loss_fn(logits, labels) + COUNTERFACTUAL_WEIGHT * counterfactual + INVARIANCE_WEIGHT * invariance
    return loss, logits, counterfactual, invariance


def _evaluate(
    model: CounterfactualOperatorSetModel,
    data: dict[str, np.ndarray],
    *,
    batch: int,
    diagnosis_only: bool,
    loss_fn: nn.Module,
) -> tuple[np.ndarray, float | None, float | None]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    counterfactuals: list[float] = []
    invariances: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch):
            stop = min(start + batch, len(data["labels"]))
            operators = torch.as_tensor(data["operators"][start:stop], device="cuda")
            responses = torch.as_tensor(data["responses"][start:stop], device="cuda")
            labels = torch.as_tensor(data["labels"][start:stop], device="cuda")
            count = stop - start
            first = torch.arange(3, device="cuda").expand(count, -1)
            second = torch.arange(3, 6, device="cuda").expand(count, -1)
            target = torch.full((count,), 6, dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, logits, counterfactual, invariance = _objectives(
                    model, operators, responses, labels, first, second, target, loss_fn,
                    diagnosis_only=diagnosis_only, mismatch=False,
                )
            probabilities.append(torch.sigmoid(logits).float().cpu())
            if counterfactual is not None:
                counterfactuals.append(float(counterfactual.float().cpu()))
            if invariance is not None:
                invariances.append(float(invariance.float().cpu()))
    return (
        torch.cat(probabilities).numpy(),
        float(np.mean(counterfactuals)) if counterfactuals else None,
        float(np.mean(invariances)) if invariances else None,
    )


def _train_variant(
    *,
    variant: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    cells_root: Path,
    batch: int,
    max_epochs: int,
    patience: int,
    seed: int,
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    weight_decays: tuple[float, ...] = WEIGHT_DECAYS,
) -> None:
    diagnosis_only = variant == "diagnosis_only"
    mismatch = variant == "mismatched_q"
    _seed(seed)
    template = CounterfactualOperatorSetModel(classes=train["labels"].shape[1]).cuda()
    initial_state = copy.deepcopy(template.state_dict())
    del template
    prevalence = torch.as_tensor(train["labels"].mean(0), device="cuda")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    )
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            path = cells_root / f"{variant}_lr{learning_rate:g}_wd{weight_decay:g}"
            if (path / "summary.json").exists():
                continue
            model = CounterfactualOperatorSetModel(classes=train["labels"].shape[1]).cuda()
            model.load_state_dict(initial_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            best_score, best_epoch, stale = -np.inf, 0, 0
            best_state = best_probability = best_mechanism = None
            history = []
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, max_epochs + 1):
                rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
                permutation = rng.permutation(len(train["labels"]))
                losses = []
                model.train()
                for start in range(0, len(permutation), batch):
                    index = permutation[start:start + batch]
                    if mismatch and len(index) < 2:
                        continue
                    operators = torch.as_tensor(train["operators"][index], device="cuda")
                    responses = torch.as_tensor(train["responses"][index], device="cuda")
                    labels = torch.as_tensor(train["labels"][index], device="cuda")
                    generator = torch.Generator(device="cuda")
                    generator.manual_seed(seed * 1_000_000 + epoch * 100_000 + start)
                    first, second, target = sample_disjoint_contexts(
                        len(index), operators.shape[1], generator, operators.device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _, _, _ = _objectives(
                            model, operators, responses, labels, first, second, target, loss_fn,
                            diagnosis_only=diagnosis_only, mismatch=mismatch,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                scheduler.step()
                probability, counterfactual, invariance = _evaluate(
                    model, validation, batch=batch, diagnosis_only=diagnosis_only, loss_fn=loss_fn
                )
                score = float(multilabel_metrics(validation["labels"], probability)["macro_auroc"])
                history.append({
                    "epoch": epoch, "loss": float(np.mean(losses)), "val_macro_auroc": score,
                    "val_counterfactual_mse": counterfactual,
                    "val_normalized_state_mse": invariance,
                })
                if score > best_score + 1e-6:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    best_probability = probability
                    best_mechanism = {
                        "counterfactual_mse": counterfactual,
                        "normalized_state_mse": invariance,
                    }
                else:
                    stale += 1
                    if stale >= patience:
                        break
                print(json.dumps({
                    "variant": variant, "learning_rate": learning_rate,
                    "weight_decay": weight_decay, "epoch": epoch,
                    "best_macro_auroc": round(best_score, 6),
                    "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                }), flush=True)
            assert best_state is not None and best_probability is not None
            path.mkdir(parents=True, exist_ok=True)
            loss_contract = {
                "diagnosis_bce": 1.0,
                "counterfactual_mse": 0.0 if diagnosis_only else COUNTERFACTUAL_WEIGHT,
                "normalized_state_mse": 0.0 if diagnosis_only else INVARIANCE_WEIGHT,
                "mismatch_q_response_pairing": mismatch,
            }
            summary = {
                "variant": variant, "learning_rate": learning_rate, "weight_decay": weight_decay,
                "best_epoch": best_epoch, "epochs_run": len(history),
                "metrics": multilabel_metrics(validation["labels"], best_probability),
                "mechanism": best_mechanism, "loss_contract": loss_contract,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "execution": "shared_host_arrays_minibatched_to_cuda", "history": history,
            }
            torch.save({"variant": variant, "seed": seed, "summary": summary, "state_dict": best_state}, path / "checkpoint.pt")
            np.savez_compressed(
                path / "predictions.npz", probability=best_probability, labels=validation["labels"],
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
    parser.add_argument("--variant", choices=tuple(get_variants_for_paper(9)), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--training-regime", default="full_only", choices=["full_only"])
    args = parser.parse_args()
    manifest = json.loads((args.representations / "manifest.json").read_text())
    if manifest.get("kind") != "paper07_measurement_operator_response_sets" or manifest.get("status") != "complete":
        raise ValueError("Paper 9 requires complete Paper 7 operator-response artifacts")
    train = _load(args.representations / "representation_train.npz")
    validation = _load(args.representations / "representation_val.npz")
    args.output.mkdir(parents=True, exist_ok=True)
    cells = args.output / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")
    _train_variant(
        variant=args.variant, train=train, validation=validation, cells_root=cells,
        batch=args.batch, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed,
        learning_rates=(3e-4,) if args.smoke else LEARNING_RATES,
        weight_decays=(1e-4,) if args.smoke else WEIGHT_DECAYS,
    )


if __name__ == "__main__":
    main()
