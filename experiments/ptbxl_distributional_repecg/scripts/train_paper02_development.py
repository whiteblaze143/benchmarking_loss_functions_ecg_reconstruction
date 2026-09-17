from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from repecg.common.metrics import multilabel_metrics
from repecg.common.models import PhaseCNN


VARIANTS = ("kernel", "moments", "gaussian")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _predict(model: nn.Module, features: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(features), batch):
            output.append(torch.sigmoid(model(features[start : start + batch])).float().cpu())
    return torch.cat(output).numpy()


def _train_one(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    batch: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor], np.ndarray]:
    _seed(seed)
    model = PhaseCNN(train_x.shape[-1]).cuda()
    prevalence = train_y.mean(dim=0)
    positive_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_probability: np.ndarray | None = None
    stale = 0
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), device=train_x.device)
        losses = []
        for start in range(0, len(train_x), batch):
            index = permutation[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(train_x[index])
                loss = loss_fn(logits, train_y[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        probability = _predict(model, val_x, batch * 2)
        score = float(multilabel_metrics(val_y, probability)["macro_auroc"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_auroc": score})
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_probability = probability
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None and best_probability is not None
    summary = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "metrics": multilabel_metrics(val_y, best_probability),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "history": history,
    }
    return summary, best_state, best_probability


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--cell-output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.representations / "manifest.json").read_text())
    if not manifest["audit"]["passed"]:
        raise RuntimeError("Nyström fidelity gate is not passed")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")
    with np.load(args.representations / "representation_train.npz") as item:
        train = {name: np.asarray(item[name]) for name in item.files}
    with np.load(args.representations / "representation_val.npz") as item:
        validation = {name: np.asarray(item[name]) for name in item.files}
    train_y = torch.from_numpy(train["labels"]).cuda()
    val_y = validation["labels"]
    if args.variant is not None:
        if args.learning_rate is None or args.weight_decay is None or args.cell_output is None:
            parser.error("cell mode requires --learning-rate, --weight-decay, and --cell-output")
        args.cell_output.mkdir(parents=True, exist_ok=True)
        variant_index = VARIANTS.index(args.variant)
        summary, state, probability = _train_one(
            torch.from_numpy(train[args.variant]).cuda(),
            train_y,
            torch.from_numpy(validation[args.variant]).cuda(),
            val_y,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch=args.batch,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed + variant_index * 100,
        )
        torch.save(
            {"variant": args.variant, "seed": args.seed, "summary": summary, "state_dict": state},
            args.cell_output / "checkpoint.pt",
        )
        np.savez_compressed(
            args.cell_output / "predictions.npz",
            probability=probability,
            labels=val_y,
            ecg_ids=validation["ecg_ids"],
            patient_ids=validation["patient_ids"],
        )
        _atomic_json(args.cell_output / "summary.json", {"variant": args.variant, **summary})
        print(json.dumps({"variant": args.variant, "macro_auroc": summary["metrics"]["macro_auroc"], "cell": str(args.cell_output)}))
        return
    searches: dict[str, list[dict[str, object]]] = {}
    for variant_index, variant in enumerate(VARIANTS):
        train_x = torch.from_numpy(train[variant]).cuda()
        val_x = torch.from_numpy(validation[variant]).cuda()
        runs = []
        best: tuple[float, dict[str, object], dict[str, torch.Tensor], np.ndarray] | None = None
        for learning_rate in (1e-4, 3e-4, 1e-3):
            for weight_decay in (1e-5, 1e-4, 1e-3):
                run_seed = args.seed + variant_index * 100
                summary, state, probability = _train_one(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch=args.batch,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    seed=run_seed,
                )
                runs.append(summary)
                score = float(summary["metrics"]["macro_auroc"])
                if best is None or score > best[0]:
                    best = (score, copy.deepcopy(summary), state, probability.copy())
                _atomic_json(args.output / "search_state.json", {**searches, variant: runs})
        assert best is not None
        score, summary, state, probability = best
        torch.save({"variant": variant, "seed": args.seed, "summary": summary, "state_dict": state}, args.output / f"{variant}_best.pt")
        columns: dict[str, np.ndarray] = {
            "ecg_id": validation["ecg_ids"],
            "patient_id": validation["patient_ids"],
        }
        for index, name in enumerate(("NORM", "MI", "STTC", "CD", "HYP")):
            columns[f"y_{name}"] = val_y[:, index]
            columns[f"p_{name}"] = probability[:, index]
        pd.DataFrame(columns).to_csv(args.output / f"predictions_{variant}.csv", index=False)
        searches[variant] = runs
        _atomic_json(args.output / f"metrics_{variant}.json", summary)
        del train_x, val_x
        torch.cuda.empty_cache()
    _atomic_json(
        args.output / "manifest.json",
        {
            "kind": "paper02_development_hyperparameter_search",
            "seed": args.seed,
            "variants": list(VARIANTS),
            "folds": {"train": [1, 2, 3, 4, 5, 6, 7], "validation": [8]},
            "selection_metric": "macro_auroc",
        },
    )
    print(json.dumps({variant: json.loads((args.output / f"metrics_{variant}.json").read_text())["metrics"]["macro_auroc"] for variant in VARIANTS}, sort_keys=True))


if __name__ == "__main__":
    main()
