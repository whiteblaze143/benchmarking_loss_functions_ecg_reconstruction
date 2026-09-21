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
from repecg.common.models import InvariantMechanismDiscoveryModel
from repecg.common.variants import get_variants_for_paper


REPRESENTATION_VARIANTS = ("kernel", "moments", "gaussian", "linear")
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _cell_path(root: Path, variant: str, learning_rate: float, weight_decay: float) -> Path:
    return root / f"{variant}_lr{learning_rate:g}_wd{weight_decay:g}"


def _save_cell(
    cell: dict[str, object],
    *,
    variant: str,
    training_regime: str = "full_only",
    seed: int,
    val_y: np.ndarray,
    ecg_ids: np.ndarray,
    patient_ids: np.ndarray,
    peak_vram_bytes: int,
    mmd_lambda: float = 0.0,
) -> None:
    path = cell["path"]
    assert isinstance(path, Path)
    path.mkdir(parents=True, exist_ok=True)
    best_probability = cell["best_probability"]
    best_state = cell["best_state"]
    assert isinstance(best_probability, np.ndarray) and isinstance(best_state, dict)
    summary = {
        "variant": variant,
        "mmd_lambda": mmd_lambda,
        "learning_rate": cell["learning_rate"],
        "weight_decay": cell["weight_decay"],
        "best_epoch": cell["best_epoch"],
        "epochs_run": len(cell["history"]),
        "metrics": multilabel_metrics(val_y, best_probability),
        "peak_vram_bytes": peak_vram_bytes,
        "execution": "single_process_shared_tensor_cuda_streams",
        "history": cell["history"],
    }
    torch.save(
        {"variant": variant, "seed": seed, "summary": summary, "state_dict": best_state},
        path / "checkpoint.pt",
    )
    np.savez_compressed(
        path / "predictions.npz",
        probability=best_probability,
        labels=val_y,
        ecg_ids=ecg_ids,
        patient_ids=patient_ids,
    )
    _atomic_json(path / "summary.json", summary)


def _generate_balanced_batches(
    env: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    envs = torch.unique(env, sorted=True)
    if len(envs) < 2:
        raise RuntimeError(
            "Paper 14 MMD training requires at least two environments per batch"
        )
    per_env = batch_size // len(envs)
    if per_env < 2:
        raise RuntimeError(
            f"Batch size {batch_size} is too small for {len(envs)} environments (needs >= 2 samples per env)"
        )

    env_indices: dict[int, torch.Tensor] = {}
    for e in envs:
        idx = (env == e).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx), generator=generator, device=env.device)
        env_indices[e.item()] = idx[perm]

    num_batches = min(len(idx) // per_env for idx in env_indices.values())
    if num_batches == 0:
        raise RuntimeError("Not enough samples per environment to form a single balanced batch")

    batches = []
    for b in range(num_batches):
        parts = [env_indices[e.item()][b * per_env : (b + 1) * per_env] for e in envs]
        batches.append(torch.cat(parts))
    return batches


def _train_variant(
    *,
    variant_obj,
    variant: str,
    training_regime: str = "full_only",
    variant_index: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_env: torch.Tensor | None = None,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    ecg_ids: np.ndarray,
    patient_ids: np.ndarray,
    cells_root: Path,
    batch: int,
    max_epochs: int,
    patience: int,
    seed: int,
    mmd_lambda: float = 1.0,
    mmd_c2: float = 1.0,
) -> None:
    if variant_obj.use_mmd:
        if train_env is None:
            raise RuntimeError(
                "Paper 14 MMD training requested, but no 'environment' was provided."
            )
        unique_envs = torch.unique(train_env)
        if len(unique_envs) < 2:
            raise RuntimeError(
                f"Paper 14 MMD training requires at least two environments, found {len(unique_envs)}"
            )

    run_seed = seed
    _seed(run_seed)
    template = InvariantMechanismDiscoveryModel(input_dim=train_x.shape[-1], classes=train_y.shape[-1], variant=variant_obj).cuda()
    initial_state = copy.deepcopy(template.state_dict())
    del template
    prevalence = train_y.mean(dim=0)
    positive_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    cells: list[dict[str, object]] = []
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            path = _cell_path(cells_root, variant, learning_rate, weight_decay)
            if (path / "summary.json").exists():
                continue
            model = InvariantMechanismDiscoveryModel(input_dim=train_x.shape[-1], classes=train_y.shape[-1], variant=variant_obj).cuda()
            model.load_state_dict(initial_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            cells.append(
                {
                    "path": path,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs),
                    "loss_fn": nn.BCEWithLogitsLoss(pos_weight=positive_weight),
                    "stream": torch.cuda.Stream(),
                    "best_score": -np.inf,
                    "best_epoch": 0,
                    "best_state": None,
                    "best_probability": None,
                    "stale": 0,
                    "history": [],
                    "active": True,
                    "saved": False,
                }
            )
    if not cells:
        print(json.dumps({"variant": variant, "status": "already_complete"}), flush=True)
        return

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        active = [cell for cell in cells if cell["active"]]
        if not active:
            break
        generator = torch.Generator(device="cuda")
        generator.manual_seed(run_seed * 1000 + epoch)

        if train_env is not None:
            batches = _generate_balanced_batches(train_env, batch, generator)
        else:
            permutation = torch.randperm(len(train_x), generator=generator, device="cuda")
            batches = [permutation[start : start + batch] for start in range(0, len(train_x), batch)]

        epoch_losses: dict[int, list[torch.Tensor]] = {id(cell): [] for cell in active}
        epoch_bce: dict[int, list[torch.Tensor]] = {id(cell): [] for cell in active}
        epoch_mmd: dict[int, list[torch.Tensor]] = {id(cell): [] for cell in active}
        epoch_var_z: dict[int, list[float]] = {id(cell): [] for cell in active}

        for batch_idx, index in enumerate(batches):
            for cell in active:
                stream = cell["stream"]
                model = cell["model"]
                optimizer = cell["optimizer"]
                loss_fn = cell["loss_fn"]
                assert isinstance(stream, torch.cuda.Stream)
                assert isinstance(model, InvariantMechanismDiscoveryModel)
                assert isinstance(optimizer, torch.optim.Optimizer)
                assert isinstance(loss_fn, nn.Module)
                with torch.cuda.stream(stream):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, z = model.forward_with_representation(train_x[index])
                        bce_loss = loss_fn(logits, train_y[index])

                        if variant_obj.use_mmd:
                            assert train_env is not None
                            batch_env = train_env[index]
                            if torch.unique(batch_env).numel() < 2:
                                raise RuntimeError("Paper 14 MMD batch contains fewer than two environments")
                            mmd_loss = model.compute_mmd_penalty(z.float(), batch_env, c2=mmd_c2)
                            loss = bce_loss + mmd_lambda * mmd_loss
                        else:
                            mmd_loss = z.new_zeros(())
                            loss = bce_loss

                    if epoch == 1 and batch_idx == 0 and variant_obj.use_mmd and float(mmd_loss.detach()) == 0.0:
                        raise RuntimeError(
                            "Paper 14 MMD is exactly zero on the first heterogeneous batch; "
                            "check environment IDs and representation routing"
                        )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    with torch.no_grad():
                        vz = float(z.float().var(dim=0).mean().cpu())

                    epoch_losses[id(cell)].append(loss.detach())
                    epoch_bce[id(cell)].append(bce_loss.detach())
                    epoch_mmd[id(cell)].append(mmd_loss.detach())
                    epoch_var_z[id(cell)].append(vz)
            torch.cuda.synchronize()

        probabilities: dict[int, torch.Tensor] = {}
        for cell in active:
            scheduler = cell["scheduler"]
            stream = cell["stream"]
            model = cell["model"]
            assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
            assert isinstance(stream, torch.cuda.Stream)
            assert isinstance(model, nn.Module)
            scheduler.step()
            with torch.cuda.stream(stream), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model.eval()
                probabilities[id(cell)] = torch.sigmoid(model(val_x)).float()
        torch.cuda.synchronize()

        for cell in active:
            probability = probabilities[id(cell)].cpu().numpy()
            score = float(multilabel_metrics(val_y, probability)["macro_auroc"])
            loss_total_val = float(torch.stack(epoch_losses[id(cell)]).mean().cpu())
            loss_bce_val = float(torch.stack(epoch_bce[id(cell)]).mean().cpu())
            loss_mmd_val = float(torch.stack(epoch_mmd[id(cell)]).mean().cpu())
            var_z_val = float(np.mean(epoch_var_z[id(cell)]))
            history = cell["history"]
            assert isinstance(history, list)
            history.append({
                "epoch": epoch,
                "loss_total": loss_total_val,
                "loss_bce": loss_bce_val,
                "loss_mmd": loss_mmd_val,
                "weighted_mmd": mmd_lambda * loss_mmd_val,
                "var_z": var_z_val,
                "val_macro_auroc": score,
            })
            if score > float(cell["best_score"]) + 1e-6:
                model = cell["model"]
                assert isinstance(model, nn.Module)
                cell["best_score"] = score
                cell["best_epoch"] = epoch
                cell["best_state"] = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                cell["best_probability"] = probability
                cell["stale"] = 0
            else:
                cell["stale"] = int(cell["stale"]) + 1
                if int(cell["stale"]) >= patience:
                    cell["active"] = False
                    _save_cell(
                        cell,
                        variant=variant,
                        training_regime=training_regime,
                        seed=seed,
                        val_y=val_y,
                        ecg_ids=ecg_ids,
                        patient_ids=patient_ids,
                        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
                        mmd_lambda=mmd_lambda,
                    )
                    cell["saved"] = True
        print(
            json.dumps(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "active_cells": sum(bool(cell["active"]) for cell in cells),
                    "scores": [round(float(cell["best_score"]), 6) for cell in cells],
                    "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                }
            ),
            flush=True,
        )

    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    for cell in cells:
        if cell["saved"]:
            continue
        _save_cell(
            cell,
            variant=variant,
            training_regime=training_regime,
            seed=seed,
            val_y=val_y,
            ecg_ids=ecg_ids,
            patient_ids=patient_ids,
            peak_vram_bytes=peak_vram_bytes,
            mmd_lambda=mmd_lambda,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--training-regime", type=str, default="full_only", choices=["full_only"])
    parser.add_argument("--mmd-lambda", type=float, default=1.0)
    parser.add_argument("--mmd-c2", type=float, default=1.0)

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cells_root = args.output / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")
    with np.load(args.representations / "representation_train.npz") as item:
        train = {name: np.asarray(item[name]) for name in item.files}
    with np.load(args.representations / "representation_val.npz") as item:
        validation = {name: np.asarray(item[name]) for name in item.files}
    train_y = torch.from_numpy(train["labels"]).cuda()
    val_y = validation["labels"]
    variant_registry = get_variants_for_paper(14)
    if args.variant not in variant_registry:
        raise ValueError(f"Variant {args.variant} not found.")
    variant_obj = variant_registry[args.variant]
    
    if "lambda_" in args.variant:
        args.mmd_lambda = float(args.variant.split("lambda_")[1])
    elif not variant_obj.use_mmd:
        args.mmd_lambda = 0.0
    
    rep_key = variant_obj.representation if variant_obj.representation != "full" else REPRESENTATION_VARIANTS[0]
    if rep_key not in train:
        rep_key = REPRESENTATION_VARIANTS[0]
        
    train_x = torch.from_numpy(train[rep_key]).cuda()
    val_x = torch.from_numpy(validation[rep_key]).cuda()

    train_env = None
    for env_key in ("environment", "environments", "device"):
        if env_key in train:
            train_env = torch.from_numpy(train[env_key]).long().cuda()
            break

    if variant_obj.use_mmd and train_env is None:
        raise RuntimeError(
            "Paper 14 MMD training requested, but no 'environment' found in training representations."
        )

    _train_variant(
        variant_obj=variant_obj,
        variant=args.variant,
        training_regime=args.training_regime,
        variant_index=0,
        train_x=train_x,
        train_y=train_y,
        train_env=train_env,
        val_x=val_x,
        val_y=val_y,
        ecg_ids=validation["ecg_ids"],
        patient_ids=validation["patient_ids"],
        cells_root=cells_root,
        batch=args.batch,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        mmd_lambda=args.mmd_lambda,
        mmd_c2=args.mmd_c2,
    )
    del train_x, val_x
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
