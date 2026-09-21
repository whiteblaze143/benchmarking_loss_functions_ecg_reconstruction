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
from repecg.common.models import LocalTokenCrossAttention
from repecg.common.variants import ExperimentVariant
from repecg.paper08_tokens import deterministic_phase_permutations


LEARNING_RATES = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)
REPRESENTATIONS = (
    "continuous", "fine_kmeans", "size_matched_kmeans", "equivalence",
    "random_merge", "frequency_matched_random_merge", "unk_pattern_only",
    "token_without_unk_signal",
)
ROUTINGS = ("dynamic_global", "static", "local_cyclic")
ROUTING_CONTROLS = ("uniform_attention", "phase_agnostic_set", "scrambled_phases", "cnn_matched_control")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact_lock(artifact: Path, lock: dict[str, object]) -> None:
    expected = lock.get("sha256")
    if not isinstance(expected, dict):
        raise ValueError("Paper 8 artifact lock has no file digests")
    actual = {name: _sha256(artifact / name) for name in expected}
    if actual != expected:
        raise RuntimeError("Paper 8 artifact bytes changed after its lock was created")


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
    design: dict[str, object],
    parameter_count: int,
) -> None:
    path = cell["path"]
    assert isinstance(path, Path)
    path.mkdir(parents=True, exist_ok=True)
    best_probability = cell["best_probability"]
    best_state = cell["best_state"]
    if best_probability is None or best_state is None:
        best_probability = np.full_like(val_y, 0.5, dtype=np.float32)
        model = cell["model"]
        assert isinstance(model, nn.Module)
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        metrics = {"macro_auroc": 0.5, "micro_auroc": 0.5}
    else:
        try:
            metrics = multilabel_metrics(val_y, best_probability)
        except Exception:
            metrics = {"macro_auroc": 0.5, "micro_auroc": 0.5}
    summary = {
        "variant": variant,
        "learning_rate": cell["learning_rate"],
        "weight_decay": cell["weight_decay"],
        "best_epoch": cell["best_epoch"],
        "epochs_run": len(cell["history"]),
        "metrics": metrics,
        "peak_vram_bytes": peak_vram_bytes,
        "execution": "single_process_shared_tensor_cuda_streams",
        "factorial_design": design,
        "parameter_count": parameter_count,
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


def _train_variant(
    *,
    variant_obj,

    variant: str,
    training_regime: str = "full_only",
    variant_index: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_ecg_ids: np.ndarray,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    ecg_ids: np.ndarray,
    patient_ids: np.ndarray,
    cells_root: Path,
    batch: int,
    max_epochs: int,
    patience: int,
    seed: int,
    codebook: np.ndarray | None,
    design: dict[str, object],
    artifact: Path,
    artifact_lock: dict[str, object],
) -> None:
    run_seed = seed + variant_index * 100
    _seed(run_seed)
    template = LocalTokenCrossAttention(input_dim=train_x.shape[-1], classes=train_y.shape[-1], variant=variant_obj).cuda()
    if codebook is not None:
        template.set_codebook(torch.from_numpy(codebook).to(template.codebook.device))
    initial_state = copy.deepcopy(template.state_dict())
    parameter_count = sum(parameter.numel() for parameter in template.parameters())
    del template
    prevalence = train_y.mean(dim=0)
    positive_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    cells: list[dict[str, object]] = []
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            path = _cell_path(cells_root, variant, learning_rate, weight_decay)
            if (path / "summary.json").exists():
                continue
            model = LocalTokenCrossAttention(input_dim=train_x.shape[-1], classes=train_y.shape[-1], variant=variant_obj).cuda()
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
        _verify_artifact_lock(artifact, artifact_lock)
        active = [cell for cell in cells if cell["active"]]
        if not active:
            break
        generator = torch.Generator(device="cuda")
        generator.manual_seed(run_seed * 1000 + epoch)
        permutation = torch.randperm(len(train_x), generator=generator, device="cuda")
        epoch_losses: dict[int, list[torch.Tensor]] = {id(cell): [] for cell in active}
        for start in range(0, len(train_x), batch):
            index = permutation[start : start + batch]
            phase_permutation = None
            if variant_obj.mechanism == "scrambled_phases":
                ids = train_ecg_ids[index.cpu().numpy()]
                phase_permutation = torch.from_numpy(
                    deterministic_phase_permutations(ids, run_seed)
                ).to(index.device)
            for cell in active:
                stream = cell["stream"]
                model = cell["model"]
                optimizer = cell["optimizer"]
                loss_fn = cell["loss_fn"]
                assert isinstance(stream, torch.cuda.Stream)
                assert isinstance(model, nn.Module)
                assert isinstance(optimizer, torch.optim.Optimizer)
                assert isinstance(loss_fn, nn.Module)
                with torch.cuda.stream(stream):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(
                            model(train_x[index], phase_permutation=phase_permutation), train_y[index]
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses[id(cell)].append(loss.detach())
            torch.cuda.synchronize()

        probabilities: dict[int, torch.Tensor] = {}
        val_phase_permutation = None
        if variant_obj.mechanism == "scrambled_phases":
            val_phase_permutation = torch.from_numpy(
                deterministic_phase_permutations(ecg_ids, run_seed)
            ).to(val_x.device)
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
                probabilities[id(cell)] = torch.sigmoid(
                    model(val_x, phase_permutation=val_phase_permutation)
                ).float()
        torch.cuda.synchronize()

        for cell in active:
            probability = probabilities[id(cell)].cpu().numpy()
            has_invalid = not np.isfinite(probability).all()
            if has_invalid:
                score = 0.5
            else:
                try:
                    score = float(multilabel_metrics(val_y, probability)["macro_auroc"])
                except Exception:
                    score = -1.0
            raw_loss = torch.stack(epoch_losses[id(cell)]).mean().cpu()
            loss_value = float(raw_loss) if torch.isfinite(raw_loss) else 10.0
            history = cell["history"]
            assert isinstance(history, list)
            history.append({"epoch": epoch, "loss": loss_value, "val_macro_auroc": max(score, 0.0)})
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
                if int(cell["stale"]) >= patience or has_invalid:
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
                        design=design,
                        parameter_count=parameter_count,
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
            design=design,
            parameter_count=parameter_count,
        )


def _categorical_features(token_ids: np.ndarray, token_count: int, *, hide_unknown: bool = False) -> np.ndarray:
    values = np.asarray(token_ids, dtype=np.int64)
    output = np.zeros((*values.shape, token_count), dtype=np.float32)
    known = values >= 0
    output[np.nonzero(known)[0], np.nonzero(known)[1], values[known]] = 1.0
    if not hide_unknown:
        output[~known, token_count - 1] = 1.0
    return output


def _routing_variant(routing: str) -> ExperimentVariant:
    mechanisms = {
        "dynamic_global": "global",
        "static": "static_attention",
        "local_cyclic": "local_banded",
        "uniform_attention": "uniform_attention",
        "phase_agnostic_set": "phase_agnostic",
        "scrambled_phases": "scrambled_phases",
        "cnn_matched_control": "cnn_matched",
    }
    return ExperimentVariant(mechanism=mechanisms[routing])


def _representation_features(
    payload: dict[str, np.ndarray], certificate: dict[str, np.ndarray], representation: str,
    random_control: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if representation == "continuous":
        return payload["continuous"].astype(np.float32, copy=False), {"input": "continuous_phase_kme"}
    if representation == "fine_kmeans":
        count = len(certificate["base_centers"])
        return _categorical_features(payload["fine_kmeans_ids"], count + 1), {"token_count": count}
    if representation == "size_matched_kmeans":
        count = len(certificate["token_prototypes"])
        return _categorical_features(payload["size_kmeans_ids"], count + 1), {"token_count": count}
    if representation in {"equivalence", "token_without_unk_signal", "unk_pattern_only"}:
        tokens = payload["equivalence_token_ids"]
        count = len(certificate["token_prototypes"])
        if representation == "unk_pattern_only":
            return (tokens < 0).astype(np.float32)[..., None], {"token_count": 0, "unknown_only": True}
        return _categorical_features(
            tokens, count + 1, hide_unknown=representation == "token_without_unk_signal"
        ), {"token_count": count, "unknown_hidden": representation == "token_without_unk_signal"}
    if representation in {"random_merge", "frequency_matched_random_merge"}:
        maps_key = (
            "random_merge_maps" if representation == "random_merge"
            else "frequency_matched_random_merge_maps"
        )
        maps = certificate[maps_key]
        if not 0 <= random_control < len(maps):
            raise ValueError(f"random control must be in 0..{len(maps) - 1}")
        mapped = maps[random_control][payload["fine_kmeans_ids"]]
        count = len(certificate["token_prototypes"])
        return _categorical_features(mapped, count + 1), {
            "token_count": count, "random_control": random_control,
            "random_control_family": representation,
        }
    raise ValueError(f"unsupported representation: {representation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--routing", choices=(*ROUTINGS, *ROUTING_CONTROLS), required=True)
    parser.add_argument("--random-control", type=int, default=0)
    parser.add_argument("--training-regime", type=str, default="full_only", choices=["full_only"])

    args = parser.parse_args()

    manifest = json.loads((args.representations / "manifest.json").read_text())
    lock_path = args.representations / "artifact_lock.json"
    if not lock_path.is_file():
        raise ValueError("Paper 8 requires a mechanically audited artifact_lock.json before training")
    artifact_lock = json.loads(lock_path.read_text())
    _verify_artifact_lock(args.representations, artifact_lock)
    if (
        manifest.get("kind") != "paper08_equivalence_token_representations"
        or manifest.get("status") != "complete"
        or manifest.get("audit", {}).get("passed") is not True
        or manifest.get("fold_firewall", {}).get("fold8_used_for_fit_or_selection") is not False
    ):
        raise ValueError("Paper 8 requires a complete audited equivalence-token artifact")
    args.output.mkdir(parents=True, exist_ok=True)
    cells_root = args.output / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")
    with np.load(args.representations / "representation_train.npz") as item:
        train = {name: np.asarray(item[name]) for name in item.files}
    with np.load(args.representations / "representation_selection.npz") as item:
        validation = {name: np.asarray(item[name]) for name in item.files}
    with np.load(args.representations / "vocabulary_certificate.npz") as item:
        certificate = {name: np.asarray(item[name]) for name in item.files}
    train_y = torch.from_numpy(train["labels"]).cuda()
    val_y = validation["labels"]
    train_features, train_design = _representation_features(
        train, certificate, args.representation, args.random_control
    )
    validation_features, validation_design = _representation_features(
        validation, certificate, args.representation, args.random_control
    )
    if train_design != validation_design:
        raise RuntimeError("representation construction differs between training and selection")
    variant_obj = _routing_variant(args.routing)
    variant = f"{args.representation}__{args.routing}"
    if "random_control" in train_design:
        variant += f"__control{args.random_control:02d}"
    design = {
        "representation": args.representation,
        "routing": args.routing,
        "selection_fold": 7,
        "pseudo_test_touched": False,
        "primary_factorial_cell": args.representation in {
            "continuous", "fine_kmeans", "size_matched_kmeans", "equivalence",
            "random_merge", "frequency_matched_random_merge",
        } and args.routing in ROUTINGS,
        **train_design,
        "artifact_lock_sha256": artifact_lock["sha256"]["vocabulary_certificate.npz"],
    }
    train_x = torch.from_numpy(train_features).cuda()
    val_x = torch.from_numpy(validation_features).cuda()
    _train_variant(
        variant_obj=variant_obj,
        variant=variant,
        training_regime=args.training_regime,
        variant_index=0,
        train_x=train_x,
        train_y=train_y,
        train_ecg_ids=train["ecg_ids"],
        val_x=val_x,
        val_y=val_y,
        ecg_ids=validation["ecg_ids"],
        patient_ids=validation["patient_ids"],
        cells_root=cells_root,
        batch=args.batch,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        codebook=None,
        design=design,
        artifact=args.representations,
        artifact_lock=artifact_lock,
    )
    del train_x, val_x
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
