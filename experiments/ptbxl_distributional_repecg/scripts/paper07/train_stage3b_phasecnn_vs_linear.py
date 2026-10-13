#!/usr/bin/env python3
"""Stage 3B: Matched Two-Arm Retraining: PhaseCNN vs Linear Response Bottleneck.

Arm A: Current PhaseCNN response encoder (r_j in R^{16x128} -> e_r in R^{128}).
Arm B: Flatten(2048) -> Linear(2048, 128) and stop there (no GELU, no LayerNorm, no dropout).

Downstream architecture, MLP-q operator branch, training contract, minibatches,
and initialization of common modules are strictly matched.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from repecg.common.metrics import multilabel_metrics
from repecg.paper07_operator import OperatorSetModel


LEARNING_RATES = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONFIGURATIONS = {
    "Q8": (0, 1, 6, 7, 8, 9, 10, 11),
    "I": (0,),
    "II": (1,),
    "I+II": (0, 1),
    "I+II+V1": (0, 1, 6),
    "I+II+V5": (0, 1, 10),
    "S6 limb": (0, 1, 2, 3, 4, 5),
    "S6 precordial": (6, 7, 8, 9, 10, 11),
}


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _macro_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    values = []
    for column in range(labels.shape[1]):
        if len(np.unique(labels[:, column])) == 2:
            values.append(roc_auc_score(labels[:, column], probability[:, column]))
    if len(values) != labels.shape[1]:
        raise ValueError("each superclass must have both outcomes")
    return float(np.mean(values))


def sample_context_target_indices(
    batch_size: int, candidates: int, generator: torch.Generator, device: torch.device,
    *, max_context_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = candidates - 1 if max_context_size is None else min(max_context_size, candidates)
    if maximum < 1:
        raise ValueError("at least one context pair is required")
    context_size = int(torch.randint(1, maximum + 1, (), generator=generator, device=device))
    order = torch.rand((batch_size, candidates), generator=generator, device=device).argsort(dim=1)
    target = order[:, context_size] if context_size < candidates else torch.zeros(batch_size, dtype=torch.long, device=device)
    return order[:, :context_size], target


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(values), device=values.device).unsqueeze(1)
    return values[rows, indices]


def _load_data(path: Path) -> dict[str, np.ndarray]:
    required = ["operators", "responses", "labels", "ecg_ids", "patient_ids"]
    with np.load(path) as item:
        missing = set(required).difference(item.files)
        if missing:
            raise ValueError(f"missing Paper 7 arrays: {sorted(missing)}")
        return {name: np.asarray(item[name]) for name in required}


def create_initial_states(num_classes: int, seed: int = 42) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Initialize both architectures and strictly match all shape-compatible downstream modules."""
    _seed(seed)
    phase_model = OperatorSetModel(classes=num_classes, operator_mode="continuous", response_mode="phase_cnn")

    _seed(seed)
    linear_model = OperatorSetModel(classes=num_classes, operator_mode="continuous", response_mode="linear")

    # Copy all shape-compatible downstream modules from phase_model to linear_model
    linear_model.operator.load_state_dict(phase_model.operator.state_dict())
    linear_model.set_encoder.load_state_dict(phase_model.set_encoder.state_dict())
    linear_model.pool_query.data.copy_(phase_model.pool_query.data)
    linear_model.pool.load_state_dict(phase_model.pool.state_dict())
    linear_model.latent.load_state_dict(phase_model.latent.state_dict())
    linear_model.head.load_state_dict(phase_model.head.state_dict())
    linear_model.decoder.load_state_dict(phase_model.decoder.state_dict())

    # Assert downstream equality
    for name, param in phase_model.named_parameters():
        if not name.startswith("response."):
            assert torch.equal(param, dict(linear_model.named_parameters())[name]), f"Common module mismatch: {name}"
        else:
            # Response module should differ in architecture
            assert name not in dict(linear_model.named_parameters()) or not torch.equal(param, dict(linear_model.named_parameters())[name])

    return (
        {k: v.detach().cpu().clone() for k, v in phase_model.state_dict().items()},
        {k: v.detach().cpu().clone() for k, v in linear_model.state_dict().items()},
    )


def _train_step(
    model: OperatorSetModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    labels: torch.Tensor,
    operators: torch.Tensor,
    responses: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        context_operators = _gather(operators, context)
        context_responses = _gather(responses, context)
        logits = model(context_operators, context_responses)
        loss = loss_fn(logits, labels)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(float("nan"), device=loss.device)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
        optimizer.step()
    return loss.detach()


def _evaluate_validation(
    model: OperatorSetModel,
    data: dict[str, np.ndarray],
    batch_size: int = 64,
    evaluation_context_size: int = 6,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch_size):
            stop = min(start + batch_size, len(data["labels"]))
            operators = torch.as_tensor(data["operators"][start:stop], device="cuda")
            responses = torch.as_tensor(data["responses"][start:stop], device="cuda")
            context = torch.arange(evaluation_context_size, device="cuda").expand(stop - start, -1)
            context_operators = _gather(operators, context)
            context_responses = _gather(responses, context)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(context_operators, context_responses)
            probabilities.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(probabilities).numpy()


def _patient_split(patient_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    patients = np.unique(patient_ids)
    order = sorted(
        range(len(patients)),
        key=lambda index: hashlib.sha256(f"p07-stage3b-probe:{seed}:{int(patients[index])}".encode()).digest(),
    )
    selected = set(patients[np.asarray(order[:max(1, round(len(order) * 0.2))])].tolist())
    val_mask = np.fromiter((value in selected for value in patient_ids), dtype=bool, count=len(patient_ids))
    if not val_mask.any() or val_mask.all():
        raise ValueError("invalid patient split")
    return ~val_mask, val_mask


def _stage_features(model: OperatorSetModel, operators: torch.Tensor, responses: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, count = operators.shape[:2]
    q = model.encode_operator(operators, None)
    phase = model.response(responses.reshape(batch * count, *responses.shape[2:])).reshape(batch, count, 128)
    token = torch.cat((q, phase), dim=-1)
    contextual = model.set_encoder(token)
    query = model.pool_query.expand(batch, -1, -1)
    pooled, _ = model.pool(query, contextual, contextual)
    latent = model.latent(pooled[:, 0])
    logits = model.head(latent)
    return {
        "E0_mean_phase_kme": responses.mean(dim=1).flatten(1),
        "E1_response_embedding": phase.mean(dim=1),
        "E4_contextual_token": contextual.mean(dim=1),
        "E5_attention_pooled": pooled[:, 0],
        "E6_latent": latent,
        "E7_frozen_logits": logits,
    }


def _extract_features(
    model: OperatorSetModel,
    responses: np.ndarray,
    operators: np.ndarray,
    indices: tuple[int, ...],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    result: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for start in range(0, len(responses), batch_size):
            stop = min(start + batch_size, len(responses))
            sub_res = torch.as_tensor(np.asarray(responses[start:stop, list(indices)]), device=device)
            sub_op = torch.as_tensor(np.asarray(operators[list(indices)]), device=device).expand(stop - start, -1, -1)
            features = _stage_features(model, sub_op, sub_res)
            for k, v in features.items():
                result.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(chunks).astype(np.float32) for k, chunks in result.items()}


def _fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, select_x: np.ndarray, select_y: np.ndarray,
    c_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> tuple[StandardScaler, Ridge, float]:
    candidates = []
    for c_val in c_values:
        scaler = StandardScaler().fit(train_x)
        probe = Ridge(alpha=1.0 / c_val).fit(scaler.transform(train_x), train_y)
        candidates.append((c_val, _macro_auc(select_y, probe.predict(scaler.transform(select_x)))))
    best_c = max(candidates, key=lambda it: (it[1], -it[0]))[0]
    scaler = StandardScaler().fit(np.concatenate((train_x, select_x)))
    probe = Ridge(alpha=1.0 / best_c).fit(
        scaler.transform(np.concatenate((train_x, select_x))),
        np.concatenate((train_y, select_y)),
    )
    return scaler, probe, float(best_c)


def paired_patient_bootstrap(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Compute paired patient-clustered bootstrap for delta = AUROC(B) - AUROC(A)."""
    patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    
    # Precompute patient to row indices map
    patient_to_indices = {p: np.where(patient_ids == p)[0] for p in patients}
    
    deltas = []
    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_indices = np.concatenate([patient_to_indices[p] for p in sampled_patients])
        
        y_samp = labels[sampled_indices]
        # Ensure all classes have both 0 and 1
        valid = True
        for col in range(y_samp.shape[1]):
            if len(np.unique(y_samp[:, col])) < 2:
                valid = False
                break
        if not valid:
            continue
            
        auc_a = _macro_auc(y_samp, probs_a[sampled_indices])
        auc_b = _macro_auc(y_samp, probs_b[sampled_indices])
        deltas.append(auc_b - auc_a)
        
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    return ci_lower, ci_upper, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step for end-to-end pipeline verification")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cells_root = args.output / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)

    # 1. Load training and validation representations
    print("Loading Paper 7 representations...", flush=True)
    train_data = _load_data(args.representations / "representation_train.npz")
    val_data = _load_data(args.representations / "representation_val.npz")
    num_classes = train_data["labels"].shape[1]

    # Positive class weighting for BCE
    prevalence = torch.as_tensor(train_data["labels"].mean(axis=0), device="cuda")
    positive_weight = ((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    # 2. Fairness initialization: shared downstream weights
    print("Initializing PhaseCNN and Linear arm initial templates with strictly matched downstream modules...", flush=True)
    phase_init_state, linear_init_state = create_initial_states(num_classes, seed=args.seed)

    # 3. Create grid cells for both arms
    arms = ("phase_cnn", "linear")
    cells = []
    for arm in arms:
        init_state = phase_init_state if arm == "phase_cnn" else linear_init_state
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                cell_name = f"{arm}_lr{lr:g}_wd{wd:g}"
                cell_dir = cells_root / cell_name
                cell_dir.mkdir(parents=True, exist_ok=True)
                
                model = OperatorSetModel(classes=num_classes, operator_mode="continuous", response_mode=arm).cuda()
                model.load_state_dict(init_state)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
                
                cells.append({
                    "name": cell_name,
                    "arm": arm,
                    "path": cell_dir,
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "best_score": -np.inf,
                    "best_epoch": 0,
                    "best_state": None,
                    "best_probability": None,
                    "stale": 0,
                    "history": [],
                    "active": True,
                })

    print(f"Created {len(cells)} cells across 2 arms ({len(cells)//2} per arm). Starting matched retraining...", flush=True)

    # 4. Matched training loop: identical minibatches and sampled contexts per step
    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")

    for epoch in range(1, args.max_epochs + 1):
        active = [c for c in cells if c["active"]]
        if not active:
            print(f"All cells early-stopped by epoch {epoch}.", flush=True)
            break

        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        permutation = rng.permutation(len(train_data["labels"]))
        epoch_losses = {id(c): [] for c in active}

        for start in range(0, len(permutation), args.batch):
            index = permutation[start:start + args.batch]
            operators = torch.as_tensor(train_data["operators"][index], device="cuda")
            responses = torch.as_tensor(train_data["responses"][index], device="cuda")
            labels = torch.as_tensor(train_data["labels"][index], device="cuda")

            generator = torch.Generator(device="cuda")
            generator.manual_seed(args.seed * 1_000_000 + epoch * 100_000 + start)
            context, _ = sample_context_target_indices(
                len(index), operators.shape[1], generator, operators.device,
                max_context_size=6,
            )

            # Feed the exact same minibatch and context to every active cell in both arms
            for cell in active:
                loss = _train_step(
                    cell["model"], cell["optimizer"], loss_fn,
                    labels, operators, responses, context,
                )
                epoch_losses[id(cell)].append(float(loss.detach().cpu()))
            if args.dry_run:
                break

        # Epoch validation evaluation
        for cell in active:
            cell["scheduler"].step()
            prob = _evaluate_validation(cell["model"], val_data, batch_size=args.batch, evaluation_context_size=6)
            if np.isnan(prob).any() or np.isinf(prob).any():
                score = -1.0
            else:
                try:
                    score = float(multilabel_metrics(val_data["labels"], prob)["macro_auroc"])
                except Exception:
                    score = -1.0

            losses = [l for l in epoch_losses[id(cell)] if not (np.isnan(l) or np.isinf(l))]
            cell["history"].append({
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "val_macro_auroc": max(score, 0.0),
            })

            if score > float(cell["best_score"]) + 1e-6:
                cell.update({
                    "best_score": score,
                    "best_epoch": epoch,
                    "best_state": {k: v.detach().cpu().clone() for k, v in cell["model"].state_dict().items()},
                    "best_probability": prob,
                    "stale": 0,
                })
            else:
                cell["stale"] += 1
                if cell["stale"] >= args.patience or score < 0:
                    cell["active"] = False

        # Status logging
        scores_by_arm = {
            "phase_cnn": [round(float(c["best_score"]), 6) for c in cells if c["arm"] == "phase_cnn"],
            "linear": [round(float(c["best_score"]), 6) for c in cells if c["arm"] == "linear"],
        }
        print(json.dumps({
            "epoch": epoch,
            "active_cells": len(active),
            "best_scores": scores_by_arm,
            "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }), flush=True)

        if args.dry_run:
            print("Dry run completed 1 step. Exiting epoch loop.", flush=True)
            break

    # 5. Save all cell checkpoints and summaries
    print("\nTraining completed. Saving cell checkpoints...", flush=True)
    for cell in cells:
        summary = {
            "name": cell["name"],
            "arm": cell["arm"],
            "learning_rate": cell["learning_rate"],
            "weight_decay": cell["weight_decay"],
            "best_epoch": cell["best_epoch"],
            "best_val_macro_auroc": cell["best_score"],
            "metrics": multilabel_metrics(val_data["labels"], cell["best_probability"]) if cell["best_probability"] is not None else {},
            "history": cell["history"],
        }
        torch.save({
            "variant": cell["name"],
            "arm": cell["arm"],
            "response_mode": cell["arm"],
            "seed": args.seed,
            "summary": summary,
            "state_dict": cell["best_state"],
        }, cell["path"] / "checkpoint.pt")
        _atomic_json(cell["path"] / "summary.json", summary)

    # 6. Model selection: Validation macro AUROC ONLY
    best_phase_cell = max([c for c in cells if c["arm"] == "phase_cnn"], key=lambda c: c["best_score"])
    best_linear_cell = max([c for c in cells if c["arm"] == "linear"], key=lambda c: c["best_score"])

    print("\n=== Model Selection (Validation Macro AUROC) ===", flush=True)
    print(f"Selected PhaseCNN Cell: {best_phase_cell['name']} (Val Macro AUROC: {best_phase_cell['best_score']:.6f}, Epoch {best_phase_cell['best_epoch']})", flush=True)
    print(f"Selected Linear Cell:   {best_linear_cell['name']} (Val Macro AUROC: {best_linear_cell['best_score']:.6f}, Epoch {best_linear_cell['best_epoch']})", flush=True)

    # 7. Permanent freezing of selected models
    device = torch.device("cuda")
    phase_model = OperatorSetModel(classes=num_classes, operator_mode="continuous", response_mode="phase_cnn").to(device)
    phase_model.load_state_dict(best_phase_cell["best_state"])
    phase_model.eval()

    linear_model = OperatorSetModel(classes=num_classes, operator_mode="continuous", response_mode="linear").to(device)
    linear_model.load_state_dict(best_linear_cell["best_state"])
    linear_model.eval()

    # Save selected checkpoints to root
    torch.save({
        "variant": best_phase_cell["name"],
        "arm": "phase_cnn",
        "response_mode": "phase_cnn",
        "seed": args.seed,
        "selected_cell": best_phase_cell["name"],
        "val_macro_auroc": best_phase_cell["best_score"],
        "state_dict": best_phase_cell["best_state"],
    }, args.output / "selected_phase_cnn.pt")

    torch.save({
        "variant": best_linear_cell["name"],
        "arm": "linear",
        "response_mode": "linear",
        "seed": args.seed,
        "selected_cell": best_linear_cell["name"],
        "val_macro_auroc": best_linear_cell["best_score"],
        "state_dict": best_linear_cell["best_state"],
    }, args.output / "selected_linear.pt")

    # 8. Primary Fold-8 Endpoint (Original Frozen Diagnostic Head E7)
    print("\n=== Evaluating Primary Fold-8 Endpoint (Original Frozen Diagnostic Head) ===", flush=True)
    val_responses = np.load(args.contexts / "validation_responses.npy", mmap_mode="r")
    val_labels = np.load(args.contexts / "validation_labels.npy", allow_pickle=False)
    val_patient_ids = np.load(args.contexts / "validation_patient_ids.npy", allow_pickle=False)
    operators = np.load(args.contexts / "operators.npy", allow_pickle=False)

    fold8_table = {}
    phase_predictions = {}
    linear_predictions = {}

    for config_name, indices in CONFIGURATIONS.items():
        # Evaluate PhaseCNN on fold 8
        extracted_phase = _extract_features(phase_model, val_responses, operators, indices, device, batch_size=args.batch)
        prob_phase = torch.sigmoid(torch.as_tensor(extracted_phase["E7_frozen_logits"])).numpy()
        phase_predictions[config_name] = prob_phase
        auc_phase = _macro_auc(val_labels, prob_phase)

        # Evaluate Linear on fold 8
        extracted_linear = _extract_features(linear_model, val_responses, operators, indices, device, batch_size=args.batch)
        prob_linear = torch.sigmoid(torch.as_tensor(extracted_linear["E7_frozen_logits"])).numpy()
        linear_predictions[config_name] = prob_linear
        auc_linear = _macro_auc(val_labels, prob_linear)

        # Paired patient-level bootstrap for delta = Linear - PhaseCNN
        ci_low, ci_high, _ = paired_patient_bootstrap(
            val_labels, prob_phase, prob_linear, val_patient_ids, n_bootstraps=1000, seed=args.seed,
        )
        delta = auc_linear - auc_phase

        fold8_table[config_name] = {
            "phase_cnn_auroc": auc_phase,
            "linear_auroc": auc_linear,
            "delta": delta,
            "bootstrap_ci_95": [ci_low, ci_high],
        }
        print(f"Config: {config_name:<14} | PhaseCNN: {auc_phase:.6f} | Linear: {auc_linear:.6f} | Delta: {delta:+.6f} | 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]", flush=True)

    # 9. Secondary Mechanistic Endpoint: Frozen Linear Probing on E0, E1, E4, E5, E6
    print("\n=== Evaluating Secondary Mechanistic Endpoint (Fitted Linear Probes) ===", flush=True)
    train_responses = np.load(args.contexts / "train_responses.npy", mmap_mode="r")
    train_labels = np.load(args.contexts / "train_labels.npy", allow_pickle=False)
    train_patient_ids = np.load(args.contexts / "train_patient_ids.npy", allow_pickle=False)
    fit_mask, select_mask = _patient_split(train_patient_ids, args.seed)

    # Fit canonical Q8 probes for PhaseCNN
    train_feat_phase = _extract_features(phase_model, train_responses, operators, CONFIGURATIONS["Q8"], device, batch_size=args.batch)
    train_feat_linear = _extract_features(linear_model, train_responses, operators, CONFIGURATIONS["Q8"], device, batch_size=args.batch)

    probes_phase = {}
    probes_linear = {}
    stages_to_probe = ("E0_mean_phase_kme", "E1_response_embedding", "E4_contextual_token", "E5_attention_pooled", "E6_latent")

    for stage in stages_to_probe:
        scaler_p, probe_p, c_p = _fit_probe(
            train_feat_phase[stage][fit_mask], train_labels[fit_mask],
            train_feat_phase[stage][select_mask], train_labels[select_mask],
        )
        probes_phase[stage] = (scaler_p, probe_p, c_p)

        scaler_l, probe_l, c_l = _fit_probe(
            train_feat_linear[stage][fit_mask], train_labels[fit_mask],
            train_feat_linear[stage][select_mask], train_labels[select_mask],
        )
        probes_linear[stage] = (scaler_l, probe_l, c_l)

    mechanistic_results = {stage: {} for stage in stages_to_probe}
    for config_name, indices in CONFIGURATIONS.items():
        feat_phase_val = _extract_features(phase_model, val_responses, operators, indices, device, batch_size=args.batch)
        feat_linear_val = _extract_features(linear_model, val_responses, operators, indices, device, batch_size=args.batch)

        for stage in stages_to_probe:
            s_p, p_p, _ = probes_phase[stage]
            pred_p = p_p.predict(s_p.transform(feat_phase_val[stage]))
            auc_p = _macro_auc(val_labels, pred_p)

            s_l, p_l, _ = probes_linear[stage]
            pred_l = p_l.predict(s_l.transform(feat_linear_val[stage]))
            auc_l = _macro_auc(val_labels, pred_l)

            mechanistic_results[stage][config_name] = {
                "phase_cnn_probe_auroc": auc_p,
                "linear_probe_auroc": auc_l,
                "delta": auc_l - auc_p,
            }

    print("\nE1 (Response Embedding e_r) Linear Probe Comparison:", flush=True)
    for cfg in CONFIGURATIONS:
        r = mechanistic_results["E1_response_embedding"][cfg]
        print(f"Config: {cfg:<14} | PhaseCNN E1: {r['phase_cnn_probe_auroc']:.6f} | Linear E1: {r['linear_probe_auroc']:.6f} | Delta: {r['delta']:+.6f}", flush=True)

    # 10. Save Complete Results JSON
    final_results = {
        "status": "complete",
        "experiment": "Stage 3B: PhaseCNN vs Linear Response Bottleneck",
        "seed": args.seed,
        "selected_models": {
            "phase_cnn": {
                "cell": best_phase_cell["name"],
                "learning_rate": best_phase_cell["learning_rate"],
                "weight_decay": best_phase_cell["weight_decay"],
                "val_macro_auroc": best_phase_cell["best_score"],
                "best_epoch": best_phase_cell["best_epoch"],
            },
            "linear": {
                "cell": best_linear_cell["name"],
                "learning_rate": best_linear_cell["learning_rate"],
                "weight_decay": best_linear_cell["weight_decay"],
                "val_macro_auroc": best_linear_cell["best_score"],
                "best_epoch": best_linear_cell["best_epoch"],
            },
        },
        "primary_fold8_frozen_head": fold8_table,
        "secondary_mechanistic_probes": mechanistic_results,
    }

    _atomic_json(args.output / "results.json", final_results)
    print(f"\nSaved all Stage 3B results to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
