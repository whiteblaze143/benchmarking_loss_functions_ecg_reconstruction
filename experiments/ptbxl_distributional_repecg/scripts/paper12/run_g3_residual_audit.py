"""Frozen Paper 12 G3 real-data residual-adequacy audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from repecg.paper12_innovations import LocationScaleInnovationModel, PhaseCoordinateStandardizer


ORIGINS = (0, 4, 8, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotate(values: torch.Tensor, origin: int) -> torch.Tensor:
    return torch.roll(values, shifts=-origin, dims=1)


def record_permutations(ecg_ids: np.ndarray, device: torch.device, seed: int) -> torch.Tensor:
    rows = []
    for ecg_id in ecg_ids:
        generator = torch.Generator().manual_seed(seed * 1_000_003 + int(ecg_id))
        while True:
            order = torch.randperm(16, generator=generator)
            if not torch.equal(order, torch.arange(16)) and not torch.equal(
                order, torch.roll(torch.arange(16), shifts=int(order[0]))
            ):
                rows.append(order)
                break
    return torch.stack(rows).to(device)


def permute_rows(values: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(values), device=values.device)[:, None]
    return values[rows, orders]


def wrong_history_indices(
    patient_ids: np.ndarray, labels: np.ndarray, *, label_matched: bool
) -> np.ndarray:
    count = len(patient_ids)
    rows = np.arange(count)[:, None]
    columns = np.arange(count)[None, :]
    cyclic_distance = (columns - rows - 1) % count
    cost = cyclic_distance.astype(np.int64)
    if label_matched:
        hamming = np.abs(labels[:, None, :] - labels[None, :, :]).sum(axis=2).astype(np.int64)
        cost = cost + hamming * (count + 1)
    cost[patient_ids[:, None] == patient_ids[None, :]] = 10**12
    target, donor = linear_sum_assignment(cost)
    result = np.empty(count, dtype=np.int64)
    result[target] = donor
    if np.any(patient_ids[result] == patient_ids) or len(np.unique(result)) != count:
        raise RuntimeError("wrong-history assignment is not a different-patient bijection")
    return result


def fit_model(
    features: torch.Tensor, *, steps: int, batch: int, seed: int, unit_scale_init: bool
) -> LocationScaleInnovationModel:
    torch.manual_seed(seed)
    model = LocationScaleInnovationModel(
        input_dim=features.shape[-1], width=64, unit_scale_init=unit_scale_init
    ).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    model.train()
    final_loss = float("nan")
    for _ in range(steps):
        index = torch.randint(len(features), (batch,), generator=generator, device=features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.density_loss(features[index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    model.final_training_loss = final_loss  # type: ignore[attr-defined]
    return model


def pearson_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean(dim=-1, keepdim=True)
    right = right - right.mean(dim=-1, keepdim=True)
    return (left * right).mean(dim=-1) / (
        left.square().mean(dim=-1).sqrt() * right.square().mean(dim=-1).sqrt()
    ).clamp_min(1e-8)


def dependence(values: torch.Tensor, *, squared: bool) -> torch.Tensor:
    values = values[:, 1:]
    if squared:
        values = values.square()
    return pearson_rows(values[:, :-1], values[:, 1:]).abs().mean(dim=1)


def patient_equal(values: torch.Tensor, patient_ids: np.ndarray) -> tuple[float, dict[int, float]]:
    array = values.detach().float().cpu().numpy()
    per_patient = {
        int(patient): float(array[patient_ids == patient].mean()) for patient in np.unique(patient_ids)
    }
    return float(np.mean(list(per_patient.values()))), per_patient


def bootstrap_difference(
    left: dict[int, float], right: dict[int, float], *, seed: int, draws: int = 2000
) -> dict[str, float]:
    patients = np.asarray(sorted(set(left) & set(right)))
    delta = np.asarray([left[int(p)] - right[int(p)] for p in patients])
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "patients": int(len(patients)),
    }


def nll_per_record(mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    innovation = (target - mean) * torch.exp(-log_scale)
    return (0.5 * innovation[:, 1:].square() + log_scale[:, 1:]).mean(dim=(1, 2))


def evaluate(
    model: LocationScaleInnovationModel,
    values: torch.Tensor,
    patient_ids: np.ndarray,
    wrong: torch.Tensor,
    wrong_label: torch.Tensor,
) -> tuple[dict[str, object], dict[str, dict[int, float]]]:
    with torch.inference_mode():
        mean, log_scale, innovation = model.components(values)
        wrong_mean, wrong_scale, _ = model.components(values, wrong)
        label_mean, label_scale, _ = model.components(values, wrong_label)
        raw_scale = model.raw_log_scale(values)
    unconditional_mean = model.unconditional_mean  # type: ignore[attr-defined]
    unconditional_scale = model.unconditional_log_scale  # type: ignore[attr-defined]
    mean_residual = values - mean
    phase_standardized = (values - unconditional_mean) * torch.exp(-unconditional_scale)
    metrics: dict[str, tuple[float, dict[int, float]]] = {
        "conditional_nll": patient_equal(nll_per_record(mean, log_scale, values), patient_ids),
        "phase_only_nll": patient_equal(nll_per_record(unconditional_mean, unconditional_scale, values), patient_ids),
        "wrong_patient_nll": patient_equal(nll_per_record(wrong_mean, wrong_scale, values), patient_ids),
        "wrong_label_nll": patient_equal(nll_per_record(label_mean, label_scale, values), patient_ids),
        "state_location_dependence": patient_equal(dependence(phase_standardized, squared=False), patient_ids),
        "innovation_location_dependence": patient_equal(dependence(innovation, squared=False), patient_ids),
        "mean_residual_scale_dependence": patient_equal(dependence(mean_residual, squared=True), patient_ids),
        "innovation_scale_dependence": patient_equal(dependence(innovation, squared=True), patient_ids),
    }
    scale_quantiles = np.quantile(
        raw_scale[:, 1:].detach().float().cpu().numpy(), [0.01, 0.05, 0.5, 0.95, 0.99]
    )
    result = {
        "metrics": {name: value[0] for name, value in metrics.items()},
        "contrasts": {
            "conditional_minus_phase_only": bootstrap_difference(metrics["conditional_nll"][1], metrics["phase_only_nll"][1], seed=1200),
            "conditional_minus_wrong_patient": bootstrap_difference(metrics["conditional_nll"][1], metrics["wrong_patient_nll"][1], seed=1201),
            "conditional_minus_wrong_label": bootstrap_difference(metrics["conditional_nll"][1], metrics["wrong_label_nll"][1], seed=1202),
            "innovation_minus_state_location": bootstrap_difference(metrics["innovation_location_dependence"][1], metrics["state_location_dependence"][1], seed=1203),
            "innovation_minus_meanres_scale": bootstrap_difference(metrics["innovation_scale_dependence"][1], metrics["mean_residual_scale_dependence"][1], seed=1204),
        },
        "calibration": {
            "mean": float(innovation[:, 1:].mean()),
            "variance": float(innovation[:, 1:].var(unbiased=False)),
        },
        "scale": {
            "lower_clamp_fraction": float((raw_scale[:, 1:] <= -4).float().mean()),
            "upper_clamp_fraction": float((raw_scale[:, 1:] >= 2).float().mean()),
            "near_lower_fraction": float((raw_scale[:, 1:] < -3.8).float().mean()),
            "near_upper_fraction": float((raw_scale[:, 1:] > 1.8).float().mean()),
            "quantiles": [float(value) for value in scale_quantiles],
        },
    }
    return result, {name: value[1] for name, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v2-standardize", action="store_true")
    args = parser.parse_args()
    if args.steps != 1000 or args.batch != 128 or args.seed != 42:
        raise ValueError("G3 fitting budget is frozen at steps=1000, batch=128, seed=42")
    device = torch.device("cuda")
    with np.load(args.representations / "representation_train.npz") as item:
        train_np = np.asarray(item["kernel"])
        train_labels = np.asarray(item["labels"])
        train_ecg = np.asarray(item["ecg_ids"])
    with np.load(args.representations / "representation_val.npz") as item:
        val_np = np.asarray(item["kernel"])
        val_labels = np.asarray(item["labels"])
        val_ecg = np.asarray(item["ecg_ids"])
        val_patient = np.asarray(item["patient_ids"])
    train = torch.from_numpy(train_np).to(device)
    validation = torch.from_numpy(val_np).to(device)
    standardization = None
    if args.v2_standardize:
        standardizer = PhaseCoordinateStandardizer.fit(train)
        raw_scale = train.double().std(dim=0, unbiased=False)
        standardization = {
            "floor": 1e-6,
            "floored_fraction": float((raw_scale <= 1e-6).float().mean()),
            "floored_fraction_by_phase": [
                float((raw_scale[phase] <= 1e-6).float().mean()) for phase in range(16)
            ],
        }
        train = standardizer.transform(train)
        validation = standardizer.transform(validation)
    wrong = torch.as_tensor(wrong_history_indices(val_patient, val_labels, label_matched=False), device=device)
    wrong_label = torch.as_tensor(wrong_history_indices(val_patient, val_labels, label_matched=True), device=device)
    results: dict[str, object] = {}
    for origin in ORIGINS:
        fit = rotate(train, origin)
        select = rotate(validation, origin)
        model = fit_model(
            fit, steps=args.steps, batch=args.batch, seed=args.seed, unit_scale_init=args.v2_standardize
        )
        model.unconditional_mean = fit.mean(dim=0, keepdim=True).detach()  # type: ignore[attr-defined]
        model.unconditional_log_scale = fit.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6).log().detach()  # type: ignore[attr-defined]
        result, patient_metrics = evaluate(model, select, val_patient, wrong, wrong_label)
        result["training_loss"] = model.final_training_loss  # type: ignore[attr-defined]
        results[str(origin)] = result
        if origin == 0:
            ordered_patient_nll = patient_metrics["conditional_nll"]
        print(json.dumps({"origin": origin, "metrics": result["metrics"]}), flush=True)

    train_order = record_permutations(train_ecg, device, args.seed)
    val_order = record_permutations(val_ecg, device, args.seed)
    perm_train = permute_rows(train, train_order)
    perm_val = permute_rows(validation, val_order)
    perm_model = fit_model(
        perm_train, steps=args.steps, batch=args.batch, seed=args.seed, unit_scale_init=args.v2_standardize
    )
    perm_model.unconditional_mean = perm_train.mean(dim=0, keepdim=True).detach()  # type: ignore[attr-defined]
    perm_model.unconditional_log_scale = perm_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6).log().detach()  # type: ignore[attr-defined]
    perm_result, perm_patient_metrics = evaluate(perm_model, perm_val, val_patient, wrong, wrong_label)
    ordered_nll = results["0"]["metrics"]["conditional_nll"]  # type: ignore[index]
    permuted_nll = perm_result["metrics"]["conditional_nll"]  # type: ignore[index]
    payload = {
        "kind": "paper12_v2_g3_residual_adequacy" if args.v2_standardize else "paper12_g3_residual_adequacy",
        "status": "OBSERVED_PENDING_GATE_RECONCILIATION",
        "kernel_fit_sha256": sha256(args.kernel_fit),
        "representation_manifest_sha256": sha256(args.representations / "manifest.json"),
        "folds": {"fit": [1, 2, 3, 4, 5, 6], "selection": [7], "unread": [8]},
        "budget": {"width": 64, "lr": 3e-4, "weight_decay": 1e-4, "batch": 128, "steps": 1000, "seed": 42},
        "records": {"fit": len(train), "selection": len(validation)},
        "standardization": standardization,
        "origins": results,
        "phase_permutation": {
            "result": perm_result,
            "ordered_nll": ordered_nll,
            "permuted_nll": permuted_nll,
            "ordered_minus_permuted": bootstrap_difference(
                ordered_patient_nll, perm_patient_metrics["conditional_nll"], seed=1205
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"]}), flush=True)


if __name__ == "__main__":
    main()
