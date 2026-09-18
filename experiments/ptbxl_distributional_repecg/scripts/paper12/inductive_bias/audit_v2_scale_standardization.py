"""Frozen synthetic V2 scale-stress and destroyer audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from repecg.paper12_innovations import LocationScaleInnovationModel, PhaseCoordinateStandardizer


def latent_world(count: int, seed: int, *, iid: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    eta = torch.randn(count, 16, 8, generator=generator, device=device)
    if iid:
        return eta.clone(), eta
    values = torch.empty_like(eta)
    values[:, 0] = eta[:, 0]
    for phase in range(1, 16):
        previous = values[:, phase - 1]
        mean = 0.62 * previous + 0.18 * torch.sin(previous) + 0.12 * torch.roll(previous, 1, -1)
        scale = 0.25 + 0.35 * torch.sigmoid(previous)
        values[:, phase] = mean + scale * eta[:, phase]
    return values, eta


def distort(values: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=values.device).manual_seed(seed)
    scales = torch.logspace(-3, 3, values.shape[-1], device=values.device)
    scales = scales[torch.randperm(values.shape[-1], generator=generator, device=values.device)]
    indices = (torch.arange(8, device=values.device)[None, :] - torch.arange(16, device=values.device)[:, None]) % 8
    phase_scale = scales[indices]
    offsets = torch.linspace(-5, 5, 16, device=values.device)[:, None] + torch.linspace(-2, 2, 8, device=values.device)
    return values * phase_scale + offsets


def fit(values: torch.Tensor, seed: int) -> LocationScaleInnovationModel:
    torch.manual_seed(seed)
    model = LocationScaleInnovationModel(input_dim=8, width=32, unit_scale_init=True).to(values.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator(device=values.device).manual_seed(seed + 1000)
    for _ in range(500):
        index = torch.randint(len(values), (512,), generator=generator, device=values.device)
        loss = model.density_loss(values[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def nll(model: LocationScaleInnovationModel, target: torch.Tensor, history_index: torch.Tensor | None = None) -> float:
    with torch.inference_mode():
        mean, log_scale, _ = model.components(target, history_index)
        innovation = (target - mean) * torch.exp(-log_scale)
    return float((0.5 * innovation[:, 1:].square() + log_scale[:, 1:]).mean())


def wrong_indices(labels: torch.Tensor, *, matched: bool) -> torch.Tensor:
    result = torch.empty(len(labels), dtype=torch.long, device=labels.device)
    for index in range(len(labels)):
        candidates = torch.arange(len(labels), device=labels.device)
        candidates = candidates[candidates != index]
        if matched:
            candidates = candidates[labels[candidates] == labels[index]]
        result[index] = candidates[(candidates - index).abs().argmin()]
    return result


def permute(values: torch.Tensor, seed: int) -> torch.Tensor:
    orders = []
    for index in range(len(values)):
        generator = torch.Generator().manual_seed(seed * 1_000_003 + index)
        orders.append(torch.randperm(16, generator=generator))
    order = torch.stack(orders).to(values.device)
    rows = torch.arange(len(values), device=values.device)[:, None]
    return values[rows, order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(48, 49, 50))
    args = parser.parse_args()
    device = torch.device("cuda")
    train_u, _ = latent_world(4096, args.seed, iid=False, device=device)
    test_u, test_eta = latent_world(2048, args.seed + 10_000, iid=False, device=device)
    train_x, test_x = distort(train_u, 1200), distort(test_u, 1200)
    standardizer = PhaseCoordinateStandardizer.fit(train_x)
    train = standardizer.transform(train_x)
    test = standardizer.transform(test_x)
    model = fit(train, args.seed)
    correct = nll(model, test)
    phase_mean = train.mean(dim=0, keepdim=True)
    phase_log_scale = train.std(dim=0, keepdim=True, unbiased=False).log()
    phase_innovation = (test - phase_mean) * torch.exp(-phase_log_scale)
    phase_only = float((0.5 * phase_innovation[:, 1:].square() + phase_log_scale[:, 1:]).mean())
    with torch.inference_mode():
        recovered = model.components(test)[2]
    recovery = float((recovered[:, 1:] - test_eta[:, 1:]).square().mean())
    labels = (test_u[:, 0, 0] > 0).long()
    wrong = nll(model, test, wrong_indices(labels, matched=False))
    wrong_label = nll(model, test, wrong_indices(labels, matched=True))
    perm_train, perm_test = permute(train, args.seed), permute(test, args.seed + 10_000)
    permuted = nll(fit(perm_train, args.seed), perm_test)

    iid_train_u, _ = latent_world(4096, args.seed + 20_000, iid=True, device=device)
    iid_test_u, _ = latent_world(2048, args.seed + 30_000, iid=True, device=device)
    iid_train_x, iid_test_x = distort(iid_train_u, 1200), distort(iid_test_u, 1200)
    iid_standardizer = PhaseCoordinateStandardizer.fit(iid_train_x)
    iid_train, iid_test = iid_standardizer.transform(iid_train_x), iid_standardizer.transform(iid_test_x)
    iid_model = fit(iid_train, args.seed + 100)
    iid_conditional = nll(iid_model, iid_test)
    iid_phase_mean = iid_train.mean(dim=0, keepdim=True)
    iid_phase_scale = iid_train.std(dim=0, keepdim=True, unbiased=False).log()
    iid_residual = (iid_test - iid_phase_mean) * torch.exp(-iid_phase_scale)
    iid_phase_only = float((0.5 * iid_residual[:, 1:].square() + iid_phase_scale[:, 1:]).mean())
    gates = {
        "affine_scale_recovery": recovery <= 0.02,
        "conditional_beats_phase_only": correct <= phase_only - 0.05,
        "wrong_patient_is_worse": wrong >= correct + 0.02,
        "wrong_label_is_worse": wrong_label >= correct + 0.02,
        "phase_permutation_is_worse": permuted >= correct + 0.02,
        "iid_has_no_material_gain": iid_conditional >= iid_phase_only - 0.01,
    }
    payload = {
        "kind": "paper12_v2_frozen_synthetic_scale_audit",
        "seed": args.seed,
        "metrics": {
            "innovation_recovery_mse": recovery,
            "conditional_nll": correct,
            "phase_only_nll": phase_only,
            "wrong_patient_nll": wrong,
            "wrong_label_nll": wrong_label,
            "phase_permuted_nll": permuted,
            "iid_conditional_nll": iid_conditional,
            "iid_phase_only_nll": iid_phase_only,
            "floored_fraction": float(standardizer.floored_fraction(train_x)),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
