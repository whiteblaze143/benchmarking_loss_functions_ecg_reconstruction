from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalPredictor(nn.Module):
    def __init__(self, dimension: int, width: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(dimension, width, batch_first=True)
        self.head = nn.Linear(width, dimension)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(sequence[:, :1])
        history = torch.cat((zeros, sequence[:, :-1]), dim=1)
        hidden, _ = self.gru(history)
        return self.head(hidden)


class ConditionalLocationScale(nn.Module):
    def __init__(self, dimension: int, width: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(dimension, width, batch_first=True)
        self.mean = nn.Linear(width, dimension)
        self.log_scale = nn.Linear(width, dimension)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(sequence[:, :1])
        history = torch.cat((zeros, sequence[:, :-1]), dim=1)
        hidden, _ = self.gru(history)
        return self.mean(hidden), self.log_scale(hidden).clamp(-4.0, 2.0)


def make_world(count: int, length: int, seed: int, device: torch.device, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    if kind == "iid":
        innovations = 0.20 * torch.randn(count, length, 4, generator=generator, device=device)
        return innovations.clone(), innovations
    state = torch.empty(count, length, 4, device=device)
    innovations = torch.empty_like(state)
    state[:, 0] = 0.4 * torch.randn(count, 4, generator=generator, device=device)
    innovations[:, 0] = state[:, 0]
    for phase in range(1, length):
        previous = state[:, phase - 1]
        conditional_mean = 0.68 * previous + 0.18 * torch.sin(1.7 * previous) + 0.12 * torch.roll(previous, shifts=1, dims=-1)
        if kind == "additive":
            uniform = torch.rand(count, 4, generator=generator, device=device).clamp_min(1e-7)
            noise = 0.20 * (-uniform.log() - 1.0)
        elif kind == "heteroscedastic":
            scale = 0.08 + 0.28 * torch.sigmoid(2.0 * previous[:, :1])
            noise = scale * torch.randn(count, 4, generator=generator, device=device)
        else:
            raise ValueError(kind)
        innovations[:, phase] = noise
        state[:, phase] = conditional_mean + noise
    return state, innovations


def fit_predictor(train: torch.Tensor, *, steps: int, seed: int) -> ConditionalPredictor:
    torch.manual_seed(seed)
    model = ConditionalPredictor(train.shape[-1]).to(train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator(device=train.device).manual_seed(seed + 100)
    for _ in range(steps):
        index = torch.randint(len(train), (512,), generator=generator, device=train.device)
        prediction = model(train[index])
        loss = F.mse_loss(prediction[:, 1:], train[index, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def fit_location_scale(train: torch.Tensor, *, steps: int, seed: int) -> ConditionalLocationScale:
    torch.manual_seed(seed)
    model = ConditionalLocationScale(train.shape[-1]).to(train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator(device=train.device).manual_seed(seed + 200)
    for _ in range(steps):
        index = torch.randint(len(train), (512,), generator=generator, device=train.device)
        mean, log_scale = model(train[index])
        target = train[index]
        standardized = (target[:, 1:] - mean[:, 1:]) * torch.exp(-log_scale[:, 1:])
        loss = (0.5 * standardized.square() + log_scale[:, 1:]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.flatten() - left.mean()
    right = right.flatten() - right.mean()
    return float((left @ right / (left.norm() * right.norm()).clamp_min(1e-8)).abs())


def evaluate_world(kind: str, seed: int, steps: int, device: torch.device) -> dict[str, float]:
    train, _ = make_world(4096, 16, seed, device, kind)
    test, true_innovation = make_world(2048, 16, seed + 1, device, kind)
    model = fit_predictor(train, steps=steps, seed=seed + 10)
    reverse_model = fit_predictor(train.flip(1), steps=steps, seed=seed + 20)
    density_model = fit_location_scale(train, steps=steps, seed=seed + 30)
    with torch.inference_mode():
        expected = model(test)
        residual = test - expected
        reverse_expected = reverse_model(test.flip(1))
        density_mean, density_log_scale = density_model(test)
        standardized = (test - density_mean) * torch.exp(-density_log_scale)
    result = {
        "forward_prediction_mse": float(F.mse_loss(expected[:, 1:], test[:, 1:])),
        "reverse_prediction_mse": float(F.mse_loss(reverse_expected[:, 1:], test.flip(1)[:, 1:])),
        "unconditional_mse": float(F.mse_loss(test[:, 1:], test[:, 1:].mean(dim=(0, 1), keepdim=True).expand_as(test[:, 1:]))),
        "innovation_recovery_mse": float(F.mse_loss(residual[:, 1:], true_innovation[:, 1:])),
        "residual_lag_correlation": correlation(residual[:, 1:], test[:, :-1]),
        "squared_residual_past_correlation": correlation(residual[:, 1:].square(), test[:, :-1]),
        "standardized_innovation_lag_correlation": correlation(standardized[:, 1:], test[:, :-1]),
        "squared_standardized_past_correlation": correlation(standardized[:, 1:].square(), test[:, :-1]),
        "location_scale_nll": float((0.5 * standardized[:, 1:].square() + density_log_scale[:, 1:]).mean()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 12 structural-innovation development audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--frozen-g1", action="store_true")
    args = parser.parse_args()
    if args.frozen_g1 and args.steps != 500:
        raise ValueError("Paper 12 frozen G1 requires exactly 500 updates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worlds = {kind: evaluate_world(kind, args.seed, args.steps, device) for kind in ("additive", "iid", "heteroscedastic")}
    gates = {
        "additive_history_is_predictive": worlds["additive"]["forward_prediction_mse"] <= 0.20 * worlds["additive"]["unconditional_mse"],
        "additive_innovation_is_recovered": worlds["additive"]["innovation_recovery_mse"] <= 0.002,
        "additive_standardized_innovation_is_history_uncorrelated": worlds["additive"]["standardized_innovation_lag_correlation"] <= 0.02,
        "heteroscedasticity_breaks_mean_only_residual": worlds["heteroscedastic"]["squared_residual_past_correlation"] >= 0.10,
        "location_scale_removes_heteroscedastic_dependence": worlds["heteroscedastic"]["squared_standardized_past_correlation"] <= 0.03,
        "iid_has_no_predictive_gain": worlds["iid"]["forward_prediction_mse"] >= 0.95 * worlds["iid"]["unconditional_mse"],
    }
    payload = {
        "kind": "paper12_structural_innovation_development_audit",
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "worlds": worlds,
        "gates": gates if args.frozen_g1 else None,
        "passed": all(gates.values()) if args.frozen_g1 else None,
        "interpretation": "Frozen Paper 12 location-scale G1." if args.frozen_g1 else "Development diagnostics only; no Paper 12 scientific gate is frozen by this run.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
