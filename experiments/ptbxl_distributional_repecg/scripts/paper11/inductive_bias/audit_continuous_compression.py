"""Development audit for Paper 11's continuous predictive-compression pivot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from repecg.paper11_predictive_state import ContinuousPredictor, SetPredictor, make_iid_sequences, make_linear_state_space_sequences


def _examples(sequence: torch.Tensor, *, prefix: int, shuffle: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    windows, targets = [], []
    for endpoint in range(prefix - 1, sequence.shape[1] - 1):
        windows.append(sequence[:, endpoint - prefix + 1 : endpoint + 1])
        targets.append(sequence[:, endpoint + 1])
    values = torch.cat(windows)
    if shuffle:
        generator = torch.Generator(device=sequence.device).manual_seed(seed)
        order = torch.rand(values.shape[:2], generator=generator, device=sequence.device).argsort(dim=-1)
        values = values[torch.arange(len(values), device=sequence.device)[:, None], order]
    return values, torch.cat(targets)


def _fit(model: nn.Module, prefix: torch.Tensor, target: torch.Tensor, *, steps: int, seed: int, wrong_future: bool = False) -> nn.Module:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    generator = torch.Generator(device=prefix.device).manual_seed(seed + 101)
    for _ in range(steps):
        index = torch.randint(len(prefix), (1024,), generator=generator, device=prefix.device)
        batch_target = target[index]
        if wrong_future:
            batch_target = batch_target[torch.randperm(len(batch_target), generator=generator, device=prefix.device)]
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(model(prefix[index]), batch_target).backward()
        optimizer.step()
    return model.eval()


def _mse(model: nn.Module, prefix: torch.Tensor, target: torch.Tensor) -> float:
    with torch.inference_mode():
        return float(F.mse_loss(model(prefix), target))


def _continuous(input_dim: int, hidden_dim: int, device: torch.device, seed: int) -> ContinuousPredictor:
    torch.manual_seed(seed)
    return ContinuousPredictor(input_dim, hidden_dim=hidden_dim).to(device)


def _probe(z_train: torch.Tensor, prefix_train: torch.Tensor, target_train: torch.Tensor, z_test: torch.Tensor, prefix_test: torch.Tensor, target_test: torch.Tensor, *, include_prefix: bool, steps: int, seed: int) -> float:
    feature_train = torch.cat((z_train, prefix_train.flatten(start_dim=1)), dim=1) if include_prefix else z_train
    feature_test = torch.cat((z_test, prefix_test.flatten(start_dim=1)), dim=1) if include_prefix else z_test
    torch.manual_seed(seed)
    probe = nn.Linear(feature_train.shape[-1], target_train.shape[-1]).to(feature_train.device)
    return _mse(_fit(probe, feature_train, target_train, steps=steps, seed=seed), feature_test, target_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 11 continuous predictive-compression development audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frozen-g1", action="store_true")
    parser.add_argument("--probe-dimension", type=int, default=4, choices=(1, 2, 4, 8, 16, 32))
    args = parser.parse_args()
    if args.frozen_g1 and args.steps != 1000:
        raise ValueError("frozen continuous G1 v2 requires exactly 1000 updates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _ = make_linear_state_space_sequences(2048, 20, args.seed, device)
    test, _ = make_linear_state_space_sequences(2048, 20, args.seed + 1, device)
    train_prefix, train_target = _examples(train, prefix=4, shuffle=False, seed=args.seed)
    test_prefix, test_target = _examples(test, prefix=4, shuffle=False, seed=args.seed)
    dimensions = (1, 2, 4, 8, 16, 32)
    models = {d: _fit(_continuous(train.shape[-1], d, device, args.seed + d), train_prefix, train_target, steps=args.steps, seed=args.seed + d) for d in dimensions}
    losses = {str(d): _mse(model, test_prefix, test_target) for d, model in models.items()}
    copy_mse = float(F.mse_loss(test_prefix[:, -1], test_target))
    mean_mse = float(F.mse_loss(test_target, test_target.mean(dim=0, keepdim=True).expand_as(test_target)))
    current = _fit(_continuous(train.shape[-1], 4, device, args.seed + 50), train_prefix[:, -1:], train_target, steps=args.steps, seed=args.seed + 50)
    shuffled_train, _ = _examples(train, prefix=4, shuffle=True, seed=args.seed)
    shuffled_test, _ = _examples(test, prefix=4, shuffle=True, seed=args.seed)
    shuffled = _fit(_continuous(train.shape[-1], 4, device, args.seed + 51), shuffled_train, train_target, steps=args.steps, seed=args.seed + 51)
    torch.manual_seed(args.seed + 52)
    set_model = _fit(SetPredictor(train.shape[-1], hidden_dim=4).to(device), train_prefix, train_target, steps=args.steps, seed=args.seed + 52)
    wrong = _fit(_continuous(train.shape[-1], 4, device, args.seed + 53), train_prefix, train_target, steps=args.steps, seed=args.seed + 53, wrong_future=True)
    probe_dimension = args.probe_dimension
    d4_model = models[probe_dimension]
    with torch.inference_mode():
        z_train = d4_model.encode_prefix(train_prefix)
        z_test = d4_model.encode_prefix(test_prefix)
    z_probe_mse = _probe(z_train, train_prefix, train_target, z_test, test_prefix, test_target, include_prefix=False, steps=args.steps, seed=args.seed + 60)
    z_prefix_probe_mse = _probe(z_train, train_prefix, train_target, z_test, test_prefix, test_target, include_prefix=True, steps=args.steps, seed=args.seed + 61)
    iid_train = make_iid_sequences(2048, 20, args.seed + 100, device)
    iid_test = make_iid_sequences(2048, 20, args.seed + 101, device)
    iid_train_p, iid_train_y = _examples(iid_train, prefix=4, shuffle=False, seed=args.seed)
    iid_test_p, iid_test_y = _examples(iid_test, prefix=4, shuffle=False, seed=args.seed)
    iid = _fit(_continuous(4, 4, device, args.seed + 102), iid_train_p, iid_train_y, steps=args.steps, seed=args.seed + 102)
    iid_mse = _mse(iid, iid_test_p, iid_test_y)
    iid_mean_mse = float(F.mse_loss(iid_test_y, iid_test_y.mean(dim=0, keepdim=True).expand_as(iid_test_y)))
    eta_d4 = float((copy_mse - losses["4"]) / max(copy_mse - losses["32"], 1e-8))
    residual_fraction = float((z_probe_mse - z_prefix_probe_mse) / max(copy_mse - losses[str(probe_dimension)], 1e-8))
    gates = {
        "d1_underfits_d4": losses["1"] > 1.10 * losses["4"],
        "d2_underfits_d4": losses["2"] > 1.05 * losses["4"],
        "d4_retains_full_predictive_gain": eta_d4 >= 0.90,
        "d4_beats_copy_current_set_and_shuffle": losses["4"] < min(copy_mse, _mse(current, test_prefix[:, -1:], test_target), _mse(set_model, test_prefix, test_target), _mse(shuffled, shuffled_test, test_target)),
        "d4_is_not_materially_worse_than_d8": losses["8"] >= 0.95 * losses["4"],
        "frozen_d4_has_small_prefix_residual": residual_fraction <= 0.05,
        "correct_pairing_beats_wrong_future": losses["4"] < 0.90 * _mse(wrong, test_prefix, test_target),
        "iid_has_no_material_predictive_gain": iid_mse >= 0.95 * iid_mean_mse,
    }
    payload = {
        "kind": "paper11_continuous_compression_development_audit",
        "device": str(device), "seed": args.seed, "steps": args.steps,
        "loss_by_dimension": losses, "beat_copy_mse": copy_mse, "mean_future_mse": mean_mse,
        "current_only_mse": _mse(current, test_prefix[:, -1:], test_target),
        "shuffled_prefix_mse": _mse(shuffled, shuffled_test, test_target),
        "set_predictor_mse": _mse(set_model, test_prefix, test_target),
        "wrong_future_mse": _mse(wrong, test_prefix, test_target),
        "probe_dimension": probe_dimension, "frozen_z_probe_mse": z_probe_mse, "frozen_z_plus_prefix_probe_mse": z_prefix_probe_mse,
        "eta_d4": eta_d4, "prefix_residual_fraction": residual_fraction,
        "iid_mse": iid_mse, "iid_mean_mse": iid_mean_mse,
        "gates": gates if args.frozen_g1 else None,
        "passed": all(gates.values()) if args.frozen_g1 else None,
        "interpretation": "Frozen continuous G1 protocol." if args.frozen_g1 else "Development calibration only; thresholds for a new continuous G1 are not frozen by this run.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
