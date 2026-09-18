from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import adjusted_rand_score
from torch import nn
from torch.nn import functional as F

from repecg.paper11_predictive_state import (
    ContinuousPredictor,
    PredictiveStateModel,
    SetPredictor,
    make_history_required_sequences,
    make_hmm_sequences,
    make_iid_sequences,
)


def _examples(sequence: torch.Tensor, *, prefix: int, horizon: int, shuffle: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if prefix < 1 or horizon < 1 or prefix + horizon >= sequence.shape[1]:
        raise ValueError("invalid prefix/horizon for sequence length")
    windows, targets = [], []
    for endpoint in range(prefix - 1, sequence.shape[1] - horizon):
        windows.append(sequence[:, endpoint - prefix + 1 : endpoint + 1])
        targets.append(sequence[:, endpoint + horizon])
    values = torch.cat(windows)
    if shuffle:
        generator = torch.Generator(device=sequence.device).manual_seed(seed)
        order = torch.rand(values.shape[:2], generator=generator, device=sequence.device).argsort(dim=-1)
        values = values[torch.arange(len(values), device=sequence.device)[:, None], order]
    return values, torch.cat(targets)


def _model(kind: str, dimension: int, states: int, device: torch.device) -> nn.Module:
    if kind == "discrete":
        return PredictiveStateModel(dimension, states).to(device)
    if kind == "continuous":
        return ContinuousPredictor(dimension).to(device)
    if kind in {"set", "current"}:
        return SetPredictor(dimension).to(device)
    raise ValueError(kind)


def _prediction(model: nn.Module, prefix: torch.Tensor) -> torch.Tensor:
    if isinstance(model, PredictiveStateModel):
        return model.predict_from_prefix(prefix)[0]
    return model.predict_from_prefix(prefix)


def _fit(kind: str, prefix: torch.Tensor, target: torch.Tensor, *, states: int, wrong_future: bool, steps: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = _model(kind, prefix.shape[-1], states, prefix.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    generator = torch.Generator(device=prefix.device).manual_seed(seed + 101)
    for _ in range(steps):
        index = torch.randint(len(prefix), (1024,), generator=generator, device=prefix.device)
        batch_target = target[index]
        if wrong_future:
            batch_target = batch_target[torch.randperm(len(batch_target), generator=generator, device=prefix.device)]
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(_prediction(model, prefix[index]), batch_target).backward()
        optimizer.step()
    return model.eval()


def _mse(model: nn.Module, prefix: torch.Tensor, target: torch.Tensor) -> float:
    with torch.inference_mode():
        return float(F.mse_loss(_prediction(model, prefix), target))


def _run_hmm(seed: int, steps: int, device: torch.device) -> dict[str, object]:
    train, _ = make_hmm_sequences(2048, 20, seed, device, states=4)
    test, latent = make_hmm_sequences(2048, 20, seed + 1, device, states=4)
    train_prefix, train_target = _examples(train, prefix=4, horizon=1, shuffle=False, seed=seed)
    test_prefix, test_target = _examples(test, prefix=4, horizon=1, shuffle=False, seed=seed)
    mean_mse = float(F.mse_loss(test_target, test_target.mean(dim=0, keepdim=True).expand_as(test_target)))
    discrete = {k: _fit("discrete", train_prefix, train_target, states=k, wrong_future=False, steps=steps, seed=seed + k) for k in (2, 4, 8)}
    continuous = _fit("continuous", train_prefix, train_target, states=4, wrong_future=False, steps=steps, seed=seed + 20)
    wrong = _fit("discrete", train_prefix, train_target, states=4, wrong_future=True, steps=steps, seed=seed + 30)
    loss = {f"k{k}": _mse(model, test_prefix, test_target) for k, model in discrete.items()}
    continuous_mse = _mse(continuous, test_prefix, test_target)
    wrong_mse = _mse(wrong, test_prefix, test_target)
    copy_mse = float(F.mse_loss(test_prefix[:, -1], test_target))
    prefix_mean_mse = float(F.mse_loss(test_prefix.mean(dim=1), test_target))
    with torch.inference_mode():
        _, q = discrete[4].predict_from_prefix(test_prefix)
        shuffled_prediction = discrete[4].decoder(q.roll(shifts=1, dims=-1) @ discrete[4].state_embedding)
        state_shuffle_mse = float(F.mse_loss(shuffled_prediction, test_target))
    state_labels = latent[:, 3 : -1].reshape(-1).cpu().numpy()
    ari = float(adjusted_rand_score(state_labels, q.argmax(dim=-1).cpu().numpy()))
    eta = float((mean_mse - loss["k4"]) / max(mean_mse - continuous_mse, 1e-8))
    state_future_means = []
    for state in range(4):
        target_subset = test_target[q.argmax(dim=-1) == state]
        if len(target_subset):
            state_future_means.append(target_subset.mean(dim=0))
    separation = min(float((left - right).norm()) for index, left in enumerate(state_future_means) for right in state_future_means[index + 1 :])
    horizons: dict[str, dict[str, float]] = {}
    for horizon in (1, 2, 4):
        train_p, train_y = _examples(train, prefix=4, horizon=horizon, shuffle=False, seed=seed)
        test_p, test_y = _examples(test, prefix=4, horizon=horizon, shuffle=False, seed=seed)
        model = _fit("discrete", train_p, train_y, states=4, wrong_future=False, steps=steps, seed=seed + 40 + horizon)
        horizons[str(horizon)] = {
            "mse": _mse(model, test_p, test_y),
            "mean_mse": float(F.mse_loss(test_y, test_y.mean(dim=0, keepdim=True).expand_as(test_y))),
        }
    return {
        "ari": ari, "mean_mse": mean_mse, "copy_mse": copy_mse, "prefix_mean_mse": prefix_mean_mse,
        "k2_mse": loss["k2"], "k4_mse": loss["k4"], "k8_mse": loss["k8"], "state_shuffle_mse": state_shuffle_mse,
        "continuous_mse": continuous_mse, "wrong_future_mse": wrong_mse, "eta_k4": eta,
        "minimum_state_future_mean_separation": separation, "horizons": horizons,
    }


def _run_history_world(seed: int, steps: int, device: torch.device) -> dict[str, float]:
    train = make_history_required_sequences(2048, 20, seed, device)
    test = make_history_required_sequences(2048, 20, seed + 1, device)
    ordered_train, target_train = _examples(train, prefix=2, horizon=1, shuffle=False, seed=seed)
    ordered_test, target_test = _examples(test, prefix=2, horizon=1, shuffle=False, seed=seed)
    shuffled_train, _ = _examples(train, prefix=2, horizon=1, shuffle=True, seed=seed)
    shuffled_test, _ = _examples(test, prefix=2, horizon=1, shuffle=True, seed=seed)
    current_train, current_target_train = _examples(train, prefix=1, horizon=1, shuffle=False, seed=seed)
    current_test, current_target_test = _examples(test, prefix=1, horizon=1, shuffle=False, seed=seed)
    ordered = _fit("discrete", ordered_train, target_train, states=4, wrong_future=False, steps=steps, seed=seed)
    shuffled = _fit("discrete", shuffled_train, target_train, states=4, wrong_future=False, steps=steps, seed=seed + 1)
    current = _fit("current", current_train, current_target_train, states=4, wrong_future=False, steps=steps, seed=seed + 2)
    set_model = _fit("set", ordered_train, target_train, states=4, wrong_future=False, steps=steps, seed=seed + 3)
    return {
        "ordered_mse": _mse(ordered, ordered_test, target_test),
        "shuffled_prefix_mse": _mse(shuffled, shuffled_test, target_test),
        "current_only_mse": _mse(current, current_test, current_target_test),
        "set_encoder_mse": _mse(set_model, ordered_test, target_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 11 strict synthetic predictive-state audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hmm = _run_hmm(args.seed, args.steps, device)
    history = _run_history_world(args.seed + 100, args.steps, device)
    train_iid = make_iid_sequences(2048, 20, args.seed + 200, device)
    test_iid = make_iid_sequences(2048, 20, args.seed + 201, device)
    iid_train_p, iid_train_y = _examples(train_iid, prefix=4, horizon=1, shuffle=False, seed=args.seed)
    iid_test_p, iid_test_y = _examples(test_iid, prefix=4, horizon=1, shuffle=False, seed=args.seed)
    iid = _fit("discrete", iid_train_p, iid_train_y, states=4, wrong_future=False, steps=args.steps, seed=args.seed + 200)
    iid_mse = _mse(iid, iid_test_p, iid_test_y)
    iid_mean = float(F.mse_loss(iid_test_y, iid_test_y.mean(dim=0, keepdim=True).expand_as(iid_test_y)))
    gates = {
        "correct_pairs_beat_wrong_pairs": hmm["k4_mse"] < 0.50 * hmm["wrong_future_mse"],
        "k4_beats_copy_and_prefix_mean": hmm["k4_mse"] < min(hmm["copy_mse"], hmm["prefix_mean_mse"]),
        "state_shuffle_hurts_prediction": hmm["state_shuffle_mse"] > 1.20 * hmm["k4_mse"],
        "cardinality_k2_underfits_k4": hmm["k2_mse"] > 1.20 * hmm["k4_mse"],
        "cardinality_k8_no_major_gain_over_k4": hmm["k8_mse"] >= 0.90 * hmm["k4_mse"],
        "discrete_retains_continuous_gain": hmm["eta_k4"] >= 0.80,
        "history_beats_current_only": history["ordered_mse"] < 0.50 * history["current_only_mse"],
        "order_beats_shuffled_prefix": history["ordered_mse"] < 0.50 * history["shuffled_prefix_mse"],
        "order_beats_set_encoder": history["ordered_mse"] < 0.50 * history["set_encoder_mse"],
        "iid_has_no_material_predictive_gain": iid_mse >= 0.95 * iid_mean,
        "state_help_persists_to_horizon_4": hmm["horizons"]["4"]["mse"] < hmm["horizons"]["4"]["mean_mse"],
    }
    payload = {
        "kind": "paper11_strict_predictive_state_synthetic_audit", "device": str(device), "steps": args.steps, "seed": args.seed,
        "hmm_k4": hmm, "history_required": history, "iid": {"mse": iid_mse, "mean_mse": iid_mean},
        "gates": gates, "passed": all(gates.values()),
        "interpretation": "Synthetic finite-state predictive-compression evidence only; no epsilon-machine, biological-causality, or real-ECG claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
