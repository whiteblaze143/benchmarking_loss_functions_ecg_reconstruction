#!/usr/bin/env python3
"""Pre-clinical mechanism gate for Paper 09's operator-conditioned model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from repecg.paper09_counterfactual import CounterfactualOperatorSetModel


SEED = 17
PAIRS = 8
PHASES = 16
RESPONSE_DIM = 8


def _world(
    records: int, phase_basis: torch.Tensor, state_map: torch.Tensor, *, operator_dependent: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # Rank two is identifiable from every permitted 1--3-pair context in the
    # noiseless synthetic world; an unrestricted 8-D state is not.
    state = torch.randn(records, state_map.shape[0], device=phase_basis.device) @ state_map
    operators = torch.randn(records, PAIRS, 8, device=phase_basis.device)
    operators = nn.functional.normalize(operators, dim=-1)
    amplitude = (
        torch.einsum("npe,ne->np", operators, state)
        if operator_dependent else state[:, :1].expand(-1, PAIRS)
    )
    return operators, amplitude[:, :, None, None] * phase_basis[None, None]


def _predict(model: CounterfactualOperatorSetModel, operators: torch.Tensor, responses: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, prediction = model(
        operators[:, :3], responses[:, :3], target_operator=target, return_counterfactual=True
    )
    return prediction


def _fit(
    operators: torch.Tensor, responses: torch.Tensor, *, mismatched: bool, epochs: int
) -> CounterfactualOperatorSetModel:
    model = CounterfactualOperatorSetModel(response_dim=RESPONSE_DIM).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(operators), device="cuda")
        for start in range(0, len(order), 128):
            rows = order[start:start + 128]
            q, response = operators[rows], responses[rows]
            context_q, target_q = q[:, :3], q[:, 6]
            if mismatched:
                context_q, target_q = context_q.roll(1, 0), target_q.roll(1, 0)
            _, prediction = model(
                context_q, response[:, :3], target_operator=target_q, return_counterfactual=True
            )
            loss = nn.functional.mse_loss(prediction, response[:, 6])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(prediction, target) / target.square().mean().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False  # A100 validation workaround.
    torch.manual_seed(SEED)
    phase_basis = torch.randn(PHASES, RESPONSE_DIM, device="cuda")
    state_map = nn.functional.normalize(torch.randn(2, 8, device="cuda"), dim=1)
    train_q, train_response = _world(768, phase_basis, state_map, operator_dependent=True)
    test_q, test_response = _world(256, phase_basis, state_map, operator_dependent=True)
    paired = _fit(train_q, train_response, mismatched=False, epochs=args.epochs)
    killed = _fit(train_q, train_response, mismatched=True, epochs=args.epochs)
    independent_q, independent_response = _world(768, phase_basis, state_map, operator_dependent=False)
    independent_test_q, independent_test_response = _world(
        256, phase_basis, state_map, operator_dependent=False
    )
    independent = _fit(independent_q, independent_response, mismatched=False, epochs=args.epochs)
    with torch.inference_mode():
        paired_prediction = _predict(paired, test_q, test_response, test_q[:, 6])
        killed_prediction = _predict(killed, test_q, test_response, test_q[:, 6])
        wrong_target_prediction = _predict(paired, test_q, test_response, test_q[:, 7])
        permutation = torch.tensor([2, 0, 1], device="cuda")
        _, permuted_prediction = paired(
            test_q[:, permutation], test_response[:, permutation],
            target_operator=test_q[:, 6], return_counterfactual=True,
        )
        independent_prediction = _predict(
            independent, independent_test_q, independent_test_response, independent_test_q[:, 6]
        )
        independent_wrong = _predict(
            independent, independent_test_q, independent_test_response, independent_test_q[:, 7]
        )
        paired_relative = _relative_mse(paired_prediction, test_response[:, 6])
        killed_relative = _relative_mse(killed_prediction, test_response[:, 6])
        wrong_target_relative = _relative_mse(wrong_target_prediction, test_response[:, 6])
        independent_relative = _relative_mse(independent_prediction, independent_test_response[:, 6])
        independent_sensitivity = float(
            nn.functional.mse_loss(independent_prediction, independent_wrong)
            / independent_test_response[:, 6].square().mean().clamp_min(1e-12)
        )
        permutation_error = float((paired_prediction - permuted_prediction).abs().max())
    gates = {
        "set_permutation_error_le_1e-5": permutation_error <= 1e-5,
        "paired_relative_mse_lt_0.50": paired_relative < 0.50,
        "mismatch_is_at_least_1_5x_worse": killed_relative >= 1.5 * paired_relative,
        "wrong_target_is_at_least_1_5x_worse": wrong_target_relative >= 1.5 * paired_relative,
        "operator_independent_relative_mse_lt_0.50": independent_relative < 0.50,
        "operator_independent_sensitivity_lt_0.05": independent_sensitivity < 0.05,
    }
    result = {
        "kind": "paper09_operator_inductive_bias_audit",
        "status": "passed" if all(gates.values()) else "failed",
        "seed": SEED,
        "epochs": args.epochs,
        "synthetic_latent_rank": int(state_map.shape[0]),
        "metrics": {
            "paired_relative_mse": paired_relative,
            "mismatched_relative_mse": killed_relative,
            "wrong_target_relative_mse": wrong_target_relative,
            "permutation_max_abs_error": permutation_error,
            "operator_independent_relative_mse": independent_relative,
            "operator_independent_target_sensitivity": independent_sensitivity,
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
