#!/usr/bin/env python3
"""Pre-clinical mechanism gate for Paper 09's operator-conditioned model.

The synthetic target is a raw linear measurement, not a Phase-KME feature:
    F_i(q) = (q^T L S_i) C.
Thus sign, homogeneity, and superposition are valid here, but are deliberately
not asserted for the nonlinear Phase-KME targets used by the clinical model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from repecg.paper09_counterfactual import CounterfactualOperatorSetModel


SEED = 17
PAIRS = 8
CONTEXT = 3
PHASES = 16
RESPONSE_DIM = 8
LATENT_RANK = 2
DERANGEMENT = torch.tensor([1, 2, 0])


def raw_measurement(
    operators: torch.Tensor, states: torch.Tensor, basis: torch.Tensor, measurement_basis: torch.Tensor
) -> torch.Tensor:
    """Evaluate F_i(q) = (q^T L S_i) C for arbitrary (not only unit) q."""
    amplitude = torch.einsum("nse,er,nr->ns", operators, measurement_basis, states)
    return amplitude[:, :, None, None] * basis[None, None]


def _world(
    records: int,
    basis: torch.Tensor,
    measurement_basis: torch.Tensor,
    *,
    operator_dependent: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.randn(records, LATENT_RANK, device=basis.device)
    operators = nn.functional.normalize(torch.randn(records, PAIRS, 8, device=basis.device), dim=-1)
    if operator_dependent:
        responses = raw_measurement(operators, states, basis, measurement_basis)
    else:
        responses = states[:, :1, None, None].expand(-1, PAIRS, PHASES, RESPONSE_DIM) * basis
    return operators, responses, states


def _predict(
    model: CounterfactualOperatorSetModel,
    operators: torch.Tensor,
    responses: torch.Tensor,
    target: torch.Tensor,
    *,
    context: int = CONTEXT,
) -> torch.Tensor:
    _, prediction = model(
        operators[:, :context], responses[:, :context], target_operator=target, return_counterfactual=True
    )
    return prediction


def _fit(
    operators: torch.Tensor, responses: torch.Tensor, *, response_mismatch: bool, epochs: int
) -> CounterfactualOperatorSetModel:
    model = CounterfactualOperatorSetModel(response_dim=RESPONSE_DIM).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    derangement = DERANGEMENT.to(device="cuda")
    for _ in range(epochs):
        order = torch.randperm(len(operators), device="cuda")
        for start in range(0, len(order), 128):
            rows = order[start:start + 128]
            q, response = operators[rows], responses[rows]
            context_response = response[:, :CONTEXT]
            if response_mismatch:
                # This is surgical: Q and F context marginals are unchanged in
                # every record; only q_j <-> F_j association is destroyed.
                context_response = context_response[:, derangement]
            _, prediction = model(
                q[:, :CONTEXT], context_response, target_operator=q[:, 6], return_counterfactual=True
            )
            loss = nn.functional.mse_loss(prediction, response[:, 6])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(prediction, target) / target.square().mean().clamp_min(1e-12))


def _oracle_prediction(
    operators: torch.Tensor,
    responses: torch.Tensor,
    target: torch.Tensor,
    basis: torch.Tensor,
    measurement_basis: torch.Tensor,
    *,
    context: int,
) -> torch.Tensor:
    observed = (responses[:, :context] * basis).sum(dim=(-1, -2)) / basis.square().sum()
    system = operators[:, :context] @ measurement_basis
    state_hat = (torch.linalg.pinv(system) @ observed.unsqueeze(-1)).squeeze(-1)
    amplitude = torch.einsum("ne,er,nr->n", target, measurement_basis, state_hat)
    return amplitude[:, None, None] * basis


def _sigma_min(operators: torch.Tensor, measurement_basis: torch.Tensor, context: int) -> torch.Tensor:
    return torch.linalg.svdvals(operators[:, :context] @ measurement_basis)[:, -1]


def _ridge_baseline(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor) -> torch.Tensor:
    train_x = torch.cat((train_x, torch.ones(len(train_x), 1, device=train_x.device)), dim=1)
    test_x = torch.cat((test_x, torch.ones(len(test_x), 1, device=test_x.device)), dim=1)
    weights = torch.linalg.solve(
        train_x.T @ train_x + 1e-3 * torch.eye(train_x.shape[1], device=train_x.device),
        train_x.T @ train_y.flatten(1),
    )
    return (test_x @ weights).reshape(len(test_x), PHASES, RESPONSE_DIM)


def _angular_targets(target: torch.Tensor) -> list[tuple[float, torch.Tensor]]:
    orthogonal = torch.randn_like(target)
    orthogonal = orthogonal - (orthogonal * target).sum(dim=1, keepdim=True) * target
    orthogonal = nn.functional.normalize(orthogonal, dim=1)
    return [
        (float(angle), torch.cos(torch.tensor(angle, device=target.device)) * target
         + torch.sin(torch.tensor(angle, device=target.device)) * orthogonal)
        for angle in (0.0, torch.pi / 6, torch.pi / 3, torch.pi / 2)
    ]


def _raw_measurement_laws(
    states: torch.Tensor, basis: torch.Tensor, measurement_basis: torch.Tensor
) -> dict[str, float]:
    q1 = torch.randn(len(states), 1, 8, device=states.device)
    q2 = torch.randn(len(states), 1, 8, device=states.device)
    alpha, beta = 1.7, -0.4
    f1 = raw_measurement(q1, states, basis, measurement_basis)
    f2 = raw_measurement(q2, states, basis, measurement_basis)
    sign = raw_measurement(-q1, states, basis, measurement_basis)
    homogeneous = raw_measurement(alpha * q1, states, basis, measurement_basis)
    superposed = raw_measurement(alpha * q1 + beta * q2, states, basis, measurement_basis)
    return {
        "sign_max_abs_error": float((sign + f1).abs().max()),
        "homogeneity_max_abs_error": float((homogeneous - alpha * f1).abs().max()),
        "superposition_max_abs_error": float((superposed - alpha * f1 - beta * f2).abs().max()),
    }


def _observability_curve(
    sigma_min: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor
) -> list[dict[str, float | int]]:
    quantiles = torch.quantile(sigma_min, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0], device="cuda"))
    result = []
    for low, high in zip(quantiles[:-1], quantiles[1:]):
        rows = (sigma_min >= low) & (sigma_min <= high)
        result.append({
            "sigma_min_low": float(low),
            "sigma_min_high": float(high),
            "records": int(rows.sum()),
            "relative_mse": _relative_mse(prediction[rows], target[rows]),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False  # A100 validation workaround.
    torch.manual_seed(SEED)
    basis = torch.randn(PHASES, RESPONSE_DIM, device="cuda")
    measurement_basis = torch.linalg.qr(torch.randn(8, LATENT_RANK, device="cuda")).Q[:, :LATENT_RANK]
    train_q, train_response, _ = _world(768, basis, measurement_basis, operator_dependent=True)
    test_q, test_response, test_state = _world(384, basis, measurement_basis, operator_dependent=True)
    paired = _fit(train_q, train_response, response_mismatch=False, epochs=args.epochs)
    mismatched = _fit(train_q, train_response, response_mismatch=True, epochs=args.epochs)
    independent_q, independent_response, _ = _world(768, basis, measurement_basis, operator_dependent=False)
    independent_test_q, independent_test_response, _ = _world(384, basis, measurement_basis, operator_dependent=False)
    independent = _fit(independent_q, independent_response, response_mismatch=False, epochs=args.epochs)

    derangement = DERANGEMENT.to(device="cuda")
    with torch.inference_mode():
        target = test_q[:, 6]
        paired_prediction = _predict(paired, test_q, test_response, target)
        mismatched_prediction = _predict(mismatched, test_q, test_response, target)
        response_permuted_prediction = _predict(paired, test_q, test_response[:, derangement], target)
        wrong_target_prediction = _predict(paired, test_q, test_response, test_q[:, 7])
        _, joint_permuted_prediction = paired(
            test_q[:, derangement], test_response[:, derangement],
            target_operator=target, return_counterfactual=True,
        )
        independent_prediction = _predict(
            independent, independent_test_q, independent_test_response, independent_test_q[:, 6]
        )
        independent_wrong = _predict(
            independent, independent_test_q, independent_test_response, independent_test_q[:, 7]
        )
        oracle_prediction = _oracle_prediction(test_q, test_response, target, basis, measurement_basis, context=CONTEXT)
        paired_relative = _relative_mse(paired_prediction, test_response[:, 6])
        mismatched_relative = _relative_mse(mismatched_prediction, test_response[:, 6])
        response_permuted_relative = _relative_mse(response_permuted_prediction, test_response[:, 6])
        wrong_target_relative = _relative_mse(wrong_target_prediction, test_response[:, 6])
        independent_relative = _relative_mse(independent_prediction, independent_test_response[:, 6])
        independent_sensitivity = _relative_mse(independent_prediction - independent_wrong, independent_test_response[:, 6])
        blind_error = _relative_mse(torch.zeros_like(test_response[:, 6]), test_response[:, 6])
        oracle_error = _relative_mse(oracle_prediction, test_response[:, 6])
        oracle_gap_closed = (blind_error - paired_relative) / max(blind_error - oracle_error, 1e-12)
        sigma_min = _sigma_min(test_q, measurement_basis, CONTEXT)
        angular_curve = []
        for angle, angular_target in _angular_targets(target):
            prediction = _predict(paired, test_q, test_response, angular_target)
            truth = raw_measurement(angular_target[:, None], test_state, basis, measurement_basis)[:, 0]
            angular_curve.append({"angle_radians": angle, "relative_mse": _relative_mse(prediction, truth)})
        one_context_prediction = _predict(paired, test_q, test_response, target, context=1)
        duplicate_q = test_q[:, :1].expand(-1, CONTEXT, -1)
        duplicate_response = test_response[:, :1].expand(-1, CONTEXT, -1, -1)
        duplicate_prediction = _predict(paired, duplicate_q, duplicate_response, target)
        duplicate_oracle = _oracle_prediction(duplicate_q, duplicate_response, target, basis, measurement_basis, context=CONTEXT)
        geometry_only = _ridge_baseline(train_q[:, :CONTEXT].flatten(1), train_response[:, 6], test_q[:, :CONTEXT].flatten(1))
        target_only = _ridge_baseline(train_q[:, 6], train_response[:, 6], target)
        laws = _raw_measurement_laws(test_state, basis, measurement_basis)

    paired.train()
    paired.zero_grad(set_to_none=True)
    _, gradient_prediction = paired(
        train_q[:8, :CONTEXT], train_response[:8, :CONTEXT],
        target_operator=train_q[:8, 6], return_counterfactual=True,
    )
    nn.functional.mse_loss(gradient_prediction, train_response[:8, 6]).backward()
    gradients = [parameter.grad for parameter in paired.parameters() if parameter.grad is not None]
    finite_nonzero_gradient = bool(gradients) and all(torch.isfinite(item).all() for item in gradients) and any(
        bool(item.abs().max() > 0) for item in gradients
    )

    response_only_change = _relative_mse(response_permuted_prediction - paired_prediction, test_response[:, 6])
    permutation_error = float((paired_prediction - joint_permuted_prediction).abs().max())
    rank_one_sigma = _sigma_min(duplicate_q, measurement_basis, CONTEXT)
    gates = {
        "joint_set_permutation_error_le_1e-5": permutation_error <= 1e-5,
        "paired_relative_mse_lt_0.50": paired_relative < 0.50,
        "surgical_mismatch_training_is_at_least_1_5x_worse": mismatched_relative >= 1.5 * paired_relative,
        "response_only_permutation_changes_prediction_ge_0.05": response_only_change >= 0.05,
        "wrong_target_is_at_least_1_5x_worse": wrong_target_relative >= 1.5 * paired_relative,
        "operator_independent_relative_mse_lt_0.50": independent_relative < 0.50,
        "operator_independent_sensitivity_lt_0.05": independent_sensitivity < 0.05,
        "all_three_context_cases_are_identifiable": bool((sigma_min > 1e-6).all()),
        "raw_sign_law_le_1e-5": laws["sign_max_abs_error"] <= 1e-5,
        "raw_homogeneity_law_le_1e-5": laws["homogeneity_max_abs_error"] <= 1e-5,
        "raw_superposition_law_le_1e-5": laws["superposition_max_abs_error"] <= 1e-5,
        "geometry_only_baseline_is_q_blind": _relative_mse(geometry_only, test_response[:, 6]) >= 0.90,
        "target_only_baseline_is_q_blind": _relative_mse(target_only, test_response[:, 6]) >= 0.90,
        "duplicate_context_is_declared_rank_deficient": bool((rank_one_sigma <= 1e-6).all()),
        "end_to_end_gradients_finite_and_nonzero": finite_nonzero_gradient,
    }
    result = {
        "kind": "paper09_operator_inductive_bias_audit",
        "status": "passed" if all(gates.values()) else "failed",
        "model_version": "paired_cross_moment_plus_gram_v1",
        "seed": SEED,
        "epochs": args.epochs,
        "synthetic_contract": {
            "latent_state": "S in R^2",
            "measurement_basis": "L in R^(8x2)",
            "raw_response": "F_i(q) = (q^T L S_i) C",
            "q_assignment": "independent of S; unit-sphere for training/evaluation",
            "measurement_laws_scope": "raw synthetic response only; not Phase-KME features",
            "context_pairs": CONTEXT,
        },
        "identifiability": {
            "condition": "rank(Q L) = 2",
            "three_context_sigma_min": {"min": float(sigma_min.min()), "median": float(sigma_min.median()), "max": float(sigma_min.max())},
            "rank_one_duplicate_sigma_min_max": float(rank_one_sigma.max()),
            "observability_curve_diagnostic": _observability_curve(sigma_min, paired_prediction, test_response[:, 6]),
            "context_count_diagnostic": {
                "m1_relative_mse": _relative_mse(one_context_prediction, test_response[:, 6]),
                "m3_duplicate_rank1_relative_mse": _relative_mse(duplicate_prediction, test_response[:, 6]),
                "m3_duplicate_rank1_oracle_relative_mse": _relative_mse(duplicate_oracle, test_response[:, 6]),
                "m3_observable_relative_mse": paired_relative,
            },
        },
        "metrics": {
            "paired_relative_mse": paired_relative,
            "surgical_mismatched_training_relative_mse": mismatched_relative,
            "response_only_permuted_relative_mse": response_permuted_relative,
            "response_only_prediction_change_relative": response_only_change,
            "wrong_target_relative_mse": wrong_target_relative,
            "joint_permutation_max_abs_error": permutation_error,
            "operator_independent_relative_mse": independent_relative,
            "operator_independent_target_sensitivity": independent_sensitivity,
            "analytic_oracle_relative_mse": oracle_error,
            "q_blind_relative_mse": blind_error,
            "oracle_gap_closed": oracle_gap_closed,
            "geometry_only_baseline_relative_mse": _relative_mse(geometry_only, test_response[:, 6]),
            "target_only_baseline_relative_mse": _relative_mse(target_only, test_response[:, 6]),
            "raw_measurement_laws": laws,
            "query_angle_curve_diagnostic": angular_curve,
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
