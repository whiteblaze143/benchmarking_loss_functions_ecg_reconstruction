"""Deterministic synthetic eligibility gates for Paper 13's surgery claim.

Protocol: P13-SURGERY-ELIGIBILITY-v1 (IMMUTABLE — failure recorded)
Pivot branch: P13-REPLACEMENT-SENSITIVITY-V2 (see INDUCTIVE_BIAS_AUDIT.md)

Four worlds are evaluated:

  World A — Directed chain with mediation (non-tautological).
    X2 = 0.9*X1 + ε2,  X3 = 0.9*X2 + ε3,  Y = 0.4*X1 + X3 + εY.
    Feature-only replacement (set X1=0, keep X2_obs, X3_obs stale) captures
    the direct X1→Y pathway (nonzero effect) but leaves the mediated
    X1→X2→X3→Y pathway unaccounted.  Counterfactual recovery error
    E[|ŷ_replace − Y_do|] is large compared to the irreducible baseline.

  World B — Confounded proxy.
    U → X (proxy),  U → Y (outcome),  X ↛ Y structurally.
    A linear predictor sensitive to X is therefore sensitive to replacement,
    but do(X=0) has zero structural effect on Y.

  World C — Redundant cell.
    Z_k = 0.5*Z_left + 0.5*Z_right + ε_k (ε_k ≪ main signal).
    A full linear predictor fitted on (Z_left, Z_right, Z_k) assigns near-zero
    weight to Z_k.  On-support conditional replacement of Z_k produces
    negligible incremental sensitivity; off-support replacement does not.
    This confirms that the replacement mechanism does not manufacture
    importance from redundancy when the reference is on-support.

  World D — Cyclic phase graph.
    Structural statement: a directed cycle has no unique forward causal order
    without an explicit dynamic SCM or cut-point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_world_a(rng: np.random.Generator, samples: int) -> dict[str, object]:
    """Directed-chain world: feature-only replacement has nonzero but wrong effect.

    Ground-truth structural equations:
      X1 ~ N(0, 1)
      X2 = 0.9*X1 + ε2,   ε2 ~ N(0, 0.3²)
      X3 = 0.9*X2 + ε3,   ε3 ~ N(0, 0.3²)
      Y  = 0.4*X1 + X3 + εY,  εY ~ N(0, 0.1²)

    Intervention: do(X1 = 0).

    Feature-only replacement:  sets X1=0 in the predictor but keeps X2_obs,
    X3_obs unchanged.  This captures the direct effect 0.4*X1 (nonzero) but
    leaves the mediated path X1→X2→X3→Y stale.

    Propagated intervention:  re-draws X2_do = 0.9*0 + ε2, X3_do = 0.9*X2_do + ε3
    using the SAME noise realisations (twin-network / abduction-action-prediction).
    Y_do = 0.4*0 + X3_do + εY.

    Decisive statistic: counterfactual recovery error
      E[|ŷ_feature_only − Y_do|]   vs.   E[|ŷ_propagated − Y_do|].
    The propagated prediction should be close to irreducible noise; the feature-
    only prediction should be substantially larger.
    """
    x1 = rng.normal(size=samples)
    noise2 = rng.normal(scale=0.3, size=samples)   # fixed — shared with counterfactual
    noise3 = rng.normal(scale=0.3, size=samples)   # fixed — shared with counterfactual
    noise_y = rng.normal(scale=0.1, size=samples)  # fixed — shared with counterfactual

    x2_obs = 0.9 * x1 + noise2
    x3_obs = 0.9 * x2_obs + noise3
    y_obs = 0.4 * x1 + x3_obs + noise_y  # observational outcome

    # True counterfactual under do(X1 = 0) — twin network with same noise draws
    x1_do = np.zeros(samples)
    x2_do = 0.9 * x1_do + noise2    # propagated via same ε2
    x3_do = 0.9 * x2_do + noise3    # propagated via same ε3
    y_do = 0.4 * x1_do + x3_do + noise_y  # true counterfactual Y

    # Feature-only predictor: uses the oracle linear model Y = 0.4*X1 + X3,
    # but with X1 replaced and X3 left at observed value.
    # This is exactly what a replacement-only operator does: it edits X1 in the
    # feature vector but does not propagate the change through the SCM.
    y_hat_feature_only = 0.4 * x1_do + x3_obs   # X1→Y direct: 0; X3 stale

    # Propagated predictor: uses propagated X3_do (structural counterfactual).
    y_hat_propagated = 0.4 * x1_do + x3_do       # = y_do − εY (irreducible)

    # Feature-only captures some effect (direct X1→Y path), but is wrong about
    # the mediated path.
    feature_only_effect = float(np.mean(np.abs(y_obs - y_hat_feature_only)))
    feature_only_recovery_error = float(np.mean(np.abs(y_hat_feature_only - y_do)))
    propagated_recovery_error = float(np.mean(np.abs(y_hat_propagated - y_do)))

    # Gate: feature-only effect must be nonzero (not tautologically zero) AND
    # feature-only recovery error must substantially exceed the propagated baseline.
    gate_pass = (
        feature_only_effect > 0.05                    # nonzero but wrong
        and feature_only_recovery_error > 0.1         # recovery failure is substantial
        and feature_only_recovery_error > 3.0 * propagated_recovery_error  # ratio test
    )

    return {
        "feature_only_effect_on_observational_prediction": feature_only_effect,
        "feature_only_counterfactual_recovery_error": feature_only_recovery_error,
        "propagated_counterfactual_recovery_error": propagated_recovery_error,
        "ratio_feature_to_propagated_recovery_error": (
            feature_only_recovery_error / max(propagated_recovery_error, 1e-9)
        ),
        "demonstrates_propagation_is_required": gate_pass,
        "interpretation": (
            "Feature-only replacement captures the direct X1→Y path (nonzero effect) "
            "but leaves the mediated path X1→X2→X3→Y stale. "
            "The counterfactual recovery error is large relative to the irreducible baseline."
        ),
    }


def _run_world_b(rng: np.random.Generator, samples: int) -> dict[str, object]:
    """Confounded proxy world: predictive sensitivity ≠ causal effect.

    U → proxy X,  U → outcome Y,  X has no structural effect on Y.
    A predictor trained on X is sensitive to X=0 substitution, but
    the true do(X=0) effect on Y is zero.
    """
    u = rng.normal(size=samples)
    proxy = u + rng.normal(scale=0.2, size=samples)
    outcome = u + rng.normal(scale=0.2, size=samples)

    # OLS coefficient of Y on X (confounded estimate)
    slope = float(np.dot(proxy, outcome) / np.dot(proxy, proxy))
    # Mean absolute sensitivity of the predictor to proxy=0 substitution
    predictor_effect = float(np.mean(np.abs(slope * proxy)))
    true_do_effect = 0.0  # structural: X has no edge to Y

    gate_pass = predictor_effect > 0.1 and true_do_effect == 0.0

    return {
        "mean_absolute_predictor_sensitivity_to_replacement": predictor_effect,
        "true_structural_do_effect": true_do_effect,
        "demonstrates_prediction_sensitivity_is_not_causality": gate_pass,
        "interpretation": (
            "A predictor trained on a confounded proxy X can exhibit substantial "
            "replacement sensitivity even when do(X) has zero structural effect on Y. "
            "This confirms that S_k(X) > 0 does not imply a causal effect."
        ),
    }


def _run_world_c(rng: np.random.Generator, samples: int) -> dict[str, object]:
    """Redundant-cell world: on-support replacement does not manufacture importance.

    Z_left, Z_right ~ N(0, 1)  (informative context cells)
    Z_k = 0.5*Z_left + 0.5*Z_right + ε_k,   ε_k ~ N(0, 0.05²)
    Y = 0.6*Z_left + 0.4*Z_right + ε_Y,      ε_Y ~ N(0, 0.1²)

    Z_k is essentially determined by its neighbours and carries no incremental
    predictive information about Y.

    A full linear predictor fitted on (Z_left, Z_right, Z_k) via OLS should
    assign a near-zero coefficient to Z_k.  Replacing Z_k with a sample drawn
    from its conditional distribution given (Z_left, Z_right) should therefore
    produce negligible incremental sensitivity in the predictor.

    Off-support replacement (Z_k replaced by a draw from a shifted distribution)
    should produce a larger change, demonstrating that it is the on-support
    constraint — not a fundamental feature of the replacement mechanism — that
    suppresses spurious importance.
    """
    z_left = rng.normal(size=samples)
    z_right = rng.normal(size=samples)
    eps_k = rng.normal(scale=0.05, size=samples)
    z_k = 0.5 * z_left + 0.5 * z_right + eps_k
    eps_y = rng.normal(scale=0.1, size=samples)
    y = 0.6 * z_left + 0.4 * z_right + eps_y

    # Fit OLS: Y ~ Z_left + Z_right + Z_k
    X_design = np.column_stack([z_left, z_right, z_k, np.ones(samples)])
    coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    w_left, w_right, w_k, _intercept = coeffs

    # On-support conditional replacement: Z_k_tilde ~ 0.5*Z_left + 0.5*Z_right + ε_new
    eps_new = rng.normal(scale=0.05, size=samples)
    z_k_on_support = 0.5 * z_left + 0.5 * z_right + eps_new

    # Off-support replacement: draw from a shifted distribution (far from train support)
    z_k_off_support = rng.normal(loc=5.0, scale=0.5, size=samples)

    # Predictor outputs
    y_hat_original = w_left * z_left + w_right * z_right + w_k * z_k
    y_hat_on_support = w_left * z_left + w_right * z_right + w_k * z_k_on_support
    y_hat_off_support = w_left * z_left + w_right * z_right + w_k * z_k_off_support

    on_support_sensitivity = float(np.mean(np.abs(y_hat_original - y_hat_on_support)))
    off_support_sensitivity = float(np.mean(np.abs(y_hat_original - y_hat_off_support)))

    # Gate: w_k must be near zero (redundancy confirmed), and on-support sensitivity
    # must be << off-support sensitivity (support constraint matters).
    w_k_is_small = abs(w_k) < 0.15
    support_constraint_matters = off_support_sensitivity > 5.0 * on_support_sensitivity
    gate_pass = w_k_is_small and support_constraint_matters

    return {
        "ols_coefficient_z_left": float(w_left),
        "ols_coefficient_z_right": float(w_right),
        "ols_coefficient_z_k": float(w_k),
        "on_support_replacement_sensitivity": on_support_sensitivity,
        "off_support_replacement_sensitivity": off_support_sensitivity,
        "on_off_support_sensitivity_ratio": (
            off_support_sensitivity / max(on_support_sensitivity, 1e-9)
        ),
        "demonstrates_no_spurious_importance_from_redundant_cell": gate_pass,
        "interpretation": (
            "When Z_k carries no incremental predictive information (near-zero OLS "
            "coefficient), on-support conditional replacement produces negligible "
            "sensitivity.  Off-support replacement produces large sensitivity driven "
            "by distribution shift, not by the cell's information content. "
            "This validates that the on-support constraint is load-bearing."
        ),
    }


def _run_world_d() -> dict[str, object]:
    """Cyclic phase graph: structural statement only (no simulation needed)."""
    return {
        "has_unique_forward_intervention_order": False,
        "reason": (
            "A directed cycle requires an explicit dynamic SCM or a cut-point. "
            "Phase order alone does not identify a forward causal direction. "
            "The rotation-invariant architecture makes this unavoidable: "
            "absolute phase identity is not available to the classifier, "
            "so phase-specific absolute causal claims cannot be grounded."
        ),
        "pivot_note": (
            "Rotation invariance is not a defect for the V2 estimand. "
            "An invariant classifier can support an equivariant sensitivity field: "
            "S_{k+s}(R_s X) = S_k(X), which is mathematically coherent. "
            "The V2 claim is local sensitivity, not absolute phase causality."
        ),
    }


def run(seed: int, samples: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    world_a = _run_world_a(rng, samples)
    world_b = _run_world_b(rng, samples)
    world_c = _run_world_c(rng, samples)
    world_d = _run_world_d()

    # Overall gate: P13-CAUSAL-SURGERY = INELIGIBLE (immutable)
    # The worlds demonstrate why the original causal surgery claim fails.
    overall_gate = {
        "status": "P13-CAUSAL-SURGERY_INELIGIBLE",
        "pivot_branch": "P13-REPLACEMENT-SENSITIVITY-V2",
        "reasons": [
            "World A: Feature-only replacement produces nonzero but wrong effect — "
            "it captures the direct X1→Y path but leaves the mediated path stale. "
            "Counterfactual recovery error substantially exceeds the propagated baseline.",
            "World B: Predictive sensitivity to proxy replacement is nonzero even when "
            "the structural do-effect is exactly zero. S_k(X) > 0 does not imply causality.",
            "World C: On-support conditional replacement of a redundant cell produces "
            "negligible sensitivity. The mechanism does not manufacture importance from "
            "redundancy when the reference is correctly constrained to the training distribution.",
            "World D: The cyclic phase architecture does not identify a forward causal order. "
            "Absolute phase identity is unavailable; a rotation-equivariant sensitivity field "
            "is a coherent and mathematically defensible alternative estimand.",
        ],
        "world_a_gate_pass": world_a["demonstrates_propagation_is_required"],
        "world_b_gate_pass": world_b["demonstrates_prediction_sensitivity_is_not_causality"],
        "world_c_gate_pass": world_c["demonstrates_no_spurious_importance_from_redundant_cell"],
    }

    return {
        "protocol": "P13-SURGERY-ELIGIBILITY-v1",
        "seed": seed,
        "samples": samples,
        "worlds": {
            "directed_chain_with_mediation": world_a,
            "confounded_proxy": world_b,
            "redundant_cell_on_support": world_c,
            "cyclic_phase_graph": world_d,
        },
        "gate": overall_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P13-SURGERY-ELIGIBILITY-v1 synthetic gate runner."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13001)
    parser.add_argument("--samples", type=int, default=200_000)
    args = parser.parse_args()
    result = run(args.seed, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
