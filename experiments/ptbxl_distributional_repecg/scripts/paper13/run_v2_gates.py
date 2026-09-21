"""V2 Synthetic Gate Runner: P13-REPLACEMENT-SENSITIVITY-V2

Tests all seven V2 gates in fully synthetic, deterministic worlds.
Real-data testing (V2-G7) is blocked until G0-G6 pass.

Gate summary:
  V2-G0  execution     — identity, locality, all controls alter the tensor
  V2-G1  equivariance  — S(R_s X) = R_s S(X); also shows equivariant vs
                         non-equivariant sampler contrast
  V2-G2  support       — context-distance, replacement-distance, donor
                         coverage diagnostics on synthetic population
  V2-G3  matched destroyers — magnitude/context-matched wrong-phase control
  V2-G4  synthetic localization — informative phase region recovered and
                         rotates correctly under cyclic shift
  V2-G5  false causality — confounded proxy stays sensitive; claim boundary
  V2-G6  estimator stability — sensitivity converges as M_draws increases

Mathematical object:
  X = (Z_0, ..., Z_{15}) ∈ R^{16 × D}
  f(X) ∈ R^C  (logits, C classes)
  S_k(X) = E_{Z̃_k ~ q_k(·|C_k(X))}[|f(X) - f(X^{k←Z̃_k})|]  (mean over C)
  D_k,c(X) = E_{Z̃_k}[ℓ_c(X^{k←Z̃_k}) - ℓ_c(X)]  (signed, per class)

Equivariance requirement:
  S_{k+s}(R_s X) = S_k(X)  for all k, s
  where R_s is cyclic shift by s along the phase axis.

  Classifier invariance f(R_s X) = f(X) is necessary but NOT sufficient.
  The sampler must also commute:
    q_{k+s}(·|R_s X) = R_s q_k(·|X)
  i.e., it must use only RELATIVE phase context, not absolute phase index k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=_NumpyEncoder) + "\n")
    tmp.replace(path)


def _roll_axis0(x: np.ndarray, s: int) -> np.ndarray:
    """Cyclic shift of x along axis 0 (phase axis). x: [16, D]."""
    return np.roll(x, shift=s, axis=0)


def _roll_batch(X: np.ndarray, s: int) -> np.ndarray:
    """Cyclic shift of X along phase axis. X: [B, 16, D]."""
    return np.roll(X, shift=s, axis=1)


# ---------------------------------------------------------------------------
# Synthetic classifier: rotation-invariant by construction (global phase pool)
# ---------------------------------------------------------------------------

class SyntheticInvariantClassifier:
    """Linear classifier over global phase-averaged features.

    f(X) = W @ mean(X, axis=phase)  →  rotation-invariant by construction.
    Uses a fixed weight matrix seeded from rng.
    """

    def __init__(self, D: int, C: int, rng: np.random.Generator):
        self.W = rng.normal(size=(C, D)) * 0.3  # [C, D]
        self.b = rng.normal(size=(C,)) * 0.1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: [B, 16, D] → logits: [B, C]."""
        z = X.mean(axis=1)  # [B, D] — global phase average → rotation invariant
        return z @ self.W.T + self.b  # [B, C]

    def sensitivity_field(
        self,
        X: np.ndarray,           # [B, 16, D]
        sampler: "ReferenceSampler",
        M: int = 16,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Estimate S_k(X) = E[|f(X) - f(X^{k←Z̃_k})|_mean-over-C] for all k.

        Returns S: [16] (averaged over batch B and classes C).
        """
        if rng is None:
            rng = np.random.default_rng(0)
        B, K, D = X.shape
        baseline = self(X)  # [B, C]
        S = np.zeros(K)
        for k in range(K):
            diffs = []
            for _ in range(M):
                Z_tilde = sampler.sample(X, k, rng)  # [B, D]
                X_rep = X.copy()
                X_rep[:, k, :] = Z_tilde
                logits_rep = self(X_rep)  # [B, C]
                diffs.append(np.abs(baseline - logits_rep).mean(axis=1))  # [B]
            S[k] = np.mean(diffs)  # scalar
        return S  # [16]


# ---------------------------------------------------------------------------
# Reference samplers
# ---------------------------------------------------------------------------

class ReferenceSampler:
    """Base class for conditional phase reference samplers."""

    def sample(self, X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """Return Z̃_k of shape [B, D], one sample per batch element."""
        raise NotImplementedError


class EquivariantLocalSampler(ReferenceSampler):
    """Rotation-equivariant sampler: uses only RELATIVE context window.

    q_k(·|X) = N(mean of ±w relative neighbors of k, σ²I)

    This is equivariant because:
      q_{k+s}(·|R_s X) uses neighbors at {k+s±w} in R_s X
                     = neighbors at {k±w} in X, shifted by s
                     = R_s q_k(·|X)   ✓

    The key property: no dependence on absolute phase index k, only
    on relative positions of neighbors.
    """

    def __init__(self, window: int = 2, sigma: float = 0.1):
        self.window = window
        self.sigma = sigma

    def sample(self, X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """X: [B, 16, D]. Returns Z̃_k: [B, D]."""
        B, K, D = X.shape
        # Collect relative neighbors (excluding k itself) using circular indexing
        neighbor_indices = [
            (k + offset) % K
            for offset in range(-self.window, self.window + 1)
            if offset != 0
        ]
        context = X[:, neighbor_indices, :].mean(axis=1)  # [B, D] relative mean
        noise = rng.normal(scale=self.sigma, size=(B, D))
        return context + noise  # [B, D]


class NonEquivariantAbsoluteSampler(ReferenceSampler):
    """Phase-specific absolute sampler — BREAKS equivariance.

    q_k(·|X) depends on the absolute index k via a fixed per-phase bias vector.
    This means q_{k+s}(·|R_s X) ≠ R_s q_k(·|X).

    Used in G1 to demonstrate that classifier invariance is not sufficient:
    the field S(R_s X) ≠ R_s S(X) when the sampler is phase-specific.
    """

    def __init__(self, K: int, D: int, rng: np.random.Generator, window: int = 2, sigma: float = 0.1):
        self.window = window
        self.sigma = sigma
        # Per-phase bias vectors — these are absolute-index-dependent (breaks equivariance)
        self.phase_bias = rng.normal(scale=0.5, size=(K, D))

    def sample(self, X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """X: [B, 16, D]. Returns Z̃_k: [B, D]."""
        B, K, D = X.shape
        neighbor_indices = [
            (k + offset) % K
            for offset in range(-self.window, self.window + 1)
            if offset != 0
        ]
        context = X[:, neighbor_indices, :].mean(axis=1)  # [B, D]
        noise = rng.normal(scale=self.sigma, size=(B, D))
        # Add absolute-phase-dependent bias — this is the equivariance-breaking term
        absolute_bias = self.phase_bias[k]  # [D], depends on absolute k
        return context + noise + absolute_bias  # [B, D]


# ---------------------------------------------------------------------------
# G0: Operator execution — identity, locality, executable controls
# ---------------------------------------------------------------------------

def run_v2_g0(rng: np.random.Generator, B: int = 32, K: int = 16, D: int = 8) -> dict:
    """V2-G0: The replacement operator R_k(X, Z̃_k) = X^{k←Z̃_k} must satisfy:

    1. Identity: R_k(X, Z_k) = X  (replacing with itself → no change)
    2. Locality: [R_k(X, Z̃_k)]_j = X_j for all j ≠ k
    3. Executable controls: all named controls actually change the tensor

    Identity → S_k^{identity}(X) = 0 to numerical precision.
    """
    X = rng.normal(size=(B, K, D))

    # --- Identity gate ---
    identity_errors = []
    for k in range(K):
        X_rep = X.copy()
        X_rep[:, k, :] = X[:, k, :]  # replace with itself
        diff = np.abs(X_rep - X).max()
        identity_errors.append(float(diff))
    identity_max_error = max(identity_errors)
    identity_pass = identity_max_error == 0.0

    # --- Locality gate ---
    locality_errors = []
    k_target = 5
    Z_tilde = rng.normal(size=(B, D))
    X_rep = X.copy()
    X_rep[:, k_target, :] = Z_tilde
    for j in range(K):
        if j == k_target:
            continue
        diff = np.abs(X_rep[:, j, :] - X[:, j, :]).max()
        locality_errors.append(float(diff))
    locality_max_error = max(locality_errors)
    locality_pass = locality_max_error == 0.0

    # --- Executable controls gate ---
    # Every named control must produce X_rep ≠ X (not a metadata-only variant)
    controls = {
        "identity_noop": X.copy(),  # this should equal X — verify identity is zero
        "conditional_replacement": X.copy(),
        "unconditional_replacement": X.copy(),
        "magnitude_matched_wrong_phase": X.copy(),
        "off_support_replacement": X.copy(),
    }

    k = 7
    # conditional_replacement: use relative context (equivariant)
    neighbors = [(k + o) % K for o in [-2, -1, 1, 2]]
    context_mean = X[:, neighbors, :].mean(axis=1)
    controls["conditional_replacement"][:, k, :] = context_mean + rng.normal(scale=0.05, size=(B, D))

    # unconditional_replacement: draw from marginal
    controls["unconditional_replacement"][:, k, :] = rng.normal(size=(B, D))

    # magnitude_matched_wrong_phase: same magnitude as target, at wrong phase j
    j_wrong = (k + 5) % K
    orig_norm = np.linalg.norm(X[:, k, :], axis=1, keepdims=True)  # [B, 1]
    wrong_phase = X[:, j_wrong, :].copy()
    wrong_norm = np.linalg.norm(wrong_phase, axis=1, keepdims=True).clip(min=1e-8)
    magnitude_matched = wrong_phase * (orig_norm / wrong_norm)
    controls["magnitude_matched_wrong_phase"][:, k, :] = magnitude_matched

    # off_support_replacement: draw from far-shifted distribution
    controls["off_support_replacement"][:, k, :] = rng.normal(loc=5.0, scale=0.3, size=(B, D))

    executable_results = {}
    for ctrl_name, X_ctrl in controls.items():
        changes_tensor = not np.allclose(X_ctrl, X)
        mean_diff = float(np.abs(X_ctrl - X).mean())
        executable_results[ctrl_name] = {
            "changes_tensor": changes_tensor,
            "mean_absolute_change": mean_diff,
        }

    # identity_noop must NOT change tensor; all others must change it
    noop_correct = not executable_results["identity_noop"]["changes_tensor"]
    all_others_executable = all(
        executable_results[c]["changes_tensor"]
        for c in executable_results
        if c != "identity_noop"
    )
    executable_pass = noop_correct and all_others_executable

    gate_pass = identity_pass and locality_pass and executable_pass

    return {
        "gate": "V2-G0",
        "description": "Operator execution: identity, locality, executable controls",
        "identity_max_error": identity_max_error,
        "locality_max_error": locality_max_error,
        "executable_controls": executable_results,
        "identity_pass": identity_pass,
        "locality_pass": locality_pass,
        "executable_pass": executable_pass,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# G1: Equivariance — S(R_s X) = R_s S(X)
# ---------------------------------------------------------------------------

def run_v2_g1(
    rng: np.random.Generator,
    B: int = 64,
    K: int = 16,
    D: int = 8,
    C: int = 5,
    shifts: list[int] | None = None,
    M: int = 32,
) -> dict:
    """V2-G1: Equivariance gate.

    Tests S(R_s X) = R_s S(X) for an equivariant sampler.
    Also demonstrates that a non-equivariant (absolute-phase) sampler breaks
    this even though the classifier itself is rotation-invariant.

    Key point: classifier invariance f(R_s X) = f(X) is necessary but not
    sufficient. The sampler must also commute with rotation.
    """
    if shifts is None:
        shifts = [1, 3, 5, 8, 11]

    clf = SyntheticInvariantClassifier(D, C, rng)
    equivariant_sampler = EquivariantLocalSampler(window=2, sigma=0.1)
    nonequiv_sampler = NonEquivariantAbsoluteSampler(K, D, rng, window=2, sigma=0.1)

    X = rng.normal(size=(B, K, D))

    # Fix rng state for deterministic reference draws — share across original and rotated
    rng_fixed_seed = 13042

    def get_S(X_input: np.ndarray, sampler: ReferenceSampler) -> np.ndarray:
        fixed_rng = np.random.default_rng(rng_fixed_seed)
        return clf.sensitivity_field(X_input, sampler, M=M, rng=fixed_rng)

    # Classifier invariance check (prerequisite, not the main gate)
    logits_orig = clf(X)
    s_test = shifts[0]
    logits_rotated = clf(_roll_batch(X, s_test))
    clf_invariance_error = float(np.abs(logits_orig - logits_rotated).max())

    S_orig_equiv = get_S(X, equivariant_sampler)  # [K]
    S_orig_nonequiv = get_S(X, nonequiv_sampler)

    shift_results = {}
    for s in shifts:
        X_rot = _roll_batch(X, s)

        # For equivariant sampler: use SAME rng seed so draws correspond
        S_rot_equiv = get_S(X_rot, equivariant_sampler)    # [K]
        S_rot_nonequiv = get_S(X_rot, nonequiv_sampler)

        # Equivariance test: S(R_s X) should equal R_s S(X) = roll(S_orig, s)
        S_expected = np.roll(S_orig_equiv, shift=s)         # R_s applied to field
        equiv_error = float(np.abs(S_rot_equiv - S_expected).max())

        S_expected_nonequiv = np.roll(S_orig_nonequiv, shift=s)
        nonequiv_error = float(np.abs(S_rot_nonequiv - S_expected_nonequiv).max())

        # Relative errors
        denom_equiv = float(np.abs(S_expected).mean()) + 1e-9
        denom_nonequiv = float(np.abs(S_expected_nonequiv).mean()) + 1e-9

        shift_results[f"shift_{s}"] = {
            "equivariant_sampler_max_error": equiv_error,
            "equivariant_sampler_relative_error": equiv_error / denom_equiv,
            "nonequivariant_sampler_max_error": nonequiv_error,
            "nonequivariant_sampler_relative_error": nonequiv_error / denom_nonequiv,
        }

    # Tolerance: equivariant sampler should achieve < 0.02 relative error
    # (small residual from finite M estimation noise shared via fixed seed)
    equiv_max_rel = max(
        v["equivariant_sampler_relative_error"] for v in shift_results.values()
    )
    nonequiv_max_rel = max(
        v["nonequivariant_sampler_relative_error"] for v in shift_results.values()
    )

    # Gate: equivariant sampler passes; non-equivariant should fail
    equivariant_gate_pass = equiv_max_rel < 0.02
    nonequivariant_correctly_fails = nonequiv_max_rel > 0.05  # must show meaningful break

    gate_pass = equivariant_gate_pass and nonequivariant_correctly_fails

    return {
        "gate": "V2-G1",
        "description": "Equivariance: S(R_s X) = R_s S(X). Tests equivariant vs non-equivariant sampler.",
        "classifier_invariance_max_error": clf_invariance_error,
        "classifier_is_rotation_invariant": clf_invariance_error < 1e-10,
        "equivariant_sampler_max_relative_error_across_shifts": equiv_max_rel,
        "nonequivariant_sampler_max_relative_error_across_shifts": nonequiv_max_rel,
        "shift_details": shift_results,
        "equivariant_sampler_gate_pass": equivariant_gate_pass,
        "nonequivariant_correctly_breaks_equivariance": nonequivariant_correctly_fails,
        "gate_pass": gate_pass,
        "key_finding": (
            "Classifier invariance f(R_s X) = f(X) is necessary but not sufficient "
            "for S(R_s X) = R_s S(X). The equivariant sampler (relative context only) "
            "achieves near-zero error; the absolute-phase sampler breaks equivariance "
            "by a substantial margin even though the classifier is unchanged."
        ),
    }


# ---------------------------------------------------------------------------
# G2: Support diagnostics
# ---------------------------------------------------------------------------

def run_v2_g2(rng: np.random.Generator, B: int = 256, K: int = 16, D: int = 8) -> dict:
    """V2-G2: Support diagnostics on a synthetic population.

    For each (patient, phase) pair, reports:
      d_context: ||C_k(X_i) - C_k(X_j)||  (context distance to donor)
      d_repl: ||Z_{i,k} - Z̃_{i,k}||       (replacement magnitude)
      donor coverage: unique donors, max reuse

    Uses a near-bijective deterministic nearest-neighbor donor rule
    (avoids unrestricted reuse per Paper 12 lesson).
    """
    # Synthetic training pool: N_train examples
    N_train = 1024
    X_train = rng.normal(size=(N_train, K, D))
    X_eval = rng.normal(size=(B, K, D))

    sampler = EquivariantLocalSampler(window=2, sigma=0.05)

    # For each eval patient and phase k, find nearest training neighbor by context
    # Enforce near-bijective: each training donor used at most ceil(B/N_train)+1 times
    max_reuse_budget = max(2, int(np.ceil(B / N_train)) + 1)

    results_per_phase = {}
    for k in range(K):
        neighbors = [(k + o) % K for o in [-2, -1, 1, 2]]

        # Context: mean of relative neighbors
        C_eval = X_eval[:, neighbors, :].mean(axis=1)     # [B, D]
        C_train = X_train[:, neighbors, :].mean(axis=1)   # [N_train, D]

        # Nearest-neighbor with reuse limit (greedy bijective assignment)
        dists = np.linalg.norm(
            C_eval[:, None, :] - C_train[None, :, :], axis=-1
        )  # [B, N_train]
        reuse_count = np.zeros(N_train, dtype=int)
        donor_ids = np.full(B, -1, dtype=int)

        for i in np.argsort(dists.min(axis=1)):  # process easiest matches first
            # Mask over-used donors
            available = (reuse_count < max_reuse_budget)
            if not available.any():
                available = np.ones(N_train, dtype=bool)  # fallback
            row_dists = dists[i].copy()
            row_dists[~available] = np.inf
            j = int(row_dists.argmin())
            donor_ids[i] = j
            reuse_count[j] += 1

        d_context = np.array([
            float(np.linalg.norm(C_eval[i] - C_train[donor_ids[i]]))
            for i in range(B)
        ])
        d_repl = np.array([
            float(np.linalg.norm(X_eval[i, k, :] - X_train[donor_ids[i], k, :]))
            for i in range(B)
        ])
        unique_donors = int(len(np.unique(donor_ids)))
        max_reuse = int(reuse_count.max())

        results_per_phase[f"phase_{k:02d}"] = {
            "d_context_median": float(np.median(d_context)),
            "d_context_p95": float(np.percentile(d_context, 95)),
            "d_repl_median": float(np.median(d_repl)),
            "d_repl_p95": float(np.percentile(d_repl, 95)),
            "unique_donors": unique_donors,
            "max_donor_reuse": max_reuse,
            "coverage_fraction": unique_donors / B,
        }

    # Summary diagnostics
    all_median_d_context = [v["d_context_median"] for v in results_per_phase.values()]
    all_max_reuse = [v["max_donor_reuse"] for v in results_per_phase.values()]
    all_coverage = [v["coverage_fraction"] for v in results_per_phase.values()]

    # Gate: max donor reuse within budget, context distances reasonable, good coverage
    gate_pass = (
        max(all_max_reuse) <= max_reuse_budget + 1
        and min(all_coverage) > 0.8
        and max(all_median_d_context) < 5.0
    )

    return {
        "gate": "V2-G2",
        "description": "Support diagnostics: context-distance, replacement-distance, donor coverage",
        "n_eval": B,
        "n_train_pool": N_train,
        "max_reuse_budget": max_reuse_budget,
        "summary": {
            "mean_d_context_median_over_phases": float(np.mean(all_median_d_context)),
            "max_donor_reuse_over_phases": int(max(all_max_reuse)),
            "min_coverage_fraction_over_phases": float(min(all_coverage)),
        },
        "per_phase_sample": {k: v for k, v in list(results_per_phase.items())[:4]},
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# G3: Magnitude/context-matched wrong-phase destroyer
# ---------------------------------------------------------------------------

def run_v2_g3(
    rng: np.random.Generator,
    B: int = 128,
    K: int = 16,
    D: int = 8,
    C: int = 5,
    M: int = 16,
) -> dict:
    """V2-G3: Matched destroyers — same perturbation magnitude, wrong phase.

    A wrong-phase control is only a valid destroyer if it matches the
    replacement magnitude of the target phase. Otherwise differences in
    sensitivity may reflect magnitude differences, not phase specificity.

    In a synthetic world where Y only depends on phase 5 and 6, the
    magnitude-matched wrong-phase sensitivity should be substantially lower
    than the target-phase sensitivity.

    This does NOT require wrong-phase sensitivity to always be lower in real
    data (Phase-KME can be redundant). It is a synthetic localization check.
    """
    # Synthetic world: only phases 5 and 6 are informative
    informative_phases = [5, 6]

    # Ground-truth linear classifier: w_k > 0 for k in {5,6}, 0 elsewhere
    W = np.zeros((C, K, D))
    for k in informative_phases:
        W[:, k, :] = rng.normal(size=(C, D)) * 0.5

    def clf_informative(X: np.ndarray) -> np.ndarray:
        """X: [B, K, D] → logits: [B, C]."""
        return np.einsum("bkd,ckd->bc", X, W)

    # Equivariant sampler
    sampler = EquivariantLocalSampler(window=2, sigma=0.3)

    X = rng.normal(size=(B, K, D))
    baseline = clf_informative(X)

    def estimate_sensitivity_matched(k_target: int, k_wrong: int) -> tuple[float, float]:
        """Estimate target and magnitude-matched wrong-phase sensitivity."""
        target_diffs = []
        wrong_diffs = []
        for _ in range(M):
            fixed_rng = np.random.default_rng(rng.integers(1 << 31))
            Z_tilde = sampler.sample(X, k_target, fixed_rng)  # [B, D]
            # Replacement magnitude for target
            d_target = np.linalg.norm(X[:, k_target, :] - Z_tilde, axis=1, keepdims=True)  # [B, 1]

            X_rep_target = X.copy()
            X_rep_target[:, k_target, :] = Z_tilde
            s_target = np.abs(clf_informative(X_rep_target) - baseline).mean(axis=1)  # [B]
            target_diffs.append(float(s_target.mean()))

            # Wrong-phase replacement: magnitude-matched
            Z_wrong_raw = sampler.sample(X, k_wrong, fixed_rng)  # [B, D]
            d_wrong = np.linalg.norm(X[:, k_wrong, :] - Z_wrong_raw, axis=1, keepdims=True).clip(min=1e-8)
            Z_wrong_matched = X[:, k_wrong, :] + (Z_wrong_raw - X[:, k_wrong, :]) * (d_target / d_wrong)

            X_rep_wrong = X.copy()
            X_rep_wrong[:, k_wrong, :] = Z_wrong_matched
            s_wrong = np.abs(clf_informative(X_rep_wrong) - baseline).mean(axis=1)  # [B]
            wrong_diffs.append(float(s_wrong.mean()))

        return float(np.mean(target_diffs)), float(np.mean(wrong_diffs))

    # Compare informative phase vs matched wrong phase
    s_target, s_wrong = estimate_sensitivity_matched(
        k_target=5, k_wrong=3  # k=3 is non-informative
    )
    ratio = s_target / max(s_wrong, 1e-9)

    gate_pass = ratio > 2.0  # informative phase should be substantially more sensitive

    return {
        "gate": "V2-G3",
        "description": "Magnitude/context-matched wrong-phase destroyer (synthetic localization check).",
        "informative_phases": informative_phases,
        "target_phase_mean_sensitivity": s_target,
        "magnitude_matched_wrong_phase_mean_sensitivity": s_wrong,
        "sensitivity_ratio_target_over_wrong": ratio,
        "gate_pass": gate_pass,
        "caveat": (
            "This gate applies to synthetic worlds with known relevance. "
            "In real Phase-KME data, redundancy across phases means wrong-phase "
            "sensitivity need not always be lower — use as descriptive control only."
        ),
    }


# ---------------------------------------------------------------------------
# G4: Synthetic localization — informative region recovered and rotates
# ---------------------------------------------------------------------------

def run_v2_g4(
    rng: np.random.Generator,
    B: int = 512,
    K: int = 16,
    D: int = 8,
    C: int = 5,
    M: int = 32,
    shifts: list[int] | None = None,
    signal_strength: float = 2.5,
) -> dict:
    """V2-G4: Synthetic localization gate (corrected).

    The previous implementation used a non-rotation-invariant classifier that
    read absolute phases 5 and 6. When the input was rotated, the classifier
    still read positions 5 and 6, so peaks never moved — this tested nothing
    about the sensitivity method.

    Correct design: the classifier IS rotation-invariant (global mean pooling),
    and localization arises through heterogeneous DATA DISTRIBUTION:

      Z_k ~ N(μ_k, I),  where μ_k = signal_vector for k in {5, 6}, else 0.

    The invariant classifier f(X) = W @ mean_k(Z_k) is sensitive to replacing
    a phase cell with a zero-signal reference (N(0,I)) in proportion to how far
    that phase deviates from zero. Since phases 5 and 6 have a signal component,
    they produce larger deviations → higher sensitivity. This is localization
    from DATA, not from model weights.

    For rotation consistency: we generate a SHIFTED WORLD where the informative
    phases are at (5+s)%K and (6+s)%K, and verify the sensitivity peaks shift
    correspondingly. This confirms the methodology responds to data structure,
    not to absolute position baked into the model.

    This simultaneously validates:
      (a) relevance localization via data distribution
      (b) world rotation consistency of the sensitivity field
      (c) the methodology does not have an absolute-phase bias
    """
    if shifts is None:
        shifts = [3, 7, 11]

    baseline_informative = [5, 6]

    # Rotation-invariant classifier: global mean pooling
    # Same W used for all worlds (not retrained per shift)
    W = rng.normal(size=(C, D)) * 0.3  # [C, D]
    b = rng.normal(size=(C,)) * 0.05

    def clf_invariant(X: np.ndarray) -> np.ndarray:
        """X: [B, K, D] → logits [B, C]. Rotation invariant."""
        z = X.mean(axis=1)  # [B, D]
        return z @ W.T + b  # [B, C]

    # Unconditional reference sampler: N(0, I) — on-support for non-informative phases,
    # but draws from the zero-mean marginal for all phases.
    def unconditional_sample(B: int, D: int, local_rng: np.random.Generator) -> np.ndarray:
        return local_rng.normal(size=(B, D))

    # Fixed signal vector for all informative phases
    signal_vector = rng.normal(size=(D,))
    signal_vector /= np.linalg.norm(signal_vector) + 1e-9
    signal_vector *= signal_strength  # [D]

    def make_world(informative_phases: list[int]) -> np.ndarray:
        """Draw X ~ N(μ_k, I) where μ_k = signal_vector for informative k."""
        X = rng.normal(size=(B, K, D))
        for k in informative_phases:
            X[:, k, :] += signal_vector[None, :]  # shift mean at informative phases
        return X

    def get_sensitivity_field(X: np.ndarray) -> np.ndarray:
        """S_k(X) for all k using unconditional reference. Returns [K]."""
        baseline = clf_invariant(X)  # [B, C]
        S = np.zeros(K)
        fixed_rng = np.random.default_rng(42)
        for k in range(K):
            diffs = []
            for _ in range(M):
                Z_tilde = unconditional_sample(B, D, fixed_rng)  # [B, D]
                X_rep = X.copy()
                X_rep[:, k, :] = Z_tilde
                diff = np.abs(clf_invariant(X_rep) - baseline).mean(axis=1)  # [B]
                diffs.append(float(diff.mean()))
            S[k] = float(np.mean(diffs))
        return S

    # Baseline world: informative at {5, 6}
    X_base = make_world(baseline_informative)
    S_base = get_sensitivity_field(X_base)

    peak_phases_base = set(int(k) for k in np.argsort(S_base)[-2:])
    correct_localization = peak_phases_base == set(baseline_informative)

    # World rotation consistency: generate world with informative phases at (5+s, 6+s)
    shift_results = {}
    for s in shifts:
        shifted_informative = [(p + s) % K for p in baseline_informative]
        X_shifted = make_world(shifted_informative)
        S_shifted = get_sensitivity_field(X_shifted)

        peak_phases_shifted = set(int(k) for k in np.argsort(S_shifted)[-2:])
        peaks_at_right_location = peak_phases_shifted == set(shifted_informative)

        # Equivariance of the method: sensitivity at shifted phases should equal
        # sensitivity at original phases (same world structure, just relocated).
        # Compare the RANK ORDER: top-2 of S_shifted should be shifted_informative.
        # Also compare magnitudes: S_shifted[shifted_k] ≈ S_base[orig_k]
        orig_info_S = float(np.mean([S_base[p] for p in baseline_informative]))
        shifted_info_S = float(np.mean([S_shifted[p] for p in shifted_informative]))
        magnitude_ratio = shifted_info_S / (orig_info_S + 1e-9)
        magnitude_consistent = 0.7 < magnitude_ratio < 1.3  # within 30%

        shift_results[f"shift_{s}"] = {
            "world_informative_phases": shifted_informative,
            "sensitivity_peak_phases": sorted(peak_phases_shifted),
            "peaks_at_correct_location": peaks_at_right_location,
            "mean_sensitivity_at_info_phases_shifted": shifted_info_S,
            "mean_sensitivity_at_info_phases_base": orig_info_S,
            "magnitude_ratio": magnitude_ratio,
            "magnitude_consistent": magnitude_consistent,
        }

    localization_gate = correct_localization
    rotation_gate = all(v["peaks_at_correct_location"] for v in shift_results.values())
    magnitude_gate = all(v["magnitude_consistent"] for v in shift_results.values())
    gate_pass = localization_gate and rotation_gate and magnitude_gate

    return {
        "gate": "V2-G4",
        "description": (
            "Synthetic localization: rotation-invariant classifier + heterogeneous data distribution. "
            "Localization is through DATA (signal at informative phases), not model weights. "
            "Rotating the world (shifting informative-phase assignment) shifts the sensitivity peaks."
        ),
        "baseline_informative_phases": baseline_informative,
        "sensitivity_field_base": {f"k{k:02d}": float(S_base[k]) for k in range(K)},
        "top_2_phases_by_sensitivity_base": sorted(peak_phases_base),
        "correct_localization": correct_localization,
        "shift_details": shift_results,
        "localization_gate_pass": localization_gate,
        "rotation_gate_pass": rotation_gate,
        "magnitude_gate_pass": magnitude_gate,
        "gate_pass": gate_pass,
        "design_note": (
            "The previous implementation used a non-invariant classifier with absolute-phase "
            "weights — its peaks were always at {5,6} regardless of input rotation, proving "
            "nothing about the sensitivity methodology. The corrected version uses global "
            "mean pooling (invariant) and tests world-rotation consistency instead."
        ),
    }
    """V2-G4: Synthetic localization gate.

    Construct a cyclic world where Y = g(Z_5, Z_6) + ε and other cells are
    noise. Verify:
    1. Sensitivity peaks at phases 5 and 6.
    2. Under shift s, peaks rotate to phases (5+s)%K and (6+s)%K.
    3. The equivariant sampler recovers this rotation correctly.

    This simultaneously validates:
      (a) relevance localization
      (b) cyclic equivariance of the sensitivity field
      (c) absence of absolute-phase dependence

    This is the most important positive synthetic gate for V2.
    """
    if shifts is None:
        shifts = [3, 7, 11]

    informative_phases = [5, 6]

    # Classifier: linear on informative phases only (known ground truth)
    W_info = rng.normal(size=(C, D)) * 0.5  # shared projection for both info phases

    def clf_localized(X: np.ndarray) -> np.ndarray:
        """X: [B, K, D] → logits [B, C]. Only uses phases 5 and 6."""
        z = X[:, informative_phases, :].sum(axis=1)  # [B, D]
        return z @ W_info.T  # [B, C]

    # Note: this classifier is NOT rotation-invariant — it uses absolute phases 5,6.
    # That is intentional for localization testing: we want a classifier that
    # knows where signal is. The equivariance of the sensitivity field still holds
    # *relative to* the classifier's structure.
    # For the equivariance test, we compare S(R_s X) vs R_s S(X).

    sampler = EquivariantLocalSampler(window=2, sigma=0.3)
    X = rng.normal(size=(B, K, D))

    def get_field(X_in: np.ndarray) -> np.ndarray:
        baseline = clf_localized(X_in)
        S = np.zeros(K)
        fixed_rng = np.random.default_rng(42)
        for k in range(K):
            diffs = []
            for _ in range(M):
                Z_tilde = sampler.sample(X_in, k, fixed_rng)
                X_rep = X_in.copy()
                X_rep[:, k, :] = Z_tilde
                diffs.append(np.abs(clf_localized(X_rep) - baseline).mean(axis=1))
            S[k] = float(np.mean(diffs))
        return S

    S_orig = get_field(X)  # [K]

    # Test 1: peaks at informative phases
    peak_phases = set(np.argsort(S_orig)[-2:])
    correct_localization = peak_phases == set(informative_phases)

    # Test 2 & 3: peaks rotate correctly under shift
    shift_results = {}
    for s in shifts:
        X_rot = _roll_batch(X, s)
        S_rot = get_field(X_rot)

        # Expected informative phases after rotation
        expected_peaks = {(p + s) % K for p in informative_phases}
        actual_peaks = set(np.argsort(S_rot)[-2:])
        peaks_rotated = actual_peaks == expected_peaks

        # Equivariance: S(R_s X) should ≈ R_s S(X) (roll of original field)
        S_expected = np.roll(S_orig, shift=s)
        equivariance_error = float(np.abs(S_rot - S_expected).max())
        denom = float(np.abs(S_expected).mean()) + 1e-9

        shift_results[f"shift_{s}"] = {
            "expected_informative_phases": sorted(expected_peaks),
            "actual_peak_phases": sorted(actual_peaks),
            "peaks_rotated_correctly": peaks_rotated,
            "equivariance_max_error": equivariance_error,
            "equivariance_relative_error": equivariance_error / denom,
        }

    localization_gate = correct_localization
    rotation_gate = all(v["peaks_rotated_correctly"] for v in shift_results.values())
    equivariance_gate = all(
        v["equivariance_relative_error"] < 0.15 for v in shift_results.values()
    )
    gate_pass = localization_gate and rotation_gate and equivariance_gate

    return {
        "gate": "V2-G4",
        "description": "Synthetic localization: informative phase region recovered and rotates correctly.",
        "informative_phases": informative_phases,
        "sensitivity_field": {f"k{k:02d}": float(S_orig[k]) for k in range(K)},
        "top_2_phases_by_sensitivity": sorted(int(k) for k in peak_phases),
        "correct_localization": correct_localization,
        "shift_details": shift_results,
        "localization_gate_pass": localization_gate,
        "rotation_gate_pass": rotation_gate,
        "equivariance_gate_pass": equivariance_gate,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# G5: False causality — confounded proxy (retained from eligibility worlds)
# ---------------------------------------------------------------------------

def run_v2_g5(rng: np.random.Generator, samples: int = 200_000) -> dict:
    """V2-G5: Claim-boundary gate (immutable, from eligibility World B).

    S_k(X) > 0 does not imply causal effect. Formally:
      S_k(X) ≠ 0  ⟹̸  Z_k causally influences Y.

    This result should be cited explicitly in the paper as a limitation
    of the replacement-sensitivity estimand and as protection for every
    real-data heatmap interpretation.
    """
    u = rng.normal(size=samples)
    proxy = u + rng.normal(scale=0.2, size=samples)
    outcome = u + rng.normal(scale=0.2, size=samples)
    slope = float(np.dot(proxy, outcome) / np.dot(proxy, proxy))
    predictor_sensitivity = float(np.mean(np.abs(slope * proxy)))
    true_do_effect = 0.0
    gate_pass = predictor_sensitivity > 0.1 and true_do_effect == 0.0

    return {
        "gate": "V2-G5",
        "description": "Claim-boundary: confounded proxy retains replacement sensitivity with zero do-effect.",
        "mean_absolute_predictor_sensitivity": predictor_sensitivity,
        "true_structural_do_effect": true_do_effect,
        "gate_pass": gate_pass,
        "claim_boundary": "S_k(X) ≠ 0  ⟹̸  Z_k causally influences Y.",
        "paper_instruction": (
            "This result must appear as a formal claim-boundary statement in the "
            "paper, not in an appendix. Every real-data heatmap should cite it."
        ),
    }


# ---------------------------------------------------------------------------
# G6: Estimator stability — convergence under increasing M
# ---------------------------------------------------------------------------

def run_v2_g6(
    rng: np.random.Generator,
    B: int = 64,
    K: int = 16,
    D: int = 8,
    C: int = 5,
    M_values: list[int] | None = None,
) -> dict:
    """V2-G6: Estimator stability.

    S_k^{(M)}(X) should converge as M increases. Report:
      ||S^{(M)} - S^{(M_max)}||_2 / (||S^{(M_max)}||_2 + eps)

    Freeze M_practical before real-data outcome analysis.
    """
    if M_values is None:
        M_values = [1, 4, 16, 64, 256]

    clf = SyntheticInvariantClassifier(D, C, rng)
    sampler = EquivariantLocalSampler(window=2, sigma=0.15)
    X = rng.normal(size=(B, K, D))

    # Use fixed reference seed for reproducibility within each M estimate
    def get_S(M: int) -> np.ndarray:
        fixed_rng = np.random.default_rng(777)
        return clf.sensitivity_field(X, sampler, M=M, rng=fixed_rng)

    M_max = max(M_values)
    S_max = get_S(M_max)
    denom = float(np.linalg.norm(S_max)) + 1e-9

    convergence = {}
    for M in M_values:
        S_M = get_S(M)
        rel_error = float(np.linalg.norm(S_M - S_max) / denom)
        convergence[f"M_{M}"] = {
            "relative_l2_error_vs_M_max": rel_error,
            "S_mean": float(S_M.mean()),
            "S_std": float(S_M.std()),
        }

    # Gate: at M=16, relative error < 0.10 (practical tolerance)
    m16_error = convergence.get("M_16", {}).get("relative_l2_error_vs_M_max", 1.0)
    gate_pass = m16_error < 0.10
    recommended_M = 16 if gate_pass else 64

    return {
        "gate": "V2-G6",
        "description": "Estimator stability: S_k^{(M)} convergence as M increases.",
        "M_max": M_max,
        "convergence": convergence,
        "relative_error_at_M16": m16_error,
        "recommended_M_practical": recommended_M,
        "gate_pass": gate_pass,
        "instruction": (
            f"Freeze M = {recommended_M} reference draws before real-data outcome analysis. "
            "Report Monte Carlo error bars for all published sensitivity maps."
        ),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(seed: int, samples: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    g0 = run_v2_g0(rng)
    g1 = run_v2_g1(rng)
    g2 = run_v2_g2(rng)
    g3 = run_v2_g3(rng)
    g4 = run_v2_g4(rng)
    g5 = run_v2_g5(rng, samples=samples)
    g6 = run_v2_g6(rng)

    gates = [g0, g1, g2, g3, g4, g5, g6]
    all_pass = all(g["gate_pass"] for g in gates)

    summary = {
        "protocol": "P13-REPLACEMENT-SENSITIVITY-V2",
        "seed": seed,
        "samples_for_g5": samples,
        "gate_results": {g["gate"]: g["gate_pass"] for g in gates},
        "all_synthetic_gates_pass": all_pass,
        "v2_g7_real_data_admissibility": (
            "BLOCKED" if not all_pass else "UNLOCKED — G0-G6 all passed"
        ),
        "thesis": (
            "A rotation-invariant ECG classifier can support a rotation-equivariant, "
            "support-constrained field of local phase-replacement sensitivity."
        ),
        "gates": {g["gate"]: g for g in gates},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P13-REPLACEMENT-SENSITIVITY-V2 synthetic gate runner (G0-G6)."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13002)
    parser.add_argument("--samples", type=int, default=200_000)
    args = parser.parse_args()
    result = run(args.seed, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)

    # Summary print
    print(f"\nP13-REPLACEMENT-SENSITIVITY-V2 Gate Results (seed={args.seed})")
    print("=" * 60)
    for gate_name, passed in result["gate_results"].items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {gate_name:<12} {status}")
    print("-" * 60)
    overall = "ALL PASS" if result["all_synthetic_gates_pass"] else "SOME FAILED"
    print(f"  OVERALL      {overall}")
    print(f"  G7 status:   {result['v2_g7_real_data_admissibility']}")
    print()


if __name__ == "__main__":
    main()
