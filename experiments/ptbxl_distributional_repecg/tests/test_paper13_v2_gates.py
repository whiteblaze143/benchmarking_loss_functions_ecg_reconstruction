"""Contract tests for the V2 gate suite: P13-REPLACEMENT-SENSITIVITY-V2.

These tests verify that the four mathematical properties required for the
V2 estimand are correctly implemented in the synthetic gate runner.

G0 — Operator execution properties (identity, locality, executable controls)
G1 — Equivariant sampler passes, non-equivariant sampler breaks equivariance
G4 — Localization through data distribution, not classifier weights
G6 — Estimator converges by M=16

The synthetic gate runner (run_v2_gates.py) provides the full simulation;
these tests verify the contract of each sub-function at small scale.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load run_v2_gates from its absolute path (scripts/ is not on pythonpath)
_GATES_MODULE_PATH = (
    Path(__file__).parent.parent
    / "scripts" / "paper13" / "run_v2_gates.py"
)
_spec = importlib.util.spec_from_file_location("run_v2_gates", _GATES_MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_v2_gates"] = _mod
_spec.loader.exec_module(_mod)

EquivariantLocalSampler = _mod.EquivariantLocalSampler
NonEquivariantAbsoluteSampler = _mod.NonEquivariantAbsoluteSampler
SyntheticInvariantClassifier = _mod.SyntheticInvariantClassifier
_roll_batch = _mod._roll_batch
run_v2_g0 = _mod.run_v2_g0
run_v2_g1 = _mod.run_v2_g1
run_v2_g4 = _mod.run_v2_g4
run_v2_g6 = _mod.run_v2_g6


# ---------------------------------------------------------------------------
# G0: Operator contract
# ---------------------------------------------------------------------------

class TestG0OperatorContract:
    """The replacement operator R_k(X, Z̃_k) = X^{k←Z̃_k} must be identity,
    local, and produce executable controls.
    """

    def test_identity_replacement_leaves_tensor_unchanged(self) -> None:
        """R_k(X, Z_k) = X: replacing phase k with itself produces no change."""
        rng = np.random.default_rng(1)
        result = run_v2_g0(rng, B=16, K=16, D=8)
        assert result["identity_max_error"] == 0.0, (
            f"Identity replacement produced nonzero change: {result['identity_max_error']}"
        )
        assert result["identity_pass"]

    def test_replacement_is_local(self) -> None:
        """R_k(X, Z̃_k)_j = X_j for all j ≠ k: no hidden propagation."""
        rng = np.random.default_rng(2)
        result = run_v2_g0(rng, B=16, K=16, D=8)
        assert result["locality_max_error"] == 0.0, (
            f"Locality violated: phase {result['locality_max_error']:.2e} change in non-target phase"
        )
        assert result["locality_pass"]

    def test_all_non_identity_controls_change_tensor(self) -> None:
        """Every named control (except identity_noop) must alter the tensor."""
        rng = np.random.default_rng(3)
        result = run_v2_g0(rng, B=16, K=16, D=8)
        for ctrl, info in result["executable_controls"].items():
            if ctrl == "identity_noop":
                assert not info["changes_tensor"], (
                    "identity_noop should NOT change the tensor"
                )
            else:
                assert info["changes_tensor"], (
                    f"Control '{ctrl}' did not change the tensor (metadata-only defect)"
                )

    def test_g0_full_gate_passes(self) -> None:
        rng = np.random.default_rng(4)
        result = run_v2_g0(rng, B=16, K=16, D=8)
        assert result["gate_pass"]


# ---------------------------------------------------------------------------
# G1: Equivariance contract
# ---------------------------------------------------------------------------

class TestG1EquivarianceContract:
    """Classifier invariance is necessary but NOT sufficient for
    S(R_s X) = R_s S(X). The sampler must also commute with rotation.
    """

    def test_classifier_is_rotation_invariant(self) -> None:
        """f(R_s X) = f(X) for the global-mean-pool classifier."""
        rng = np.random.default_rng(10)
        clf = SyntheticInvariantClassifier(8, 5, rng)
        X = rng.normal(size=(32, 16, 8))
        for s in [1, 3, 8, 13]:
            out_orig = clf(X)
            out_rot = clf(_roll_batch(X, s))
            np.testing.assert_allclose(
                out_orig, out_rot, atol=1e-12,
                err_msg=f"Classifier not rotation-invariant at shift s={s}"
            )

    def test_equivariant_sampler_uses_only_relative_context(self) -> None:
        """q_{k+s}(·|R_s X) must equal the distribution of R_s q_k(·|X).

        Test: the CONTEXT used by the equivariant sampler at phase k in X
        equals the context at phase k+s in R_s X. This is the key commutation
        property that makes the sensitivity field equivariant.
        """
        rng = np.random.default_rng(11)
        sampler = EquivariantLocalSampler(window=2, sigma=0.0)  # sigma=0 → deterministic
        # X: [B=1, K=16, D=8]
        X = rng.normal(size=(1, 16, 8))
        K = 16
        s = 5

        # The equivariant sampler at phase k in X uses:
        #   mean of neighbors at relative offsets ±1, ±2
        # At phase k+s in R_s X:
        #   R_s X at position j = original phase at (j-s)%K
        #   Neighbors of (k+s) in R_s X = original phases at (k+o)%K — same window as k in X
        X_rot = _roll_batch(X, s)  # [1, 16, 8]
        for k in range(K):
            k_shifted = (k + s) % K
            # sigma=0 → sample = context mean (deterministic)
            ctx_orig = sampler.sample(X, k, np.random.default_rng(0))[0]       # [D]
            ctx_rot = sampler.sample(X_rot, k_shifted, np.random.default_rng(0))[0]
            np.testing.assert_allclose(
                ctx_orig, ctx_rot, atol=1e-12,
                err_msg=(
                    f"Equivariant sampler context at k={k} in X differs from "
                    f"k+s={k_shifted} in R_s X — sampler is NOT equivariant"
                )
            )

    def test_non_equivariant_sampler_breaks_equivariance(self) -> None:
        """The absolute-phase sampler has per-phase biases that break commutation."""
        rng = np.random.default_rng(12)
        sampler = NonEquivariantAbsoluteSampler(16, 8, rng, window=2, sigma=0.0)
        # X: [B=1, K=16, D=8]
        X = rng.normal(size=(1, 16, 8))
        s = 5
        X_rot = _roll_batch(X, s)
        K = 16

        # For at least one k, the context at k in X should differ from
        # the context at k+s in R_s X (due to absolute phase bias).
        mismatches = 0
        for k in range(K):
            k_shifted = (k + s) % K
            ctx_orig = sampler.sample(X, k, np.random.default_rng(0))[0]       # [D]
            ctx_rot = sampler.sample(X_rot, k_shifted, np.random.default_rng(0))[0]
            if not np.allclose(ctx_orig, ctx_rot, atol=1e-6):
                mismatches += 1

        assert mismatches > 0, (
            "Non-equivariant sampler should show context mismatches across phases "
            "due to absolute-phase bias vectors, but all contexts matched."
        )

    def test_g1_full_gate_passes(self) -> None:
        rng = np.random.default_rng(13)
        result = run_v2_g1(rng, B=32, K=16, D=8, C=5, shifts=[1, 3, 7], M=16)
        assert result["classifier_is_rotation_invariant"]
        assert result["equivariant_sampler_gate_pass"], (
            f"Equivariant sampler did not pass equivariance gate. "
            f"Max relative error: {result['equivariant_sampler_max_relative_error_across_shifts']:.4f}"
        )
        assert result["nonequivariant_correctly_breaks_equivariance"], (
            "Non-equivariant sampler should break equivariance but didn't. "
            "The contrast test is not informative."
        )
        assert result["gate_pass"]


# ---------------------------------------------------------------------------
# G4: Localization via data distribution (not classifier weights)
# ---------------------------------------------------------------------------

class TestG4LocalizationContract:
    """Sensitivity peaks arise from DATA distribution differences, not from
    absolute-position-specific model weights.
    """

    def test_sensitivity_peaks_at_informative_phases(self) -> None:
        """With signal at phases {5, 6}, sensitivity should peak there."""
        rng = np.random.default_rng(40)
        result = run_v2_g4(rng, B=256, K=16, D=8, C=5, M=16, shifts=[3], signal_strength=2.5)
        assert result["correct_localization"], (
            f"Sensitivity did not peak at informative phases {result['baseline_informative_phases']}. "
            f"Got peaks at: {result['top_2_phases_by_sensitivity_base']}"
        )

    def test_peaks_shift_when_world_rotates(self) -> None:
        """When the informative phases in the world shift by s, the peaks shift by s."""
        rng = np.random.default_rng(41)
        result = run_v2_g4(rng, B=256, K=16, D=8, C=5, M=16, shifts=[3, 7], signal_strength=2.5)
        for shift_key, v in result["shift_details"].items():
            assert v["peaks_at_correct_location"], (
                f"{shift_key}: world informative phases={v['world_informative_phases']}, "
                f"but sensitivity peaked at {v['sensitivity_peak_phases']}. "
                "The method has an absolute-phase bias."
            )

    def test_sensitivity_magnitude_is_shift_independent(self) -> None:
        """The magnitude of the peak sensitivity should be similar across world rotations."""
        rng = np.random.default_rng(42)
        result = run_v2_g4(rng, B=256, K=16, D=8, C=5, M=16, shifts=[3, 7, 11], signal_strength=2.5)
        for shift_key, v in result["shift_details"].items():
            assert v["magnitude_consistent"], (
                f"{shift_key}: magnitude ratio {v['magnitude_ratio']:.2f} is outside [0.7, 1.3]. "
                "The sensitivity magnitude should be approximately shift-independent."
            )


# ---------------------------------------------------------------------------
# G6: Estimator stability
# ---------------------------------------------------------------------------

class TestG6EstimatorStability:

    def test_sensitivity_converges_by_m16(self) -> None:
        """At M=16 reference draws, relative L2 error vs M=256 should be < 0.10."""
        rng = np.random.default_rng(60)
        result = run_v2_g6(rng, B=32, K=16, D=8, C=5, M_values=[1, 4, 16, 64, 256])
        assert result["relative_error_at_M16"] < 0.10, (
            f"At M=16, relative error vs M_max = {result['relative_error_at_M16']:.4f} > 0.10. "
            "Estimator has not converged. Use M=64 or higher."
        )
        assert result["gate_pass"]
        assert result["recommended_M_practical"] == 16
