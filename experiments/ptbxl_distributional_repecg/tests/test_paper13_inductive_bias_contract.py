"""Execution-contract tests for the current Paper 13 implementation.

These tests intentionally document what the code does.  They do not validate
any causal interpretation; the causal surgery claim has been declared
INELIGIBLE (P13-CAUSAL-SURGERY = INELIGIBLE).

Tests document three execution-contract findings from INDUCTIVE_BIAS_AUDIT.md:

  Finding 1/2  — G0 control variants execute the same forward algorithm during
                 training. wrong_phase_surgery is metadata-only.
  Finding 3    — no_structural_propagation is a 90/10 output blend, not a
                 structural no-propagation control.
  Finding 4    — Rotation invariance kills absolute phase identity, but does
                 NOT kill local replacement sensitivity: replacing Z_k != Z_k
                 can still change the prediction because the cell content
                 changes even if its address cannot be identified.

The fourth test (test_replacement_sensitivity_can_be_nonzero_despite_rotation_invariance)
is the key addition: it demonstrates the precise mathematical statement from the audit,
namely that f(X^{k<-r}) - f(X) != 0 even though f(R_s X) = f(X) for all s.
This is the foundation for the V2 estimand (rotation-equivariant sensitivity field).
"""

import torch
import torch.nn as nn

from repecg.common.models import CounterfactualSurgeryModel
from repecg.common.variants import get_variants_for_paper


def _same_weights(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def test_training_controls_do_not_change_the_forward_algorithm() -> None:
    """Finding 1/2: full, no_propagation, wrong_phase_surgery share the same forward pass.

    The surgery operator is not a training objective in any variant.
    wrong_phase_surgery differs only in metadata (ExperimentVariant.control).
    """
    variants = get_variants_for_paper(13)
    full = CounterfactualSurgeryModel(8, width=16, variant=variants["full"])
    no_propagation = CounterfactualSurgeryModel(
        8, width=16, variant=variants["no_propagation"]
    )
    wrong_phase = CounterfactualSurgeryModel(
        8, width=16, variant=variants["wrong_phase_surgery"]
    )
    _same_weights(full, no_propagation)
    _same_weights(full, wrong_phase)
    x = torch.randn(4, 16, 8)

    torch.testing.assert_close(full(x), no_propagation(x))
    torch.testing.assert_close(full(x), wrong_phase(x))


def test_diagnostic_forward_is_phase_rotation_invariant() -> None:
    """Finding 4 (part 1): circular convolution + global pooling = rotation invariance.

    The classifier cannot use the absolute phase index of a cell as a diagnostic
    feature.  f(R_s X) == f(X) for any cyclic shift s.
    """
    model = CounterfactualSurgeryModel(8, width=16).eval()
    x = torch.randn(4, 16, 8)

    with torch.no_grad():
        baseline = model(x)
        rotated = model(torch.roll(x, shifts=5, dims=1))

    torch.testing.assert_close(baseline, rotated, rtol=1e-5, atol=1e-6)


def test_no_propagation_surgery_is_an_output_blend() -> None:
    """Finding 3: no_structural_propagation is a 90/10 classifier-output blend.

    This is not a structural no-propagation control.  It mixes the mean backbone
    states from the original and replaced inputs, then passes through the head.
    The 90/10 blend has no principled causal interpretation.
    """
    variants = get_variants_for_paper(13)
    model = CounterfactualSurgeryModel(
        8, width=16, variant=variants["no_propagation"]
    ).eval()
    x = torch.randn(4, 16, 8)
    reference = torch.randn(4, 8)
    replaced = x.clone()
    replaced[:, 3] = reference

    with torch.no_grad():
        observed = model.do_surgery(x, 3, reference)
        expected = 0.9 * model(x) + 0.1 * model(replaced)

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_replacement_sensitivity_can_be_nonzero_despite_rotation_invariance() -> None:
    """Finding 4 (part 2): local replacement sensitivity field can still exist.

    Rotation invariance means f(R_s X) == f(X) — absolute phase address is
    unavailable.  But replacing cell Z_k with a different vector r != Z_k
    changes the *content* around that cell, so f(X^{k<-r}) - f(X) can be
    nonzero:

        absolute phase identity unavailable
        BUT local replacement sensitivity field can still exist.

    This is the foundation for the V2 estimand:
        S_k(X) = E_{Z_k~}[ d(f(X), f(X^{k<-Z_k~})) ]
    which is rotation-equivariant:
        S_{k+s}(R_s X) = S_k(X).

    This test confirms that, with high probability over random inputs and model
    weights, at least one cell position exhibits nonzero replacement sensitivity.
    """
    torch.manual_seed(130042)
    model = CounterfactualSurgeryModel(8, width=16).eval()
    x = torch.randn(4, 16, 8)

    # Replace cell 3 with a distinctly different vector (off-content replacement)
    reference = torch.randn(4, 8) * 3.0  # large magnitude to ensure nonzero effect

    with torch.no_grad():
        original_prediction = model(x)
        replaced = x.clone()
        replaced[:, 3] = reference
        replaced_prediction = model(replaced)

    sensitivity = (original_prediction - replaced_prediction).abs().mean()

    assert sensitivity.item() > 1e-4, (
        f"Expected nonzero replacement sensitivity but got {sensitivity.item():.2e}. "
        "The local replacement-sensitivity field S_k(X) should be nonzero when "
        "Z_k is replaced by a substantially different vector, even though the "
        "classifier is rotation-invariant (absolute phase identity unavailable). "
        "This would indicate a bug in the replacement path."
    )
