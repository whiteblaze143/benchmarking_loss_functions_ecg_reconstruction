from __future__ import annotations

import numpy as np

from repecg.paper11_predictive_state import CausalTokenConfig, build_morphology_tokens, causal_qrs_anchors, eligible_next_token_examples


def _synthetic_ecg() -> np.ndarray:
    rng = np.random.default_rng(11)
    signal = 0.01 * rng.normal(size=(5000, 12))
    pulse = np.asarray([-0.2, 0.4, 1.5, 0.4, -0.2])
    for peak in range(1100, 4700, 400):
        signal[peak - 2 : peak + 3, 1] += pulse
    return signal


def test_suffix_intervention_cannot_change_confirmed_anchors_or_tokens():
    signal = _synthetic_ecg()
    config = CausalTokenConfig()
    original = build_morphology_tokens(signal, np.zeros(12), np.ones(12), config)
    cutoff = 3200
    altered = signal.copy()
    altered[cutoff + 1 :] = -7.0
    rebuilt = build_morphology_tokens(altered, np.zeros(12), np.ones(12), config)
    left = [token for token in original if token.availability_sample <= cutoff]
    right = [token for token in rebuilt if token.availability_sample <= cutoff]
    assert [(x.anchor_sample, x.confirmation_sample) for x in left] == [(x.anchor_sample, x.confirmation_sample) for x in right]
    assert all(np.array_equal(x.values, y.values) for x, y in zip(left, right, strict=True))


def test_target_region_perturbation_cannot_change_prefix_tokens():
    signal = _synthetic_ecg()
    original = build_morphology_tokens(signal, np.zeros(12), np.ones(12))
    examples = [row for row in eligible_next_token_examples(original, prefix_length=3) if row["eligible"]]
    assert examples
    example = examples[0]
    target = original[int(example["target_token_id"])]
    altered = signal.copy()
    altered[target.token_start : target.token_stop] = 13.0
    rebuilt = build_morphology_tokens(altered, np.zeros(12), np.ones(12))
    for token_id in example["prefix_token_ids"]:
        assert np.array_equal(original[int(token_id)].values, rebuilt[int(token_id)].values)


def test_fixed_support_latency_and_strict_target_separation():
    tokens = build_morphology_tokens(_synthetic_ecg(), np.zeros(12), np.ones(12))
    assert tokens
    assert all(token.values.shape == (250, 12) for token in tokens)
    assert all(0 <= token.confirmation_sample - token.anchor_sample <= 125 for token in tokens)
    for row in eligible_next_token_examples(tokens, prefix_length=3):
        if row["eligible"]:
            assert row["target_raw_min"] > row["prefix_raw_max"]
            assert row["target_raw_min"] > row["prefix_availability_max"]


def test_detector_is_prefix_invariant_at_every_confirmed_cutoff():
    signal = _synthetic_ecg()
    full = causal_qrs_anchors(signal)
    assert full
    for event in full[:4]:
        prefix = signal[: event.confirmation_sample + 1]
        observed = causal_qrs_anchors(prefix)
        expected = tuple(item for item in full if item.confirmation_sample <= event.confirmation_sample)
        assert observed == expected
