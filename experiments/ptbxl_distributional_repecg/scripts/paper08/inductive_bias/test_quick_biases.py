from __future__ import annotations

import numpy as np
import torch

from repecg.common.variants import ExperimentVariant
from repecg.paper08_tokens import (
    PhaseTokenAttentionLayer,
    PhaseTokenTransformer,
    build_cyclic_banded_mask,
    deterministic_phase_permutations,
)


def test_local_layer_has_no_cls_global_bypass() -> None:
    torch.manual_seed(1)
    layer = PhaseTokenAttentionLayer(32, 4, 64, dropout=0.0, mode="local_banded")
    layer.eval()
    mask = build_cyclic_banded_mask(16, 2)
    baseline = torch.randn(1, 17, 32)
    perturbed = baseline.clone()
    perturbed[:, 9] += 100.0
    with torch.inference_mode():
        left, _ = layer(baseline, mask=mask)
        right, _ = layer(perturbed, mask=mask)
    # Phase 0 (token 1) cannot see opposite phase 8 (token 9).
    assert torch.equal(left[:, 1], right[:, 1])
    assert torch.all(mask[1:, 0] < -1e8)


def test_static_routing_is_input_independent_and_capacity_near_matched() -> None:
    torch.manual_seed(2)
    dynamic = PhaseTokenTransformer(8, d_model=64, nhead=4, num_layers=2, mode="global")
    torch.manual_seed(2)
    static = PhaseTokenTransformer(8, d_model=64, nhead=4, num_layers=2, mode="static")
    dynamic.eval()
    static.eval()
    with torch.inference_mode():
        _, first = static(torch.randn(3, 16, 8), return_attention=True)
        _, second = static(torch.randn(3, 16, 8) * 20, return_attention=True)
    for left, right in zip(first, second):
        assert torch.equal(left[0], right[0])
    dynamic_count = sum(value.numel() for value in dynamic.parameters() if value.requires_grad)
    static_count = sum(value.numel() for value in static.parameters() if value.requires_grad)
    assert abs(static_count - dynamic_count) / dynamic_count < 0.05


def test_phase_agnostic_transformer_is_permutation_invariant() -> None:
    torch.manual_seed(3)
    model = PhaseTokenTransformer(8, d_model=64, nhead=4, num_layers=2, mode="phase_agnostic")
    model.eval()
    values = torch.randn(2, 16, 8)
    permutation = torch.tensor([7, 3, 12, 0, 15, 2, 9, 5, 1, 14, 6, 11, 4, 10, 8, 13])
    with torch.inference_mode():
        original = model(values)
        shuffled = model(values[:, permutation])
    assert torch.allclose(original, shuffled, atol=1e-6, rtol=1e-6)


def test_dynamic_routing_is_record_dependent() -> None:
    torch.manual_seed(4)
    model = PhaseTokenTransformer(8, d_model=64, nhead=4, num_layers=1, mode="global")
    model.eval()
    with torch.inference_mode():
        _, first = model(torch.zeros(1, 16, 8), return_attention=True)
        _, second = model(torch.randn(1, 16, 8) * 5, return_attention=True)
    assert torch.linalg.vector_norm(first[0] - second[0]) > 1e-3


def test_record_specific_scramble_is_frozen_and_non_global() -> None:
    record_ids = np.array([11, 12, 13, 11])
    first = deterministic_phase_permutations(record_ids, 42)
    second = deterministic_phase_permutations(record_ids, 42)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], first[3])
    assert not np.array_equal(first[0], first[1])
    assert all(np.array_equal(np.sort(row), np.arange(16)) for row in first)


def test_attention_checks_do_not_claim_equivalence_token_readiness() -> None:
    # The UCB-equivalence vocabulary needs a separate fitted artifact. A raw
    # PhaseTokenTransformer intentionally carries no such readiness flag.
    model = PhaseTokenTransformer(8)
    assert not hasattr(model, "equivalence_vocabulary_ready")


def test_cnn_control_is_not_an_aliased_transformer() -> None:
    control = PhaseTokenTransformer(8, variant=ExperimentVariant(mechanism="cnn_matched"))
    primary = PhaseTokenTransformer(8, variant=ExperimentVariant(mechanism="global"))
    assert control.mode == "cnn_matched"
    assert control.cnn is not None
    assert len(control.layers) == 0
    assert primary.cnn is None
    assert len(primary.layers) == 2
    x = torch.randn(2, 16, 8)
    assert control(x).shape == primary(x).shape == (2, 5)
    control_count = sum(p.numel() for p in control.parameters() if p.requires_grad)
    primary_count = sum(p.numel() for p in primary.parameters() if p.requires_grad)
    assert abs(control_count - primary_count) / primary_count < 0.15
