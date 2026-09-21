from __future__ import annotations

import torch

from repecg.braid_field import (
    BraidFieldClassifier,
    BraidFieldConfig,
    SoftBraidReadout,
    canonical_q8_geometry,
    make_interpolated_query_bank,
)


def test_query_bank_is_unit_norm_and_finite():
    q, coords = make_interpolated_query_bank(4, 5)
    assert q.shape == (20, 8)
    assert coords.shape == (20, 2)
    assert torch.isfinite(q).all()
    assert torch.allclose(q.norm(dim=-1), torch.ones(20), atol=1e-6)


def test_soft_braid_output_shapes_and_finite():
    readout = SoftBraidReadout(slots_per_sign=2, braid_dim=32, event_dim=16)
    q, coords = make_interpolated_query_bank(4, 4)
    field = torch.randn(3, 16, len(q))
    out = readout(field, coords)
    assert out["braid_embedding"].shape == (3, 32)
    assert out["event_embedding"].shape == (3, 16)
    assert out["tracks"].shape == (3, 16, 4, 2)
    assert torch.isfinite(out["braid_embedding"]).all()
    assert torch.isfinite(out["event_embedding"]).all()


def test_braid_field_forward_all_variants():
    q8, _ = canonical_q8_geometry()
    query_ops, query_coords = make_interpolated_query_bank(3, 3)
    responses = torch.randn(2, 8, 16, 32)
    operators = q8.unsqueeze(0).expand(2, -1, -1)
    for variant in BraidFieldClassifier.VARIANTS:
        config = BraidFieldConfig(
            response_dim=32,
            classes=5,
            slots_per_sign=2,
            braid_dim=32,
            event_dim=16,
            mc_samples=2,
            variant=variant,
        )
        model = BraidFieldClassifier(config)
        out = model(operators, responses, query_ops, query_coords)
        assert out["logits"].shape == (2, 5)
        assert out["field"].shape == (2, 16, 9)
        assert torch.isfinite(out["logits"]).all()
        out["logits"].sum().backward()


def test_single_operator_context_supported():
    q8, _ = canonical_q8_geometry()
    query_ops, query_coords = make_interpolated_query_bank(3, 3)
    model = BraidFieldClassifier(
        BraidFieldConfig(
            response_dim=16,
            slots_per_sign=2,
            braid_dim=16,
            event_dim=8,
            variant="field_braid_event",
        )
    )
    out = model(
        q8[:1].unsqueeze(0),
        torch.randn(1, 1, 16, 16),
        query_ops,
        query_coords,
    )
    assert out["logits"].shape == (1, 5)
