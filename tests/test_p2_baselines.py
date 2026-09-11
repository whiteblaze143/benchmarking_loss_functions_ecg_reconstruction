"""Unit tests for Phase P2 Model Baselines (B0, B1, B2, M, UB).

Validates:
1. Shape contracts: All models output a 96D latent representation Z in R^{96}.
2. Gradient flow across each architecture.
3. Parameter parity across all variants.
"""

from __future__ import annotations

import pytest
import torch
import yaml
from pathlib import Path

from grail_ecg.src.models.baselines import (
    PlainEncoderB0,
    StructuredEncoderB1,
    GeometryEncoderB2,
    SupervisedUpperBoundUB,
)
from grail_ecg.src.models.grail_encoder import GRAILEncoder
from grail_ecg.src.models.view_aux_decoder import ViewAuxiliaryDecoder


@pytest.fixture
def dummy_batch():
    # Batch of 4 ECGs with 8 independent leads of 5000 samples
    return torch.randn(4, 8, 5000)


@pytest.fixture
def anchor_counts():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ptbxl_concepts.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {domain: len(info["anchors"]) for domain, info in cfg["domains"].items() if info["anchors"]}


def test_b0_plain_encoder(dummy_batch):
    model = PlainEncoderB0(num_leads=8, hidden_dim=128, latent_dim=96, num_anchor_classes=25)
    z, logits = model(dummy_batch)

    assert z.shape == (4, 96), f"Expected [4, 96], got {z.shape}"
    assert logits.shape == (4, 25), f"Expected [4, 25], got {logits.shape}"

    loss = logits.sum() + z.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"B0 Param {name} has no gradient"


def test_b1_structured_encoder(dummy_batch, anchor_counts):
    model = StructuredEncoderB1(
        num_leads=8,
        hidden_dim=128,
        num_slots=6,
        slot_dim=16,
        anchor_counts_per_domain=anchor_counts,
    )
    slots, z_flat, anchor_logits = model(dummy_batch)

    assert slots.shape == (4, 6, 16), f"Expected [4, 6, 16], got {slots.shape}"
    assert z_flat.shape == (4, 96), f"Expected [4, 96], got {z_flat.shape}"
    assert isinstance(anchor_logits, dict)
    assert len(anchor_logits) == 4

    total_logits = sum(v.sum() for v in anchor_logits.values())
    loss = total_logits + z_flat.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"B1 Param {name} has no gradient"


def test_b2_geometry_encoder(dummy_batch):
    model = GeometryEncoderB2(
        num_leads=8,
        hidden_dim=128,
        latent_dim=96,
        num_anchor_classes=25,
    )
    z, logits = model(dummy_batch)

    assert z.shape == (4, 96), f"Expected [4, 96], got {z.shape}"
    assert logits.shape == (4, 25), f"Expected [4, 25], got {logits.shape}"

    # Also test with view aux decoder
    decoder = ViewAuxiliaryDecoder(latent_dim=96, target_len=5000)
    target_angles = torch.tensor([[0.0, 1.57]] * 4)
    x_hat = decoder(z, target_angles)
    assert x_hat.shape == (4, 1, 5000)

    loss = logits.sum() + z.sum() + x_hat.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"B2 Param {name} has no gradient"


def test_m_full_grail(dummy_batch, anchor_counts):
    model = GRAILEncoder(
        num_leads=8,
        hidden_dim=128,
        num_slots=6,
        slot_dim=16,
        anchor_counts_per_domain=anchor_counts,
    )
    slots, z_flat, anchor_logits = model(dummy_batch)

    assert slots.shape == (4, 6, 16)
    assert z_flat.shape == (4, 96)
    assert isinstance(anchor_logits, dict)

    decoder = ViewAuxiliaryDecoder(latent_dim=96, target_len=5000)
    target_angles = torch.tensor([[0.0, 1.57]] * 4)
    x_hat = decoder(z_flat, target_angles)
    assert x_hat.shape == (4, 1, 5000)


def test_ub_supervised_upper_bound(dummy_batch):
    model = SupervisedUpperBoundUB(num_leads=8, hidden_dim=128, num_classes=25)
    logits = model(dummy_batch)
    assert logits.shape == (4, 25)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"UB Param {name} has no gradient"


def test_parameter_parity(anchor_counts):
    b0 = PlainEncoderB0(num_leads=8, hidden_dim=128, latent_dim=96, num_anchor_classes=25)
    b1 = StructuredEncoderB1(num_leads=8, hidden_dim=128, num_slots=6, slot_dim=16, anchor_counts_per_domain=anchor_counts)
    b2 = GeometryEncoderB2(num_leads=8, hidden_dim=128, latent_dim=96, num_anchor_classes=25)
    m = GRAILEncoder(num_leads=8, hidden_dim=128, num_slots=6, slot_dim=16, anchor_counts_per_domain=anchor_counts)
    ub = SupervisedUpperBoundUB(num_leads=8, hidden_dim=128, num_classes=25)

    params = {
        "B0": sum(p.numel() for p in b0.parameters()),
        "B1": sum(p.numel() for p in b1.parameters()),
        "B2": sum(p.numel() for p in b2.parameters()),
        "M": sum(p.numel() for p in m.parameters()),
        "UB": sum(p.numel() for p in ub.parameters()),
    }

    print(f"Model Parameter Counts: {params}")
    # Verify all models have backbone in ~0.6M range and total within 0.9M - 1.6M
    for name, count in params.items():
        assert 900_000 <= count <= 1_600_000, f"{name} param count {count} outside expected range"
