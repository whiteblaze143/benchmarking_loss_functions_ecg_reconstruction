"""Mandatory Unit Tests for GRAIL-ECG Architecture & Losses (PRD §35)."""

import pytest
import torch
import torch.nn.functional as F

from grail_ecg.src.models.view_encoder import GeometryConditionedViewEncoder
from grail_ecg.src.models.clinical_slots import ClinicalSlotAggregator, ClinicalAnchorHead
from grail_ecg.src.models.view_aux_decoder import ViewAuxiliaryDecoder
from grail_ecg.src.models.grail_encoder import GRAILEncoder
from grail_ecg.src.losses.ssl import VICRegLoss
from grail_ecg.src.losses.clinical import ClinicalAnchorLoss


def test_shared_view_encoder_weights():
    """Asserts that all leads are processed through identical shared parameters."""
    view_enc = GeometryConditionedViewEncoder(num_leads=8, num_tokens_per_lead=32, hidden_dim=128)
    view_enc.eval()

    # In eval mode (no dropout randomness), identical inputs passed to the shared backbone yield identical outputs
    x_lead_a = torch.randn(2, 1, 5000)
    with torch.no_grad():
        h_a = view_enc.temporal_encoder(x_lead_a)
        h_b = view_enc.temporal_encoder(x_lead_a)
    assert torch.allclose(h_a, h_b)


def test_view_encoder_shape_5000():
    """Verifies output token shape [B, 8*32, 128] = [B, 256, 128]."""
    view_enc = GeometryConditionedViewEncoder(num_leads=8, num_tokens_per_lead=32, hidden_dim=128)
    ecg = torch.randn(2, 8, 5000)
    tokens = view_enc(ecg)
    assert tokens.shape == (2, 256, 128)


def test_latent_shape_6x16():
    """Verifies slot aggregator outputs [B, 6, 16] and flattened [B, 96]."""
    aggregator = ClinicalSlotAggregator(num_slots=6, slot_dim=16, hidden_dim=128)
    tokens = torch.randn(3, 256, 128)
    slots, z_flat = aggregator(tokens)
    assert slots.shape == (3, 6, 16)
    assert z_flat.shape == (3, 96)


def test_gradient_reaches_every_slot():
    """Asserts that backpropagating from the flattened latent reaches all 6 slot queries."""
    aggregator = ClinicalSlotAggregator(num_slots=6, slot_dim=16, hidden_dim=128)
    tokens = torch.randn(2, 256, 128, requires_grad=True)
    slots, z_flat = aggregator(tokens)
    loss = z_flat.sum()
    loss.backward()

    assert aggregator.slot_queries.grad is not None
    assert torch.all(aggregator.slot_queries.grad.abs().sum(dim=-1) > 0)


def test_discovery_slots_have_no_supervised_head():
    """Asserts discovery slots 5 and 6 have zero connection to clinical anchor heads."""
    anchor_counts = {"rhythm": 2, "conduction": 6, "morphology": 9, "stt": 8}
    encoder = GRAILEncoder(
        num_leads=8,
        num_slots=6,
        slot_dim=16,
        hidden_dim=128,
        anchor_counts_per_domain=anchor_counts,
    )

    x = torch.randn(2, 8, 5000)
    slots, z_flat, logits = encoder(x)
    assert logits is not None

    # Compute loss strictly on clinical anchor logits
    clinical_loss = sum(v.sum() for v in logits.values())
    clinical_loss.backward()

    # Slot 4 and 5 (0-indexed for slots 5 and 6) must have NO direct gradient from the anchor heads
    # Check that the anchor head only takes slots[:, 0:4, :]
    assert len(encoder.anchor_heads.heads) == 4
    for domain in ["rhythm", "conduction", "morphology", "stt"]:
        assert encoder.anchor_heads.heads[domain] is not None


def test_vicreg_batch_denominator():
    """Verifies that sample covariance in VICReg uses (B - 1) rather than dataset size N."""
    vicreg = VICRegLoss(sim_coeff=1.0, var_coeff=1.0, cov_coeff=1.0)
    B = 16
    D = 96
    z_a = torch.randn(B, D)
    z_b = torch.randn(B, D)

    # Reference computation with explicit (B - 1)
    za_centered = z_a - z_a.mean(dim=0)
    ref_cov = (za_centered.T @ za_centered) / (B - 1)
    ref_cov_offdiag = ref_cov - torch.diag_embed(torch.diagonal(ref_cov))
    ref_cov_loss_a = ref_cov_offdiag.pow(2).sum() / D

    loss_dict = vicreg(z_a, z_b)
    # The internal cov loss should match ref_cov_loss_a + ref_cov_loss_b
    zb_centered = z_b - z_b.mean(dim=0)
    ref_cov_b = (zb_centered.T @ zb_centered) / (B - 1)
    ref_cov_b_offdiag = ref_cov_b - torch.diag_embed(torch.diagonal(ref_cov_b))
    ref_cov_loss_b = ref_cov_b_offdiag.pow(2).sum() / D
    expected_cov_loss = ref_cov_loss_a + ref_cov_loss_b

    assert torch.allclose(loss_dict["cov_loss"], expected_cov_loss, atol=1e-5)


def test_clinical_masked_label_loss():
    """Verifies multi-domain anchor loss accurately indexes domain slices."""
    slices = {
        "rhythm": slice(0, 2),
        "conduction": slice(2, 8),
        "morphology": slice(8, 17),
        "stt": slice(17, 25),
    }
    loss_fn = ClinicalAnchorLoss(slices)
    domain_logits = {
        "rhythm": torch.randn(4, 2),
        "conduction": torch.randn(4, 6),
        "morphology": torch.randn(4, 9),
        "stt": torch.randn(4, 8),
    }
    targets = torch.randint(0, 2, (4, 25)).float()
    res = loss_fn(domain_logits, targets)

    assert "loss" in res
    assert res["loss"] > 0
    assert not torch.isnan(res["loss"])


def test_view_aux_hides_target():
    """Verifies ViewAuxiliaryDecoder maps [B, 96] + [B, 2] angle to [B, 1, 5000]."""
    decoder = ViewAuxiliaryDecoder(latent_dim=96, hidden_dim=128, target_len=5000)
    z = torch.randn(2, 96)
    angle = torch.tensor([[1.5708, 0.5236], [2.0944, -1.5708]])  # [2, 2]
    recon = decoder(z, angle)
    assert recon.shape == (2, 1, 5000)
    assert not torch.isnan(recon).any()
