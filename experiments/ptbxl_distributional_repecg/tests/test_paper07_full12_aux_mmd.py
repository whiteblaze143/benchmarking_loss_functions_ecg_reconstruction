from __future__ import annotations

import torch

from repecg.common.models import biased_imq_mmd2
from repecg.paper07_operator import Full12LeadWaveformDecoder, OperatorSetModel
from scripts.paper07.train_paper07_full12_aux_mmd import _sample_subsets, heldout_waveform_mse


def test_full12_decoder_preserves_waveform_contract_and_gradients() -> None:
    torch.manual_seed(2026)
    decoder = Full12LeadWaveformDecoder(seed_samples=25, channels=8)
    latent = torch.randn(3, 256, requires_grad=True)
    waveform = decoder(latent, samples=100)
    assert waveform.shape == (3, 12, 100)
    loss = waveform.square().mean()
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert decoder.seed.weight.grad is not None and torch.isfinite(decoder.seed.weight.grad).all()


def test_full12_primary_bce_mmd_and_reconstruction_reach_encoder() -> None:
    torch.manual_seed(2026)
    encoder = OperatorSetModel(response_dim=8, classes=5, operator_mode="continuous")
    decoder = Full12LeadWaveformDecoder(seed_samples=25, channels=8)
    normalizer = torch.nn.LayerNorm(256, elementwise_affine=False)
    operators = torch.randn(4, 12, 8)
    operators = operators / operators.norm(dim=-1, keepdim=True)
    responses = torch.randn(4, 12, 16, 8)
    labels = torch.randint(0, 2, (4, 5)).float()
    targets = torch.randn(4, 12, 100)
    order, subset_mask, heldout = _sample_subsets(4, torch.Generator().manual_seed(11), torch.device("cpu"))
    rows = torch.arange(4).unsqueeze(1)
    h_subset = encoder.encode_context(operators[rows, order], responses[rows, order], subset_mask)
    h_full = encoder.encode_context(operators, responses)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(encoder.head(h_subset), labels)
    mmd = biased_imq_mmd2(normalizer(h_subset), normalizer(h_full), c2=1.0)
    reconstruction = heldout_waveform_mse(decoder(h_subset, samples=100), targets, heldout)
    (bce + mmd + 0.1 * reconstruction).backward()
    assert encoder.response.input.weight.grad is not None
    assert torch.isfinite(encoder.response.input.weight.grad).all()
    assert decoder.seed.weight.grad is not None


def test_full12_subset_sampler_never_observes_every_lead_or_scores_observed_leads() -> None:
    order, padding, heldout = _sample_subsets(32, torch.Generator().manual_seed(19), torch.device("cpu"))
    assert order.shape == padding.shape == heldout.shape == (32, 12)
    assert torch.all(heldout.any(dim=1))
    assert torch.all((~heldout).any(dim=1))
    target = torch.zeros(32, 12, 10)
    reconstruction = target.clone()
    reconstruction[~heldout] = 100.0
    assert heldout_waveform_mse(reconstruction, target, heldout).item() == 0.0
