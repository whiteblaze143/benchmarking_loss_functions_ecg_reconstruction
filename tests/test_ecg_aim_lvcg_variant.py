import torch
from torch import nn

from unified_latents.engineering.models.ecg_aim_lvcg_variant import (
    ECGAIMLVCGVariant,
    LVCGStyleLeadIEmbedding,
)
from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import build_wavelet_ecg_aim


class _DummyECGAIM(nn.Module):
    def forward(self, masked_ecg, **kwargs):
        return {"y_pred": masked_ecg.clone()}


def _inputs():
    lead_i = torch.randn(2, 1, 256)
    bounds = torch.tensor(
        [[[0, 64], [64, 128], [128, 192]], [[0, 80], [80, 160], [0, 0]]]
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])
    return lead_i, bounds, mask


def test_embedding_shapes_and_gradients():
    model = LVCGStyleLeadIEmbedding(
        beat_len=32,
        state_dim=16,
        rhythm_dim=8,
        max_beats=4,
        stem_channels=8,
        stage_channels=(8, 8),
        stage_blocks=(1, 1),
        dropout=0.0,
    )
    lead_i, bounds, mask = _inputs()
    result = model(lead_i, bounds, mask)
    assert result["ecg_emb"].shape == (2, 40)
    assert result["beat_states"].shape == (2, 3, 16)
    assert torch.count_nonzero(result["beat_states"][~mask]) == 0
    assert result["beat_reconstruction"].shape == (2, 3, 32)
    assert torch.isfinite(result["embedding_ssl_loss"])
    (result["ecg_emb"].square().mean() + result["embedding_ssl_loss"]).backward()
    assert model.lead_i_to_field.weight.grad is not None
    assert model.beat_decoder.proj.weight.grad is not None


def test_wrapper_preserves_reconstruction_and_embedding_api():
    model = ECGAIMLVCGVariant(
        _DummyECGAIM(),
        beat_len=32,
        state_dim=16,
        rhythm_dim=8,
        max_beats=4,
        stem_channels=8,
        stage_channels=(8, 8),
        stage_blocks=(1, 1),
        dropout=0.0,
    )
    lead_i, bounds, mask = _inputs()
    masked_ecg = torch.zeros(2, 12, 256)
    masked_ecg[:, 0] = lead_i[:, 0]
    result = model(masked_ecg, bounds, mask)
    assert torch.equal(result["y_pred"][:, 0], lead_i[:, 0])
    assert result["y_pred"].shape == (2, 12, 256)
    normalized = model.ext_ecg_emb(masked_ecg, bounds, mask, normalize=True)
    torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(2))


def test_non_left_aligned_mask_fails_loudly():
    model = LVCGStyleLeadIEmbedding(
        beat_len=32,
        state_dim=16,
        rhythm_dim=8,
        max_beats=4,
        stem_channels=8,
        stage_channels=(8, 8),
        stage_blocks=(1, 1),
        dropout=0.0,
    )
    lead_i, bounds, _ = _inputs()
    bad_mask = torch.tensor([[True, False, True], [True, True, False]])
    try:
        model(lead_i, bounds, bad_mask)
    except ValueError as exc:
        assert "left-aligned" in str(exc)
    else:
        raise AssertionError("non-left-aligned mask must fail")


def test_triplication_control_has_no_field_adapter():
    model = LVCGStyleLeadIEmbedding(
        beat_len=32, state_dim=16, rhythm_dim=8, max_beats=4,
        stem_channels=8, stage_channels=(8, 8), stage_blocks=(1, 1),
        dropout=0.0, field_mode="triplicate",
    )
    lead_i, bounds, mask = _inputs()
    result = model(lead_i, bounds, mask)
    assert model.lead_i_to_field is None
    assert torch.isfinite(result["embedding_ssl_loss"])


def test_wrapper_preserves_real_ecg_aim_forward():
    backbone = build_wavelet_ecg_aim(
        target_len=256,
        patch_size=32,
        width=32,
        encoder_depth=1,
        decoder_depth=1,
        heads=4,
        use_wavelet_branch=False,
        use_delineation_head=False,
        predict_fiducials=False,
        random_mask_ratio=0.0,
        temporal_mask_ratio=0.0,
    )
    model = ECGAIMLVCGVariant(
        backbone,
        beat_len=32,
        state_dim=16,
        rhythm_dim=8,
        max_beats=4,
        stem_channels=8,
        stage_channels=(8, 8),
        stage_blocks=(1, 1),
        dropout=0.0,
    ).eval()
    lead_i, bounds, mask = _inputs()
    masked_ecg = torch.zeros(2, 12, 256)
    masked_ecg[:, 0] = lead_i[:, 0]
    lead_indices = torch.zeros(2, 1, dtype=torch.long)
    with torch.no_grad():
        expected = backbone(masked_ecg, lead_indices=lead_indices, compute_delineation=False, compute_ssl=False)
        actual = model(
            masked_ecg,
            bounds,
            mask,
            lead_indices=lead_indices,
            compute_delineation=False,
            compute_ssl=False,
        )
    torch.testing.assert_close(actual["y_pred"], expected["y_pred"], rtol=0, atol=0)
