from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from unified_latents.engineering.experimental.alitok_vae_exp import AliTokECGAIM as OriginalAliTokECGAIM

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "unified_latents/engineering/experimental/aim_1_lead.py"
_SPEC = importlib.util.spec_from_file_location("ecg_aim_spatial_test_module", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AliTokECGAIM = _MODULE.AliTokECGAIM
AliTokECGAIMSpatial = _MODULE.AliTokECGAIMSpatial
AliTokECGAIMPanoramaAuthor = _MODULE.AliTokECGAIMPanoramaAuthor
AliTokECGAIMExactThetaFactorial = _MODULE.AliTokECGAIMExactThetaFactorial
ECGSpatialAdaptiveAxialBlock = _MODULE.ECGSpatialAdaptiveAxialBlock
PANORAMA_TO_ECGAIM = _MODULE.PANORAMA_TO_ECGAIM
panorama_angles_ecgaim_order = _MODULE.panorama_angles_ecgaim_order
panorama_features = _MODULE.panorama_features


def _base(**kwargs):
    defaults = dict(target_len=100, patch_size=10, width=32, encoder_depth=1, decoder_depth=1, heads=4)
    defaults.update(kwargs)
    return AliTokECGAIM(**defaults)


def _spatial(**kwargs):
    defaults = dict(target_len=100, patch_size=10, width=32, encoder_depth=1, decoder_depth=1, heads=4)
    defaults.update(kwargs)
    return AliTokECGAIMSpatial(**defaults)


def _author(**kwargs):
    defaults = dict(target_len=100, patch_size=10, width=32, encoder_depth=1, decoder_depth=1, heads=4)
    defaults.update(kwargs)
    return AliTokECGAIMPanoramaAuthor(**defaults)


def _exact_theta(**kwargs):
    defaults = dict(target_len=100, patch_size=10, width=32, encoder_depth=1, decoder_depth=1, heads=4)
    defaults.update(kwargs)
    return AliTokECGAIMExactThetaFactorial(**defaults)


def test_baseline_shared_condition_matches_legacy_positions():
    model = _base().eval()
    inherited = torch.zeros(2, 12, model.num_patches, dtype=torch.bool)
    condition = model._lead_condition(inherited)
    new_positions = condition[:, :, None] + model.time_embedding[None, None]
    legacy_positions = model.lead_embedding[None, :, None] + model.time_embedding[None, None]
    torch.testing.assert_close(new_positions, legacy_positions.expand(2, -1, -1, -1))


def test_working_copy_learned_mode_is_numerically_identical_to_original():
    torch.manual_seed(7)
    original = OriginalAliTokECGAIM(
        target_len=100, patch_size=10, width=32, encoder_depth=1, decoder_depth=1, heads=4
    ).eval()
    working = _base().eval()
    working.load_state_dict(original.state_dict(), strict=True)
    x_full = torch.randn(2, 12, 100)
    indices = torch.tensor([[1], [1]])
    x = torch.zeros_like(x_full)
    x[:, 1] = x_full[:, 1]
    with torch.inference_mode():
        expected = original(x, y_full=x_full, lead_indices=indices)["y_pred"]
        actual = working(x, y_full=x_full, lead_indices=indices)["y_pred"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_panorama_mapping_and_exact_feature_shape():
    assert PANORAMA_TO_ECGAIM == (0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7)
    angles = panorama_angles_ecgaim_order()
    assert angles.shape == (12, 2)
    torch.testing.assert_close(angles[0], torch.tensor([torch.pi / 2, torch.pi / 2]))
    torch.testing.assert_close(angles[2], torch.tensor([torch.pi * 5 / 6, -torch.pi / 2]))
    assert panorama_features(angles).shape == (12, 12)


def test_author_control_uses_released_encoder_and_separate_ungained_projections():
    model = _author()
    angles = panorama_angles_ecgaim_order().unsqueeze(0)
    expected = model.theta_encoder(angles)
    actual = model._author_theta_features(1, angles)
    torch.testing.assert_close(actual, expected)
    assert type(model.theta_encoder).__module__ == "electrocardio_panorama_theta_encoder"
    assert isinstance(model.source_theta_projection, torch.nn.Linear)
    assert isinstance(model.query_theta_projection, torch.nn.Linear)
    assert model.source_theta_projection is not model.query_theta_projection
    assert not hasattr(model, "spatial_gain")


def test_exact_theta_factorial_uses_released_encoder_without_fixed_gains():
    model = _exact_theta()
    angles = panorama_angles_ecgaim_order().unsqueeze(0)
    expected = model.theta_encoder(angles)
    projected = model.theta_projection(expected)
    inherited = torch.ones(1, 12, model.num_patches, dtype=torch.bool)
    inherited[:, 1] = False
    torch.testing.assert_close(model._theta_vectors(inherited), projected)
    assert type(model.theta_encoder).__module__ == "electrocardio_panorama_theta_encoder"
    assert not hasattr(model, "spatial_gain")
    assert not hasattr(model, "relative_gain")


def test_exact_theta_learned_id_is_an_independent_additive_factor():
    torch.manual_seed(19)
    without_learned = _exact_theta(use_learned_lead_id=False).eval()
    with_learned = _exact_theta(use_learned_lead_id=True).eval()
    with_learned.load_state_dict(without_learned.state_dict(), strict=True)
    inherited = torch.ones(2, 12, without_learned.num_patches, dtype=torch.bool)
    inherited[:, 1] = False
    delta = with_learned._lead_condition(inherited) - without_learned._lead_condition(inherited)
    expected = with_learned.lead_embedding.unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(delta, expected)


def test_geometry_buffer_is_fixed_and_persistent():
    model = _spatial(lead_conditioning_mode="panorama")
    assert "panorama_theta" not in dict(model.named_parameters())
    assert "panorama_theta" in dict(model.named_buffers())
    assert "panorama_theta" in model.state_dict()
    assert not model.panorama_theta.requires_grad


def test_capacity_matched_and_permuted_controls_preserve_parameter_count():
    standard = _spatial(lead_conditioning_mode="panorama")
    capacity = _spatial(
        lead_conditioning_mode="panorama", geometry_control="fixed_random"
    )
    permuted = _spatial(
        lead_conditioning_mode="panorama", geometry_control="permuted"
    )
    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in (standard, capacity, permuted)
    }
    assert len(parameter_counts) == 1
    reference = panorama_features(capacity.panorama_theta)
    assert capacity.geometry_control_features.shape == reference.shape
    assert not torch.allclose(capacity.geometry_control_features, reference)
    torch.testing.assert_close(
        capacity.geometry_control_features.mean(), reference.mean(), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        capacity.geometry_control_features.std(), reference.std(), rtol=0, atol=1e-6
    )
    expected_permuted = reference[permuted.geometry_lead_permutation]
    captured = {}

    def hook(_module, inputs, _output):
        captured["features"] = inputs[0].detach()

    handle = permuted.panorama_encoder.register_forward_hook(hook)
    inherited = torch.ones(1, 12, permuted.num_patches, dtype=torch.bool)
    inherited[:, 1] = False
    permuted._spatial_vectors(inherited)
    handle.remove()
    torch.testing.assert_close(captured["features"], expected_permuted)


def test_relative_condition_shape_and_source_pooling():
    model = _spatial(lead_conditioning_mode="panorama_hybrid", use_relative_geometry=True)
    captured = {}

    def hook(_module, inputs, _output):
        captured["relative"] = inputs[0].detach()

    handle = model.relative_geometry_encoder.register_forward_hook(hook)
    inherited = torch.ones(2, 12, model.num_patches, dtype=torch.bool)
    inherited[0, 1] = False
    inherited[1, [0, 1]] = False
    condition = model._lead_condition(inherited)
    handle.remove()
    assert condition.shape == (2, 12, model.width)
    spatial = model._spatial_vectors(inherited)
    relative = captured["relative"]
    torch.testing.assert_close(relative[0, :, : model.width], spatial[0, 1].expand(12, -1))
    expected_mean = spatial[1, [0, 1]].mean(dim=0)
    torch.testing.assert_close(relative[1, :, : model.width], expected_mean.expand(12, -1))


def test_film_zero_initialization_is_identity_modulation():
    block = ECGSpatialAdaptiveAxialBlock(width=32, heads=4)
    condition = torch.randn(2, 12, 32)
    normalized = torch.randn(2, 12, 5, 32)
    modulated = block._modulate(normalized, condition[:, :, None].expand(-1, -1, 5, -1), block.ffn_mod)
    torch.testing.assert_close(modulated, normalized)


def test_forward_and_imputation_contract_for_screen_modes():
    x_full = torch.randn(2, 12, 100)
    indices = torch.tensor([[1], [1]])
    x = torch.zeros_like(x_full)
    x[:, 1] = x_full[:, 1]
    variants = (
        dict(lead_conditioning_mode="learned"),
        dict(lead_conditioning_mode="panorama"),
        dict(lead_conditioning_mode="panorama_hybrid"),
        dict(lead_conditioning_mode="panorama_hybrid", use_relative_geometry=True),
        dict(
            lead_conditioning_mode="panorama_hybrid",
            use_relative_geometry=True,
            use_spatial_film=True,
        ),
    )
    for variant in variants:
        model = _spatial(**variant).eval()
        result = model(x, y_full=x_full, lead_indices=indices)
        assert result["y_pred"].shape == x_full.shape
        assert torch.isfinite(result["y_pred"]).all()
        assert {"loss", "decoder_loss", "y_pred", "z_regressed"} <= result.keys()
        imputed = model.impute_from_regressor(x, lead_indices=indices)
        assert imputed["available"] is True
        assert imputed["y_pred"].shape == x_full.shape
        assert torch.isfinite(imputed["y_pred"]).all()


def test_author_control_forward_and_imputation_contract():
    x_full = torch.randn(2, 12, 100)
    indices = torch.tensor([[1], [1]])
    x = torch.zeros_like(x_full)
    x[:, 1] = x_full[:, 1]
    model = _author().eval()
    result = model(x, y_full=x_full, lead_indices=indices)
    assert result["y_pred"].shape == x_full.shape
    assert torch.isfinite(result["y_pred"]).all()
    imputed = model.impute_from_regressor(x, lead_indices=indices)
    assert imputed["available"] is True
    assert imputed["y_pred"].shape == x_full.shape
    assert torch.isfinite(imputed["y_pred"]).all()


def test_all_eight_exact_theta_factorial_cells_forward():
    x_full = torch.randn(1, 12, 100)
    indices = torch.tensor([[1]])
    x = torch.zeros_like(x_full)
    x[:, 1] = x_full[:, 1]
    observed_cells = set()
    for learned in (False, True):
        for relative in (False, True):
            for film in (False, True):
                model = _exact_theta(
                    use_learned_lead_id=learned,
                    use_relative_geometry=relative,
                    use_spatial_film=film,
                ).eval()
                result = model(x, y_full=x_full, lead_indices=indices)
                assert result["y_pred"].shape == x_full.shape
                assert torch.isfinite(result["y_pred"]).all()
                assert model.use_learned_lead_id is learned
                assert model.use_relative_geometry is relative
                assert model.use_spatial_film is film
                observed_cells.add((learned, relative, film))
    assert len(observed_cells) == 8
