import torch

from geometry import (
    LEAD_DIRECTIONS,
    constrained_oracle,
    geometry_diagnostics,
    project_vcg,
    residual_subspace_oracle,
    standard_from_independent,
)


def test_geometry_shape_rank_and_conditioning():
    result = geometry_diagnostics()
    assert result["shape"] == [8, 3]
    assert result["rank"] == 3
    assert result["missing_yz_rank"] == 2
    assert result["missing_yz_condition_number"] < 10
    assert result["trainable_geometry_parameters"] == 0


def test_projection_preserves_observed_lead_exactly():
    source = torch.randn(4, 12, 97, dtype=torch.float64)
    reconstruction, _ = constrained_oracle(source)
    torch.testing.assert_close(reconstruction[:, 0], source[:, 0], rtol=0, atol=0)


def test_dependent_limb_algebra_is_exact():
    independent = torch.randn(3, 8, 101, dtype=torch.float64)
    result = standard_from_independent(independent)
    torch.testing.assert_close(result[:, 2], result[:, 1] - result[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 3], -(result[:, 0] + result[:, 1]) / 2, rtol=0, atol=0)
    torch.testing.assert_close(result[:, 4], result[:, 0] - result[:, 1] / 2, rtol=0, atol=0)
    torch.testing.assert_close(result[:, 5], result[:, 1] - result[:, 0] / 2, rtol=0, atol=0)


def test_gradients_reach_only_predicted_yz_coordinates():
    lead_i = torch.randn(2, 1, 89, dtype=torch.float64)
    yz = torch.randn(2, 2, 89, dtype=torch.float64, requires_grad=True)
    vcg = torch.cat((lead_i, yz), dim=1)
    loss = project_vcg(vcg)[:, 1:].square().mean()
    loss.backward()
    assert yz.grad is not None
    assert torch.count_nonzero(yz.grad) == yz.numel()
    assert not isinstance(LEAD_DIRECTIONS, torch.nn.Parameter)


def test_residual_subspace_oracle_preserves_i_and_recovers_in_subspace_signal():
    generator = torch.Generator().manual_seed(7)
    raw_basis = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(raw_basis, mode="reduced").Q
    coefficients = torch.linspace(-0.4, 0.7, 7, dtype=torch.float64)
    lead_i = torch.randn(3, 1, 113, generator=generator, dtype=torch.float64)
    latent = torch.randn(3, 2, 113, generator=generator, dtype=torch.float64)
    missing = coefficients[None, :, None] * lead_i + torch.einsum("lr,brt->blt", basis, latent)
    source = standard_from_independent(torch.cat((lead_i, missing), dim=1))
    reconstruction, recovered = residual_subspace_oracle(source, coefficients, basis)
    torch.testing.assert_close(reconstruction, source, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(recovered, latent, rtol=1e-12, atol=1e-12)
