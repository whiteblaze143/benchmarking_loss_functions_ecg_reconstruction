import itertools

import pytest
import torch

from unified_latents.engineering.experimental.Multi_Scale_VAE import (
    baseline_loss_function,
)

from scripts.common_loss import (
    CompositeLoss,
    FactorialLossConfig,
    PaperParityCompositeLoss,
    mmd_loss,
)


@pytest.mark.parametrize("bits", itertools.product((0, 1), repeat=4))
def test_all_factorial_masks_activate_only_requested_components(bits):
    mask = "".join(str(bit) for bit in bits)
    config = FactorialLossConfig.from_mask(mask)
    criterion = CompositeLoss(**config.effective_weights())
    generator = torch.Generator().manual_seed(42)
    target = torch.randn(4, 3, 32, generator=generator)
    pred = (target + 0.2 * torch.randn(4, 3, 32, generator=generator)).requires_grad_()

    total, mse, corr, mmd, deriv = criterion(pred, target)
    component_values = (mse, corr, mmd, deriv)
    weights = tuple(config.effective_weights().values())
    assert mse.item() > 0
    for component_index, (enabled, weight, value) in enumerate(
        zip(bits, weights, component_values)
    ):
        if not enabled and component_index != 0:
            assert value.requires_grad is False
            continue
        weighted_value = weight * value
        component_grad = torch.autograd.grad(
            weighted_value, pred, retain_graph=True, allow_unused=False
        )[0]
        if enabled:
            assert torch.isfinite(component_grad).all()
            assert component_grad.abs().sum().item() > 0
        else:
            assert component_grad.abs().sum().item() == 0

    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad.abs().sum().item() > 0) is any(bits)


def test_factorial_mask_validation():
    for invalid in ("", "00", "000", "00000", "10x0"):
        with pytest.raises(ValueError):
            FactorialLossConfig.from_mask(invalid)


def test_mmd_gradient_is_live_at_full_ecg_dimensionality():
    """Regression test for fixed-bandwidth RBF underflow on real ECG shapes."""
    generator = torch.Generator().manual_seed(42)
    target = 0.35 * torch.randn(4, 9, 5000, generator=generator)
    pred = (
        target + 0.20 * torch.randn(4, 9, 5000, generator=generator)
    ).requires_grad_()

    value = mmd_loss(pred, target)
    gradient = torch.autograd.grad(value, pred)[0]

    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    assert value.item() > 0
    assert gradient.abs().sum().item() > 1e-4


@pytest.mark.parametrize("mask", ("1000", "1011", "1101", "1110", "1111"))
def test_paper_parity_objective_is_finite(mask):
    target = torch.linspace(-1, 1, 64).reshape(1, 1, -1).repeat(3, 2, 1)
    pred = (target.roll(1, dims=-1) * 0.9).requires_grad_()
    total, *_ = PaperParityCompositeLoss(mask)(pred, target)
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(pred.grad).all()
def test_msvae_decoder_loss_remains_attached_for_lambda_mse_reweighting():
    target = torch.randn(2, 12, 32)
    reconstruction = torch.randn(2, 12, 32, requires_grad=True)
    mu = torch.randn(2, 8, requires_grad=True)
    log_var = torch.randn(2, 8, requires_grad=True)
    losses = baseline_loss_function(reconstruction, target, mu, log_var)
    assert losses["recons_loss"].requires_grad
    gradient = torch.autograd.grad(losses["recons_loss"], reconstruction)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_msvae_mse_toggle_changes_reconstruction_gradient():
    target = torch.randn(2, 12, 32)
    reconstruction = torch.randn(2, 12, 32, requires_grad=True)
    mu = torch.randn(2, 8, requires_grad=True)
    log_var = torch.randn(2, 8, requires_grad=True)
    losses = baseline_loss_function(reconstruction, target, mu, log_var)

    mse_on = losses["loss"]
    mse_off = losses["loss"] - losses["recons_loss"]
    gradient_on = torch.autograd.grad(
        mse_on, reconstruction, retain_graph=True, allow_unused=True
    )[0]
    gradient_off = torch.autograd.grad(
        mse_off, reconstruction, allow_unused=True
    )[0]

    assert gradient_on is not None
    assert torch.isfinite(gradient_on).all()
    assert gradient_on.abs().sum() > 0
    assert gradient_off is None or torch.equal(
        gradient_off, torch.zeros_like(reconstruction)
    )
