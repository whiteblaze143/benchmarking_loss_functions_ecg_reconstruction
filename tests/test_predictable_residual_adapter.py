from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.train_1lead_wavelet_ssl_mtl import INDEPENDENT_MISSING, ResidualOutputAdapter


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = "dummy"
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.delineation_head = None

    def forward(self, x, **_):
        raw = self.scale * torch.arange(12, device=x.device, dtype=x.dtype)[None, :, None].expand_as(x)
        return {"y_pred": raw, "limb_consistency_loss": raw.new_tensor(3.0), "seg_logits": None}

    def update_byol_target(self, *_args, **_kwargs):
        return None


def fixtures():
    generator = torch.Generator().manual_seed(11)
    coefficients = torch.linspace(-0.3, 0.6, 7)
    basis = torch.linalg.qr(torch.randn(7, 5, generator=generator), mode="reduced").Q
    signal = torch.randn(3, 12, 41, generator=generator)
    signal[:, 1:] = 0
    return coefficients, basis, signal


def test_direct_adapter_preserves_independent_raw_outputs_and_observed_i():
    coefficients, basis, signal = fixtures()
    model = ResidualOutputAdapter(DummyBase(), "direct", coefficients, basis)
    result = model(signal)
    torch.testing.assert_close(result["y_pred"][:, 0], signal[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result["y_pred"][:, INDEPENDENT_MISSING], result["y_pred_raw"][:, INDEPENDENT_MISSING])
    assert result["predicted_residual_latent"] is None


def test_frozen_pca_output_lies_in_frozen_subspace_and_has_no_basis_parameter():
    coefficients, basis, signal = fixtures()
    model = ResidualOutputAdapter(DummyBase(), "frozen_pca", coefficients, basis)
    result = model(signal)
    residual = result["y_pred"][:, INDEPENDENT_MISSING] - coefficients[None, :, None] * signal[:, :1]
    projection_error = residual - torch.einsum("lk,bkt->blt", basis, torch.einsum("lk,blt->bkt", basis, residual))
    torch.testing.assert_close(projection_error, torch.zeros_like(projection_error), atol=5e-6, rtol=0)
    assert "basis_fixed" in dict(model.named_buffers())
    assert "basis_parameter" not in dict(model.named_parameters())


def test_learned_basis_is_orthonormal_and_receives_gradient():
    coefficients, basis, signal = fixtures()
    model = ResidualOutputAdapter(DummyBase(), "learned", coefficients, basis)
    output = model(signal)
    q = model.output_basis()
    torch.testing.assert_close(q.T @ q, torch.eye(5), atol=2e-6, rtol=0)
    output["y_pred"][:, INDEPENDENT_MISSING].square().mean().backward()
    assert model.basis_parameter.grad is not None
    assert torch.count_nonzero(model.basis_parameter.grad) > 0


def test_adapter_derives_dependent_limb_channels_exactly():
    coefficients, basis, signal = fixtures()
    prediction = ResidualOutputAdapter(DummyBase(), "learned", coefficients, basis)(signal)["y_pred"]
    torch.testing.assert_close(prediction[:, 2], prediction[:, 1] - prediction[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(prediction[:, 3], -(prediction[:, 0] + prediction[:, 1]) / 2, rtol=0, atol=0)
    torch.testing.assert_close(prediction[:, 4], prediction[:, 0] - prediction[:, 1] / 2, rtol=0, atol=0)
    torch.testing.assert_close(prediction[:, 5], prediction[:, 1] - prediction[:, 0] / 2, rtol=0, atol=0)
