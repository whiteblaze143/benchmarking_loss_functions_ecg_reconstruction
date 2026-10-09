from __future__ import annotations

import pytest
import torch

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.observable_encoder import (
    LinearFrame,
    ObservableEncoder,
    OSOnlyModel,
    OSWithoutPModel,
    LearnedQWaveformModel,
    OSSetOpFused,
)


def test_linear_frame_dct_and_time_cells():
    T = 1000
    L = 24
    frame_dct = LinearFrame(T=T, L=L, mode="dct")
    assert frame_dct.H.shape == (L, T)
    assert len(frame_dct.frame_hash) == 64

    # Orthonormality check for DCT rows: H @ H^T should be identity
    inner = frame_dct.H @ frame_dct.H.T
    eye = torch.eye(L)
    assert torch.allclose(inner, eye, atol=1e-5), "DCT frame rows are not orthonormal"

    # Forward pass shape
    x = torch.randn(4, T, 8)
    z = frame_dct(x)
    assert z.shape == (4, L, 8)

    # Time cells mode
    frame_tc = LinearFrame(T=T, L=16, mode="time_cells")
    assert frame_tc.H.shape == (16, T)
    z_tc = frame_tc(x)
    assert z_tc.shape == (4, 16, 8)


def test_observable_encoder_shapes():
    L, d = 24, 8
    encoder = ObservableEncoder(L=L, d=d, hidden_dim=256, out_dim=256, include_projector=True)
    Z = torch.randn(5, L, d)
    P = torch.randn(5, d, d)
    out = encoder(Z, P)
    assert out.shape == (5, 256)

    # Without projector
    encoder_no_p = ObservableEncoder(L=L, d=d, hidden_dim=256, out_dim=256, include_projector=False)
    out_no_p = encoder_no_p(Z, P)
    assert out_no_p.shape == (5, 256)


def test_os_only_model_forward():
    B, T, m, d = 3, 1000, 4, 8
    model = OSOnlyModel(T=T, L=24, d=d, classes=5, frame_mode="dct")

    q = torch.randn(B, m, d)
    y = torch.randn(B, T, m)

    logits, latent = model(q, y, return_latent=True)
    assert logits.shape == (B, 5)
    assert latent.shape == (B, 256)
    assert torch.isfinite(logits).all()


def test_capacity_matching_os_vs_learned_q():
    """Verify that OSOnlyModel and LearnedQWaveformModel have matched capacity."""
    T, L, d = 1000, 24, 8
    model_b = OSOnlyModel(T=T, L=L, d=d, classes=5)
    model_d = LearnedQWaveformModel(T=T, L=L, d=d, classes=5)

    params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    params_d = sum(p.numel() for p in model_d.parameters() if p.requires_grad)

    print(f"Model B (OS-Only) params: {params_b:,}")
    print(f"Model D (Learned-Q) params: {params_d:,}")

    # The parameters should be within 1.5x of each other (both ~130k-150k)
    ratio = params_b / params_d
    assert 0.7 <= ratio <= 1.4, f"Parameter mismatch: B={params_b}, D={params_d}, ratio={ratio:.2f}"


def test_learned_q_waveform_model_forward():
    B, T, m, d = 3, 1000, 5, 8
    model = LearnedQWaveformModel(T=T, L=24, d=d, classes=5)

    q = torch.randn(B, m, d)
    y = torch.randn(B, T, m)

    logits, latent = model(q, y, return_latent=True)
    assert logits.shape == (B, 5)
    assert latent.shape == (B, 256)
    assert torch.isfinite(logits).all()


def test_os_setop_fused_zero_residual():
    """Verify that OSSetOpFused initializes with zero residual adapter."""
    B, m, d, T = 2, 4, 8, 1000
    canonical = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    fused = OSSetOpFused(canonical, T=T, L=24, d=d, freeze_canonical=True)

    ops = torch.randn(B, m, d)
    resps = torch.randn(B, m, 16, 128)
    raw_y = torch.randn(B, T, m)

    # Before adapter training, residual weight is zero
    assert torch.all(fused.w_obs == 0.0)

    fused.eval()
    canonical.eval()
    with torch.no_grad():
        logits, z_total = fused(ops, resps, raw_y, return_latent=True)
        z_morph = canonical.encode_context(ops, resps)

    # z_total should match z_morph exactly because w_obs is 0
    assert torch.allclose(z_total, z_morph, atol=1e-6)
