"""Any-Pairs Auxiliary View Decoder for GRAIL-ECG.

Reconstructs a single masked electrical view conditioned on target lead angle (theta, phi)
from the clinical latent representation:
    x_hat_q = D_view(Z, theta_q, phi_q) in R^{5000}
"""

from __future__ import annotations

import torch
import torch.nn as nn
from grail_ecg.src.geometry.theta_encoder import ThetaEncoder


class ViewAuxiliaryDecoder(nn.Module):
    """Thin GeoVT-style auxiliary view decoder."""

    def __init__(self, latent_dim: int = 96, hidden_dim: int = 128, target_len: int = 5000):
        super().__init__()
        self.target_len = target_len
        self.latent_dim = latent_dim

        self.theta_encoder = ThetaEncoder(omega=1.0)
        self.angle_proj = nn.Linear(12, 32, bias=False)

        # Combine latent (96) + angle (32) = 128
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256 * 20),  # 20 temporal seed tokens of dim 256
        )

        # Transposed 1D convolutions to upsample from 20 -> 5000 samples
        # 20 * 5 = 100 * 5 = 500 * 5 = 2500 * 2 = 5000
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=7, stride=5, padding=1, output_padding=0),  # ~100
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=5, padding=1, output_padding=0),   # ~500
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=5, padding=1, output_padding=0),   # ~2500
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=2, padding=2, output_padding=0),   # ~5000
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, z: torch.Tensor, target_angle: torch.Tensor) -> torch.Tensor:
        """Args:

            z: [B, 96] clinical latent representation.
            target_angle: [B, 2] spherical angles (theta, phi) in radians.

        Returns:
            [B, 1, 5000] reconstructed target lead waveform.
        """
        B = z.shape[0]
        geom = self.theta_encoder(target_angle)  # [B, 12]
        geom_emb = self.angle_proj(geom)         # [B, 32]

        fused = torch.cat([z, geom_emb], dim=-1) # [B, 128]
        seeds = self.fusion(fused).reshape(B, 256, 20)

        out = self.upsample(seeds)  # [B, 1, T_out]

        # Precise crop / pad to exactly target_len 5000
        if out.shape[-1] != self.target_len:
            out = nn.functional.interpolate(out, size=self.target_len, mode="linear", align_corners=False)

        return out
