"""ThetaEncoder: Harmonic Angle Embedding Module for Lead Geometry.

Maps 2D spherical electrode angles (theta, phi) into a 12-dimensional
harmonic representation via trigonometric basis functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ThetaEncoder(nn.Module):
    """Encodes [..., 2] angles (theta, phi) into [..., 12] harmonic features."""

    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Args:

            theta: [..., 2] tensor containing (theta, phi) angles in radians.

        Returns:
            [..., 12] tensor of geometric harmonic features.
        """
        if theta.shape[-1] != 2:
            raise ValueError(f"Expected last dimension 2 for (theta, phi), got shape {tuple(theta.shape)}")

        th = theta[..., 0:1]
        ph = theta[..., 1:2]
        sum_th = th + ph
        sub_th = th - ph

        # [..., 4] base features
        base = torch.cat([th, ph, sum_th, sub_th], dim=-1)

        sin_feat = torch.sin(base * self.omega)
        cos_feat = torch.cos(base * self.omega)

        # Total 4 + 4 + 4 = 12 features
        out = torch.cat([base, sin_feat, cos_feat], dim=-1)
        return out
