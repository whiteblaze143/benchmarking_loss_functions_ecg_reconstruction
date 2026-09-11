"""Shared 1D ResNet Temporal View Encoder with Geometric Conditioning.

Processes each independent lead using identical, shared weights:
    h_i = E_view(x_i)
Injects harmonic ThetaEncoder lead geometry and learned temporal position embeddings:
    t_{i, tau} = h_{i, tau} + W_g g_i + p_tau
"""

from __future__ import annotations

import torch
import torch.nn as nn
from grail_ecg.src.geometry.theta_encoder import ThetaEncoder
from grail_ecg.src.geometry.lead_geometry import get_angles_tensor, INDEPENDENT_8_LEADS


class Conv1dBlock(nn.Module):
    """Basic 1D Residual Convolutional Block with Dropout."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class SharedResNet1D(nn.Module):
    """Shared 1D ResNet Temporal Backbone across all leads."""

    def __init__(self, hidden_dim: int = 128, num_tokens_per_lead: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens_per_lead

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # 3 Stages of residual blocks
        self.stage1 = nn.Sequential(
            Conv1dBlock(64, 64, stride=2),
            Conv1dBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            Conv1dBlock(64, 128, stride=2),
            Conv1dBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            Conv1dBlock(128, hidden_dim, stride=2),
            Conv1dBlock(hidden_dim, hidden_dim, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool1d(num_tokens_per_lead)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: [B * L, 1, T]

        Returns:
            [B * L, hidden_dim, num_tokens]
        """
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.pool(h)  # [B * L, 128, 32]
        return h


class GeometryConditionedViewEncoder(nn.Module):
    """End-to-end view encoder: Shared ResNet + ThetaEncoder + Position Embeddings."""

    def __init__(
        self,
        num_leads: int = 8,
        num_tokens_per_lead: int = 32,
        hidden_dim: int = 128,
        angles: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.num_tokens = num_tokens_per_lead
        self.hidden_dim = hidden_dim

        # Shared temporal backbone
        self.temporal_encoder = SharedResNet1D(hidden_dim=hidden_dim, num_tokens_per_lead=num_tokens_per_lead)

        # Geometry encoder
        self.theta_encoder = ThetaEncoder(omega=1.0)
        self.geom_proj = nn.Linear(12, hidden_dim, bias=False)

        # Register fixed canonical angles if provided
        if angles is None:
            angles = get_angles_tensor(INDEPENDENT_8_LEADS[:num_leads])
        self.register_buffer("angles", angles)

        # Learned temporal positional embeddings [1, 1, 32, 128]
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 1, num_tokens_per_lead, hidden_dim) * 0.02)

    def forward(
        self,
        ecg_leads: torch.Tensor,
        custom_angles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:

            ecg_leads: [B, L, T] where L is number of leads (typically 8) and T=5000.
            custom_angles: Optional [B, L, 2] or [L, 2] angles. Defaults to self.angles.

        Returns:
            tokens: [B, L * 32, 128] spatio-temporally conditioned tokens.
        """
        B, L, T = ecg_leads.shape

        # Flatten leads into batch dimension to enforce shared weights
        x_flat = ecg_leads.reshape(B * L, 1, T)
        h_flat = self.temporal_encoder(x_flat)  # [B * L, 128, 32]
        h = h_flat.reshape(B, L, self.hidden_dim, self.num_tokens).permute(0, 1, 3, 2)  # [B, L, 32, 128]

        # Geometry embeddings
        angles = custom_angles if custom_angles is not None else self.angles
        if angles.ndim == 2:
            angles = angles.unsqueeze(0).expand(B, -1, -1)  # [B, L, 2]

        geom_harmonics = self.theta_encoder(angles)  # [B, L, 12]
        g_tilde = self.geom_proj(geom_harmonics).unsqueeze(2)  # [B, L, 1, 128]

        # Add temporal tokens + geometry + positional embeddings
        t_tokens = h + g_tilde + self.temporal_pos_embed  # [B, L, 32, 128]

        # Flatten into token sequence [B, L * 32, 128]
        return t_tokens.reshape(B, L * self.num_tokens, self.hidden_dim)
