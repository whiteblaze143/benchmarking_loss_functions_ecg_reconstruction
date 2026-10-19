"""SetOperator Architectural Ablation Models for EchoNext Campaign.

Implements all 5 ablation groups:
- Group A: Operator Geometry (A0 baseline, A1 G1, A2 capacity control, A3 no-q, A4 projective-only, A5 diagonal-quadratic, A6 G2, A7 diagonal attention)
- Group B: Phase-KME Morphology (B1 global-KME, B2 phase-position-aware, B3 noncircular, B4 no-local-conv, B5 disordered, B6 raw-block)
- Group C: Set Reasoning (C1 no-transformer, C2 mean pooling, C3 1-layer, C4 1-head, C5 4-layer)
- Group D: Clinical Modalities (D1 ECG-only, D2 tabular-only, D3 late fusion)
- Group E: Architectural Interactions (E1 G1 + phase-aware, E2 best combinations)
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from repecg.common.models import CircularResidualBlock, StandardResidualBlock
from repecg.geometry_setoperator.geometry_features import gamma, vech_sqrt2, compute_operator_gram
from repecg.geometry_setoperator.geometry_attention import (
    GeometryTransformerEncoder,
    GeometryTransformerEncoderLayer,
)


class PhaseAwareReadout(nn.Module):
    """Phase-position-aware readout replacing uniform mean pooling over cardiac phase cells.
    
    Initializes phase logits to zeros, so softmax weights are exactly 1/16 at initialization,
    strictly matching uniform global mean pooling at step 0 while enabling the model
    to learn absolute cardiac phase sensitivity relative to the R-peak.
    """
    def __init__(self, num_phases: int = 16):
        super().__init__()
        self.phase_logits = nn.Parameter(torch.zeros(num_phases))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (..., channels, num_phases)
        weights = F.softmax(self.phase_logits, dim=0)  # (16,)
        return torch.sum(features * weights, dim=-1)   # (..., channels)


class ModulatedPhaseCNN(nn.Module):
    """Configurable PhaseCNN supporting circular, noncircular, phase-position-aware, and per-phase MLP modes."""

    def __init__(
        self,
        input_dim: int = 128,
        width: int = 128,
        blocks: int = 2,
        classes: int = 128,
        phase_mode: Literal[
            "standard_circular",
            "phase_position_aware",
            "noncircular_conv",
            "no_local_conv",
            "global_kme",
            "phase_disordered",
        ] = "standard_circular",
    ):
        super().__init__()
        self.phase_mode = phase_mode
        self.input_dim = input_dim
        self.width = width

        if phase_mode == "no_local_conv":
            # Per-phase MLP without local temporal convolutional inductive bias
            self.input = None
            self.blocks = None
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, classes),
            )
        else:
            self.input = nn.Conv1d(input_dim, width, 1)
            if phase_mode == "noncircular_conv":
                self.blocks = nn.Sequential(*(StandardResidualBlock(width) for _ in range(blocks)))
            else:
                # standard_circular, phase_position_aware, global_kme, phase_disordered all use circular blocks
                self.blocks = nn.Sequential(*(CircularResidualBlock(width) for _ in range(blocks)))

            if phase_mode == "phase_position_aware":
                self.readout = PhaseAwareReadout(num_phases=16)
            else:
                self.readout = None

            self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, 16, 128)
        if self.phase_mode == "global_kme":
            # Pool across all 16 phase cells before CNN, broadcast back
            phase_features = phase_features.mean(dim=1, keepdim=True).expand(-1, 16, -1)

        elif self.phase_mode == "phase_disordered" and self.training:
            # Random permutation of the 16 phase cells per batch
            perm = torch.randperm(16, device=phase_features.device)
            phase_features = phase_features[:, perm, :]

        if self.phase_mode == "no_local_conv":
            # Independent per-phase MLP followed by global mean pooling
            features = self.mlp(phase_features)  # (batch, 16, classes)
            return features.mean(dim=1)          # (batch, classes)

        x = phase_features.transpose(1, 2)  # (batch, 128, 16)
        features = self.blocks(self.input(x))  # (batch, width, 16)

        if self.phase_mode == "phase_position_aware":
            pooled = self.readout(features)   # (batch, width)
        else:
            pooled = features.mean(dim=-1)    # (batch, width)

        return self.head(pooled)              # (batch, classes)


class AblationSetOperatorEchoNext(nn.Module):
    """Comprehensive configurable SetOperator architecture for EchoNext campaign."""

    def __init__(
        self,
        operator_mode: str = "continuous",
        geometry_mode: str = "baseline",
        phase_mode: str = "standard_circular",
        set_encoder_mode: str = "standard_transformer",
        pooling_mode: str = "learned_query",
        tabular_mode: str = "native_additive",
        operator_dim: int = 8,
        response_dim: int = 128,
        tabular_dim: int = 7,
        classes: int = 1,
    ):
        super().__init__()
        self.operator_mode = operator_mode
        self.geometry_mode = geometry_mode
        self.phase_mode = phase_mode
        self.set_encoder_mode = set_encoder_mode
        self.pooling_mode = pooling_mode
        self.tabular_mode = tabular_mode
        self.tabular_dim = tabular_dim

        # 1. Response Encoder
        self.response = ModulatedPhaseCNN(
            input_dim=response_dim,
            width=128,
            blocks=2,
            classes=128,
            phase_mode=phase_mode,
        )

        # 2. Operator Encoder
        if geometry_mode == "q_ablated" or operator_mode == "q_ablated":
            self.operator = None
            self.register_buffer("unknown_operator", torch.zeros(64), persistent=True)
        elif geometry_mode == "control_capacity":
            self.operator = nn.Sequential(nn.Linear(operator_dim, 96), nn.GELU(), nn.Linear(96, 64))
        elif geometry_mode in {"G1", "G1_G2"}:
            self.operator = nn.Sequential(nn.Linear(44, 64), nn.GELU(), nn.Linear(64, 64))
        elif geometry_mode == "projective_only":
            self.operator = nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.Linear(64, 64))
        elif geometry_mode == "diagonal_quadratic":
            self.operator = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 64))
        else:
            # baseline, G2, control_diagonal_bias
            self.operator = nn.Sequential(nn.Linear(operator_dim, 64), nn.GELU(), nn.Linear(64, 64))

        # 3. Set Encoder
        if set_encoder_mode == "no_transformer":
            self.set_encoder = nn.Sequential(
                nn.Linear(192, 192),
                nn.GELU(),
                nn.Linear(192, 192),
            )
        elif geometry_mode in {"G2", "G1_G2", "control_diagonal_bias"}:
            layer = GeometryTransformerEncoderLayer(d_model=192, nhead=4, dim_feedforward=384, activation="gelu")
            self.set_encoder = GeometryTransformerEncoder(layer, num_layers=2)
        elif set_encoder_mode == "transformer_depth_1":
            layer = nn.TransformerEncoderLayer(d_model=192, nhead=4, batch_first=True, dim_feedforward=384, activation="gelu")
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=1)
        elif set_encoder_mode == "transformer_depth_4":
            layer = nn.TransformerEncoderLayer(d_model=192, nhead=4, batch_first=True, dim_feedforward=384, activation="gelu")
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=4)
        elif set_encoder_mode == "transformer_1head":
            layer = nn.TransformerEncoderLayer(d_model=192, nhead=1, batch_first=True, dim_feedforward=384, activation="gelu")
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            # standard_transformer
            layer = nn.TransformerEncoderLayer(d_model=192, nhead=4, batch_first=True, dim_feedforward=384, activation="gelu")
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)

        # 4. Set Pooling
        if pooling_mode == "mean":
            self.pool_query = None
            self.pool = None
        else:
            self.pool_query = nn.Parameter(torch.zeros(1, 1, 192))
            self.pool = nn.MultiheadAttention(192, 4, batch_first=True)

        # 5. Diagnostic Latent
        self.latent = nn.Sequential(nn.Linear(192, 256), nn.GELU())

        # 6. Tabular / Patient Context
        if tabular_mode == "native_additive":
            self.patient_context = nn.Sequential(nn.Linear(tabular_dim, 64), nn.GELU(), nn.Linear(64, 256))
        else:
            self.patient_context = None

        # 7. Classification Head
        if tabular_mode == "late_concat":
            self.head = nn.Linear(256 + tabular_dim, classes)
        else:
            self.head = nn.Linear(256, classes)

    def encode_operator(self, operators: torch.Tensor) -> torch.Tensor:
        if self.geometry_mode == "q_ablated" or self.operator_mode == "q_ablated":
            return self.unknown_operator.expand(*operators.shape[:-1], 64)
        elif self.geometry_mode in {"G1", "G1_G2"}:
            return self.operator(gamma(operators))
        elif self.geometry_mode == "projective_only":
            proj_features = vech_sqrt2(operators.unsqueeze(-1) @ operators.unsqueeze(-2))
            return self.operator(proj_features)
        elif self.geometry_mode == "diagonal_quadratic":
            diag_quad = torch.cat([operators, operators ** 2], dim=-1)
            return self.operator(diag_quad)
        else:
            return self.operator(operators)

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, count = responses.shape[:2]

        q = self.encode_operator(operators)
        resp_flat = responses.reshape(batch * count, responses.shape[2], responses.shape[3])
        r = self.response(resp_flat).reshape(batch, count, 128)

        tokens = torch.cat([q, r], dim=-1)  # (batch, count, 192)

        if self.set_encoder_mode == "no_transformer":
            encoded = self.set_encoder(tokens)
        elif self.geometry_mode in {"G2", "G1_G2", "control_diagonal_bias"}:
            _, R = compute_operator_gram(operators)
            if self.geometry_mode == "control_diagonal_bias":
                eye = torch.eye(count, device=operators.device, dtype=operators.dtype).unsqueeze(0).expand(batch, -1, -1)
                R = R * eye
            encoded = self.set_encoder(tokens, R=R, src_key_padding_mask=mask)
        else:
            encoded = self.set_encoder(tokens, src_key_padding_mask=mask)

        # Pooling
        if self.pooling_mode == "mean":
            if mask is not None:
                valid_mask = (~mask).unsqueeze(-1).float()
                pooled_token = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
            else:
                pooled_token = encoded.mean(dim=1)
        else:
            query = self.pool_query.expand(batch, -1, -1)
            pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
            pooled_token = pooled[:, 0]

        latent = self.latent(pooled_token)

        if self.tabular_mode == "native_additive":
            assert tabular is not None
            latent = latent + self.patient_context(tabular)
            logits = self.head(latent)
        elif self.tabular_mode == "late_concat":
            assert tabular is not None
            fused = torch.cat([latent, tabular], dim=-1)
            logits = self.head(fused)
        else:
            logits = self.head(latent)

        return {"logits": logits, "latent": latent}


class TabularMLPEchoNext(nn.Module):
    """Clinical tabular-only baseline model for EchoNext SHD detection."""

    def __init__(self, tabular_dim: int = 7, hidden_dim: int = 64, classes: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, classes),
        )

    def forward(
        self,
        operators: Optional[torch.Tensor] = None,
        responses: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert tabular is not None, "tabular features required"
        logits = self.mlp(tabular)
        return {"logits": logits, "latent": tabular}
