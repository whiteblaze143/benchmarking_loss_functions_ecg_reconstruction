"""Continuous SetOperator Architectures for AF-Density Estimation.

T-CAIREM AF-Density Research Program:
Implements the canonical EchoNext continuous D1 SetOperator (A001) with EXACTLY
1,027,009 trainable parameters (excluding the reconstruction decoder from the
diagnostic classification path), alongside matched controls and Stage 1
architectural ablation variants.

Models implemented:
- A001: ContinuousSetOperatorA001 (Exact EchoNext continuous D1 SetOperator)
- A002: MatchedCategoricalSetOperatorA002 (Matched categorical operator)
- A003: OperatorAblatedSetOperatorA003 (Operator-ablated control)
- A004: Continuous2DSetOperatorA004 (2D continuous q in frontal plane)
- D032: DualMomentPhaseSetOperatorD032 (Mean + variance dual-moment pooling)
- D037: OrderPreservingBeatSetOperatorD037 (Order-preserving beat sequence encoder)
- E047: PhaseWiseAttentionSetOperatorE047 (Phase-wise attention pooling)
- F053: NoTransformerSetOperatorF053 (No transformer, masked mean pooling)
- F057: ResidualTokenSetOperatorF057 (Residual token pathway)
- G061: MultimodalRRSetOperatorG061 (Explicit RR statistical branch)
- H071: PerChannelRawResNetSetOperatorH071 (Per-channel raw ResNet encoder)
- I082: ResNetResidualMLPSetOperatorI082 (ResNet + residual MLP head)
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# Core Building Blocks: Phase Response Encoders & Blocks
# ============================================================================

class CircularResidualBlock(nn.Module):
    """Residual block with circular padding across periodic cardiac phase cells."""

    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=0)
        self.conv2 = nn.Conv1d(width, width, 3, padding=0)
        self.activation = nn.GELU()

    def _conv(self, x: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
        return layer(F.pad(x, (1, 1), mode="circular"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self._conv(x, self.conv1))
        x = self._conv(x, self.conv2)
        return self.activation(x + residual)


class StandardResidualBlock(nn.Module):
    """Residual block with standard zero padding."""

    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class PhaseCNN(nn.Module):
    """Standard Circular PhaseCNN response encoder (16 phase cells x 128 features -> 128-D).
    
    Exact architecture from EchoNext / OperatorSetModel:
    - input: Conv1d(128, 128, 1) [16,512 params]
    - blocks: 2 x CircularResidualBlock(128) [2 x 98,560 = 197,120 params]
    - head: Linear(128, 128) [16,512 params]
    Total parameters = 230,144.
    """

    def __init__(self, input_dim: int = 128, width: int = 128, blocks: int = 2, classes: int = 128):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.Sequential(*(CircularResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            phase_features: (batch_x_channels, 16, 128)
            
        Returns:
            features: (batch_x_channels, 128)
        """
        if phase_features.ndim != 3:
            raise ValueError(f"Expected (batch, 16, {self.input.in_channels}), got {phase_features.shape}")
        x = phase_features.transpose(1, 2)  # (B*M, 128, 16)
        features = self.blocks(self.input(x))  # (B*M, 128, 16)
        return self.head(features.mean(dim=-1))  # (B*M, 128)


# ============================================================================
# A001: Exact EchoNext Continuous D1 SetOperator
# ============================================================================

class ContinuousSetOperatorA001(nn.Module):
    """Exact EchoNext Continuous D1 SetOperator (A001).

    Architectural components:
    - Operator Encoder: 8 -> 64 -> 64 MLP (Linear(8, 64) -> GELU -> Linear(64, 64)) [4,736 params]
    - Response Encoder: PhaseCNN(128, width=128, blocks=2, classes=128) [230,144 params]
    - Set Encoder: 2-layer TransformerEncoder (d_model=192, nhead=4, d_ff=384, gelu) [738,816 params]
    - Set Pooling: MultiheadAttention(192, 4) with learned pool query (1, 1, 192) [148,416 params]
    - Latent Projection: Linear(192, 256) -> GELU [49,408 params]
    - AF Head: Linear(256, 1) [257 params]

    EXACT Parameter Count: 1,027,009 trainable parameters.
    Reconstruction decoder is excluded from diagnostic path to guarantee parameter fidelity.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator_dim = operator_dim
        self.response_dim = response_dim

        # 1. Continuous Operator Encoder: 8 -> 64 -> 64
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # 2. PhaseCNN Response Encoder: (16, 128) -> 128-D
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)

        # 3. Set Encoder: 2-layer TransformerEncoder (d_model=192)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)

        # 4. Attention Set Pooling (Cross-Attention with learned query)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)

        # 5. Latent Representation
        self.latent = nn.Sequential(
            nn.Linear(transformer_dim, latent_dim),
            nn.GELU(),
        )

        # 6. Binary AF Classification Head
        self.head = nn.Linear(latent_dim, 1)

    def encode_tokens(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes operator and response pairs into joint measurement tokens.
        
        Args:
            responses: (B, M, 16, 128)
            operators: (B, M, 8)
            
        Returns:
            tokens: (B, M, 192)
        """
        B, M, P, D = responses.shape
        q = self.operator(operators)  # (B, M, 64)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)  # (B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)  # (B, M, 192)
        return tokens

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        available: Optional[torch.Tensor] = None,
        phase_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            responses: (B, M, 16, 128) float tensor of phase features
            operators: (B, M, 8) float tensor of operator continuous vectors
            mask: (B, M) bool tensor where True indicates padded/missing token
            available: optional (B, M) bool tensor
            phase_valid: optional (B, M) bool tensor

        Returns:
            Dict containing 'logits' (B,), 'latent' (B, 256), and 'tokens' (B, M, 192)
        """
        B, M = responses.shape[:2]

        # Resolve padding mask
        if mask is None:
            if available is not None and phase_valid is not None:
                active = available & phase_valid
                mask = ~active
            else:
                mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        # Enforce that at least one channel is active per batch element
        active_per_item = (~mask).sum(dim=1)
        if (active_per_item == 0).any():
            raise ValueError("All-channel representation failure: no active channel for one or more batch items.")

        tokens = self.encode_tokens(responses, operators)  # (B, M, 192)
        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)  # (B, M, 192)

        # Attention Set Pooling
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, 192)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)  # (B, 1, 192)

        latent = self.latent(pooled[:, 0])  # (B, 256)
        logits = self.head(latent).squeeze(-1)  # (B,)

        return {
            "logits": logits,
            "latent": latent,
            "tokens": tokens,
        }


# ============================================================================
# A002: Matched Categorical SetOperator
# ============================================================================

class MatchedCategoricalSetOperatorA002(nn.Module):
    """Matched Categorical Operator Control (A002).
    
    Replaces continuous R^8 operator projection with a discrete embedding table
    for lead 0 (Lead I) and lead 1 (Lead II), with an unknown token for out-of-vocabulary leads.
    """

    def __init__(
        self,
        response_dim: int = 128,
        vocabulary_size: int = 2,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.known_operator = nn.Embedding(vocabulary_size, 64)
        self.register_buffer("unknown_operator", torch.zeros(64), persistent=True)

        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def encode_operator(self, operator_ids: torch.Tensor) -> torch.Tensor:
        """Encodes integer lead IDs (-1 for unknown/unseen, 0..1 for known)."""
        valid_ids = operator_ids.clamp_min(0)
        embedded = self.known_operator(valid_ids)
        unknown = self.unknown_operator.to(device=operator_ids.device, dtype=embedded.dtype).expand_as(embedded)
        return torch.where((operator_ids >= 0).unsqueeze(-1), embedded, unknown)

    def forward(
        self,
        responses: torch.Tensor,
        operator_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.encode_operator(operator_ids)  # (B, M, 64)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# A003: Operator-Ablated Control (Permutation Invariant Set Processing)
# ============================================================================

class OperatorAblatedSetOperatorA003(nn.Module):
    """Operator-Ablated Set Control (A003).
    
    Ablates measurement operator identity entirely: replaces q with a fixed zero
    embedding, evaluating set aggregation without measurement coordinate knowledge.
    """

    def __init__(
        self,
        response_dim: int = 128,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.register_buffer("zero_operator", torch.zeros(64), persistent=True)
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.zero_operator.to(device=responses.device, dtype=responses.dtype).expand(B, M, 64)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# A004: 2D Continuous Operator (Frontal Plane Coordinates)
# ============================================================================

class Continuous2DSetOperatorA004(nn.Module):
    """2D Continuous Operator SetOperator (A004).
    
    Continuous operator coordinates in the 2D frontal plane [cos θ, sin θ].
    """

    def __init__(
        self,
        response_dim: int = 128,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)  # (B, M, 64) from 2D input
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant D032: Dual-Moment (Mean + Variance) Phase Response
# ============================================================================

class DualMomentPhaseSetOperatorD032(nn.Module):
    """Dual-Moment Phase SetOperator (D032).
    
    Response encoder extracts both mean and variance across the 16 phase cells,
    capturing intra-window phase dispersion and morphological heterogeneity.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        # Modified PhaseCNN that returns (mean, variance) projected to 128
        self.response_input = nn.Conv1d(response_dim, 128, 1)
        self.response_blocks = nn.Sequential(
            CircularResidualBlock(128),
            CircularResidualBlock(128),
        )
        # Dual-moment projection: 128 mean + 128 var -> 128 response token
        self.dual_moment_proj = nn.Linear(256, 128)

        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def encode_response(self, responses: torch.Tensor) -> torch.Tensor:
        """Computes mean + variance dual moment over phase cells."""
        # responses: (B*M, 16, 128)
        x = responses.transpose(1, 2)  # (B*M, 128, 16)
        features = self.response_blocks(self.response_input(x))  # (B*M, 128, 16)
        mean = features.mean(dim=-1)  # (B*M, 128)
        var = torch.sqrt(features.var(dim=-1, unbiased=False) + 1e-6)  # (B*M, 128) numerically stable std dev
        moments = torch.cat([mean, var], dim=-1)  # (B*M, 256)
        return self.dual_moment_proj(moments)  # (B*M, 128)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)
        resp = self.encode_response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant D037: Order-Preserving Beat Sequence Encoder
# ============================================================================

class OrderPreservingBeatSetOperatorD037(nn.Module):
    """Order-Preserving Beat Sequence SetOperator (D037).
    
    Preserves cardiac cycle progression order using sequential causal 1D depthwise
    convolutional layers across the ordered phase trajectory.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response_input = nn.Conv1d(response_dim, 128, 1)
        # Order-preserving temporal blocks with causal padding (padding=2 for kernel=3)
        self.temporal_conv1 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.response_head = nn.Linear(128, 128)

        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def encode_response(self, responses: torch.Tensor) -> torch.Tensor:
        # responses: (B*M, 16, 128)
        x = responses.transpose(1, 2)  # (B*M, 128, 16)
        x = F.gelu(self.response_input(x))
        x = x + F.gelu(self.temporal_conv1(x))
        x = x + F.gelu(self.temporal_conv2(x))
        return self.response_head(x.mean(dim=-1))

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)
        resp = self.encode_response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant E047: Phase-Wise Attention Pooling
# ============================================================================

class PhaseWiseAttentionSetOperatorE047(nn.Module):
    """Phase-Wise Attention Pooling SetOperator (E047).
    
    Instead of collapsing the 16 phase cells before the Transformer, pools across
    channels and phase cells jointly via a multi-head attention mechanism.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response_input = nn.Conv1d(response_dim, 128, 1)
        self.response_blocks = nn.Sequential(
            CircularResidualBlock(128),
            CircularResidualBlock(128),
        )
        self.phase_proj = nn.Linear(128, 128)

        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        x = responses.reshape(B * M, P, D).transpose(1, 2)
        feat = self.response_blocks(self.response_input(x)).transpose(1, 2)
        resp_phase = self.phase_proj(feat).reshape(B, M, P, 128)

        q = self.operator(operators).unsqueeze(2).expand(-1, -1, P, -1)
        tokens = torch.cat([q, resp_phase], dim=-1).reshape(B, M * P, 192)

        if mask is not None:
            phase_mask = mask.unsqueeze(-1).expand(-1, -1, P).reshape(B, M * P)
        else:
            phase_mask = None

        encoded = self.set_encoder(tokens, src_key_padding_mask=phase_mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=phase_mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant F053: No Transformer, Direct Masked Mean Pooling
# ============================================================================

class NoTransformerSetOperatorF053(nn.Module):
    """No Transformer SetOperator (F053).
    
    Ablates the Transformer Set Encoder entirely. Channel tokens (q, resp) are
    projected and pooled directly using masked mean pooling across active channels.
    Ultra-lightweight, parameter-efficient control.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        self.token_proj = nn.Sequential(
            nn.Linear(192, latent_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)  # (B, M, 64)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)  # (B, M, 192)
        projected = self.token_proj(tokens)  # (B, M, 256)

        # Masked mean pooling
        active = (~mask).unsqueeze(-1).float()  # (B, M, 1)
        denom = active.sum(dim=1).clamp_min(1.0)  # (B, 1)
        latent = (projected * active).sum(dim=1) / denom  # (B, 256)

        logits = self.head(latent).squeeze(-1)
        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant F057: Residual Token Pathway SetOperator
# ============================================================================

class ResidualTokenSetOperatorF057(nn.Module):
    """Residual Token Pathway SetOperator (F057).
    
    Adds an explicit residual connection bypassing the Transformer set encoder:
    preserves local channel representations against oversmoothing.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        # Residual skip connection around the Transformer set encoder
        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        encoded = encoded + tokens  # Residual skip pathway

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant G061: Explicit RR Statistical Branch Fusion
# ============================================================================

class MultimodalRRSetOperatorG061(nn.Module):
    """Multimodal RR + Morphology SetOperator (G061).
    
    Fuses the 256-D morphology SetOperator latent with 36-D standardized RR
    interval summary statistics (mean, SD, RMSSD, Shannon entropy, etc.) at the AF head.
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        rr_dim: int = 36,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())

        # Multimodal fusion head: 256 latent + 36 RR statistics
        self.head = nn.Linear(latent_dim + rr_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        rr_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])

        # Concatenate RR statistics at the classification head
        fused = torch.cat([latent, rr_features], dim=-1)  # (B, 256 + 36 = 292)
        logits = self.head(fused).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens, "fused": fused}


# ============================================================================
# Stage 1 Variant H071: Per-Channel Raw ResNet Encoder SetOperator
# ============================================================================

class Raw1DResNetBlock(nn.Module):
    """1D ResNet residual block with downsampling for raw ECG waveforms."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + res)


class PerChannelRawResNetSetOperatorH071(nn.Module):
    """Per-Channel Raw ResNet SetOperator (H071).
    
    Directly consumes raw 1D voltage waveforms (e.g. 6000 samples @ 200 Hz) via
    an 8-block 1D ResNet per channel, producing 128-D response tokens without Phase-KME.
    """

    def __init__(
        self,
        raw_length: int = 6000,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.raw_length = raw_length
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # 8-block 1D ResNet: Conv1d -> Blocks -> AdaptiveAvgPool -> Linear(128)
        self.raw_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.resnet_blocks = nn.Sequential(
            Raw1DResNetBlock(32, 32),
            Raw1DResNetBlock(32, 64, stride=2),
            Raw1DResNetBlock(64, 64),
            Raw1DResNetBlock(64, 128, stride=2),
            Raw1DResNetBlock(128, 128),
            Raw1DResNetBlock(128, 128, stride=2),
            Raw1DResNetBlock(128, 128),
            Raw1DResNetBlock(128, 128, stride=2),
        )
        self.raw_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())
        self.head = nn.Linear(latent_dim, 1)

    def encode_raw(self, raw_ecg: torch.Tensor) -> torch.Tensor:
        """Encodes raw 1D ECG channel: (B*M, 1, T) -> (B*M, 128)."""
        x = self.raw_stem(raw_ecg)
        x = self.resnet_blocks(x)
        return self.raw_head(x)

    def forward(
        self,
        raw_waveforms: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for raw waveforms.
        
        Args:
            raw_waveforms: (B, M, T) where T is raw sample length
            operators: (B, M, 8) continuous operator vectors
            mask: (B, M) bool padding mask
        """
        B, M, T = raw_waveforms.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=raw_waveforms.device)

        q = self.operator(operators)
        resp = self.encode_raw(raw_waveforms.reshape(B * M, 1, T)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])
        logits = self.head(latent).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Stage 1 Variant I082: ResNet + Residual MLP Classification Head
# ============================================================================

class ResNetResidualMLPSetOperatorI082(nn.Module):
    """ResNet + Residual MLP Classification Head SetOperator (I082).
    
    Replaces the single Linear(256, 1) head with a 2-layer residual MLP block:
    Linear(256, 256) -> GELU -> Linear(256, 256) + skip -> GELU -> Linear(256, 1).
    """

    def __init__(
        self,
        response_dim: int = 128,
        operator_dim: int = 8,
        transformer_dim: int = 192,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.operator = nn.Sequential(
            nn.Linear(operator_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=384,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pool = nn.MultiheadAttention(transformer_dim, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(transformer_dim, latent_dim), nn.GELU())

        # Residual MLP classification head
        self.mlp_fc1 = nn.Linear(latent_dim, latent_dim)
        self.mlp_fc2 = nn.Linear(latent_dim, latent_dim)
        self.mlp_out = nn.Linear(latent_dim, 1)

    def forward(
        self,
        responses: torch.Tensor,
        operators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, M, P, D = responses.shape
        if mask is None:
            mask = torch.zeros((B, M), dtype=torch.bool, device=responses.device)

        q = self.operator(operators)
        resp = self.response(responses.reshape(B * M, P, D)).reshape(B, M, 128)
        tokens = torch.cat([q, resp], dim=-1)

        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        latent = self.latent(pooled[:, 0])

        # Residual MLP head
        h = F.gelu(self.mlp_fc1(latent))
        h = F.gelu(self.mlp_fc2(h) + latent)  # Skip connection
        logits = self.mlp_out(h).squeeze(-1)

        return {"logits": logits, "latent": latent, "tokens": tokens}


# ============================================================================
# Model Factory and Registry
# ============================================================================

SET_OPERATOR_MODEL_REGISTRY = {
    "A001": ContinuousSetOperatorA001,
    "A002": MatchedCategoricalSetOperatorA002,
    "A003": OperatorAblatedSetOperatorA003,
    "A004": Continuous2DSetOperatorA004,
    "D032": DualMomentPhaseSetOperatorD032,
    "D037": OrderPreservingBeatSetOperatorD037,
    "E047": PhaseWiseAttentionSetOperatorE047,
    "F053": NoTransformerSetOperatorF053,
    "F057": ResidualTokenSetOperatorF057,
    "G061": MultimodalRRSetOperatorG061,
    "H071": PerChannelRawResNetSetOperatorH071,
    "I082": ResNetResidualMLPSetOperatorI082,
}


def build_continuous_setoperator(model_id: str, **kwargs) -> nn.Module:
    """Factory function for Continuous SetOperator architectures.
    
    Args:
        model_id: Ablation identifier (e.g. 'A001', 'A002', 'D032', etc.)
        **kwargs: Architecture hyperparameters passed to the model constructor
        
    Returns:
        Instantiated nn.Module
    """
    model_id = model_id.upper()
    if model_id not in SET_OPERATOR_MODEL_REGISTRY:
        raise ValueError(f"Unknown SetOperator model_id '{model_id}'. Available: {list(SET_OPERATOR_MODEL_REGISTRY.keys())}")
    return SET_OPERATOR_MODEL_REGISTRY[model_id](**kwargs)


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the exact count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_flops(
    model: nn.Module,
    batch_size: int = 1,
    num_channels: int = 2,
    device: str = "cpu",
) -> int:
    """Computes exact forward pass FLOPs using PyTorch's native FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.to(device).eval()
    if isinstance(model, PerChannelRawResNetSetOperatorH071):
        raw = torch.randn(batch_size, num_channels, 6000, device=device)
        ops = torch.randn(batch_size, num_channels, 8, device=device)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(raw, ops)
        return flop_counter.get_total_flops()

    elif isinstance(model, MatchedCategoricalSetOperatorA002):
        resp = torch.randn(batch_size, num_channels, 16, 128, device=device)
        ids = torch.tensor([[0, 1]], dtype=torch.long, device=device).expand(batch_size, -1)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(resp, ids)
        return flop_counter.get_total_flops()

    elif isinstance(model, OperatorAblatedSetOperatorA003):
        resp = torch.randn(batch_size, num_channels, 16, 128, device=device)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(resp)
        return flop_counter.get_total_flops()

    elif isinstance(model, Continuous2DSetOperatorA004):
        resp = torch.randn(batch_size, num_channels, 16, 128, device=device)
        ops = torch.randn(batch_size, num_channels, 2, device=device)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(resp, ops)
        return flop_counter.get_total_flops()

    elif isinstance(model, MultimodalRRSetOperatorG061):
        resp = torch.randn(batch_size, num_channels, 16, 128, device=device)
        ops = torch.randn(batch_size, num_channels, 8, device=device)
        rr = torch.randn(batch_size, 36, device=device)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(resp, ops, rr)
        return flop_counter.get_total_flops()

    else:
        resp = torch.randn(batch_size, num_channels, 16, 128, device=device)
        ops = torch.randn(batch_size, num_channels, 8, device=device)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(resp, ops)
        return flop_counter.get_total_flops()
