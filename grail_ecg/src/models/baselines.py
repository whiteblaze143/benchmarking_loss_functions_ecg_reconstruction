"""Phase P2 Architectural Baseline Models.

Defined in PRD Addendum §8:
- B0: Plain representation (shared CNN backbone + global pooling, no geometry, no slots).
- B1: Structured latent (clinical slots, but no geometry).
- B2: Geometry latent (8-view angle embedding + geometry tokens, no named slots).
- UB: Supervised upper bound (same temporal backbone, end-to-end multi-task supervision).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from grail_ecg.src.models.view_encoder import SharedResNet1D, GeometryConditionedViewEncoder
from grail_ecg.src.models.clinical_slots import ClinicalSlotAggregator, ClinicalAnchorHead


class PlainEncoderB0(nn.Module):
    """B0: Plain SSL Baseline.
    
    Shared temporal ResNet1D across 8 leads -> global average pooling -> 96D latent.
    No geometry embeddings, no structured slots.
    """

    def __init__(
        self,
        num_leads: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        num_anchor_classes: int = 25,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.temporal_encoder = SharedResNet1D(hidden_dim=hidden_dim, num_tokens_per_lead=32)
        self.proj = nn.Linear(hidden_dim, latent_dim)
        self.anchor_head = nn.Linear(latent_dim, num_anchor_classes)

    def forward(self, ecg_leads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            ecg_leads: [B, num_leads, 5000]
        Returns:
            z_flat: [B, 96]
            logits: [B, num_anchor_classes]
        """
        B, L, T = ecg_leads.shape
        x_flat = ecg_leads.reshape(B * L, 1, T)
        h_flat = self.temporal_encoder(x_flat)  # [B * L, 128, 32]
        h = h_flat.reshape(B, L, self.hidden_dim, 32)
        # Global average pool across leads and time
        pooled = h.mean(dim=(1, 3))  # [B, 128]
        z_flat = self.proj(pooled)   # [B, 96]
        logits = self.anchor_head(z_flat)
        return z_flat, logits


class StructuredEncoderB1(nn.Module):
    """B1: Structured Latent Baseline.
    
    Shared ResNet1D temporal encoder + temporal positional embeddings.
    6 learned clinical slot queries cross-attending over 256 tokens -> 6 x 16D = 96D.
    NO geometry embeddings (theta/phi).
    """

    def __init__(
        self,
        num_leads: int = 8,
        num_tokens_per_lead: int = 32,
        hidden_dim: int = 128,
        num_slots: int = 6,
        slot_dim: int = 16,
        anchor_counts_per_domain: dict[str, int] | None = None,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.num_tokens = num_tokens_per_lead
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.latent_dim = num_slots * slot_dim

        self.temporal_encoder = SharedResNet1D(hidden_dim=hidden_dim, num_tokens_per_lead=num_tokens_per_lead)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 1, num_tokens_per_lead, hidden_dim) * 0.02)

        self.slot_aggregator = ClinicalSlotAggregator(
            num_slots=num_slots,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
        )

        if anchor_counts_per_domain is not None:
            self.anchor_heads = ClinicalAnchorHead(
                anchor_counts_per_domain=anchor_counts_per_domain,
                slot_dim=slot_dim,
            )
        else:
            self.anchor_heads = None

    def forward(
        self,
        ecg_leads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        B, L, T = ecg_leads.shape
        x_flat = ecg_leads.reshape(B * L, 1, T)
        h_flat = self.temporal_encoder(x_flat)
        h = h_flat.reshape(B, L, self.hidden_dim, self.num_tokens).permute(0, 1, 3, 2)  # [B, L, 32, 128]

        # Add positional embedding only, NO geometry embedding
        tokens = (h + self.temporal_pos_embed).reshape(B, L * self.num_tokens, self.hidden_dim)

        slots, z_flat = self.slot_aggregator(tokens)
        anchor_logits = None
        if self.anchor_heads is not None:
            anchor_logits = self.anchor_heads(slots)

        return slots, z_flat, anchor_logits


class GeometryEncoderB2(nn.Module):
    """B2: Geometry Latent Baseline.
    
    Shared ResNet1D + ThetaEncoder angle embeddings + temporal positional embeddings -> 256 tokens.
    Cross-attentive single query aggregation into a 96D flat latent space.
    NO structured/named slots.
    """

    def __init__(
        self,
        num_leads: int = 8,
        num_tokens_per_lead: int = 32,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        num_anchor_classes: int = 25,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.view_encoder = GeometryConditionedViewEncoder(
            num_leads=num_leads,
            num_tokens_per_lead=num_tokens_per_lead,
            hidden_dim=hidden_dim,
        )

        # Single global query token cross-attending to 256 tokens
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)
        self.anchor_head = nn.Linear(latent_dim, num_anchor_classes)

    def forward(
        self,
        ecg_leads: torch.Tensor,
        custom_angles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = ecg_leads.shape[0]
        tokens = self.view_encoder(ecg_leads, custom_angles=custom_angles)  # [B, 256, 128]

        q = self.query.expand(B, -1, -1)  # [B, 1, 128]
        attn_out, _ = self.cross_attn(query=q, key=tokens, value=tokens)
        out = self.norm(q + attn_out).squeeze(1)  # [B, 128]

        z_flat = self.proj(out)  # [B, 96]
        logits = self.anchor_head(z_flat)
        return z_flat, logits


class SupervisedUpperBoundUB(nn.Module):
    """UB: End-to-end fully supervised upper bound.
    
    Same shared ResNet1D temporal backbone + multi-head attention pooling -> 25 classes.
    Trained end-to-end purely with multi-label BCE loss.
    """

    def __init__(
        self,
        num_leads: int = 8,
        hidden_dim: int = 128,
        num_classes: int = 25,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim

        self.temporal_encoder = SharedResNet1D(hidden_dim=hidden_dim, num_tokens_per_lead=32)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ecg_leads: torch.Tensor) -> torch.Tensor:
        B, L, T = ecg_leads.shape
        x_flat = ecg_leads.reshape(B * L, 1, T)
        h_flat = self.temporal_encoder(x_flat)
        h = h_flat.reshape(B, L, self.hidden_dim, 32)
        pooled = h.mean(dim=(1, 3))  # [B, 128]
        return self.classifier(pooled)
