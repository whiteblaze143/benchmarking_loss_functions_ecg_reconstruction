"""Phase P2 Architectural Baseline Models and Factorial Experiment Architecture.

Defined in PRD v2 §10 & §11:
- UB: Supervised upper bound (same temporal backbone, end-to-end multi-task supervision).
- B0: True SSL-only baseline (no clinical loss, no geometry, no slots).
- B1: SSL + Clinical shaping (no geometry, no slots).
- B2: SSL + Clinical + Structured slots (no geometry).
- B3: SSL + Clinical + Geometry (no structured slots).
- M: Full GRAIL (Geometry + Structured slots + Clinical + View Auxiliary).

Plus FactorialGRAILEncoder supporting all 2^3 = 8 combinations of (G, S, V).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from grail_ecg.src.models.view_encoder import SharedResNet1D, GeometryConditionedViewEncoder
from grail_ecg.src.models.clinical_slots import ClinicalSlotAggregator, ClinicalAnchorHead
from grail_ecg.src.models.view_aux_decoder import ViewAuxiliaryDecoder


class PlainSSLBaselineB0(nn.Module):
    """B0: True Plain SSL Baseline.
    
    Shared temporal ResNet1D across 8 leads -> global average pooling -> 96D latent.
    Purely self-supervised. Contains NO clinical anchor head and NO clinical loss.
    """

    def __init__(
        self,
        num_leads: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 96,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.temporal_encoder = SharedResNet1D(hidden_dim=hidden_dim, num_tokens_per_lead=32)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, ecg_leads: torch.Tensor) -> torch.Tensor:
        """Args:
            ecg_leads: [B, num_leads, 5000]
        Returns:
            z_flat: [B, 96]
        """
        B, L, T = ecg_leads.shape
        x_flat = ecg_leads.reshape(B * L, 1, T)
        h_flat = self.temporal_encoder(x_flat)  # [B * L, 128, 32]
        h = h_flat.reshape(B, L, self.hidden_dim, 32)
        # Global average pool across leads and time
        pooled = h.mean(dim=(1, 3))  # [B, 128]
        z_flat = self.proj(pooled)   # [B, 96]
        return z_flat


class PlainClinicalSSLBaselineB1(nn.Module):
    """B1: Plain SSL + Clinical Multitask Baseline.
    
    Same backbone as B0 + linear anchor classification head.
    NO geometry, NO structured slots.
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
        B, L, T = ecg_leads.shape
        x_flat = ecg_leads.reshape(B * L, 1, T)
        h_flat = self.temporal_encoder(x_flat)
        h = h_flat.reshape(B, L, self.hidden_dim, 32)
        pooled = h.mean(dim=(1, 3))  # [B, 128]
        z_flat = self.proj(pooled)   # [B, 96]
        logits = self.anchor_head(z_flat)
        return z_flat, logits


class StructuredBaselineB2(nn.Module):
    """B2: Structured Slots Baseline (Slots=1, Geometry=0).
    
    Shared temporal ResNet1D + temporal positional embeddings -> 6 slots x 16D = 96D.
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
        h = h_flat.reshape(B, L, self.hidden_dim, self.num_tokens).permute(0, 1, 3, 2)

        # Add temporal positional embedding only, NO geometry embedding
        tokens = (h + self.temporal_pos_embed).reshape(B, L * self.num_tokens, self.hidden_dim)

        slots, z_flat = self.slot_aggregator(tokens)
        anchor_logits = None
        if self.anchor_heads is not None:
            anchor_logits = self.anchor_heads(slots)

        return slots, z_flat, anchor_logits


class GeometryBaselineB3(nn.Module):
    """B3: Geometry Latent Baseline (Geometry=1, Slots=0).
    
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
        tokens = self.view_encoder(ecg_leads, custom_angles=custom_angles)

        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=tokens, value=tokens)
        out = self.norm(q + attn_out).squeeze(1)

        z_flat = self.proj(out)
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


class FactorialGRAILEncoder(nn.Module):
    """Universal Factorial Architecture supporting all 2^3 = 8 combinations of (G, S, V).
    
    Factors:
    - G (use_geometry: bool): Include ThetaEncoder spherical coordinates (theta, phi).
    - S (use_slots: bool): Aggregate into 6 clinical slots (96D) vs single query flat 96D.
    - V (use_view_aux: bool): Whether auxiliary view decoder is trained.
    """

    def __init__(
        self,
        use_geometry: bool = True,
        use_slots: bool = True,
        use_view_aux: bool = False,
        num_leads: int = 8,
        num_tokens_per_lead: int = 32,
        hidden_dim: int = 128,
        num_slots: int = 6,
        slot_dim: int = 16,
        anchor_counts_per_domain: dict[str, int] | None = None,
        total_anchor_classes: int = 25,
    ):
        super().__init__()
        self.use_geometry = use_geometry
        self.use_slots = use_slots
        self.use_view_aux = use_view_aux
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim
        self.latent_dim = num_slots * slot_dim  # 96

        if use_geometry:
            self.view_encoder = GeometryConditionedViewEncoder(
                num_leads=num_leads,
                num_tokens_per_lead=num_tokens_per_lead,
                hidden_dim=hidden_dim,
            )
        else:
            self.temporal_encoder = SharedResNet1D(hidden_dim=hidden_dim, num_tokens_per_lead=num_tokens_per_lead)
            self.temporal_pos_embed = nn.Parameter(torch.randn(1, 1, num_tokens_per_lead, hidden_dim) * 0.02)

        if use_slots:
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
        else:
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.proj = nn.Linear(hidden_dim, self.latent_dim)
            self.flat_anchor_head = nn.Linear(self.latent_dim, total_anchor_classes)

        if use_view_aux:
            self.view_aux_decoder = ViewAuxiliaryDecoder(
                latent_dim=self.latent_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.view_aux_decoder = None

    def forward(
        self,
        ecg_leads: torch.Tensor,
        custom_angles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | dict[str, torch.Tensor] | None]:
        B = ecg_leads.shape[0]

        # 1. Spatio-temporal tokens
        if self.use_geometry:
            tokens = self.view_encoder(ecg_leads, custom_angles=custom_angles)
        else:
            L, T = ecg_leads.shape[1], ecg_leads.shape[2]
            x_flat = ecg_leads.reshape(B * L, 1, T)
            h_flat = self.temporal_encoder(x_flat)
            h = h_flat.reshape(B, L, self.hidden_dim, 32).permute(0, 1, 3, 2)
            tokens = (h + self.temporal_pos_embed).reshape(B, L * 32, self.hidden_dim)

        # 2. Aggregation
        if self.use_slots:
            slots, z_flat = self.slot_aggregator(tokens)
            anchor_logits = self.anchor_heads(slots) if self.anchor_heads is not None else None
            return slots, z_flat, anchor_logits
        else:
            q = self.query.expand(B, -1, -1)
            attn_out, _ = self.cross_attn(query=q, key=tokens, value=tokens)
            out = self.norm(q + attn_out).squeeze(1)
            z_flat = self.proj(out)
            logits = self.flat_anchor_head(z_flat)
            return None, z_flat, logits
