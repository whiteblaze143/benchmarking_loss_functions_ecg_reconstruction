"""GRAIL-ECG End-to-End Clinical State Encoder.

Architecture:
    X_8L -> GeometryConditionedViewEncoder -> ClinicalSlotAggregator -> Z in R^{96}
"""

from __future__ import annotations

import torch
import torch.nn as nn
from grail_ecg.src.models.view_encoder import GeometryConditionedViewEncoder
from grail_ecg.src.models.clinical_slots import ClinicalSlotAggregator, ClinicalAnchorHead
from grail_ecg.src.geometry.lead_geometry import get_angles_tensor, INDEPENDENT_8_LEADS


class GRAILEncoder(nn.Module):
    """Geometry- and Representer-Anchored Clinical ECG Latent Encoder."""

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
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.latent_dim = num_slots * slot_dim

        # View encoder
        self.view_encoder = GeometryConditionedViewEncoder(
            num_leads=num_leads,
            num_tokens_per_lead=num_tokens_per_lead,
            hidden_dim=hidden_dim,
        )

        # Slot aggregator
        self.slot_aggregator = ClinicalSlotAggregator(
            num_slots=num_slots,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
        )

        # Optional multi-task supervised heads for anchor concepts (slots 1-4)
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
        custom_angles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Args:

            ecg_leads: [B, num_leads, 5000]
            custom_angles: Optional [B, num_leads, 2] or [num_leads, 2]

        Returns:
            slots: [B, 6, 16] structured slots
            z_flat: [B, 96] flattened public representation
            anchor_logits: Dict of anchor prediction logits or None
        """
        tokens = self.view_encoder(ecg_leads, custom_angles=custom_angles)
        slots, z_flat = self.slot_aggregator(tokens)

        anchor_logits = None
        if self.anchor_heads is not None:
            anchor_logits = self.anchor_heads(slots)

        return slots, z_flat, anchor_logits
