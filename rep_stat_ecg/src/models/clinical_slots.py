"""Clinical Latent Slot Aggregator for GRAIL-ECG.

Aggregates 256 spatio-temporal tokens into 6 structured clinical slots
via learned cross-attention queries:
    Z = [z_rhythm, z_conduction, z_morphology, z_stt, z_res1, z_res2] in R^{6 x 16} -> R^{96}
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Transformer Cross-Attention Block with Pre-LayerNorm and FFN."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Args:

            query: [B, num_queries, hidden_dim]
            key_value: [B, num_tokens, hidden_dim]

        Returns:
            [B, num_queries, hidden_dim]
        """
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        x = query + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ClinicalSlotAggregator(nn.Module):
    """Cross-attentive slot aggregator mapping spatio-temporal tokens to 6 slots x 16D = 96D."""

    def __init__(
        self,
        num_slots: int = 6,
        slot_dim: int = 16,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim

        # Learned slot query tokens [1, num_slots, hidden_dim]
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)

        # 2 Cross-Attention Blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, ffn_dim=hidden_dim * 4)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        # Project 128 -> 16 per slot
        self.slot_proj = nn.Linear(hidden_dim, slot_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:

            tokens: [B, 256, 128] spatio-temporal tokens from view encoder.

        Returns:
            slots: [B, 6, 16] structured slot representation.
            z_flat: [B, 96] flattened public clinical latent representation.
        """
        B = tokens.shape[0]
        q = self.slot_queries.expand(B, -1, -1)  # [B, 6, 128]

        for block in self.blocks:
            q = block(query=q, key_value=tokens)

        q = self.final_norm(q)
        slots = self.slot_proj(q)  # [B, 6, 16]
        z_flat = slots.reshape(B, self.num_slots * self.slot_dim)  # [B, 96]
        return slots, z_flat


class ClinicalAnchorHead(nn.Module):
    """Supervised multi-task heads for slots 1-4.

    Slots 5-6 (discovery) receive ZERO supervised gradients.
    """

    def __init__(
        self,
        anchor_counts_per_domain: dict[str, int],
        slot_dim: int = 16,
    ):
        super().__init__()
        # 4 structured slot domains: rhythm, conduction, morphology, stt
        self.domains = ["rhythm", "conduction", "morphology", "stt"]
        self.heads = nn.ModuleDict()

        for domain in self.domains:
            n_classes = anchor_counts_per_domain.get(domain, 0)
            if n_classes > 0:
                self.heads[domain] = nn.Linear(slot_dim, n_classes)
            else:
                self.heads[domain] = None

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:

            slots: [B, 6, 16]

        Returns:
            dict of logits for each domain.
        """
        out = {}
        for idx, domain in enumerate(self.domains):
            head = self.heads[domain]
            if head is not None:
                slot_repr = slots[:, idx, :]  # [B, 16]
                out[domain] = head(slot_repr)
        return out
