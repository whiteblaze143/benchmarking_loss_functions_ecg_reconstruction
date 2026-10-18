"""Geometry-Aware Transformer Attention for SetOperator (NEF-Net Inspired).

Implements:
1. GeometryMultiheadAttention: Multi-head attention with learned per-head scalar bias beta * R_ij,
   where R_ij = (q_i^T q_j)^2 is the operator directional similarity.
   Initialized with beta = 0.0, reproducing standard multi-head self-attention with machine precision.
2. GeometryTransformerEncoderLayer: Transformer encoder layer compatible with PyTorch state_dict.
3. GeometryTransformerEncoder: Stack of geometry-aware encoder layers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryMultiheadAttention(nn.Module):
    """Multihead Attention with learned operator relational geometry bias.
    
    Attention logits:
        S_{ij}^{(h)} = (Q_i^{(h)} K_j^{(h)}) / sqrt(d_k) + beta_h * R_{ij}
        
    At beta_h = 0.0, this exactly reproduces standard PyTorch MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Match PyTorch MultiheadAttention parameter naming exactly for checkpoint compatibility
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Learned geometric attention scalar per head, initialized strictly to 0.0
        self.beta = nn.Parameter(torch.zeros(num_heads))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.constant_(self.beta, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim)
            value: (batch, seq_len, embed_dim)
            key_padding_mask: (batch, seq_len) boolean tensor, True for padded/invalid keys
            R: (batch, seq_len, seq_len) relational alignment matrix, R_ij = (q_i^T q_j)^2
            need_weights: whether to return attention weights
            
        Returns:
            Tuple (output, attention_weights):
            - output: (batch, seq_len, embed_dim)
            - attention_weights: (batch, num_heads, seq_len, seq_len) if need_weights else None
        """
        B, S, _ = query.shape

        # Compute Q, K, V projections simultaneously
        if query is key and key is value:
            # Self-attention optimization
            proj = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = proj.chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = (
                self.in_proj_bias.chunk(3, dim=0) if self.in_proj_bias is not None else (None, None, None)
            )
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention logits: (batch, num_heads, S, S)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add geometric relational bias: beta_h * R_ij
        if R is not None:
            # R is (batch, S, S) -> unsqueeze to (batch, 1, S, S)
            # beta is (num_heads,) -> view to (1, num_heads, 1, 1)
            R_expanded = R.unsqueeze(1)
            beta_expanded = self.beta.view(1, self.num_heads, 1, 1)
            attn_logits = attn_logits + (beta_expanded * R_expanded)

        # Apply key padding mask before softmax
        if key_padding_mask is not None:
            # key_padding_mask is (batch, S) where True means padded/ignore
            # Broadcast to (batch, 1, 1, S)
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask_expanded, float("-inf"))

        # Softmax over keys (last dimension)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Handle edge case where an entire row is masked (-inf everywhere -> NaN in softmax)
        # Replace NaNs with zeros
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Dropout on attention weights if in training mode
        if self.dropout > 0.0 and self.training:
            attn_weights_dropped = F.dropout(attn_weights, p=self.dropout, training=self.training)
        else:
            attn_weights_dropped = attn_weights

        # Attention output: (batch, num_heads, S, head_dim)
        attn_out = torch.matmul(attn_weights_dropped, v)

        # Transpose and concatenate heads: (batch, S, embed_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)

        # Final linear projection
        output = self.out_proj(attn_out)

        if need_weights:
            return output, attn_weights
        return output, None


class GeometryTransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer with geometry-aware self-attention."""

    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = GeometryMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=True
        )

        # Feedforward network matching nn.TransformerEncoderLayer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass the input through the geometry-aware encoder layer.

        Args:
            src: (batch, seq_len, d_model)
            src_mask: Not used for set encoder (bidirectional)
            src_key_padding_mask: (batch, seq_len) boolean tensor
            R: (batch, seq_len, seq_len) relational alignment matrix
        """
        # Multi-head self-attention block
        src2, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask, R=R, need_weights=False
        )
        # Post-LN residual connection 1
        x = self.norm1(src + self.dropout1(src2))

        # Feedforward block
        ffn = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Post-LN residual connection 2
        out = self.norm2(x + self.dropout2(ffn))
        return out


class GeometryTransformerEncoder(nn.Module):
    """Stack of GeometryTransformerEncoderLayer instances."""

    def __init__(self, encoder_layer: GeometryTransformerEncoderLayer, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryTransformerEncoderLayer(
                d_model=encoder_layer.self_attn.embed_dim,
                nhead=encoder_layer.self_attn.num_heads,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask, R=R)
        return output
