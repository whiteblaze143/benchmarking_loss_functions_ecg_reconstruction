from __future__ import annotations

import math
import hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from repecg.common.variants import ExperimentVariant


def deterministic_phase_permutations(record_ids: np.ndarray, seed: int) -> np.ndarray:
    """Frozen record-specific phase destroyer, independent of model inputs."""
    result = np.empty((len(record_ids), 16), dtype=np.int64)
    for index, record_id in enumerate(np.asarray(record_ids).reshape(-1)):
        token = f"paper08-phase-permutation:{seed}:{int(record_id)}".encode()
        record_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        result[index] = np.random.default_rng(record_seed).permutation(16)
    return result


def build_cyclic_banded_mask(num_phases: int = 16, bandwidth: int = 2, device: torch.device | None = None) -> torch.Tensor:
    """
    Builds attention mask for 1 CLS token + num_phases tokens on cyclic torus C_{num_phases}.
    CLS token (index 0) pools from all tokens. Phase tokens cannot attend to CLS,
    which prevents a two-layer global-information bypass around the local mask.
    Phase tokens i, j in {1, ..., num_phases} interact iff dist_C(i, j) <= bandwidth.
    Mask has 0.0 for allowed attention, -1e9 for disallowed attention.
    Shape: (1 + num_phases, 1 + num_phases).
    """
    total = 1 + num_phases
    mask = torch.full((total, total), -1e9, device=device)
    
    # CLS pools globally, but phase tokens do not read the global CLS state.
    mask[0, :] = 0.0
    mask[0, 0] = 0.0
    
    # Phase token cyclic distance
    for i in range(1, total):
        phi_i = i - 1
        for j in range(1, total):
            phi_j = j - 1
            dist = min(abs(phi_i - phi_j), num_phases - abs(phi_i - phi_j))
            if dist <= bandwidth:
                mask[i, j] = 0.0
    return mask


class PhaseTokenAttentionLayer(nn.Module):
    """
    Custom Transformer encoder layer supporting:
    - Standard dynamic multi-head self-attention
    - Additive spatial attention masks (e.g. cyclic banded)
    - Learned static routing matrix (independent of Q, K)
    - Uniform attention (weights = 1 / N)
    - Extraction of attention weight tensors [B, H, N, N]
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        mode: str = "global",
        bandwidth: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.mode = mode
        self.bandwidth = bandwidth

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if mode == "static":
            # Input-independent learned routing, while retaining and using the
            # same Q/K projections as dynamic attention for a matched control.
            self.static_tokens = nn.Parameter(torch.randn(1, 17, d_model) * 0.02)
            self.register_parameter("static_weights", None)
        elif mode == "uniform":
            self.register_parameter("static_tokens", None)
            self.register_parameter("static_weights", None)
            self.q_proj.requires_grad_(False)
            self.k_proj.requires_grad_(False)
        else:
            self.register_parameter("static_tokens", None)
            self.register_parameter("static_weights", None)


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, C = x.shape
        residual = x
        x_norm = self.norm1(x)

        if self.mode == "static":
            routing = self.static_tokens[:, :N]
            q_static = self.q_proj(routing).view(1, N, self.nhead, self.head_dim).transpose(1, 2)
            k_static = self.k_proj(routing).view(1, N, self.nhead, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q_static, k_static.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1).expand(B, -1, -1, -1)
            v = self.v_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
            out = torch.matmul(attn_weights, v)  # [B, H, N, head_dim]
            out = out.transpose(1, 2).contiguous().view(B, N, C)
        elif self.mode == "uniform":
            attn_weights = torch.full((B, self.nhead, N, N), 1.0 / N, device=x.device, dtype=x.dtype)
            v = self.v_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(B, N, C)
        else:
            q = self.q_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn_weights = F.softmax(scores, dim=-1)
            attn_dropped = self.dropout(attn_weights)
            out = torch.matmul(attn_dropped, v)
            out = out.transpose(1, 2).contiguous().view(B, N, C)

        out = residual + self.dropout(self.out_proj(out))
        # Feedforward block
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(out)))))
        out = out + self.dropout(ff_out)

        return (out, attn_weights if return_attention else None)


class PhaseTokenTransformer(nn.Module):
    """
    Phase-Token Transformer for repECG cardiac phase representations.
    Processes [B, 16, input_dim] phase tokens into multi-label diagnostic logits [B, classes].
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        classes: int = 5,
        variant: ExperimentVariant | None = None,
        mode: str | None = None,
        bandwidth: int = 2,
    ):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.input_dim = input_dim
        self.d_model = d_model
        self.classes = classes
        self.bandwidth = bandwidth

        # Resolve mode from variant or explicit argument
        if mode is not None:
            self.mode = mode
        elif self.variant.mechanism == "local_banded":
            self.mode = "local_banded"
        elif self.variant.mechanism == "static_attention":
            self.mode = "static"
        elif self.variant.mechanism == "uniform_attention":
            self.mode = "uniform"
        elif self.variant.mechanism == "phase_agnostic":
            self.mode = "phase_agnostic"
        elif self.variant.mechanism == "cnn_matched":
            self.mode = "cnn_matched"
        else:
            self.mode = "global"

        self.input_proj = nn.Linear(input_dim, d_model)
        if self.mode == "cnn_matched":
            self.register_parameter("cls_token", None)
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding for 1 CLS + 16 phase tokens
        if self.mode in {"phase_agnostic", "cnn_matched"}:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 17, d_model) * 0.02)

        # Build attention layers
        self.layers = nn.ModuleList([] if self.mode == "cnn_matched" else [
            PhaseTokenAttentionLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                dropout=0.1, mode=self.mode, bandwidth=bandwidth,
            ) for _ in range(num_layers)
        ])
        self.cnn = None if self.mode != "cnn_matched" else nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="circular"),
                nn.GELU(),
            ) for _ in range(5)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Codebook buffer for vector quantization (e.g. kmeans_dictionary)
        self.register_buffer("codebook", torch.zeros(64, input_dim), persistent=True)
        self.register_buffer("codebook_ready", torch.tensor(False), persistent=True)

        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Linear(d_model, classes)

    def set_codebook(self, centers: torch.Tensor) -> None:
        values = torch.as_tensor(centers, dtype=self.codebook.dtype, device=self.codebook.device)
        if values.shape != self.codebook.shape or not torch.isfinite(values).all():
            raise ValueError(f"codebook must be finite with shape {tuple(self.codebook.shape)}")
        self.codebook.copy_(values)
        self.codebook_ready.fill_(True)

    def forward(
        self,
        phase_features: torch.Tensor,
        return_attention: bool = False,
        scramble_phases: bool = False,
        phase_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # Fast path for linear probe
        if self.variant.head == "linear":
            logits = self.head(phase_features.mean(dim=1))
            return (logits, []) if return_attention else logits

        # Codebook vector quantization
        if self.variant.mechanism == "kmeans_dictionary":
            if not bool(self.codebook_ready):
                raise RuntimeError("kmeans token variant requires a fitted train-only codebook")
            distances = torch.cdist(phase_features.float(), self.codebook.float())
            nearest = distances.argmin(dim=-1)
            phase_features = self.codebook[nearest].to(phase_features.dtype)

        # Optional phase scrambling stress test
        if phase_permutation is not None:
            if phase_permutation.shape != phase_features.shape[:2]:
                raise ValueError("phase_permutation must have shape (batch,16)")
            rows = torch.arange(len(phase_features), device=phase_features.device).unsqueeze(1)
            phase_features = phase_features[rows, phase_permutation]
        elif scramble_phases:
            perm = torch.randperm(phase_features.shape[1], device=phase_features.device)
            phase_features = phase_features[:, perm]

        B, P, _ = phase_features.shape
        assert P == 16, f"Expected 16 phase cells, got {P}"

        x_proj = self.input_proj(phase_features)  # [B, 16, d_model]
        if self.mode == "cnn_matched":
            assert self.cnn is not None
            pooled = self.norm(self.cnn(x_proj.transpose(1, 2)).mean(dim=-1))
            logits = self.head(pooled)
            return (logits, []) if return_attention else logits

        assert self.cls_token is not None
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, d_model]
        tokens = torch.cat([cls, x_proj], dim=1)  # [B, 17, d_model]

        if self.pos_embedding is not None:
            tokens = tokens + self.pos_embedding

        # Mask construction for local banded mode
        mask = None
        if self.mode == "local_banded":
            mask = build_cyclic_banded_mask(num_phases=16, bandwidth=self.bandwidth, device=tokens.device)

        attention_maps = []
        out = tokens
        for layer in self.layers:
            out, attn = layer(out, mask=mask, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_maps.append(attn)

        out = self.norm(out)
        cls_rep = out[:, 0]
        logits = self.head(cls_rep)

        if return_attention:
            return logits, attention_maps
        return logits
