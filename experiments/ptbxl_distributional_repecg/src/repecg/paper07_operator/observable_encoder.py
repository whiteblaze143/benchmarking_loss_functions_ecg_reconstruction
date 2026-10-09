from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .observable_subspace import ObservableState, observable_state


class LinearFrame(nn.Module):
    """Deterministic linear temporal operator H: R^T -> R^L.

    Transforms projected waveform X_obs in R^(T x d) to temporal features Z in R^(L x d)
    without using task labels or nonlinear parameters.
    """

    def __init__(self, T: int = 1000, L: int = 24, mode: str = "dct"):
        super().__init__()
        self.T = T
        self.L = L
        self.mode = mode

        H_mat = self._build_frame_matrix(T, L, mode)
        self.register_buffer("H", H_mat, persistent=True)

        # Compute SHA256 of frozen transformation matrix for audit tracking
        h_bytes = H_mat.cpu().numpy().tobytes()
        self.frame_hash = hashlib.sha256(h_bytes).hexdigest()

    @staticmethod
    def _build_frame_matrix(T: int, L: int, mode: str) -> torch.Tensor:
        if mode == "dct":
            # Orthonormal DCT-II basis vectors
            # H[k, t] = sqrt(2/T) * cos(pi * (2t + 1) * k / (2T)) for k > 0
            # H[0, t] = sqrt(1/T)
            t = torch.arange(T, dtype=torch.float32)
            k = torch.arange(L, dtype=torch.float32).unsqueeze(1)
            angles = math.pi * (2.0 * t + 1.0) * k / (2.0 * T)
            H = torch.cos(angles) * math.sqrt(2.0 / T)
            H[0] = 1.0 / math.sqrt(T)
            return H
        elif mode == "time_cells":
            # Uniform boxcar window averages
            H = torch.zeros((L, T), dtype=torch.float32)
            step = T / L
            for i in range(L):
                start = int(round(i * step))
                end = int(round((i + 1) * step))
                if end > start:
                    H[i, start:end] = 1.0 / (end - start)
            return H
        else:
            raise ValueError(f"Unknown frame mode: {mode}. Supported: 'dct', 'time_cells'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear frame H.

        Args:
            x: Input waveform [B, T, d] or [T, d]

        Returns:
            Z = H @ x with shape [B, L, d] or [L, d]
        """
        # H is [L, T], x is [..., T, d]
        # output is [..., L, d]
        return torch.matmul(self.H, x)


class ObservableEncoder(nn.Module):
    """Small MLP encoder operating on observable state Z in R^(L x d) and P_Q in R^(d x d)."""

    def __init__(
        self,
        L: int = 24,
        d: int = 8,
        hidden_dim: int = 256,
        out_dim: int = 256,
        *,
        include_projector: bool = True,
        include_rank: bool = False,
        strict_masking: bool = False,
    ):
        super().__init__()
        self.L = L
        self.d = d
        self.include_projector = include_projector
        self.include_rank = include_rank
        self.strict_masking = strict_masking

        in_dim = L * d
        if include_projector:
            in_dim += d * d
        if include_rank:
            in_dim += 1

        self.in_dim = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        Z: torch.Tensor,
        P: torch.Tensor,
        rank: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observable state.

        Args:
            Z: Projected frame features [B, L, d]
            P: Orthogonal projector [B, d, d]
            rank: Optional rank [B]

        Returns:
            E_obs: Embedding in R^[B, out_dim]
        """
        if self.strict_masking:
            # Enforce exact internal observability subspace projection Z = Z @ P
            Z = torch.matmul(Z, P)

        B = Z.shape[0]
        feats = [Z.reshape(B, self.L * self.d)]

        if self.include_projector:
            feats.append(P.reshape(B, self.d * self.d))

        if self.include_rank:
            if rank is None:
                rank_feat = torch.diagonal(P, dim1=-2, dim2=-1).sum(-1, keepdim=True) / self.d
            else:
                rank_feat = rank.float().unsqueeze(-1) / self.d
            feats.append(rank_feat)

        x = torch.cat(feats, dim=-1)
        x = self.norm(x)
        return self.mlp(x)


class OSOnlyModel(nn.Module):
    """End-to-end OS-Only model (Model B in PRD).

    Evaluates zero-shot inductive bias of observable-subspace quotient representation.
    """

    def __init__(
        self,
        T: int = 1000,
        L: int = 24,
        d: int = 8,
        classes: int = 5,
        *,
        frame_mode: str = "dct",
        include_projector: bool = True,
        strict_masking: bool = False,
        rcond: float = 1e-5,
    ):
        super().__init__()
        self.T = T
        self.L = L
        self.d = d
        self.rcond = rcond
        self.include_projector = include_projector

        self.frame = LinearFrame(T=T, L=L, mode=frame_mode)
        self.encoder = ObservableEncoder(
            L=L,
            d=d,
            hidden_dim=256,
            out_dim=256,
            include_projector=include_projector,
            strict_masking=strict_masking,
        )
        self.head = nn.Linear(256, classes)

    def forward(
        self,
        q: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            q: Operators [B, m, d]
            y: Signals [B, T, m]
            mask: Optional valid measurements mask [B, m]
            return_latent: If True, returns (logits, latent)
        """
        state = observable_state(q, y, rcond=self.rcond, mask=mask, d=self.d)
        Z = self.frame(state.projected)
        latent = self.encoder(Z, state.projector, state.rank)
        logits = self.head(latent)

        if return_latent:
            return logits, latent
        return logits


class OSWithoutPModel(OSOnlyModel):
    """OS model with projector P removed from encoder input (Model C in PRD).

    Isolates the effect of the observability mask.
    """

    def __init__(
        self,
        T: int = 1000,
        L: int = 24,
        d: int = 8,
        classes: int = 5,
        *,
        frame_mode: str = "dct",
        rcond: float = 1e-5,
    ):
        super().__init__(
            T=T,
            L=L,
            d=d,
            classes=classes,
            frame_mode=frame_mode,
            include_projector=False,
            strict_masking=False,
            rcond=rcond,
        )


class LearnedQWaveformModel(nn.Module):
    """Capacity-matched learned-q control model (Model D in PRD).

    Uses the same linear temporal frame H and capacity, but processes (Q, Y)
    directly through a learned operator embedding without quotient projection.
    """

    def __init__(
        self,
        T: int = 1000,
        L: int = 24,
        d: int = 8,
        classes: int = 5,
        *,
        frame_mode: str = "dct",
    ):
        super().__init__()
        self.T = T
        self.L = L
        self.d = d

        self.frame = LinearFrame(T=T, L=L, mode=frame_mode)

        # Per-channel mapping: [q_j, w_j] in R^(d + L)
        # DeepSets architecture matched in parameter count to Model B
        in_dim = d + L
        self.phi = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.head = nn.Linear(256, classes)

    def forward(
        self,
        q: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            q: Operators [B, m, d]
            y: Signals [B, T, m]
        """
        # Apply temporal frame to each channel: W_Y is [B, L, m]
        W_Y = self.frame(y)
        # Transpose to [B, m, L]
        W_Y = W_Y.transpose(1, 2)

        # Concatenate operator and waveform features per channel: [B, m, d + L]
        channel_feats = torch.cat([q, W_Y], dim=-1)

        # Per-channel mapping
        phi_out = self.phi(channel_feats)  # [B, m, 256]

        if mask is not None:
            # Masked pooling
            mask_expanded = mask.unsqueeze(-1).float()
            phi_sum = (phi_out * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = phi_sum / counts
        else:
            pooled = phi_out.mean(dim=1)  # [B, 256]

        latent = self.rho(pooled)
        logits = self.head(latent)

        if return_latent:
            return logits, latent
        return logits


class OSSetOpFused(nn.Module):
    """Fused Model: Canonical morphology SetOperator + residual OS branch.

    z_total = z_morph + W_obs @ E_obs, with W_obs initialized to zero
    for a safe, auditable residual initialization.
    """

    def __init__(
        self,
        canonical_model: nn.Module,
        T: int = 1000,
        L: int = 24,
        d: int = 8,
        *,
        frame_mode: str = "dct",
        freeze_canonical: bool = True,
        rcond: float = 1e-5,
    ):
        super().__init__()
        self.canonical = canonical_model
        if freeze_canonical:
            for p in self.canonical.parameters():
                p.requires_grad = False

        self.os_branch = OSOnlyModel(
            T=T, L=L, d=d, classes=5, frame_mode=frame_mode, rcond=rcond
        )
        # Safe zero-residual adapter
        self.w_obs = nn.Parameter(torch.zeros(256, 256))
        self.head = nn.Linear(256, 5)

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        raw_y: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Canonical morphology representation
        z_morph = self.canonical.encode_context(operators, responses, mask=mask)

        # OS observable representation
        state = observable_state(operators, raw_y, rcond=self.os_branch.rcond, mask=mask, d=self.os_branch.d)
        Z = self.os_branch.frame(state.projected)
        e_obs = self.os_branch.encoder(Z, state.projector, state.rank)

        # Fused latent: z_total = z_morph + e_obs @ W_obs^T
        z_total = z_morph + torch.matmul(e_obs, self.w_obs)
        logits = self.head(z_total)

        if return_latent:
            return logits, z_total
        return logits
