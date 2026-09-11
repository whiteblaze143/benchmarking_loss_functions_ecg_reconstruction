"""ECG -> VCG Lift, Projection, and Non-Dipolar Residual Auditing.

Implements:
- v(t) = A^+ E(t)
- E_proj(t) = A v(t)
- r(t) = E(t) - E_proj(t)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from .geometry import get_lead_direction_matrix, compute_tikhonov_svd_pinv, CANONICAL_12_LEADS


class VCGLift(nn.Module):
    """Linear VCG lift and projection with non-dipolar residual tracking."""

    def __init__(
        self,
        leads: tuple[str, ...] = CANONICAL_12_LEADS,
        lam: float = 1e-6,
    ):
        super().__init__()
        self.leads = leads
        self.lam = lam

        A = get_lead_direction_matrix(leads)
        A_pinv = compute_tikhonov_svd_pinv(A, lam=lam)

        self.register_buffer("A", A)            # [L, 3]
        self.register_buffer("A_pinv", A_pinv)  # [3, L]

    def lift(self, E: torch.Tensor) -> torch.Tensor:
        """Lifts ECG [..., L, T] into 3D VCG trajectory [..., 3, T]."""
        if E.shape[-2] != self.A.shape[0]:
            raise ValueError(f"Expected {self.A.shape[0]} leads at dim -2, got {E.shape[-2]}")
        # v = A_pinv @ E: [3, L] @ [..., L, T] -> [..., 3, T]
        v = torch.einsum("cl,...lt->...ct", self.A_pinv, E)
        return v

    def project(self, v: torch.Tensor) -> torch.Tensor:
        """Projects 3D VCG trajectory [..., 3, T] to multi-lead ECG [..., L, T]."""
        if v.shape[-2] != 3:
            raise ValueError(f"Expected 3 channels at dim -2, got {v.shape[-2]}")
        # E = A @ v: [L, 3] @ [..., 3, T] -> [..., L, T]
        E = torch.einsum("lc,...ct->...lt", self.A, v)
        return E

    def compute_residual(self, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lifts ECG, projects back, and computes residual.

        Returns:
            v: [..., 3, T] VCG trajectory
            E_proj: [..., L, T] projected ECG
            r_norm: [..., T] pointwise L2 norm of residual ||E - E_proj||
        """
        v = self.lift(E)
        E_proj = self.project(v)
        r = E - E_proj
        r_norm = torch.linalg.norm(r, dim=-2)
        return v, E_proj, r_norm
