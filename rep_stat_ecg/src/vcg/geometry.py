"""LVCG Physical 12-Lead Direction Matrix and SVD Tikhonov Left-Inverse.

Table 7 (LVCG, ICML 2026 / arXiv:2605.31249):
Standard 12-lead direction vectors used to form the lead direction matrix A.
Each row is u_l^T = [ux, uy, uz].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Canonical 12-lead names in PTB-XL / MIMIC order
CANONICAL_12_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
)

# Table 7: Standard 12-lead direction vectors [ux, uy, uz]
TABLE_7_LEAD_DIRECTIONS: dict[str, list[float]] = {
    "I":   [ 1.00000,  0.00000,  0.00000],
    "II":  [ 0.50000,  0.86603,  0.00000],
    "III": [-0.50000,  0.86603,  0.00000],
    "aVR": [-0.86603, -0.50000,  0.00000],
    "aVL": [ 0.86603, -0.50000,  0.00000],
    "aVF": [ 0.00000,  1.00000,  0.00000],
    "V1":  [-0.33682,  0.17365,  0.92542],
    "V2":  [ 0.33682,  0.17365,  0.92542],
    "V3":  [ 0.75441,  0.17365,  0.63302],
    "V4":  [ 0.96985,  0.17365,  0.17101],
    "V5":  [ 0.92542,  0.17365, -0.33682],
    "V6":  [ 0.63302,  0.17365, -0.75441],
}


def get_lead_direction_matrix(
    leads: tuple[str, ...] = CANONICAL_12_LEADS,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Returns the [L, 3] lead direction matrix A for the specified leads."""
    arr = np.array([TABLE_7_LEAD_DIRECTIONS[l] for l in leads], dtype=np.float32)
    return torch.tensor(arr, dtype=dtype, device=device)


def compute_tikhonov_svd_pinv(
    A: torch.Tensor,
    lam: float = 1e-6,
    rcond: float = 1e-5,
) -> torch.Tensor:
    """Computes regularized left-inverse A_lam^+ = W diag(sigma_i / (sigma_i^2 + lam)) U^T.

    Args:
        A: [L, 3] lead direction matrix with L >= 3.
        lam: Tikhonov regularization coefficient.
        rcond: Relative threshold below which singular values are zeroed out.

    Returns:
        [3, L] pseudoinverse matrix A_pinv.
    """
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"Expected A to have shape [L, 3], got {tuple(A.shape)}")
    L = A.shape[0]
    if L < 3:
        raise ValueError(f"Need at least 3 leads to invert 3D VCG, got {L}")

    # Thin SVD: A = U Sigma W^T where U is [L, 3], Sigma is [3], W is [3, 3]
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    W = Vh.mH  # [3, 3]

    # Shrinkage factor: sigma_i / (sigma_i^2 + lam)
    mask = S > (S[0] * rcond)
    shrinkage = torch.zeros_like(S)
    shrinkage[mask] = S[mask] / (S[mask] ** 2 + lam)

    # A_pinv = W @ diag(shrinkage) @ U^T: [3, 3] @ [3, 3] @ [3, L] -> [3, L]
    A_pinv = W @ torch.diag(shrinkage) @ U.mH
    return A_pinv
