"""Deterministic Geometry Feature Expansion for SetOperator (NEF-Net Inspired).

Implements:
1. vech_sqrt2: Isometric vectorization of symmetric matrices preserving Frobenius inner product.
2. gamma(q): Deterministic map R^8 -> R^44 combining signed orientation q with projective rank-1 outer product qq^T.
3. compute_operator_gram: Operator Gram matrix G = Q Q^T and sign-insensitive relational alignment R = G ⊙ G.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def vech_sqrt2(matrix: torch.Tensor) -> torch.Tensor:
    """Extract unique entries of symmetric 8x8 matrices scaled by sqrt(2) on off-diagonals.
    
    Preserves Frobenius norm: ||vech_sqrt2(M)||_2 == ||M||_F.
    For an 8x8 symmetric matrix, extracts 8 diagonal and 28 off-diagonal entries (total 36).
    
    Args:
        matrix: Tensor of shape (..., 8, 8)
        
    Returns:
        Tensor of shape (..., 36)
    """
    if matrix.shape[-2:] != (8, 8):
        raise ValueError(f"Expected matrix of shape (..., 8, 8), got {matrix.shape}")

    # Row and column indices for lower triangular part (including diagonal)
    tril_indices = torch.tril_indices(8, 8, offset=0, device=matrix.device)
    rows, cols = tril_indices[0], tril_indices[1]

    # Extract elements
    elements = matrix[..., rows, cols]  # (..., 36)

    # Scale off-diagonal elements by sqrt(2)
    # Diagonal elements are where rows == cols
    is_diag = (rows == cols).to(dtype=matrix.dtype, device=matrix.device)
    scale = torch.where(
        is_diag.bool(),
        torch.tensor(1.0, dtype=matrix.dtype, device=matrix.device),
        torch.tensor(math.sqrt(2.0), dtype=matrix.dtype, device=matrix.device),
    )
    return elements * scale


def gamma(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Deterministic geometric feature expansion: q -> [q; vech_sqrt2(q q^T)] in R^44.
    
    Args:
        q: Measurement operator tensor of shape (..., 8)
        eps: Small constant for numerical stability during normalization check
        
    Returns:
        Tensor of shape (..., 44) containing:
        - First 8 features: unit-normalized signed operator q / ||q||_2
        - Next 36 features: isometric vectorization of rank-one projective matrix (q / ||q||_2)(q / ||q||_2)^T
    """
    if q.shape[-1] != 8:
        raise ValueError(f"Expected operator of shape (..., 8), got {q.shape}")

    if not torch.isfinite(q).all():
        raise ValueError("Operator contains NaN or infinite values")

    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    if (norm < eps).any():
        raise ValueError("Operator contains zero or near-zero norm vectors")

    # Unit-normalize operator
    q_unit = q / norm

    # Rank-1 outer product: q_unit q_unit^T in R^(..., 8, 8)
    q_outer = q_unit.unsqueeze(-1) @ q_unit.unsqueeze(-2)

    # Isometric vectorization preserving Frobenius norm
    q_vech = vech_sqrt2(q_outer)  # (..., 36)

    return torch.cat([q_unit, q_vech], dim=-1)  # (..., 44)


def compute_operator_gram(
    operators: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute operator Gram matrix G and relational alignment matrix R.
    
    Args:
        operators: Tensor of shape (batch, set_size, 8)
        mask: Optional boolean padding mask of shape (batch, set_size), True = padded
        eps: Minimum norm epsilon
        
    Returns:
        Tuple (G, R):
        - G: Unit-normalized Gram matrix (batch, set_size, set_size), G_ij = q_i^T q_j in [-1, 1]
        - R: Relational redundancy matrix (batch, set_size, set_size), R_ij = G_ij^2 in [0, 1]
    """
    if operators.shape[-1] != 8:
        raise ValueError(f"Expected operators of shape (batch, set_size, 8), got {operators.shape}")

    batch, set_size, _ = operators.shape
    device = operators.device
    dtype = operators.dtype

    # Safe normalization: for padded rows, avoid division by zero
    norms = torch.norm(operators, p=2, dim=-1, keepdim=True)  # (batch, set_size, 1)
    safe_norms = torch.clamp(norms, min=eps)
    q_norm = operators / safe_norms

    # Gram matrix: G_ij = q_i^T q_j
    G = torch.bmm(q_norm, q_norm.transpose(1, 2))  # (batch, set_size, set_size)

    # Relational alignment: R_ij = G_ij^2
    R = G.square()

    # For valid unit vectors, enforce diagonal self-alignment R_ii = 1.0 exactly
    # and clamp to [0, 1] for numerical float rounding
    diag_indices = torch.arange(set_size, device=device)
    R[:, diag_indices, diag_indices] = 1.0
    R = torch.clamp(R, min=0.0, max=1.0)
    G = torch.clamp(G, min=-1.0, max=1.0)

    # If mask is provided (True = padded), zero out padded rows and columns in R and G
    if mask is not None:
        valid_mask = (~mask).to(dtype=dtype)  # (batch, set_size)
        pair_valid = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)  # (batch, set_size, set_size)
        G = G * pair_valid
        R = R * pair_valid

    return G, R
