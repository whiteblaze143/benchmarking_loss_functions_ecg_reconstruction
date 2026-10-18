#!/usr/bin/env python3
"""Module A: Canonical Measurement Information Layer.

Implements the exact linear acquisition mathematics for continuous ECG measurement operators:
In the noiseless case:
    Y = Q X
    P_Q = Q^\\dagger Q           (orthogonal projection matrix onto row_space(Q))
    X_obs = Q^\\dagger Y = P_Q X   (canonical observable multichannel signal)
    P_perp = I - P_Q             (nullspace projection matrix)

In the Gaussian noisy case:
    Y = Q X + epsilon,  epsilon ~ N(0, Sigma_eps)
    G = Q^T Sigma_eps^{-1} Q     (quadratic information matrix)
    h = Q^T Sigma_eps^{-1} Y     (linear information vector)

Key Invariance Guarantees:
1. Basis Invariance: For any invertible M in GL(m), (M Q)^\\dagger (M Y) == Q^\\dagger Y.
2. Orthogonal Invariance: For any R in O(m), canonical representations are invariant.
3. Permutation Invariance: Row permutations of (Q, Y) produce identical X_obs and P_Q.
4. Polarity Invariance: Flipping sign of lead i and row i leaves X_obs and P_Q invariant.
5. Algebraic Redundancy: Linearly dependent measurement rows do not alter X_obs or P_Q.
6. Information Monotonicity: Genuinely independent measurements monotonically expand P_Q.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def compute_noiseless_observable_np(
    Q: np.ndarray,
    Y: np.ndarray,
    rcond: float = 1e-5,
    normalize_rows: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (X_obs, P_Q, P_perp) for noiseless linear measurements (NumPy).

    Args:
        Q: Measurement operator matrix, shape (..., m, d).
        Y: Measured lead signals, shape (..., m, T).
        rcond: Cutoff for small singular values in pseudoinverse.
        normalize_rows: If True, normalizes each measurement row by ||q_i||_2,
            ensuring numerical invariance to arbitrary sensor unit scaling.

    Returns:
        X_obs: Canonical observable signal in R^(..., d, T).
        P_Q: Orthogonal projection matrix in R^(..., d, d).
        P_perp: Nullspace projector (I - P_Q) in R^(..., d, d).
    """
    if normalize_rows:
        row_norms = np.linalg.norm(Q, axis=-1, keepdims=True)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        Q_calc = Q / row_norms
        Y_calc = Y / row_norms
    else:
        Q_calc = Q
        Y_calc = Y

    # Moore-Penrose pseudoinverse: shape (..., d, m)
    Q_pinv = np.linalg.pinv(Q_calc, rcond=rcond)
    
    # Projection matrix P_Q = Q^pinv @ Q: shape (..., d, d)
    P_Q = Q_pinv @ Q_calc
    # Enforce exact symmetry numerically
    P_Q = 0.5 * (P_Q + np.swapaxes(P_Q, -1, -2))
    
    # Observable signal: X_obs = Q^pinv @ Y: shape (..., d, T)
    X_obs = Q_pinv @ Y_calc
    
    d = Q.shape[-1]
    eye_d = np.eye(d, dtype=Q.dtype)
    P_perp = eye_d - P_Q
    
    return X_obs, P_Q, P_perp


def compute_noiseless_observable_torch(
    Q: torch.Tensor,
    Y: torch.Tensor,
    rcond: float = 1e-5,
    normalize_rows: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (X_obs, P_Q, P_perp) for noiseless linear measurements (PyTorch).

    Args:
        Q: Measurement operator matrix, shape (..., m, d).
        Y: Measured lead signals, shape (..., m, T).
        rcond: Cutoff for small singular values.
        normalize_rows: If True, normalizes each row by ||q_i||_2.

    Returns:
        X_obs: Canonical observable signal in R^(..., d, T).
        P_Q: Orthogonal projection matrix in R^(..., d, d).
        P_perp: Nullspace projector (I - P_Q) in R^(..., d, d).
    """
    if normalize_rows:
        row_norms = torch.norm(Q, dim=-1, keepdim=True)
        row_norms = torch.where(row_norms < 1e-12, torch.ones_like(row_norms), row_norms)
        Q_calc = Q / row_norms
        Y_calc = Y / row_norms
    else:
        Q_calc = Q
        Y_calc = Y

    # Moore-Penrose pseudoinverse: shape (..., d, m)
    Q_pinv = torch.linalg.pinv(Q_calc, rcond=rcond)
    
    # Projection matrix P_Q = Q^pinv @ Q: shape (..., d, d)
    P_Q = torch.matmul(Q_pinv, Q_calc)
    # Symmetrize
    P_Q = 0.5 * (P_Q + P_Q.transpose(-1, -2))
    
    # Observable signal: X_obs = Q^pinv @ Y: shape (..., d, T)
    X_obs = torch.matmul(Q_pinv, Y_calc)
    
    d = Q.shape[-1]
    eye_d = torch.eye(d, dtype=Q.dtype, device=Q.device)
    P_perp = eye_d - P_Q
    
    return X_obs, P_Q, P_perp


def compute_gaussian_information_np(
    Q: np.ndarray,
    Y: np.ndarray,
    Sigma_eps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian information terms (G, h) under linear measurement model (NumPy).

    Model: Y = Q X + epsilon,  epsilon ~ N(0, Sigma_eps)
    G = Q^T Sigma_eps^{-1} Q  (Fisher Information Matrix / Quadratic Likelihood Term)
    h = Q^T Sigma_eps^{-1} Y  (Linear Information Term)

    Args:
        Q: Measurement operator, shape (m, d).
        Y: Measured signals, shape (m, T).
        Sigma_eps: Noise covariance, shape (m, m). Defaults to identity I_m.

    Returns:
        G: Quadratic information matrix, shape (d, d).
        h: Linear information term, shape (d, T).
    """
    m, d = Q.shape
    if Sigma_eps is None:
        # Default: isotropic unit noise
        G = Q.T @ Q
        h = Q.T @ Y
    else:
        # Solve Sigma_eps @ W = Q -> W = Sigma_eps^{-1} Q
        W_Q = np.linalg.solve(Sigma_eps, Q)
        W_Y = np.linalg.solve(Sigma_eps, Y)
        G = Q.T @ W_Q
        h = Q.T @ W_Y
    return G, h


def compute_gaussian_information_torch(
    Q: torch.Tensor,
    Y: torch.Tensor,
    Sigma_eps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Gaussian information terms (G, h) under linear measurement model (PyTorch)."""
    if Sigma_eps is None:
        G = torch.matmul(Q.transpose(-1, -2), Q)
        h = torch.matmul(Q.transpose(-1, -2), Y)
    else:
        # Solve Sigma_eps @ W = Q
        W_Q = torch.linalg.solve(Sigma_eps, Q)
        W_Y = torch.linalg.solve(Sigma_eps, Y)
        G = torch.matmul(Q.transpose(-1, -2), W_Q)
        h = torch.matmul(Q.transpose(-1, -2), W_Y)
    return G, h


def evaluate_concept_observability(
    P_Q: Union[np.ndarray, torch.Tensor],
    concept_direction: Union[np.ndarray, torch.Tensor],
    tol_observable: float = 1e-4,
    tol_unobservable: float = 0.90,
) -> Dict[str, Any]:
    """Evaluate whether a clinical concept direction is observable from P_Q.

    For a linear ECG finding characterized by direction vector v in R^d:
    - observable fraction: eta = ||P_Q v||^2 / ||v||^2
    - nullspace residue:  rho = ||(I - P_Q) v||^2 / ||v||^2 = 1 - eta

    Classification:
        * 'OBSERVABLE': rho < tol_observable (finding lies strictly in measured subspace)
        * 'PARTIALLY_OBSERVABLE': tol_observable <= rho < tol_unobservable
        * 'UNOBSERVABLE': rho >= tol_unobservable (finding is virtually unconstrained)

    Args:
        P_Q: Orthogonal projection matrix, shape (d, d).
        concept_direction: Vector v in R^d.
        tol_observable: Threshold below which residue is considered fully observed.
        tol_unobservable: Threshold above which residue is considered unsupported.

    Returns:
        Dict with observability metrics and status string.
    """
    is_torch = isinstance(P_Q, torch.Tensor)
    if is_torch:
        v = concept_direction.to(P_Q.device, dtype=P_Q.dtype)
        v_norm_sq = torch.sum(v ** 2)
        if v_norm_sq < 1e-12:
            return {"status": "ZERO_VECTOR", "observable_fraction": 0.0, "nullspace_residue": 1.0}
        
        P_v = torch.matmul(P_Q, v)
        obs_norm_sq = torch.sum(P_v ** 2)
        eta = (obs_norm_sq / v_norm_sq).item()
    else:
        v = np.asarray(concept_direction, dtype=P_Q.dtype)
        v_norm_sq = np.sum(v ** 2)
        if v_norm_sq < 1e-12:
            return {"status": "ZERO_VECTOR", "observable_fraction": 0.0, "nullspace_residue": 1.0}
        P_v = P_Q @ v
        obs_norm_sq = np.sum(P_v ** 2)
        eta = float(obs_norm_sq / v_norm_sq)

    eta = max(0.0, min(1.0, eta))
    rho = 1.0 - eta

    if rho < tol_observable:
        status = "OBSERVABLE"
    elif rho < tol_unobservable:
        status = "PARTIALLY_OBSERVABLE"
    else:
        status = "UNOBSERVABLE"

    return {
        "status": status,
        "observable_fraction": eta,
        "nullspace_residue": rho,
    }


class MeasurementInformationLayer(nn.Module):
    """PyTorch module implementing Module A: Measurement Information Layer.

    Maps variable operator sets (Q, Y) into canonical observable signals and projection:
        Input:
            Q: (batch_size, m, d)
            Y: (batch_size, m, T)
        Output:
            X_obs: (batch_size, d, T)  Canonical observable multi-lead signal
            P_Q:   (batch_size, d, d)  Measured subspace projector
            rank:  (batch_size,)       Observed rank
    """

    def __init__(self, canonical_dim: int = 8, rcond: float = 1e-5):
        super().__init__()
        self.canonical_dim = canonical_dim
        self.rcond = rcond
        self.register_buffer("eye_d", torch.eye(canonical_dim, dtype=torch.float32))

    def forward(
        self,
        Q: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute observable signal and subspace projector.

        Args:
            Q: (B, m, d) or (m, d)
            Y: (B, m, T) or (m, T)

        Returns:
            X_obs: (B, d, T)
            P_Q: (B, d, d)
            P_perp: (B, d, d)
        """
        is_unbatched = Q.ndim == 2
        if is_unbatched:
            Q = Q.unsqueeze(0)
            Y = Y.unsqueeze(0)

        X_obs, P_Q, P_perp = compute_noiseless_observable_torch(
            Q, Y, rcond=self.rcond, normalize_rows=True
        )

        if is_unbatched:
            return X_obs.squeeze(0), P_Q.squeeze(0), P_perp.squeeze(0)
        return X_obs, P_Q, P_perp
