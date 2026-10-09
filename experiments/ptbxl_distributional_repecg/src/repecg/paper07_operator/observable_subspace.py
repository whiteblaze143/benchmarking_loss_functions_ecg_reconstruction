from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ObservableState:
    """Algebraically exact observable representation of linear ECG measurements.

    Attributes:
        projected: Minimum-coordinate-norm observable waveform X_obs in R^[B, T, d].
                   In exact noiseless model: X_obs = X @ P_Q and X_obs @ P_Q = X_obs.
        projector: Orthogonal projector P_Q = Q^+ Q onto row(Q) in R^[B, d, d].
                   Satisfies P^2 = P and P^T = P.
        rank: Observable subspace numerical rank r in [B] (int64).
        singular_values: Singular values of Q in [B, d].
    """
    projected: torch.Tensor        # [B, T, d]
    projector: torch.Tensor        # [B, d, d]
    rank: torch.Tensor             # [B]
    singular_values: torch.Tensor  # [B, d]


def _single_sample_observable_state(
    q: torch.Tensor,
    y: torch.Tensor,
    *,
    rcond: float = 1e-5,
    d: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute observable state for a single recording (unbatched).

    q: [m, d]
    y: [T, m]
    """
    m = q.shape[0]
    T = y.shape[0]
    orig_dtype = q.dtype
    orig_device = q.device

    if m == 0:
        # No measurements available
        proj_wave = torch.zeros((T, d), dtype=orig_dtype, device=orig_device)
        projector = torch.zeros((d, d), dtype=orig_dtype, device=orig_device)
        rank = torch.tensor(0, dtype=torch.int64, device=orig_device)
        s_vals = torch.zeros((d,), dtype=orig_dtype, device=orig_device)
        return proj_wave, projector, rank, s_vals

    # Compute SVD in float64 for machine precision stability
    q64 = q.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)

    # SVD: Q = U @ diag(S) @ Vh
    # U: [m, k], S: [k], Vh: [k, d] where k = min(m, d)
    U, S, Vh = torch.linalg.svd(q64, full_matrices=False)
    k = S.shape[0]

    # Explicit threshold based on declared rcond
    s_max = S[0] if k > 0 else torch.tensor(0.0, dtype=torch.float64, device=orig_device)
    cutoff = rcond * s_max
    keep = S > cutoff
    num_rank = keep.sum()

    # Moore-Penrose pseudo-inverse singular values: 1/sigma if sigma > cutoff else 0
    sinv = torch.where(keep, 1.0 / torch.clamp(S, min=1e-18), torch.zeros_like(S))

    # Q_pinv = V @ diag(sinv) @ U^T = Vh^T @ (sinv * U^T)
    # Vh^T: [d, k], (sinv[:, None] * U^T): [k, m]
    # Q_pinv: [d, m]
    Q_pinv = (Vh.transpose(-2, -1) * sinv.unsqueeze(-2)) @ U.transpose(-2, -1)

    # Orthogonal projector onto row(Q): P_Q = Q_pinv @ Q in R^[d, d]
    P_Q = Q_pinv @ q64
    # Enforce exact symmetry numerically
    P_Q = 0.5 * (P_Q + P_Q.transpose(-2, -1))

    # Minimum-coordinate-norm observable waveform: X_obs = Y @ (Q_pinv)^T in R^[T, d]
    X_obs = y64 @ Q_pinv.transpose(-2, -1)

    # Pad singular values to fixed size d for consistent tensor shape
    s_padded = torch.zeros((d,), dtype=torch.float64, device=orig_device)
    if k > 0:
        s_padded[:k] = S

    return (
        X_obs.to(dtype=orig_dtype),
        P_Q.to(dtype=orig_dtype),
        num_rank.to(dtype=torch.int64),
        s_padded.to(dtype=orig_dtype),
    )


def observable_state(
    q: torch.Tensor,
    y: torch.Tensor,
    *,
    rcond: float = 1e-5,
    mask: Optional[torch.Tensor] = None,
    d: int = 8,
) -> ObservableState:
    """Compute the algebraically exact ObservableState from measurement operators and signals.

    Args:
        q: Measurement operators [B, m, d] or [m, d]
        y: Measured signals [B, T, m] or [T, m]
        rcond: SVD relative singular value cutoff threshold (default 1e-5)
        mask: Optional boolean mask [B, m] or [m] where True indicates valid measurements
        d: Coordinate convention dimension (default 8)

    Returns:
        ObservableState with fields:
            projected: [B, T, d]
            projector: [B, d, d]
            rank: [B]
            singular_values: [B, d]
    """
    is_unbatched = (q.ndim == 2)
    if is_unbatched:
        q_b = q.unsqueeze(0)
        y_b = y.unsqueeze(0)
        mask_b = mask.unsqueeze(0) if mask is not None else None
    else:
        q_b = q
        y_b = y
        mask_b = mask

    B, m, q_dim = q_b.shape
    if q_dim != d:
        raise ValueError(f"Expected q last dimension to be d={d}, got {q_dim}")
    if y_b.shape[0] != B or y_b.shape[2] != m:
        raise ValueError(f"Shape mismatch: q is [{B}, {m}, {d}], y is {list(y_b.shape)}")

    T = y_b.shape[1]

    # Process each sample in batch (handles variable/masked valid measurements cleanly)
    proj_list = []
    proj_mat_list = []
    rank_list = []
    s_val_list = []

    for b in range(B):
        q_item = q_b[b]
        y_item = y_b[b]
        if mask_b is not None:
            valid = mask_b[b].bool()
            q_item = q_item[valid]
            y_item = y_item[:, valid]

        x_obs, p_q, r_val, s_vals = _single_sample_observable_state(
            q_item, y_item, rcond=rcond, d=d
        )
        proj_list.append(x_obs)
        proj_mat_list.append(p_q)
        rank_list.append(r_val)
        s_val_list.append(s_vals)

    projected = torch.stack(proj_list, dim=0)
    projector = torch.stack(proj_mat_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
    singular_values = torch.stack(s_val_list, dim=0)

    return ObservableState(
        projected=projected,
        projector=projector,
        rank=rank,
        singular_values=singular_values,
    )


def compute_projector(
    q: torch.Tensor,
    *,
    rcond: float = 1e-5,
    d: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute only the orthogonal projector P_Q and rank from Q.

    Args:
        q: [B, m, d] or [m, d]
        rcond: SVD cutoff threshold
        d: Subspace dimension (default 8)

    Returns:
        (projector [B, d, d], rank [B])
    """
    dummy_y = torch.zeros(
        (q.shape[0], 1, q.shape[1]) if q.ndim == 3 else (1, q.shape[0]),
        dtype=q.dtype,
        device=q.device,
    )
    state = observable_state(q, dummy_y, rcond=rcond, d=d)
    return state.projector, state.rank
