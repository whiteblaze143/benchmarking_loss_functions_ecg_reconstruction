"""Milestone R026: Canonical-Frame SetOperator (CF-SetOperator).

Pure mathematical front-end for observable measurement canonicalization:
1. Observable Subspace Projector: P_Q = Q^+ Q in R^{d x d}
2. Deterministic Pivoted Gram-Schmidt Canonical Frame: B_Q = C(P_Q) in R^{r x d}
3. Canonical Waveform Recovery: Y_canon = Y @ C_Q^T, where C_Q = B_Q @ Q^+ in R^{r x m}

No extra trainable parameters, no extra Transformer, no graph, no DCT encoder.
Directly composes with the existing Phase-KME SetOperator.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch


def observable_projector(Q: np.ndarray, tol: float = 1e-7) -> Tuple[np.ndarray, int]:
    """Compute the orthogonal projector P_Q onto row(Q) and numerical rank r.

    Args:
        Q: Measurement matrix [m, d].
        tol: Relative singular value cutoff for rank determination.

    Returns:
        P: Orthogonal projector [d, d], P = Q^+ Q = V_r V_r^T.
        rank: Numerical rank r = dim(row(Q)).
    """
    Q_64 = np.asarray(Q, dtype=np.float64)
    m, d = Q_64.shape
    u, s, vh = np.linalg.svd(Q_64, full_matrices=False)
    
    if len(s) == 0 or s[0] <= 0.0:
        return np.zeros((d, d), dtype=np.float64), 0
        
    cutoff = tol * s[0]
    rank = int(np.sum(s > cutoff))
    
    if rank == 0:
        return np.zeros((d, d), dtype=np.float64), 0
        
    v_r = vh[:rank].T  # [d, rank]
    P = v_r @ v_r.T    # [d, d]
    
    # Enforce exact symmetry
    P = 0.5 * (P + P.T)
    return P, rank


def canonical_frame(
    P: np.ndarray,
    rank: Optional[int] = None,
    tol: float = 1e-7,
    tie_tol: float = 1e-11,
) -> np.ndarray:
    """Compute a deterministic orthonormal canonical basis B_Q for range(P).

    Uses deterministic pivoted Gram-Schmidt on canonical coordinate basis vectors e_1..e_d:
    At each step:
    1. Project all canonical coordinate axes into the residual unspanned subspace:
       u_i = P e_i - sum_{j < k} (b_j)_i b_j
    2. Compute residual norms s_i = ||u_i||_2
    3. Select pivot index with maximum residual norm (ties within tie_tol broken by lowest coordinate index)
    4. Normalize b_k = u_{i*} / ||u_{i*}||_2
    5. Enforce deterministic sign: (b_k)_{i*} > 0

    Args:
        P: Orthogonal projector [d, d].
        rank: Target rank (if None, deduced from trace(P) or tol).
        tol: Residual norm cutoff for termination.
        tie_tol: Tolerance for considering candidate norms tied.

    Returns:
        B: Orthonormal canonical frame [r, d], B B^T = I_r.
    """
    P_64 = np.asarray(P, dtype=np.float64)
    d = P_64.shape[0]
    
    if rank is None:
        rank = int(np.round(np.trace(P_64)))
        
    if rank <= 0:
        return np.empty((0, d), dtype=np.float64)
        
    B = []
    for k in range(rank):
        # Candidate vectors: project standard basis e_i into residual subspace
        residuals = []
        for i in range(d):
            u_i = P_64[:, i].copy()
            for b_j in B:
                u_i -= b_j[i] * b_j
            residuals.append(u_i)
            
        norms = np.array([np.linalg.norm(u) for u in residuals], dtype=np.float64)
        max_norm = float(np.max(norms))
        
        if max_norm < tol:
            break
            
        # Tie-breaking: all candidates within tie_tol of max_norm are considered tied; pick lowest index
        tied_candidates = np.where(max_norm - norms <= tie_tol)[0]
        best_i = int(tied_candidates[0])
        
        b_k = residuals[best_i] / norms[best_i]
        
        # Deterministic sign convention: component along pivot index best_i must be strictly positive
        if b_k[best_i] < 0.0:
            b_k = -b_k
        elif np.isclose(b_k[best_i], 0.0, atol=1e-12):
            # Fallback if pivot component is zero: first non-zero component positive
            for val in b_k:
                if abs(val) > 1e-12:
                    if val < 0.0:
                        b_k = -b_k
                    break
                    
        B.append(b_k)
        
    if len(B) == 0:
        return np.empty((0, d), dtype=np.float64)
        
    return np.array(B, dtype=np.float64)


def canonicalize_measurements(
    Q: np.ndarray,
    Y: np.ndarray,
    tol: float = 1e-7,
    tie_tol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize an arbitrary linear measurement collection (Q, Y).

    Transforms:
        Q in R^{m x d}, Y in R^{T x m}
    into:
        B_Q in R^{r x d}, Y_canon in R^{T x r}
    such that:
        row(B_Q) = row(Q), B_Q B_Q^T = I_r, Y_canon = X B_Q^T.

    Args:
        Q: Measurement matrix [m, d].
        Y: Observed signal array [T, m] (or batch [N, T, m]).
        tol: Rank cutoff tolerance.
        tie_tol: Tie breaking tolerance.

    Returns:
        B_Q: Canonical frame [r, d].
        Y_canon: Canonical waveforms [T, r] (or [N, T, r]).
        C_Q: Transformation matrix [r, m], B_Q = C_Q @ Q.
    """
    Q_64 = np.asarray(Q, dtype=np.float64)
    P, rank = observable_projector(Q_64, tol=tol)
    B = canonical_frame(P, rank=rank, tol=tol, tie_tol=tie_tol)
    
    r = len(B)
    if r == 0:
        if Y.ndim == 2:
            return B, np.empty((Y.shape[0], 0), dtype=Y.dtype), np.empty((0, Q.shape[0]), dtype=np.float64)
        else:
            return B, np.empty((Y.shape[0], Y.shape[1], 0), dtype=Y.dtype), np.empty((0, Q.shape[0]), dtype=np.float64)
            
    # Compute C_Q = B @ pinv(Q)
    Q_pinv = np.linalg.pinv(Q_64, rcond=tol)  # [d, m]
    C = B @ Q_pinv                           # [r, m]
    
    # Y_canon = Y @ C^T
    if Y.ndim == 2:
        Y_canon = Y @ C.T
    elif Y.ndim == 3:
        Y_canon = np.einsum("ntm,rm->ntr", Y, C)
    else:
        raise ValueError(f"Expected Y with 2 or 3 dimensions, got ndim={Y.ndim}")
        
    return B, Y_canon.astype(Y.dtype), C


def canonicalize_beats(
    Q: np.ndarray,
    beats_mv: np.ndarray,
    tol: float = 1e-7,
    tie_tol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize phase-normalized beat waveforms.

    Args:
        Q: Measurement matrix [m, d].
        beats_mv: Beat tensor [n_beats, 256, m].
        tol: Rank cutoff tolerance.
        tie_tol: Tie breaking tolerance.

    Returns:
        B_Q: Canonical frame [r, d].
        beats_canon: Canonical beat tensor [n_beats, 256, r].
        C_Q: Transformation matrix [r, m].
    """
    B, beats_canon, C = canonicalize_measurements(Q, beats_mv, tol=tol, tie_tol=tie_tol)
    return B, beats_canon, C


def deployment_validity_condition(
    Q: np.ndarray,
    gamma_max: float = 10.0,
    tol: float = 1e-7,
) -> Tuple[bool, float, float]:
    """Evaluate deterministic deployment validity condition g(Q) <= gamma_max.

    Under independent isotropic channel noise E ~ N(0, sigma^2 I), the transformed
    noise covariance satisfies:
        Cov(E_tilde) = sigma^2 C_Q C_Q^T, where C_Q = B_Q Q^+.
        lambda_max(Cov(E_tilde)) <= sigma^2 ||C_Q||_2^2 = sigma^2 sigma_min(Q)^{-2}.

    For an allowable SNR loss budget of L dB (variance amplification A_max = 10^(L/10)):
        gamma_max = sqrt(A_max) = 10^(L / 20).
        Examples:
          L = 6 dB  -> gamma_max = 2.0  (A_max = 4)
          L = 12 dB -> gamma_max = 4.0  (A_max = 16)
          L = 20 dB -> gamma_max = 10.0 (A_max = 100)

    Mathematical distinction from condition number kappa(Q):
        ||B_Q Q^+||_2 = 1 / sigma_min(Q), whereas kappa(Q) = sigma_max(Q) / sigma_min(Q).
        These are NOT generally equivalent unless sigma_max(Q) = 1.
        Therefore, ||B_Q Q^+||_2 <= gamma_max is the exact deployment gate, while kappa(Q)
        serves as a scale-independent diagnostic condition number.

    Args:
        Q: Measurement matrix [m, d].
        gamma_max: Maximum tolerated noise amplification factor (10^(L/20)).
        tol: Rank cutoff tolerance for SVD.

    Returns:
        is_valid: True if g(Q) = ||B_Q Q^+||_2 <= gamma_max.
        g_Q: Exact noise amplification norm ||B_Q Q^+||_2 = 1 / sigma_min(Q).
        kappa_Q: Diagnostic condition number sigma_max(Q) / sigma_min(Q).
    """
    Q_64 = np.asarray(Q, dtype=np.float64)
    _, s, _ = np.linalg.svd(Q_64, full_matrices=False)
    if len(s) == 0 or s[0] <= 0.0:
        return False, float("inf"), float("inf")

    cutoff = tol * s[0]
    pos_s = s[s > cutoff]
    if len(pos_s) == 0:
        return False, float("inf"), float("inf")

    sigma_min = float(pos_s[-1])
    sigma_max = float(pos_s[0])

    g_Q = 1.0 / sigma_min
    kappa_Q = sigma_max / sigma_min

    is_valid = bool(g_Q <= gamma_max)
    return is_valid, g_Q, kappa_Q
