"""Transparent Q-VCG Representation Extractors (R0, R1, R2, R3).

Adheres to PRD Sections 30-37:
- R0: Basic VCG summary baseline (mean, cov, axis ranges, speed, curvature, RR).
- R1: Spatial domain representation (M domains: occupancy, physical dwell, transitions).
- R2: Quotient motif representation (K motifs: occupancy, physical dwell, transitions, run-compression).
- R3: Quotient + Where factorized representation ([Z_motif, Z_location]).
"""
from __future__ import annotations

import numpy as np
from .tokenizer import QVCGTokenizer


def extract_r0_vcg_summary(
    v_physical: np.ndarray,
    v_dot: np.ndarray,
    xi: np.ndarray,
    rr_intervals_s: np.ndarray | None = None,
) -> np.ndarray:
    """Extracts R0 basic VCG summary features (PRD Section 35).

    Args:
        v_physical: [3, T] physical VCG coordinates in mV.
        v_dot: [3, T] physical velocity in mV/s.
        xi: [3, T] intrinsic descriptor [s, rho, kappa].
        rr_intervals_s: Optional array of RR intervals in seconds.

    Returns:
        feature_vector: [D_0] array.
    """
    # 1. Mean VCG vector (3)
    mu_v = np.mean(v_physical, axis=-1)

    # 2. Covariance of v (6 upper triangle elements)
    cov_v = np.cov(v_physical)
    cov_upper = cov_v[np.triu_indices(3)]

    # 3. Axis ranges (3)
    axis_ranges = np.ptp(v_physical, axis=-1)

    # 4. Mean and range of ||v|| (2)
    norm_v = np.linalg.norm(v_physical, axis=0)
    mean_norm = float(np.mean(norm_v))
    range_norm = float(np.ptp(norm_v))

    # 5. Speed mean and std (2)
    speed = np.exp(xi[0])
    mean_speed = float(np.mean(speed))
    std_speed = float(np.std(speed))

    # 6. Curvature summary (3)
    curv = xi[2]
    mean_curv = float(np.mean(curv))
    std_curv = float(np.std(curv))
    p90_curv = float(np.percentile(curv, 90))

    # 7. RR interval mean and std (2)
    if rr_intervals_s is not None and len(rr_intervals_s) > 0:
        mean_rr = float(np.mean(rr_intervals_s))
        std_rr = float(np.std(rr_intervals_s)) if len(rr_intervals_s) > 1 else 0.0
    else:
        mean_rr = 0.8
        std_rr = 0.0

    feats = np.concatenate([
        mu_v,
        cov_upper,
        axis_ranges,
        [mean_norm, range_norm],
        [mean_speed, std_speed],
        [mean_curv, std_curv, p90_curv],
        [mean_rr, std_rr],
    ])
    return feats.astype(np.float32)


def compute_motif_occupancy_and_dwell(
    tokens: np.ndarray,
    num_classes: int,
    total_duration_s: float,
    ood_token: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes occupancy mass pi and physical dwell time tau.

    Occupancy satisfies: sum(pi_k) + pi_OOD = 1.0.
    Dwell is measured in seconds.
    """
    total_samples = len(tokens)
    if total_samples == 0:
        return np.zeros(num_classes + 1, dtype=np.float32), np.zeros(num_classes + 1, dtype=np.float32)

    dt = total_duration_s / total_samples

    counts = np.bincount(tokens, minlength=num_classes + 1)
    # Truncate to num_classes + 1 (last slot is OOD)
    counts = counts[:num_classes + 1]

    occupancy = (counts / total_samples).astype(np.float32)
    dwell = (counts * dt).astype(np.float32)
    return occupancy, dwell


def compute_transition_matrix(
    tokens: np.ndarray,
    num_classes: int,
    run_compress: bool = False,
) -> np.ndarray:
    """Computes transition count matrix between consecutive tokens."""
    if len(tokens) <= 1:
        return np.zeros((num_classes, num_classes), dtype=np.float32)

    seq = tokens
    if run_compress:
        # Collapse consecutive identical tokens (e.g. 2, 2, 2, 5 -> 2, 5)
        mask = np.concatenate([[True], seq[1:] != seq[:-1]])
        seq = seq[mask]

    if len(seq) <= 1:
        return np.zeros((num_classes, num_classes), dtype=np.float32)

    src = seq[:-1]
    dst = seq[1:]

    # Filter out OOD transitions
    valid = (src < num_classes) & (dst < num_classes)
    src_v = src[valid]
    dst_v = dst[valid]

    T = np.zeros((num_classes, num_classes), dtype=np.float32)
    np.add.at(T, (src_v, dst_v), 1.0)

    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T_norm = T / row_sums
    return T_norm.astype(np.float32)


def extract_r1_spatial_domain_features(
    domains: np.ndarray,
    num_domains: int,
    total_duration_s: float,
) -> np.ndarray:
    """Extracts R1 features for unmerged spatial domains (PRD Section 30)."""
    occ, dwell = compute_motif_occupancy_and_dwell(domains, num_domains, total_duration_s)
    T = compute_transition_matrix(domains, num_domains, run_compress=False)
    # Flatten transition matrix (or use row sums/diagonals + top transitions)
    # Using occupancy, dwell, and transition diagonals + row/col marginals
    t_diag = np.diag(T)
    t_flat = T.flatten()
    features = np.concatenate([occ, dwell, t_diag, t_flat])
    return features.astype(np.float32)


def extract_r2_quotient_motif_features(
    motifs: np.ndarray,
    num_motifs: int,
    total_duration_s: float,
) -> np.ndarray:
    """Extracts R2 features for quotient motifs (PRD Section 30)."""
    occ, dwell = compute_motif_occupancy_and_dwell(motifs, num_motifs, total_duration_s)
    T_std = compute_transition_matrix(motifs, num_motifs, run_compress=False)
    T_run = compute_transition_matrix(motifs, num_motifs, run_compress=True)

    t_diag = np.diag(T_std)
    t_flat = T_std.flatten()
    t_run_flat = T_run.flatten()

    features = np.concatenate([occ, dwell, t_diag, t_flat, t_run_flat])
    return features.astype(np.float32)


def extract_r3_quotient_plus_where(
    v_physical: np.ndarray,
    motifs: np.ndarray,
    num_motifs: int,
    total_duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts R3 factorized representation [Z_motif, Z_location] (PRD Section 30).

    Returns:
        z_r3: Concatenated [Z_motif, Z_location]
        z_motif: Motif branch features only
        z_where: Spatial location features only
    """
    z_motif = extract_r2_quotient_motif_features(motifs, num_motifs, total_duration_s)

    # Z_location: compute mean coordinate for each motif k in physical mV
    z_where = np.zeros(num_motifs * 3, dtype=np.float32)

    occ, _ = compute_motif_occupancy_and_dwell(motifs, num_motifs, total_duration_s)

    for k in range(num_motifs):
        mask = (motifs == k)
        if np.any(mask):
            mean_coord = np.mean(v_physical[:, mask], axis=1)  # [3]
            # Weight by motif occupancy pi_k
            z_where[k * 3 : (k + 1) * 3] = mean_coord * occ[k]

    z_r3 = np.concatenate([z_motif, z_where])
    return z_r3.astype(np.float32), z_motif.astype(np.float32), z_where.astype(np.float32)
