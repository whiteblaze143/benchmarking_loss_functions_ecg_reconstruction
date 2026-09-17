from __future__ import annotations

import numpy as np


def fit_macro_basis(
    values: np.ndarray,
    return_complement: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit population mean and top-3 spatial principal eigenvectors (macro basis) in physical mV.
    
    values: shape (N, leads) e.g. (N, 8) in physical mV
    Returns:
      train_mean: shape (leads,)
      basis_macro: shape (leads, 3) satisfying basis_macro.T @ basis_macro = I_3
      basis_perp (optional): shape (leads, 5) orthonormal complement satisfying basis_macro.T @ basis_perp = 0
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be a 2D matrix (N, leads)")
    mean = arr.mean(axis=0)
    centered = arr - mean
    covariance = centered.T @ centered / max(len(arr) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis_macro = eigenvectors[:, order[:3]]
    if return_complement:
        basis_perp = eigenvectors[:, order[3:]]
        return mean, basis_macro, basis_perp
    return mean, basis_macro


def decompose_macro_residual(
    x: np.ndarray,
    train_mean: np.ndarray,
    basis: np.ndarray,
    basis_perp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose multilead voltages into 3D macro spatial coordinates and non-macro residual.
    
    x: shape (..., leads) in physical mV
    basis: shape (leads, 3)
    basis_perp (optional): shape (leads, 5) orthonormal complement
    
    Returns:
      macro: shape (..., 3)
      residual: shape (..., leads) satisfying basis.T @ residual = 0
      u (if basis_perp provided): shape (..., 5) genuine 5D residual coordinates
    """
    centered = np.asarray(x, dtype=np.float64) - train_mean
    leads = centered.shape[-1]
    if basis.shape != (leads, 3):
        raise ValueError(f"macro basis must have shape ({leads}, 3)")
    macro = centered @ basis
    residual = centered - macro @ basis.T
    if basis_perp is not None:
        if basis_perp.shape != (leads, leads - 3):
            raise ValueError(f"basis_perp must have shape ({leads}, {leads - 3})")
        u = centered @ basis_perp
        return macro, residual, u
    return macro, residual


def soft_membership(macro: np.ndarray, anchors: np.ndarray, sigma2: float) -> np.ndarray:
    """Compute soft RBF membership of 3D macro states over global macro anchors.
    
    macro: shape (N, 3)
    anchors: shape (K, 3)
    sigma2: bandwidth squared
    Returns:
      membership: shape (N, K) with values in Delta^{K-1}
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    m = np.asarray(macro, dtype=np.float64)
    a = np.asarray(anchors, dtype=np.float64)
    d2 = ((m[:, None, :] - a[None, :, :]) ** 2).sum(axis=-1)
    logits = -d2 / sigma2
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def anchor_conditioned_mmd(
    feature_cells: list[np.ndarray],
    membership_cells: list[np.ndarray],
    *,
    min_effective_mass: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Anchor-Conditioned MMD (AC-MMD) distance matrix and overlap mass matrix.
    
    D_{ACMMD}^2(g, h) = (1 / O_{gh}) * sum_k sqrt(p_g(k)*p_h(k)) * ||m_{g,k} - m_{h,k}||^2
    where O_{gh} = sum_{k in K_shared} sqrt(p_g(k)*p_h(k))
    
    Returns:
      distance: shape (G, G)
      overlap_mass: shape (G, G)
    """
    diagnostics = conditional_distance_diagnostics(
        feature_cells, membership_cells, min_effective_mass=min_effective_mass
    )
    return diagnostics["distance"], diagnostics["overlap_mass"]


def conditional_distance(
    feature_cells: list[np.ndarray],
    membership_cells: list[np.ndarray],
    *,
    min_effective_mass: float = 2.0,
) -> np.ndarray:
    """Legacy backward-compatible wrapper returning conditional distance matrix."""
    dist, _ = anchor_conditioned_mmd(
        feature_cells, membership_cells, min_effective_mass=min_effective_mass
    )
    return dist


def conditional_distance_diagnostics(
    feature_cells: list[np.ndarray],
    membership_cells: list[np.ndarray],
    *,
    min_effective_mass: float = 2.0,
) -> dict[str, object]:
    """Compute AC-MMD distance matrix alongside support overlap diagnostics."""
    if len(feature_cells) != len(membership_cells):
        raise ValueError("feature and membership cell counts differ")
    groups = len(feature_cells)
    if groups < 2:
        raise ValueError("at least 2 phase cells required")

    means = []
    prevalence = []
    effective = []
    for features, membership in zip(feature_cells, membership_cells, strict=True):
        if len(features) != len(membership):
            raise ValueError("cell feature and membership rows differ")
        mass = membership.sum(axis=0)
        means.append((membership.T @ features) / (mass[:, None] + 1e-8))
        prevalence.append(mass / max(len(membership), 1))
        effective.append(mass**2 / (np.square(membership).sum(axis=0) + 1e-12))

    means = np.stack(means)
    prevalence = np.stack(prevalence)
    effective = np.stack(effective)

    distance = np.zeros((groups, groups), dtype=np.float64)
    overlap_mass = np.zeros((groups, groups), dtype=np.float64)
    shared_counts = np.zeros((groups, groups), dtype=np.int32)

    for g in range(groups):
        for h in range(g + 1, groups):
            active = (effective[g] >= min_effective_mass) & (effective[h] >= min_effective_mass)
            shared_counts[g, h] = shared_counts[h, g] = int(np.sum(active))
            raw = np.where(active, np.sqrt(prevalence[g] * prevalence[h]), 0.0)
            denominator = float(raw.sum())
            overlap_mass[g, h] = overlap_mass[h, g] = denominator
            if denominator < 1e-8:
                raise ValueError(f"ineligible: no shared effective macrostate for cells {g},{h}")
            per_state = np.square(means[g] - means[h]).sum(axis=1)
            dist_val = float(raw @ per_state / denominator)
            distance[g, h] = distance[h, g] = dist_val

    return {
        "distance": distance,
        "overlap_mass": overlap_mass,
        "shared_counts": shared_counts,
        "effective_anchors_per_cell": effective,
    }


def within_macro_phase_shuffle(
    feature_cells: list[np.ndarray],
    membership_cells: list[np.ndarray],
    seed: int = 42,
) -> list[np.ndarray]:
    """Conditionally permute residual features across phase cells within macro anchor clusters.
    
    Preserves P(r | z) and macrostate distribution while breaking phase coupling R perp G | Z.
    """
    rng = np.random.default_rng(seed)
    n_phases = len(feature_cells)
    k_anchors = membership_cells[0].shape[1]

    # Combine all points with their (phase, hard_cluster) tag
    cluster_bins: dict[int, list[tuple[int, int, np.ndarray]]] = {k: [] for k in range(k_anchors)}
    for g in range(n_phases):
        feats = feature_cells[g]
        mems = membership_cells[g]
        best_anchor = np.argmax(mems, axis=-1)
        for idx in range(len(feats)):
            cluster_bins[best_anchor[idx]].append((g, idx, feats[idx]))

    # For each cluster bin, permute feature vectors across their original slots
    shuffled_cells = [np.empty_like(feature_cells[g]) for g in range(n_phases)]
    for k in range(k_anchors):
        bin_items = cluster_bins[k]
        if not bin_items:
            continue
        slots = [(g, idx) for g, idx, _ in bin_items]
        features = [feat for _, _, feat in bin_items]
        perm = rng.permutation(len(features))
        for (g, idx), p_idx in zip(slots, perm):
            shuffled_cells[g][idx] = features[p_idx]

    return shuffled_cells


def non_adjacent_circular_mask(phases: int = 16) -> np.ndarray:
    """Allowed mask: excludes self-diagonal and immediately adjacent phases on circle."""
    indices = np.arange(phases)
    separation = np.abs(indices[:, None] - indices[None])
    circular_dist = np.minimum(separation, phases - separation)
    return circular_dist > 1


def recurrence_from_distances(
    distances: np.ndarray,
    tau: float,
    *,
    phases: int = 16,
    mask_type: str = "non_adjacent",
    laplacian_normalize: bool = True,
) -> np.ndarray:
    """Convert distance matrix to recurrence matrix under chosen mask and normalization."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if mask_type == "non_adjacent":
        allowed = non_adjacent_circular_mask(phases)
    elif mask_type == "all_pairs":
        allowed = np.ones((phases, phases), dtype=bool)
    elif mask_type == "exclude_diagonal":
        allowed = ~np.eye(phases, dtype=bool)
    else:
        raise ValueError(f"unknown mask_type: {mask_type}")

    affinity = np.where(allowed, np.exp(-distances / tau), 0.0)
    if not laplacian_normalize:
        return affinity.astype(np.float32)

    degree = affinity.sum(axis=-1).clip(min=1e-8)
    inverse_root = 1.0 / np.sqrt(degree)
    result = inverse_root[:, None] * affinity * inverse_root[None, :]
    return result.astype(np.float32)
