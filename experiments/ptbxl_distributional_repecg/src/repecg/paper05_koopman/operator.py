from __future__ import annotations

import numpy as np


def soft_observables(means: np.ndarray, anchors: np.ndarray, tau: float) -> np.ndarray:
    """Compute soft RBF observable coordinates over anchor centers.
    
    Returns array with values in Delta^{d-1} such that sum_k psi_{tk} = 1.0 and psi_{tk} >= 0.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    means = np.asarray(means, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if means.ndim == 1:
        means = means[None, :]
    d2 = ((means[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=-1)
    logits = -d2 / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    probs = weights / weights.sum(axis=1, keepdims=True)
    return probs.astype(np.float64)


def median_anchor_bandwidth(means: np.ndarray, anchors: np.ndarray) -> float:
    """Training-only median squared distance to the nearest fitted anchor."""
    values = np.asarray(means, dtype=np.float64)
    centers = np.asarray(anchors, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("means and anchors must be compatible matrices")
    nearest_squared = ((values[:, None] - centers[None]) ** 2).sum(axis=-1).min(axis=1)
    tau = float(np.median(nearest_squared))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("training-median anchor bandwidth is not positive and finite")
    return tau


def effective_anchor_count(observables: np.ndarray) -> np.ndarray:
    """Compute n_eff = 1 / sum_k psi_{tk}^2 for each time step."""
    values = np.asarray(observables, dtype=np.float64)
    sum_sq = np.sum(values ** 2, axis=-1)
    return 1.0 / np.maximum(sum_sq, 1e-12)


def effective_ranks(z_minus: np.ndarray, eps: float = 1e-4) -> tuple[int, float]:
    """Compute thresholded effective rank r_eff and entropy rank r_entropy of snapshot matrix.
    
    z_minus: shape (d, M)
    """
    s = np.linalg.svd(z_minus, compute_uv=False)
    if len(s) == 0 or s[0] < 1e-12:
        return 0, 0.0
    r_eff = int(np.sum(s > eps * s[0]))
    p = s / np.sum(s)
    p_nz = p[p > 1e-15]
    r_entropy = float(np.exp(-np.sum(p_nz * np.log(p_nz))))
    return r_eff, r_entropy


def koopman_operator(
    observables: np.ndarray | tuple[np.ndarray, np.ndarray],
    *,
    ridge: float | None = None,
    alpha: float = 1e-3,
    ridge_min: float = 1e-6,
) -> np.ndarray:
    """Estimate record-specific Koopman operator via normalized sample moments.
    
    Accepts either:
      - observables: 2D array of consecutive states (T, d) where z_minus = obs[:-1].T, z_plus = obs[1:].T
      - (z_minus, z_plus): tuple of snapshot matrices where z_minus, z_plus in R^{d x M}
    
    Formulation:
      G = (1/M) Z_- Z_-^T
      A = (1/M) Z_+ Z_-^T
      lambda = max(alpha * tr(G) / d, ridge_min)  (if ridge is None)
      K = A (G + lambda I)^{-1}
    """
    if isinstance(observables, tuple):
        z_minus = np.asarray(observables[0], dtype=np.float64)
        z_plus = np.asarray(observables[1], dtype=np.float64)
        if z_minus.ndim != 2 or z_plus.ndim != 2 or z_minus.shape != z_plus.shape:
            raise ValueError("z_minus and z_plus must have matching 2D shapes (d, M)")
        d, m = z_minus.shape
    else:
        values = np.asarray(observables, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("observables must be a 2D matrix (T, d)")
        d = values.shape[1]
        m = len(values) - 1
        if m < d:
            raise ValueError("ineligible: fewer transitions than observable dimensions")
        z_minus = values[:-1].T
        z_plus = values[1:].T

    if m < d:
        raise ValueError("ineligible: fewer transitions than observable dimensions")

    # Normalized sample moments
    gram = (z_minus @ z_minus.T) / m
    cross = (z_plus @ z_minus.T) / m

    if ridge is None:
        tr_gram = np.trace(gram)
        lambda_reg = max(alpha * float(tr_gram) / d, ridge_min)
    else:
        lambda_reg = max(float(ridge), ridge_min)

    reg_gram = gram + lambda_reg * np.eye(d, dtype=np.float64)
    # K = cross @ inv(reg_gram) <=> K @ reg_gram = cross <=> reg_gram.T @ K.T = cross.T
    return np.linalg.solve(reg_gram.T, cross.T).T


def soft_markov_operator(z_minus: np.ndarray, z_plus: np.ndarray) -> np.ndarray:
    """Compute empirical column-normalized soft Markov transition matrix.
    
    C = Z_+ Z_-^T
    T_{ij} = C_{ij} / sum_r C_{rj}
    Satisfies T_{ij} >= 0 and sum_i T_{ij} = 1.
    """
    z_minus = np.asarray(z_minus, dtype=np.float64)
    z_plus = np.asarray(z_plus, dtype=np.float64)
    c = z_plus @ z_minus.T
    col_sums = c.sum(axis=0, keepdims=True)
    col_sums[col_sums < 1e-12] = 1.0
    return c / col_sums


def analytical_independence_null(
    z_minus: np.ndarray,
    *,
    ridge: float | None = None,
    alpha: float = 1e-3,
    ridge_min: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute analytical independence null operator K_iid and excess operator K_excess.
    
    Under serial independence: E[psi_{t+1} psi_t^T] ~ psi_bar psi_bar^T
    K_iid = psi_bar psi_bar^T (G + lambda I)^{-1}
    """
    z_minus = np.asarray(z_minus, dtype=np.float64)
    d, m = z_minus.shape
    psi_bar = z_minus.mean(axis=1, keepdims=True)
    gram = (z_minus @ z_minus.T) / m
    if ridge is None:
        tr_gram = np.trace(gram)
        lambda_reg = max(alpha * float(tr_gram) / d, ridge_min)
    else:
        lambda_reg = max(float(ridge), ridge_min)

    reg_gram = gram + lambda_reg * np.eye(d, dtype=np.float64)
    cross_iid = psi_bar @ psi_bar.T
    k_iid = np.linalg.solve(reg_gram.T, cross_iid.T).T
    return k_iid, lambda_reg


def mass_conservation_defect(operator: np.ndarray) -> float:
    """Compute epsilon_mass = ||1^T K - 1^T||_2 / sqrt(d)."""
    k = np.asarray(operator, dtype=np.float64)
    d = k.shape[0]
    ones = np.ones(d, dtype=np.float64)
    mapped = ones @ k
    return float(np.linalg.norm(mapped - ones) / np.sqrt(d))


def negative_entry_fraction(operator: np.ndarray) -> float:
    """Compute fraction of entries K_{ij} < 0."""
    k = np.asarray(operator, dtype=np.float64)
    return float(np.mean(k < 0))


def invariant_matrix_summaries(
    operator: np.ndarray,
    z_minus: np.ndarray | None = None,
    z_plus: np.ndarray | None = None,
) -> np.ndarray:
    """Compute permutation-invariant spectral and algebraic summaries of operator.
    
    Returns 10 stable invariant scalars:
      [rho(K), ||K||_F, ||K||_2, tr(K), tr(K^2), tr(K^3), tr(K^4),
       ||K^T K - K K^T||_F, epsilon_mass, f_neg]
    """
    k = np.asarray(operator, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(k)
    rho = float(np.max(np.abs(eigenvalues)))
    frob = float(np.linalg.norm(k, "fro"))
    spectral_norm = float(np.linalg.norm(k, 2))
    tr1 = float(np.trace(k))
    k2 = k @ k
    tr2 = float(np.trace(k2))
    k3 = k2 @ k
    tr3 = float(np.trace(k3))
    k4 = k3 @ k
    tr4 = float(np.trace(k4))
    nonnormal = float(np.linalg.norm(k.T @ k - k @ k.T, "fro"))
    eps_mass = mass_conservation_defect(k)
    f_neg = negative_entry_fraction(k)
    return np.array(
        [rho, frob, spectral_norm, tr1, tr2, tr3, tr4, nonnormal, eps_mass, f_neg],
        dtype=np.float32,
    )


def operator_descriptor(
    operator: np.ndarray,
    observables: np.ndarray | None = None,
    z_minus: np.ndarray | None = None,
    z_plus: np.ndarray | None = None,
) -> np.ndarray:
    """Compute standard 68-dimensional descriptor for Paper 05 representation matching.
    
    Contains 2 * d eigenvalues (magnitudes and angles) + 4 scalar summaries.
    """
    operator = np.asarray(operator, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(operator)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]

    if observables is not None and len(observables) > 1:
        values = np.asarray(observables, dtype=np.float64)
        one = float(
            np.linalg.norm(values[1:].T - operator @ values[:-1].T)
            / max(float(np.linalg.norm(values[1:])), 1e-12)
        )
        if len(values) > 4:
            four = float(
                np.linalg.norm(values[4:].T - np.linalg.matrix_power(operator, 4) @ values[:-4].T)
                / max(float(np.linalg.norm(values[4:])), 1e-12)
            )
        else:
            four = 0.0
    elif z_minus is not None and z_plus is not None:
        one = float(
            np.linalg.norm(z_plus - operator @ z_minus)
            / max(float(np.linalg.norm(z_plus)), 1e-12)
        )
        four = 0.0
    else:
        one = 0.0
        four = 0.0

    nonnormal = float(np.linalg.norm(operator.T @ operator - operator @ operator.T))
    rho = float(np.max(np.abs(eigenvalues)))
    return np.concatenate(
        (np.abs(eigenvalues), np.angle(eigenvalues), [rho, one, four, nonnormal])
    ).astype(np.float32)


def extract_dual_koopman_operators(
    beat_cell_means: np.ndarray,
    anchors: np.ndarray,
    tau: float,
    *,
    ridge: float | None = None,
    alpha: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Extract decoupled intra-cycle (K_phase) and cycle-to-cycle (K_cycle) Koopman operators.
    
    beat_cell_means: shape (B, 16, D_in) or (B * 16, D_in)
    """
    means = np.asarray(beat_cell_means, dtype=np.float64)
    if means.ndim == 2:
        n_cells = len(means)
        if n_cells % 16 != 0:
            raise ValueError(f"beat_cell_means count {n_cells} is not a multiple of 16")
        b_count = n_cells // 16
        means = means.reshape(b_count, 16, -1)
    elif means.ndim == 3:
        b_count = len(means)
    else:
        raise ValueError("beat_cell_means must have 2 or 3 dimensions")

    if b_count < 2:
        raise ValueError("dual Koopman extraction requires at least 2 beats")

    # Compute soft observables: shape (B, 16, d)
    flat_means = means.reshape(b_count * 16, -1)
    flat_obs = soft_observables(flat_means, anchors, tau)
    obs = flat_obs.reshape(b_count, 16, -1)

    # 1. Phase-progression operator K_phase (transitions g -> g+1 within each beat)
    # Z_phase^- : shape (d, 15 * B)
    # Z_phase^+ : shape (d, 15 * B)
    z_phase_minus = obs[:, :15, :].reshape(-1, anchors.shape[0]).T
    z_phase_plus = obs[:, 1:16, :].reshape(-1, anchors.shape[0]).T

    # 2. Cycle-to-cycle operator K_cycle (transitions b -> b+1 at matched phase)
    # Z_cycle^- : shape (d, 16 * (B - 1))
    # Z_cycle^+ : shape (d, 16 * (B - 1))
    z_cycle_minus = obs[: b_count - 1, :, :].reshape(-1, anchors.shape[0]).T
    z_cycle_plus = obs[1: b_count, :, :].reshape(-1, anchors.shape[0]).T

    k_phase = koopman_operator((z_phase_minus, z_phase_plus), ridge=ridge, alpha=alpha)
    k_cycle = koopman_operator((z_cycle_minus, z_cycle_plus), ridge=ridge, alpha=alpha)

    occupancy = flat_obs.mean(axis=0)

    return {
        "k_phase": k_phase,
        "k_cycle": k_cycle,
        "occupancy": occupancy,
        "z_phase_minus": z_phase_minus,
        "z_phase_plus": z_phase_plus,
        "z_cycle_minus": z_cycle_minus,
        "z_cycle_plus": z_cycle_plus,
    }


def split_half_operator_reproducibility(k_odd: np.ndarray, k_even: np.ndarray) -> float:
    """Compute operator reproducibility score S_K = 1 - ||K_odd - K_even||_F / (||K_odd||_F + ||K_even||_F)."""
    diff = float(np.linalg.norm(k_odd - k_even, "fro"))
    denom = float(np.linalg.norm(k_odd, "fro") + np.linalg.norm(k_even, "fro"))
    if denom < 1e-12:
        return 1.0
    return 1.0 - (diff / denom)


def split_half_cross_validation_error(
    k_odd: np.ndarray,
    k_even: np.ndarray,
    z_odd_minus: np.ndarray,
    z_odd_plus: np.ndarray,
    z_even_minus: np.ndarray,
    z_even_plus: np.ndarray,
) -> float:
    """Compute out-of-sample split-half cross-validation error e_cross."""
    norm_even_plus = max(float(np.linalg.norm(z_even_plus, "fro")), 1e-12)
    norm_odd_plus = max(float(np.linalg.norm(z_odd_plus, "fro")), 1e-12)
    err_odd_to_even = float(np.linalg.norm(z_even_plus - k_odd @ z_even_minus, "fro")) / norm_even_plus
    err_even_to_odd = float(np.linalg.norm(z_odd_plus - k_even @ z_odd_minus, "fro")) / norm_odd_plus
    return 0.5 * (err_odd_to_even + err_even_to_odd)


def chronological_permutation(length: int, record_id: int, seed: int) -> np.ndarray:
    """Single chronological permutation (historical baseline)."""
    if length < 2:
        raise ValueError("chronology permutation requires at least two states")
    sequence = np.random.SeedSequence([seed, int(record_id)])
    return np.random.default_rng(sequence).permutation(length)


def targeted_chronological_permutations(
    b_count: int,
    record_id: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate 3 targeted destructive permutations for a record with B beats x 16 phases.
    
    1. phase_order_shuffled: permutes the 16 phase indices independently within each beat.
       (Preserves beat identities and beat occupancy, destroys intra-cycle phase progression).
    2. beat_order_shuffled: permutes the B beats.
       (Preserves within-beat phase sequence, destroys beat-to-beat cycle chronology).
    3. global_shuffled: permutes all B * 16 cells globally.
       (Preserves complete 1-point occupancy, destroys all serial dependence).
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(record_id)]))
    n_total = b_count * 16

    # 1. Phase-order shuffled
    grid_phase = np.arange(n_total).reshape(b_count, 16)
    for b in range(b_count):
        grid_phase[b] = rng.permutation(grid_phase[b])
    phase_order_perm = grid_phase.reshape(-1)

    # 2. Beat-order shuffled
    grid_beat = np.arange(n_total).reshape(b_count, 16)
    beat_perm = rng.permutation(b_count)
    grid_beat = grid_beat[beat_perm]
    beat_order_perm = grid_beat.reshape(-1)

    # 3. Global cell shuffled
    global_perm = rng.permutation(n_total)

    return {
        "phase_order_shuffled": phase_order_perm,
        "beat_order_shuffled": beat_order_perm,
        "global_shuffled": global_perm,
    }
