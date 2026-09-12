"""QVCG-H: A Hilbert Atlas of Cardiac Electrical Dynamics from repSpat.

Implements the continuous distributional RKHS geometry underlying MMD,
overlapping maximal-clique compatibility covers, physical-functional
isotonic fold residual analysis, and QAP matrix permutation testing.

Reference: PRD QVCG-H v1.0 (Senanayake & Jeganathan, 2024 / Q-VCG M3H).
"""
from __future__ import annotations

import itertools
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression


def validate_squared_distance_matrix(
    D2: np.ndarray, tol_sym: float = 1e-12, tol_nonneg: float = 1e-12
) -> None:
    """Validates that D2 is a symmetric 2D matrix with zero diagonal and valid non-negative entries."""
    if D2.ndim != 2 or D2.shape[0] != D2.shape[1]:
        raise ValueError(f"D2 must be a 2D square matrix, got shape {D2.shape}")
    if not np.allclose(D2, D2.T, atol=tol_sym):
        raise ValueError("D2 must be symmetric to numerical tolerance")
    if not np.allclose(np.diag(D2), 0.0, atol=tol_sym):
        raise ValueError("D2 diagonal must be zero to numerical tolerance")
    if np.any(D2 < -tol_nonneg):
        raise ValueError(f"D2 elements must be non-negative within tolerance {tol_nonneg}, min is {D2.min()}")


def double_center_distance_matrix(D2: np.ndarray) -> np.ndarray:
    """Computes the centered Gram matrix B = -0.5 * J @ D2 @ J.
    
    Here J = I - (1/G) * 1 * 1^T is the centering operator.
    Guarantees B @ 1 = 0 and rank(B) <= G - 1.
    """
    G = len(D2)
    J = np.eye(G, dtype=np.float64) - np.ones((G, G), dtype=np.float64) / float(G)
    B = -0.5 * (J @ D2.astype(np.float64) @ J)
    # Enforce exact numerical symmetry
    return (B + B.T) / 2.0


def eigendecompose_hilbert_gram(
    B: np.ndarray, tolerance: float = 1e-6, eig_tol_factor: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Eigendecomposes centered Gram matrix B.
    
    Because B is centered (B @ 1 = 0), rank(B) <= G - 1.
    Positive rank is determined using scale-aware threshold:
        tau_lambda = eig_tol_factor * max_i |lambda_i|.
    
    Returns:
        eigenvalues: Sorted descending eigenvalues.
        eigenvectors: Sorted eigenvectors corresponding to eigenvalues.
        diagnostics: Dictionary containing:
            - lambda_min: Minimum eigenvalue.
            - lambda_max: Maximum eigenvalue.
            - lambda_abs_min: Minimum absolute eigenvalue.
            - eigenvalue_tolerance: tau_lambda threshold.
            - n_positive: Number of eigenvalues > tau_lambda.
            - n_numerical_zero: Number of eigenvalues with |lambda| <= tau_lambda.
            - n_negative: Number of eigenvalues < -tau_lambda.
            - negative_eigenmass: Fraction of total absolute mass from material negative eigenvalues (< -tau_lambda).
            - positive_rank: Number of positive eigenvalues exceeding tau_lambda (guaranteed <= G - 1).
        coordinates: Functional Hilbert coordinates Z_H of dimension (G, positive_rank).
    
    Raises:
        RuntimeError: If material negative eigenmass exceeds tolerance, or positive_rank > G - 1.
    """
    G = len(B)
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    max_abs_eig = float(np.max(np.abs(eigvals))) if len(eigvals) > 0 else 0.0
    tau_lambda = eig_tol_factor * max_abs_eig

    pos_mask = eigvals > tau_lambda
    zero_mask = np.abs(eigvals) <= tau_lambda
    neg_mask = eigvals < -tau_lambda

    n_pos = int(np.sum(pos_mask))
    n_zero = int(np.sum(zero_mask))
    n_neg = int(np.sum(neg_mask))

    total_abs = float(np.sum(np.abs(eigvals)))
    neg_mass = float(np.sum(np.abs(eigvals[neg_mask]))) / total_abs if total_abs > 0 else 0.0

    if neg_mass > tolerance:
        raise RuntimeError(
            f"Material negative eigenmass {neg_mass:.6e} exceeds declared tolerance {tolerance:.6e}"
        )

    if n_pos > G - 1:
        raise RuntimeError(
            f"Mathematical violation: Centered Gram matrix positive rank {n_pos} exceeds theoretical bound {G - 1}"
        )

    pos_eigvals = eigvals[:n_pos]
    pos_eigvecs = eigvecs[:, :n_pos]
    coordinates = pos_eigvecs * np.sqrt(pos_eigvals)

    diagnostics = {
        "lambda_min": float(np.min(eigvals)),
        "lambda_max": float(np.max(eigvals)),
        "lambda_abs_min": float(np.min(np.abs(eigvals))),
        "eigenvalue_tolerance": float(tau_lambda),
        "n_positive": n_pos,
        "n_numerical_zero": n_zero,
        "n_negative": n_neg,
        "negative_eigenmass": float(neg_mass),
        "positive_rank": n_pos,
    }

    return eigvals, eigvecs, diagnostics, coordinates


def choose_dimension_by_eigenmass(
    eigenvalues: np.ndarray, threshold: float = 0.95, eig_tol_factor: float = 1e-12
) -> int:
    """Selects smallest dimension r* explaining at least threshold of positive eigenmass."""
    max_abs = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) > 0 else 0.0
    tau_lambda = eig_tol_factor * max_abs
    pos_eigvals = eigenvalues[eigenvalues > tau_lambda]
    total_pos = float(np.sum(pos_eigvals))
    if total_pos <= 0:
        return 1
    cum_mass = np.cumsum(pos_eigvals) / total_pos
    r = int(np.searchsorted(cum_mass, threshold) + 1)
    return min(r, len(pos_eigvals))


def reconstruct_squared_distances(D2_true: np.ndarray, Z: np.ndarray) -> float:
    """Computes maximum absolute error between input D2 and reconstructed Euclidean distances."""
    diff = Z[:, None, :] - Z[None, :, :]
    D2_recon = np.sum(diff ** 2, axis=-1)
    return float(np.max(np.abs(D2_true - D2_recon)))


def enumerate_compatibility_cliques(
    pair_table: pd.DataFrame, n_domains: int = 64
) -> list[list[int]]:
    """Finds all maximal cliques of the repSpat non-rejection graph G_A.
    
    Pairs with similarity_edge == True (q_bh >= 0.05) form edges.
    """
    G_A = nx.Graph()
    G_A.add_nodes_from(range(n_domains))

    if "similarity_edge" in pair_table.columns:
        edge_rows = pair_table[pair_table["similarity_edge"]]
    elif "q_bh_imq" in pair_table.columns:
        edge_rows = pair_table[pair_table["q_bh_imq"] >= 0.05]
    elif "q_bh" in pair_table.columns:
        edge_rows = pair_table[pair_table["q_bh"] >= 0.05]
    else:
        raise KeyError("Pair table must contain 'similarity_edge', 'q_bh_imq', or 'q_bh'")

    for row in edge_rows.itertuples(index=False):
        d1 = int(getattr(row, "domain_1", getattr(row, "domain_i", None)))
        d2 = int(getattr(row, "domain_2", getattr(row, "domain_j", None)))
        G_A.add_edge(d1, d2)

    cliques = list(nx.find_cliques(G_A))
    sorted_cliques = [sorted(c) for c in cliques]
    sorted_cliques.sort(key=lambda c: (-len(c), c))
    return sorted_cliques


def build_physical_adjacency(registry_df: pd.DataFrame) -> nx.Graph:
    """Builds physical 26-neighbor domain adjacency graph from voxel lattice registry.
    
    Edge (g, h) in E_P iff there exists occupied voxel v in G_g and occupied voxel u in G_h
    such that u is one of the 26 lattice neighbors of v on the 3D grid.
    """
    locations = {
        (int(row.ix), int(row.iy), int(row.iz)): int(row.domain_id)
        for row in registry_df.itertuples(index=False)
    }
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    edges: set[tuple[int, int]] = set()
    for (x, y, z), domain in locations.items():
        for dx, dy, dz in offsets:
            neighbor = locations.get((x + dx, y + dy, z + dz))
            if neighbor is not None and neighbor != domain:
                edges.add(tuple(sorted((domain, neighbor))))

    G_P = nx.Graph()
    all_domains = set(registry_df["domain_id"].unique())
    G_P.add_nodes_from(all_domains)
    G_P.add_edges_from(edges)
    return G_P


def classify_physical_recurrence(
    clique: list[int], physical_graph: nx.Graph
) -> tuple[bool, int]:
    """Tests if a compatibility set is spatially recurrent (physically disconnected).
    
    A compatibility set C is NONLOCAL_COMPATIBILITY_SET iff |C| >= 2 and #CC(G_P[C]) >= 2.
    Singletons (|C| < 2) are local.
    """
    if len(clique) < 2:
        return False, len(clique)

    subgraph = physical_graph.subgraph(clique)
    n_cc = nx.number_connected_components(subgraph)
    is_nonlocal = bool(n_cc >= 2)
    return is_nonlocal, n_cc


def fit_physical_functional_isotonic(d_p: np.ndarray, d_h: np.ndarray) -> IsotonicRegression:
    """Fits non-decreasing isotonic regression D_H = f(D_P)."""
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(d_p, d_h)
    return iso


def compute_fold_residuals(
    d_p: np.ndarray, d_h: np.ndarray, iso_model: IsotonicRegression
) -> np.ndarray:
    """Calculates descriptive fold residuals: F_gh = D_H(g,h) - f_hat(D_P(g,h)).
    
    Negative F_gh: functionally closer than expected given physical separation.
    Positive F_gh: functionally farther than expected given physical separation.
    """
    d_h_pred = iso_model.predict(d_p)
    return d_h - d_h_pred


def exact_qap_permutation_test(D_P: np.ndarray, D_H: np.ndarray) -> dict:
    """Exhaustively enumerates all N! permutations for small N (N <= 7)."""
    N = len(D_P)
    iu = np.triu_indices(N, k=1)
    d_p_vec = D_P[iu]
    d_h_vec = D_H[iu]

    r_p = rankdata(d_p_vec).astype(np.float64)
    r_p_cent = r_p - np.mean(r_p)
    norm_p = float(np.sqrt(np.sum(r_p_cent ** 2)))

    r_h_obs = rankdata(d_h_vec).astype(np.float64)
    r_h_obs_cent = r_h_obs - np.mean(r_h_obs)
    norm_h_obs = float(np.sqrt(np.sum(r_h_obs_cent ** 2)))
    rho_obs = float(np.sum(r_p_cent * r_h_obs_cent) / (norm_p * norm_h_obs)) if norm_p * norm_h_obs > 0 else 0.0

    all_perms = list(itertools.permutations(range(N)))
    n_total = len(all_perms)
    more_extreme = 0
    null_rhos = []

    for perm in all_perms:
        perm_arr = np.array(perm)
        D_H_perm = D_H[np.ix_(perm_arr, perm_arr)]
        d_h_perm_vec = D_H_perm[iu]
        r_h = rankdata(d_h_perm_vec).astype(np.float64)
        r_h_cent = r_h - np.mean(r_h)
        norm_h = float(np.sqrt(np.sum(r_h_cent ** 2)))
        rho_p = float(np.sum(r_p_cent * r_h_cent) / (norm_p * norm_h)) if norm_p * norm_h > 0 else 0.0
        null_rhos.append(rho_p)
        if abs(rho_p) >= abs(rho_obs) - 1e-12:
            more_extreme += 1

    p_exact = float(more_extreme) / float(n_total)
    return {
        "rho_obs": rho_obs,
        "p_exact": p_exact,
        "n_perms": n_total,
        "more_extreme_count": more_extreme,
    }


def qap_matrix_test(
    D_P: np.ndarray,
    D_H: np.ndarray,
    n_perm: int = 9999,
    seed: int = 42,
) -> dict:
    """Runs a Monte Carlo Quadratic Assignment Procedure (QAP) matrix permutation test.
    
    Tests association between physical distance D_P and functional distance D_H
    via full domain-label permutation: D_H^(pi) = P_pi D_H P_pi^T.
    Uses the standard (1 + count) / (B + 1) Monte Carlo convention.
    """
    if D_P.shape != D_H.shape or D_P.ndim != 2 or D_P.shape[0] != D_P.shape[1]:
        raise ValueError("D_P and D_H must be square matrices of matching shapes")

    N = len(D_P)
    iu = np.triu_indices(N, k=1)
    d_p_vec = D_P[iu]
    d_h_vec = D_H[iu]

    r_p = rankdata(d_p_vec).astype(np.float64)
    r_p_cent = r_p - np.mean(r_p)
    norm_p = float(np.sqrt(np.sum(r_p_cent ** 2)))

    r_h_obs = rankdata(d_h_vec).astype(np.float64)
    r_h_obs_cent = r_h_obs - np.mean(r_h_obs)
    norm_h_obs = float(np.sqrt(np.sum(r_h_obs_cent ** 2)))
    rho_obs = float(np.sum(r_p_cent * r_h_obs_cent) / (norm_p * norm_h_obs)) if norm_p * norm_h_obs > 0 else 0.0

    rng = np.random.RandomState(seed)
    more_extreme = 0
    perm_rhos: list[float] = []

    for _ in range(n_perm):
        perm = rng.permutation(N)
        D_H_perm = D_H[np.ix_(perm, perm)]
        d_h_perm_vec = D_H_perm[iu]

        r_h = rankdata(d_h_perm_vec).astype(np.float64)
        r_h_cent = r_h - np.mean(r_h)
        norm_h = float(np.sqrt(np.sum(r_h_cent ** 2)))

        if norm_h > 0 and norm_p > 0:
            rho_perm = float(np.sum(r_p_cent * r_h_cent) / (norm_p * norm_h))
        else:
            rho_perm = 0.0

        perm_rhos.append(rho_perm)
        if abs(rho_perm) >= abs(rho_obs) - 1e-12:
            more_extreme += 1

    p_qap = float(1 + more_extreme) / float(n_perm + 1)
    perm_rhos_arr = np.array(perm_rhos)

    return {
        "rho_obs": rho_obs,
        "p_qap": p_qap,
        "n_perm": n_perm,
        "more_extreme_count": more_extreme,
        "null_rho_min": float(np.min(perm_rhos_arr)),
        "null_rho_max": float(np.max(perm_rhos_arr)),
        "null_rho_mean": float(np.mean(perm_rhos_arr)),
        "null_rho_std": float(np.std(perm_rhos_arr)),
        "seed": seed,
    }
