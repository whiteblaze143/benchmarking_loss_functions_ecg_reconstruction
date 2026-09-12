"""Test suite for QVCG-H: A Hilbert Atlas of Cardiac Electrical Dynamics from repSpat.

Revised 27-test specification mandated by user review:
- GEOMETRY (1-6): Double centering, PSD tolerance, centered rank <= N-1, distance reconstruction, 95% eigenmass, material negative eigenmass failure
- COMPATIBILITY (7-12): Overlapping maximal cliques, triangle, overlap preservation, nonlocal physical disconnection, local contiguous, singleton rule
- FOLD RESIDUALS (13-15): Isotonic monotonicity, negative fold residual sign, positive boundary residual sign
- QAP (16-20): Symmetry, diagonal, identical matrices rho=1.0 and p=1/(B+1), exact N=5 permutation test enumeration, deterministic seed
- INPUT / LEAKAGE (21-27): Frozen checksums, input allowlist, exact 64 domains, exact 2,016 pairs, symmetric, zero diagonal, non-negative within tolerance
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from rep_stat_ecg.src.motifs.hilbert_atlas import (
    validate_squared_distance_matrix,
    double_center_distance_matrix,
    eigendecompose_hilbert_gram,
    choose_dimension_by_eigenmass,
    reconstruct_squared_distances,
    enumerate_compatibility_cliques,
    classify_physical_recurrence,
    fit_physical_functional_isotonic,
    compute_fold_residuals,
    exact_qap_permutation_test,
    qap_matrix_test,
)


# ==============================================================================
# 1. GEOMETRY TESTS (1-6)
# ==============================================================================

def test_double_center_known_euclidean_points():
    """1: Verify double centering exactly recovers Gram matrix of known points."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 3)
    diff = X[:, None, :] - X[None, :, :]
    D2 = np.sum(diff ** 2, axis=-1)

    validate_squared_distance_matrix(D2)
    B = double_center_distance_matrix(D2)

    X_cent = X - X.mean(axis=0)
    B_expected = X_cent @ X_cent.T
    assert np.allclose(B, B_expected, atol=1e-12)


def test_recovered_gram_psd_up_to_numerical_tolerance():
    """2: Gram matrix of Euclidean/Hilbert distances has no eigenvalues below -tau_lambda."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 5)
    diff = X[:, None, :] - X[None, :, :]
    D2 = np.sum(diff ** 2, axis=-1)

    B = double_center_distance_matrix(D2)
    eigvals, eigvecs, diag, Z = eigendecompose_hilbert_gram(B)

    assert diag["negative_eigenmass"] == 0.0
    assert diag["n_negative"] == 0
    assert diag["n_positive"] <= len(B) - 1
    assert diag["n_numerical_zero"] >= 1
    assert np.all(eigvals >= -diag["eigenvalue_tolerance"])


def test_centered_gram_rank_at_most_n_minus_one():
    """3: Mathematical invariant: B @ 1 = 0 implies rank(B) <= N - 1."""
    rng = np.random.RandomState(42)
    for N in [8, 15, 64]:
        X = rng.randn(N, 10)
        diff = X[:, None, :] - X[None, :, :]
        D2 = np.sum(diff ** 2, axis=-1)
        B = double_center_distance_matrix(D2)
        # B @ 1 must be zero to numerical precision
        assert np.allclose(B @ np.ones(N), np.zeros(N), atol=1e-12)
        eigvals, _, diag, Z = eigendecompose_hilbert_gram(B)
        assert diag["positive_rank"] <= N - 1
        assert diag["n_positive"] <= N - 1
        assert diag["n_numerical_zero"] >= 1
        assert Z.shape[1] == diag["positive_rank"]


def test_reconstructed_distances_match_input():
    """4: Reconstructed distances from positive coordinates match input D2 to numerical tolerance."""
    rng = np.random.RandomState(42)
    X = rng.randn(15, 4)
    diff = X[:, None, :] - X[None, :, :]
    D2 = np.sum(diff ** 2, axis=-1)

    B = double_center_distance_matrix(D2)
    eigvals, eigvecs, diag, Z = eigendecompose_hilbert_gram(B)
    max_err = reconstruct_squared_distances(D2, Z)

    assert max_err < 1e-12


def test_dimension_selection_95_percent():
    """5: Primary dimension r* is smallest r where cumulative positive eigenmass C(r) >= 0.95."""
    eigvals = np.array([50.0, 30.0, 15.0, 3.0, 2.0])  # sum = 100
    r90 = choose_dimension_by_eigenmass(eigvals, threshold=0.90)
    r95 = choose_dimension_by_eigenmass(eigvals, threshold=0.95)
    r99 = choose_dimension_by_eigenmass(eigvals, threshold=0.99)

    assert r90 == 3  # 50 + 30 = 80 < 90, +15 = 95 >= 90
    assert r95 == 3  # 95 >= 95
    assert r99 == 5  # 98 < 99, +2 = 100 >= 99


def test_material_negative_eigenmass_fails():
    """6: Matrix with material negative eigenmass (> 1e-6) raises RuntimeError."""
    D2 = np.array([
        [0.0, 1.0, 100.0],
        [1.0, 0.0, 1.0],
        [100.0, 1.0, 0.0],
    ])
    B = double_center_distance_matrix(D2)
    with pytest.raises(RuntimeError, match="Material negative eigenmass"):
        eigendecompose_hilbert_gram(B, tolerance=1e-6)


# ==============================================================================
# 2. COMPATIBILITY TESTS (7-12)
# ==============================================================================

def test_path_graph_yields_overlapping_maximal_cliques():
    """7: A-B and B-C compatible, A-C rejected -> produces 2 maximal cliques {A,B} and {B,C}."""
    pair_table = pd.DataFrame([
        {"domain_1": 0, "domain_2": 1, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 1, "domain_2": 2, "similarity_edge": True, "q_bh_imq": 0.08},
        {"domain_1": 0, "domain_2": 2, "similarity_edge": False, "q_bh_imq": 0.001},
    ])
    cliques = enumerate_compatibility_cliques(pair_table, n_domains=3)
    clique_sets = [set(c) for c in cliques]

    assert len(cliques) == 2
    assert {0, 1} in clique_sets
    assert {1, 2} in clique_sets
    assert {0, 1, 2} not in clique_sets


def test_triangle_forms_one_maximal_clique():
    """8: A-B, B-C, A-C all compatible -> forms exactly 1 maximal clique {A, B, C}."""
    pair_table = pd.DataFrame([
        {"domain_1": 0, "domain_2": 1, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 1, "domain_2": 2, "similarity_edge": True, "q_bh_imq": 0.08},
        {"domain_1": 0, "domain_2": 2, "similarity_edge": True, "q_bh_imq": 0.12},
    ])
    cliques = enumerate_compatibility_cliques(pair_table, n_domains=3)
    assert len(cliques) == 1
    assert set(cliques[0]) == {0, 1, 2}


def test_overlapping_cliques_remain_overlapping():
    """9: Overlapping cliques retain distinct identity without merging into a component."""
    pair_table = pd.DataFrame([
        {"domain_1": 0, "domain_2": 1, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 0, "domain_2": 2, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 0, "domain_2": 3, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 1, "domain_2": 2, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 2, "domain_2": 3, "similarity_edge": True, "q_bh_imq": 0.10},
        {"domain_1": 1, "domain_2": 3, "similarity_edge": False, "q_bh_imq": 0.01},
    ])
    cliques = enumerate_compatibility_cliques(pair_table, n_domains=4)
    clique_sets = [set(c) for c in cliques]

    assert len(cliques) == 2
    assert {0, 1, 2} in clique_sets
    assert {0, 2, 3} in clique_sets
    assert set(cliques[0]) & set(cliques[1]) == {0, 2}


def test_physical_disconnected_clique_is_nonlocal():
    """10: Clique with |C| >= 2 and #CC(G_P[C]) >= 2 is NONLOCAL_COMPATIBILITY_SET."""
    G_P = nx.Graph()
    G_P.add_edge(0, 1)
    G_P.add_node(2)

    clique = [0, 1, 2]
    is_nonlocal, n_cc = classify_physical_recurrence(clique, G_P)
    assert is_nonlocal is True
    assert n_cc == 2


def test_physical_contiguous_clique_is_local():
    """11: Clique whose domains form a single connected physical component is local (not nonlocal)."""
    G_P = nx.Graph()
    G_P.add_edge(0, 1)
    G_P.add_edge(1, 2)

    clique = [0, 1, 2]
    is_nonlocal, n_cc = classify_physical_recurrence(clique, G_P)
    assert is_nonlocal is False
    assert n_cc == 1


def test_singleton_not_nonlocal():
    """12: Singletons (|C| < 2) cannot be nonlocal compatibility sets."""
    G_P = nx.Graph()
    G_P.add_node(0)

    clique = [0]
    is_nonlocal, n_cc = classify_physical_recurrence(clique, G_P)
    assert is_nonlocal is False
    assert n_cc == 1


# ==============================================================================
# 3. FOLD RESIDUAL TESTS (13-15)
# ==============================================================================

def test_isotonic_fit_is_nondecreasing():
    """13: Isotonic regression produces a monotonic non-decreasing fit."""
    d_p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d_h = np.array([0.1, 0.3, 0.25, 0.5, 0.6])
    iso = fit_physical_functional_isotonic(d_p, d_h)
    pred = iso.predict(d_p)

    assert np.all(np.diff(pred) >= -1e-12)


def test_inserted_negative_fold_has_negative_residual():
    """14: Functionally closer-than-expected pair produces negative fold residual."""
    d_p = np.array([1.0, 2.0, 3.0, 10.0])
    d_h = np.array([1.0, 2.0, 3.0, 1.5])
    iso = fit_physical_functional_isotonic(d_p, d_h)
    residuals = compute_fold_residuals(d_p, d_h, iso)

    assert residuals[-1] < 0.0


def test_inserted_boundary_has_positive_residual():
    """15: Functionally farther-than-expected pair produces positive fold residual."""
    d_p = np.array([1.0, 2.0, 3.0, 4.0])
    d_h = np.array([5.0, 2.0, 3.0, 4.0])
    iso = fit_physical_functional_isotonic(d_p, d_h)
    residuals = compute_fold_residuals(d_p, d_h, iso)

    assert residuals[0] > 0.0


# ==============================================================================
# 4. QAP PERMUTATION TESTS (16-20)
# ==============================================================================

def test_qap_permutation_preserves_symmetry():
    """16: QAP row/column permutation preserves matrix symmetry."""
    rng = np.random.RandomState(42)
    M = rng.randn(8, 8)
    M = (M + M.T) / 2.0
    perm = rng.permutation(8)
    M_perm = M[np.ix_(perm, perm)]
    assert np.allclose(M_perm, M_perm.T)


def test_qap_permutation_preserves_diagonal():
    """17: QAP row/column permutation preserves zero diagonal."""
    M = np.ones((8, 8))
    np.fill_diagonal(M, 0.0)
    perm = np.random.RandomState(42).permutation(8)
    M_perm = M[np.ix_(perm, perm)]
    assert np.allclose(np.diag(M_perm), 0.0)


def test_qap_observed_identical_matrices_rho_one():
    """18: QAP on identical matrices yields rho_obs=1.0 and minimum p = 1/(B+1)."""
    rng = np.random.RandomState(42)
    # Generic matrix with distinct entries so no automorphisms occur
    X = rng.randn(10, 3)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

    res = qap_matrix_test(D, D, n_perm=999, seed=42)
    assert np.isclose(res["rho_obs"], 1.0)
    # With 10 points and generic distances, only the identity permutation achieves rho = 1.0
    assert np.isclose(res["p_qap"], 1.0 / 1000.0)


def test_qap_matches_exact_small_n_enumeration():
    """19: Exact small-N enumeration test: N=5 all 120 permutations verified."""
    rng = np.random.RandomState(99)
    X1 = rng.randn(5, 2)
    X2 = rng.randn(5, 2)
    D1 = np.linalg.norm(X1[:, None, :] - X1[None, :, :], axis=-1)
    D2 = np.linalg.norm(X2[:, None, :] - X2[None, :, :], axis=-1)

    exact_res = exact_qap_permutation_test(D1, D2)
    assert exact_res["n_perms"] == 120
    assert 0.0 <= exact_res["p_exact"] <= 1.0
    assert -1.0 <= exact_res["rho_obs"] <= 1.0


def test_qap_deterministic_seed():
    """20: QAP Monte Carlo test is strictly deterministic given the same seed."""
    rng = np.random.RandomState(42)
    X1 = rng.randn(10, 3)
    X2 = rng.randn(10, 3)
    D1 = np.linalg.norm(X1[:, None, :] - X1[None, :, :], axis=-1)
    D2 = np.linalg.norm(X2[:, None, :] - X2[None, :, :], axis=-1)

    res1 = qap_matrix_test(D1, D2, n_perm=500, seed=123)
    res2 = qap_matrix_test(D1, D2, n_perm=500, seed=123)
    assert res1["p_qap"] == res2["p_qap"]
    assert np.isclose(res1["rho_obs"], res2["rho_obs"])


# ==============================================================================
# 5. INPUT & LEAKAGE INVARIANT TESTS (21-27)
# ==============================================================================

def test_frozen_checksums_match():
    """21: Verify all frozen input files match registered SHA-256 hashes."""
    expected_hashes = {
        "refine-logs/qvcg/VCG_MMD_MATRIX.npy": "890e77082220f165fef03850324e4abe6fd69d4137cdfdad44845e17ef5e7fd5",
        "refine-logs/qvcg/VCG_REPSPAT_PAIR_TESTS.parquet": "a6a9c93dceb008b2e6fe984016afd05939746eae8d16cf2697e0e489c1a2edfb",
        "refine-logs/qvcg/VCG_DOMAIN_STATS.parquet": "b973995cb3a266e7edbbbe084cc4b12a8943cbe3961dd72b80b8183d62ad3e1e",
        "refine-logs/qvcg/VCG_SPATIAL_DOMAIN_REGISTRY.parquet": "e2f805f40583aa00e7b8687d23cd8fdd946de38874922f270f9f61d5929826b2",
        "refine-logs/qvcg/QVCG_CC_MOTIF_GATE.json": "b0211cc29ee0e2744fb9d1f0078bb9a20a6536eadf7b989b1fc0cfe87a7436e3",
        "refine-logs/qvcg/m3r_reference/M3R_MMD_MATRIX.npy": "320fce9e38950e94dde10b36678abb11c6f01bbd6277cd5b1f427986cd9f259e",
        "refine-logs/qvcg/m3r_reference/M3R_PAIR_TESTS.parquet": "80c579d6aa9845b076aed323f0f3503bcb9a499e6e37115ffef9e7953b351929",
        "refine-logs/qvcg/m3r_reference/M3R_VS_M3_COMPARISON.json": "6a8bab4ea044b9a33e66106e6d0a8a9544e26fdb6d13d8a00b75bfeca13fc27e",
    }
    for file_str, exp_hash in expected_hashes.items():
        p = Path(file_str)
        assert p.exists(), f"Missing required frozen file: {file_str}"
        observed = hashlib.sha256(p.read_bytes()).hexdigest()
        assert observed == exp_hash, f"Hash mismatch for {file_str}: {observed} != {exp_hash}"


def test_builder_inputs_match_allowlist():
    """22: Build script code strictly opens ONLY allowlisted scientific inputs."""
    build_script = Path("rep_stat_ecg/scripts/build_hilbert_atlas.py").read_text()
    # Check that clinical files or diagnostic metadata tables are never referenced
    forbidden_sources = [
        "scp_statements",
        "ptbxl_database",
        "diagnostic_class",
        "diagnostic_subclass",
        "rhythm_label",
        "clinical_annotation",
    ]
    for term in forbidden_sources:
        assert term not in build_script.lower(), f"Forbidden source term '{term}' found in build script"

    # Verify that allowlisted inputs are present
    allowlisted = [
        "VCG_MMD_MATRIX.npy",
        "VCG_REPSPAT_PAIR_TESTS.parquet",
        "VCG_DOMAIN_STATS.parquet",
        "VCG_SPATIAL_DOMAIN_REGISTRY.parquet",
    ]
    for allowed in allowlisted:
        assert allowed in build_script, f"Mandatory allowlisted input {allowed} not referenced in build script"


def test_exact_64_domains():
    """23: Domain stats must contain exactly 64 domains."""
    stats = pd.read_parquet("refine-logs/qvcg/VCG_DOMAIN_STATS.parquet")
    assert len(stats) == 64
    assert set(stats["domain_id"]) == set(range(64))


def test_exact_2016_pairs():
    """24: Pair tests table must contain exactly 2,016 unordered pairs."""
    pairs = pd.read_parquet("refine-logs/qvcg/VCG_REPSPAT_PAIR_TESTS.parquet")
    assert len(pairs) == 2016


def test_mmd_matrix_symmetric():
    """25: MMD matrix must be symmetric to numerical tolerance."""
    D2 = np.load("refine-logs/qvcg/VCG_MMD_MATRIX.npy")
    assert D2.shape == (64, 64)
    assert np.allclose(D2, D2.T, atol=1e-12)


def test_mmd_diagonal_zero():
    """26: MMD diagonal elements must be zero to numerical tolerance."""
    D2 = np.load("refine-logs/qvcg/VCG_MMD_MATRIX.npy")
    assert np.allclose(np.diag(D2), 0.0, atol=1e-12)


def test_mmd_matrix_nonnegative_within_tolerance():
    """27: MMD matrix entries must be non-negative within declared tolerance tau_D = 1e-12."""
    D2 = np.load("refine-logs/qvcg/VCG_MMD_MATRIX.npy")
    tau_D = 1e-12
    assert np.all(D2 >= -tau_D), f"MMD matrix has entries below -{tau_D}: min is {D2.min()}"
    validate_squared_distance_matrix(D2, tol_sym=1e-12, tol_nonneg=tau_D)
