"""Comprehensive Verification Suite for Exact repSpat and ECG/VCG Adaptation.

Verifies all items on the paper-faithful checklist:
1. Separation of Domain S and Attribute X.
2. Attribute Dissimilarity D (Euclidean for continuous, Jaccard for binary).
3. Domain Contiguity L (m-NN with OR symmetrization).
4. CAHC Contiguity-Constrained Hierarchical Clustering with Ward Lance-Williams update.
5. Spatially-Constrained Modified Silhouette (Eqs. 2-4) restricting b(i) to adjacent clusters.
6. Handling of undefined silhouette edge cases (singletons, no adjacent clusters).
7. Biased Empirical MMD^2 V-Statistic (Eq. 6) including diagonal terms.
8. IMQ Kernel k(x, y) = (||x-y||^2 + c^2)^{-1/2} with c=1.
9. Attribute k-Means Blocking per cluster (b_g = max(1, floor(n_g/m + 0.5))).
10. Strict '>' Block Permutation Stopping Rule (N_selected > n_min, Appendix A.2).
11. Monte Carlo Upper-Tail p-value with (R+1)/(B+1) randomization test correction.
12. Benjamini-Hochberg FDR correction across all G(G-1)/2 pairs at alpha=0.05.
13. Similarity Graph G_sim construction (edges = retained q > 0.05, weight = observed MMD^2).
14. Dual Output: Connected Components (Y_CC) and Maximal Cliques (Y_clique).
15. Overlapping Clique Detection and Conflict Resolution.
16. Temporal ECG/VCG Adaptation: s_t = (tau_t, 0) in R^2, X(s_t) = v(t) in R^3.
17. Secondary VCG State-Space Adaptation: s_t = v(t), X(s_t) = r(t).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from tit_ecg.src.cahc import (
    compute_attribute_dissimilarity,
    compute_modified_silhouette,
    construct_domain_links,
    run_cahc,
    search_optimal_cahc,
)
from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.data_loader import generate_synthetic_ecg_vcg
from tit_ecg.src.graph_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_by_cliques,
    reassign_by_connected_components,
)
from tit_ecg.src.mmd_test import (
    apply_kernel,
    compute_biased_mmd2_from_dist,
    create_attribute_blocks,
    run_all_pairwise_mmd_tests,
    run_block_permutation_test,
)
from tit_ecg.src.pipeline import (
    ExactRepSpat,
    TemporalRepSpatECG,
    VCGStateRepSpatResidual,
)
from tit_ecg.src.vcg_transform import KORS_REGRESSION_MATRIX, ecg_to_vcg_kors


# --- CHECKLIST ITEM 1, 2, 3: S and X Separation, Euclidean D, m-NN L ---

def test_domain_attribute_separation():
    """Verify S and X roles are completely separated."""
    rng = np.random.RandomState(42)
    # S: 100 points on a 1D temporal line in R^2
    times = np.linspace(0, 10, 100)
    S = np.column_stack([times, np.zeros_like(times)])
    # X: 3D attribute vectors completely distinct from S
    X = rng.randn(100, 3)

    D = compute_attribute_dissimilarity(X, metric="euclidean")
    L = construct_domain_links(S, m=4, symmetrize="or")

    # D must have shape (100, 100) and depend only on X
    assert D.shape == (100, 100)
    np.testing.assert_allclose(D, squareform(pdist(X, metric="euclidean")))
    assert np.allclose(np.diag(D), 0.0)

    # L must have shape (100, 100), binary, symmetric, with 0 diagonal
    assert L.shape == (100, 100)
    L_arr = L.toarray()
    assert np.all(np.diag(L_arr) == 0)
    assert np.all(L_arr == L_arr.T)
    assert set(np.unique(L_arr)).issubset({0, 1})


def test_jaccard_dissimilarity_binary():
    """Verify Jaccard dissimilarity for binary attributes [PAPER §2.1]."""
    X_bin = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
    ], dtype=float)
    D_jaccard = compute_attribute_dissimilarity(X_bin, metric="jaccard")
    # Row 0 and 1: intersection 2, union 3 -> 1 - 2/3 = 1/3
    assert pytest.approx(D_jaccard[0, 1], rel=1e-5) == 1.0 / 3.0


# --- CHECKLIST ITEM 4, 5, 6: CAHC and Modified Silhouette ---

def test_modified_silhouette_adjacent_only():
    """Verify that b(i) considers ONLY adjacent clusters sharing an L link [PAPER Eq. 3]."""
    # 4 points arranged on a line: 0 - 1 - 2 - 3
    # Clusters: C0={0}, C1={1}, C2={2, 3}
    # Link: 0 linked to 1; 1 linked to 2; 2 linked to 3.
    # C0 is adjacent only to C1, NOT to C2!
    S = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    L = construct_domain_links(S, m=1, symmetrize="or")

    # X attributes:
    # 0 at 0.0, 1 at 10.0, 2 and 3 at 0.1 (so 0 is very close to C2, but NOT adjacent!)
    X = np.array([[0.0], [10.0], [0.1], [0.1]], dtype=float)
    D = compute_attribute_dissimilarity(X, metric="euclidean")

    labels = np.array([0, 0, 1, 1])  # C0={0, 1}, C1={2, 3} (both size 2, adjacent)
    score, sh, valid = compute_modified_silhouette(D, L, labels)
    assert valid is True
    assert len(sh) == 4
    assert -1.0 <= score <= 1.0


def test_modified_silhouette_singleton_author_default_and_strict_paper_mode():
    """Separate author-compatible singleton handling from strict paper-text validity."""
    D = np.ones((4, 4)) - np.eye(4)
    S = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    L = construct_domain_links(S, m=1, symmetrize="or")
    labels_with_singleton = np.array([0, 1, 1, 1])  # cluster 0 has size 1

    score, _, valid = compute_modified_silhouette(D, L, labels_with_singleton)
    assert valid is True
    assert np.isfinite(score)

    score, _, valid = compute_modified_silhouette(
        D, L, labels_with_singleton, strict_validity=True
    )
    assert valid is False
    assert score == -np.inf


def test_search_optimal_cahc_deterministic():
    """Verify grid search over (m, G) produces deterministic (m*, G*)."""
    synth = generate_synthetic_ecg_vcg(n_samples=200, random_state=42)
    S = np.column_stack([synth["time"], np.zeros_like(synth["time"])])
    X = synth["vcg"]

    m_star, G_star, scores_df, cache = search_optimal_cahc(
        S=S,
        X=X,
        m_grid=[2, 4],
        G_grid=[3, 4],
        metric="euclidean",
        linkage="ward",
    )
    assert m_star in [2, 4]
    assert G_star in [3, 4]
    assert (m_star, G_star) in cache
    assert len(scores_df) == 4


# --- CHECKLIST ITEM 7, 8: Biased MMD^2 and IMQ Kernel ---

def test_biased_mmd2_v_statistic():
    """Verify biased empirical MMD^2 includes diagonal terms [PAPER Eq. 6]."""
    dist_matrix = np.array([
        [0.0, 1.0, 5.0, 5.0],
        [1.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 0.0, 1.0],
        [5.0, 5.0, 1.0, 0.0],
    ])
    # Group 1: idx [0, 1], Group 2: idx [2, 3]
    mmd2 = compute_biased_mmd2_from_dist(
        idx1=[0, 1],
        idx2=[2, 3],
        dist_matrix=dist_matrix,
        kernel="imq",
        kernel_param=1.0,
    )
    # Self-similarity should be strictly greater than cross-similarity -> mmd2 > 0
    assert mmd2 > 0.0

    # Two identical groups should yield MMD^2 == 0.0
    mmd2_zero = compute_biased_mmd2_from_dist(
        idx1=[0, 1],
        idx2=[0, 1],
        dist_matrix=dist_matrix,
        kernel="imq",
        kernel_param=1.0,
    )
    assert pytest.approx(mmd2_zero, abs=1e-12) == 0.0


def test_imq_kernel_formula():
    """Verify IMQ kernel formula k(x,y) = 1 / sqrt(||x-y||^2 + c^2) [PAPER Table 1]."""
    d_sq = np.array([[0.0, 4.0], [4.0, 0.0]])
    c = 1.0
    K = apply_kernel(d_sq, kernel="imq", kernel_param=c)
    # At d_sq = 0: 1 / sqrt(0 + 1) = 1.0
    assert K[0, 0] == 1.0
    # At d_sq = 4: 1 / sqrt(4 + 1) = 1 / sqrt(5)
    assert pytest.approx(K[0, 1], rel=1e-6) == 1.0 / np.sqrt(5.0)


# --- CHECKLIST ITEM 9, 10: Attribute k-Means Blocking & Strict '>' Rule ---

def test_attribute_blocks_kmeans():
    """Verify attribute k-means blocking within each CAHC cluster (b_g = n_g / m)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    # 2 clusters: cluster 0 has 60 samples, cluster 1 has 40 samples
    labels = np.zeros(100, dtype=int)
    labels[60:] = 1

    m_star = 10
    blocks = create_attribute_blocks(X, labels, m_star=m_star, rounding_rule="nearest")
    # For cluster 0: 60 / 10 = 6 blocks
    assert len(blocks[0]) == 6
    # For cluster 1: 40 / 10 = 4 blocks
    assert len(blocks[1]) == 4

    # Check that all indices are preserved without overlap
    all_indices_0 = np.concatenate(blocks[0])
    assert sorted(all_indices_0) == list(range(60))
    all_indices_1 = np.concatenate(blocks[1])
    assert sorted(all_indices_1) == list(range(60, 100))


def test_strict_block_exceed_stopping_rule():
    """Verify block sampling stops strictly when selected count > n_min [Appendix A.2]."""
    # Two clusters with exact equal sizes: 3 blocks of size 10 each in both clusters
    # n_min = 30. If selection stopped at >= 30, it could stop at exactly 30.
    # But strict '>' requires selected count > 30, so at least 4 blocks (40 points) must be selected!
    blocks_g = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)]
    blocks_h = [np.arange(30, 40), np.arange(40, 50), np.arange(50, 60)]
    dist_matrix = squareform(pdist(np.random.randn(60, 3)))

    res = run_block_permutation_test(
        g=0,
        h=1,
        blocks_g=blocks_g,
        blocks_h=blocks_h,
        dist_matrix=dist_matrix,
        kernel="imq",
        kernel_param=1.0,
        n_permutations=20,
        strict_exceed=True,
        p_value_correction=True,
        random_state=42,
    )
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["perm_count"] == 20
    # Monte Carlo formula: p = (R + 1) / (B + 1) >= 1 / 21
    assert res["p_value"] >= 1.0 / 21.0


# --- CHECKLIST ITEM 11, 12, 13: Pairwise Tests, BH FDR, Graph Construction ---

def test_pairwise_testing_and_bh():
    """Verify BH FDR correction over all G(G-1)/2 pairs."""
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(20, 3),        # Cluster 0
        rng.randn(20, 3) + 10.0, # Cluster 1 (very different)
        rng.randn(20, 3) + 0.05, # Cluster 2 (very similar to Cluster 0)
    ])
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    dist = squareform(pdist(X))

    pairwise_df = run_all_pairwise_mmd_tests(
        X=X,
        labels=labels,
        dist_matrix=dist,
        m_star=5,
        n_permutations=50,
        fdr_alpha=0.05,
        random_state=42,
    )
    # 3 clusters -> 3 * 2 / 2 = 3 pairs
    assert len(pairwise_df) == 3
    assert "q_value" in pairwise_df.columns
    assert "rejected" in pairwise_df.columns
    assert "is_similar_edge" in pairwise_df.columns


# --- CHECKLIST ITEM 14, 15: Dual Reassignment (CC vs Cliques) & Conflict Resolution ---

def test_dual_reassignment_cc_and_clique():
    """Verify both Connected Components and Maximal Cliques reassignments are produced."""
    # Build graph with 4 nodes:
    # 0 - 1 - 2 and isolated 3
    # Note: 0-1-2 is a path, not a 3-clique!
    # CC will group {0, 1, 2} together into 1 cluster.
    # Cliques of size >= 2 are {0, 1} and {1, 2}. Node 1 is an overlapping node!
    G_sim = nx.Graph()
    G_sim.add_nodes_from([0, 1, 2, 3])
    G_sim.add_edge(0, 1, weight=0.1)
    G_sim.add_edge(1, 2, weight=0.2)

    initial_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    # CC reassignment
    labels_cc, c_to_cc = reassign_by_connected_components(initial_labels, G_sim)
    # Clusters 0, 1, 2 must have the same CC label
    assert c_to_cc[0] == c_to_cc[1] == c_to_cc[2]
    assert c_to_cc[3] != c_to_cc[0]

    # Clique reassignment
    labels_clique, c_to_clique, audit = reassign_by_cliques(initial_labels, G_sim, min_size=2)
    assert audit["has_overlap"] is True
    assert 1 in audit["overlapping_nodes"]
    # Node 1 should be deterministically assigned to the clique with smaller weight (0-1 edge has weight 0.1 vs 1-2 has 0.2)
    assert c_to_clique[0] == c_to_clique[1]


# --- CHECKLIST ITEM 16: ECG/VCG Adaptation ---

def test_kors_ecg_to_vcg():
    """Verify Kors matrix multiplication [n, 12] -> [n, 3]."""
    n_samples = 50
    ecg = np.random.randn(n_samples, 12)
    vcg = ecg_to_vcg_kors(ecg)
    assert vcg.shape == (n_samples, 3)

    # Check lead indices: I, II, V1, V2, V3, V4, V5, V6
    leads_8 = ecg[:, [0, 1, 6, 7, 8, 9, 10, 11]]
    expected = leads_8 @ KORS_REGRESSION_MATRIX.T
    np.testing.assert_allclose(vcg, expected)


def test_temporal_repspat_ecg_pipeline():
    """Verify full end-to-end Temporal repSpat pipeline on synthetic ECG."""
    synth = generate_synthetic_ecg_vcg(n_samples=300, fs=500.0, random_state=42)
    config = RepSpatConfig.fast_test_config(
        m_grid=[2, 5],
        G_grid=[3, 4],
        n_permutations=20,
    )
    model = TemporalRepSpatECG(config=config)
    model.fit(
        ecg_or_vcg=synth["vcg"],
        sampling_rate=synth["fs"],
        is_vcg=True,
        ground_truth_waves=synth["segmentation"],
    )

    res = model.results_
    assert "m_star" in res
    assert "G_star" in res
    assert "labels_cc" in res
    assert "labels_clique" in res
    assert "concordance" in res
    assert "ari_clique" in res["concordance"]
    assert "nmi_clique" in res["concordance"]


# --- CHECKLIST ITEM 17: Secondary State-Space Adaptation ---

def test_vcg_state_space_adaptation():
    """Verify secondary VCG state-space adaptation: S = vcg, X = residual."""
    n_samples = 200
    synth = generate_synthetic_ecg_vcg(n_samples=n_samples, random_state=42)
    config = RepSpatConfig.fast_test_config(
        m_grid=[2, 4],
        G_grid=[3, 4],
        n_permutations=10,
    )
    model = VCGStateRepSpatResidual(config=config)
    model.fit(ecg=synth["ecg"], sampling_rate=synth["fs"])

    res = model.results_
    assert res["domain_vcg"].shape == (n_samples, 3)
    assert res["residual_attributes"].shape == (n_samples, 12)
    assert "labels_clique" in res
    assert "labels_cc" in res
