"""Automated unit test suite for KirkVCG repSpat."""
from __future__ import annotations

import numpy as np
import pytest

from kirkvcg_rep_stat.src.vcg.transform import (
    ecg_to_vcg_kors,
    ecg_to_vcg_dower,
    KORS_MATRIX,
    DOWER_MATRIX,
)
from kirkvcg_rep_stat.src.vcg.kinematics import compute_vcg_kinematics
from kirkvcg_rep_stat.src.features.ecg_features import (
    extract_sample_features,
    standardize_features,
    binarize_features,
)
from kirkvcg_rep_stat.src.clustering.distance import compute_attribute_distances
from kirkvcg_rep_stat.src.clustering.vcg_adjacency import (
    construct_vcg_adjacency,
    construct_hybrid_adjacency,
)
from kirkvcg_rep_stat.src.clustering.cahc import vcg_constrained_hac
from kirkvcg_rep_stat.src.clustering.silhouette import (
    compute_vcg_modified_silhouette,
    optimize_vcg_hyperparameters,
)
from kirkvcg_rep_stat.src.testing.mmd import compute_mmd2, imq_kernel
from kirkvcg_rep_stat.src.testing.block_permutation import (
    partition_into_blocks,
    block_permutation_test,
    pairwise_cluster_testing,
)
from kirkvcg_rep_stat.src.testing.multiple_testing import fdr_correction_bh
from kirkvcg_rep_stat.src.graph.clique_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_cliques_to_reps,
)
from kirkvcg_rep_stat.src.encoder.encoder import KirkVCGRepSpatEncoder
from kirkvcg_rep_stat.src.pipeline import KirkVCGRepSpat
from kirkvcg_rep_stat.src.data.synthetic_vcg import generate_synthetic_vcg_simulation


def test_kors_transform():
    # Synthetic 12-lead ECG of length 100
    ecg = np.random.randn(100, 12)
    vcg = ecg_to_vcg_kors(ecg)
    assert vcg.shape == (100, 3)
    assert not np.isnan(vcg).any()

    # Test transposed input orientation
    vcg_t = ecg_to_vcg_kors(ecg.T)
    assert vcg_t.shape == (100, 3)
    np.testing.assert_allclose(vcg, vcg_t)


def test_dower_transform():
    ecg = np.random.randn(80, 12)
    vcg = ecg_to_vcg_dower(ecg)
    assert vcg.shape == (80, 3)
    assert not np.isnan(vcg).any()


def test_vcg_kinematics():
    # Parametric circle in 3D VCG space: s(t) = [cos(t), sin(t), 0.5*t]
    t = np.linspace(0, 4 * np.pi, 200)
    vcg = np.column_stack([np.cos(t), np.sin(t), 0.5 * t])

    kin = compute_vcg_kinematics(vcg, fs=50.0)
    assert kin["velocity"].shape == (200, 3)
    assert kin["speed"].shape == (200,)
    assert kin["acceleration"].shape == (200, 3)
    assert kin["curvature"].shape == (200,)
    assert kin["torsion"].shape == (200,)
    assert np.all(kin["speed"] > 0)
    assert not np.isnan(kin["curvature"]).any()


def test_feature_extraction():
    ecg = np.random.randn(150, 12)
    features = extract_sample_features(ecg, fs=500.0)

    assert "continuous" in features
    assert "binary" in features
    assert "vcg" in features
    assert features["continuous"].shape[0] == 150
    assert features["binary"].shape[0] == 150
    assert features["vcg"].shape == (150, 3)

    # Check binary marker values {0, 1}
    u_vals = np.unique(features["binary"])
    assert set(u_vals).issubset({0.0, 1.0})


def test_distance_computation():
    X = np.random.randn(50, 6)
    # Continuous Euclidean
    D_euc = compute_attribute_distances(X, metric="euclidean")
    assert D_euc.shape == (50, 50)
    np.testing.assert_allclose(np.diag(D_euc), 0.0)
    assert np.all(D_euc >= 0.0)
    assert np.allclose(D_euc, D_euc.T)

    # Binary Jaccard
    X_bin = (X > 0).astype(float)
    D_jac = compute_attribute_distances(X_bin, metric="jaccard")
    assert D_jac.shape == (50, 50)
    assert np.all((D_jac >= 0.0) & (D_jac <= 1.0))
    np.testing.assert_allclose(np.diag(D_jac), 0.0)


def test_vcg_adjacency():
    vcg = np.random.randn(60, 3)
    # Pure 3D VCG adjacency
    L = construct_vcg_adjacency(vcg, m_neighbors=4)
    assert L.shape == (60, 60)
    assert np.all(L.diagonal() == 0)
    # Symmetry check
    diff = L - L.T
    assert diff.nnz == 0

    # Hybrid adjacency
    L_hyb = construct_hybrid_adjacency(vcg, m_neighbors=4, temporal_window=1)
    assert L_hyb.shape == (60, 60)
    assert L_hyb.nnz >= L.nnz


def test_cahc_and_silhouette():
    X = np.random.randn(80, 5)
    vcg = np.random.randn(80, 3)

    labels, model, conn = vcg_constrained_hac(X, vcg, n_clusters=4, m_neighbors=3)
    assert len(labels) == 80
    assert len(np.unique(labels)) == 4
    assert set(np.unique(labels)) == {1, 2, 3, 4}

    # Modified silhouette
    D = compute_attribute_distances(X)
    score, s_i = compute_vcg_modified_silhouette(D, labels, conn)
    assert -1.0 <= score <= 1.0
    assert len(s_i) == 80


def test_mmd_and_imq():
    rng = np.random.default_rng(42)
    # Identical distributions
    X1 = rng.normal(0, 1, size=(40, 4))
    X2 = rng.normal(0, 1, size=(40, 4))
    # Different distribution
    X3 = rng.normal(5, 1, size=(40, 4))

    mmd_same = compute_mmd2(X1, X2, kernel="imq")
    mmd_diff = compute_mmd2(X1, X3, kernel="imq")
    assert mmd_diff > mmd_same
    assert mmd_same >= 0.0


def test_block_permutation():
    rng = np.random.default_rng(42)
    X1 = rng.normal(0, 1, size=(30, 4))
    X2 = rng.normal(0, 1, size=(30, 4))

    blocks = partition_into_blocks(X1, m_neighbors=4)
    assert len(blocks) >= 1

    stat, p_val, null = block_permutation_test(
        X1, X2, m_neighbors=4, n_permutations=50, random_state=42
    )
    assert len(null) == 50
    assert 0.0 <= p_val <= 1.0


def test_fdr_correction():
    p_vals = [0.001, 0.004, 0.02, 0.06, 0.40]
    reject, p_adj = fdr_correction_bh(p_vals, alpha=0.05)
    assert len(p_adj) == 5
    assert np.all(p_adj[1:] >= p_adj[:-1])  # Monotonicity
    assert reject[0] == True


def test_clique_reassignment():
    import pandas as pd
    # Suppose cluster 1, 2, 3 are mutually similar (forming clique of 3)
    # Cluster 4 is unique
    pairwise_data = [
        {"cluster_1": 1, "cluster_2": 2, "obs_mmd_sq": 0.01, "adj_p": 0.80},
        {"cluster_1": 1, "cluster_2": 3, "obs_mmd_sq": 0.02, "adj_p": 0.75},
        {"cluster_1": 2, "cluster_2": 3, "obs_mmd_sq": 0.015, "adj_p": 0.85},
        {"cluster_1": 1, "cluster_2": 4, "obs_mmd_sq": 0.50, "adj_p": 0.001},
        {"cluster_1": 2, "cluster_2": 4, "obs_mmd_sq": 0.55, "adj_p": 0.002},
        {"cluster_1": 3, "cluster_2": 4, "obs_mmd_sq": 0.60, "adj_p": 0.001},
    ]
    df = pd.DataFrame(pairwise_data)
    G = build_similarity_graph(df, all_clusters=[1, 2, 3, 4], alpha=0.05)
    cliques = extract_maximal_cliques(G, min_size=3)

    assert len(cliques) == 1
    assert cliques[0] == [1, 2, 3]

    initial_labels = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    rep_labels, mapping, meta = reassign_cliques_to_reps(initial_labels, cliques)

    # 1, 2, 3 should share the same REP label
    assert rep_labels[0] == rep_labels[2] == rep_labels[4]
    # 4 should have a different label
    assert rep_labels[6] != rep_labels[0]


def test_encoder():
    X = np.random.randn(100, 5)
    vcg = np.random.randn(100, 3)
    labels = np.array([1] * 40 + [2] * 40 + [3] * 20)

    encoder = KirkVCGRepSpatEncoder(max_states=5)
    res = encoder.encode(labels, X, vcg=vcg)

    assert "occupancy" in res
    assert np.isclose(sum(res["occupancy"].values()), 1.0)
    assert res["transition_matrix"].shape == (5, 5)
    assert "rep_centroids_vcg" in res
    assert len(res["flat_feature_vector"]) > 0


def test_end_to_end_synthetic():
    sim = generate_synthetic_vcg_simulation(
        n_beats=6, samples_per_beat=80, eta=0.4, p_features=6, random_state=42
    )
    X = sim["X"]
    vcg = sim["vcg"]

    pipeline = KirkVCGRepSpat(
        metric="euclidean",
        m_neighbors=3,
        n_clusters=5,
        kernel="imq",
        kernel_param=1.0,
        n_permutations=40,
        alpha=0.05,
        min_clique_size=3,
        random_state=42,
    )

    labels = pipeline.fit_predict(X, vcg)
    summary = pipeline.get_summary()

    assert len(labels) == len(X)
    assert summary["initial_clusters_count"] == 5
    assert "maximal_cliques" in summary
    assert pipeline.is_fitted_ == True
