"""Unit test suite for temporal_rep_stat_ecg.

Verifies:
1. Distance functions (Euclidean and Jaccard)
2. Temporal adjacency construction (1D k-NN)
3. Constrained Agglomerative Hierarchical Clustering (CAHC)
4. Temporally-informed modified silhouette score
5. IMQ kernel and empirical MMD^2 statistic
6. Attribute-based k-means blocking and block permutation
7. Benjamini-Hochberg FDR correction
8. Similarity graph and maximal clique extraction (size >= 3)
9. End-to-end TemporalRepSpat pipeline
10. Synthetic AR generator
"""
import numpy as np
import pytest
import networkx as nx

from temporal_rep_stat_ecg.src.clustering.distance import compute_attribute_distances
from temporal_rep_stat_ecg.src.clustering.cahc import (
    construct_temporal_adjacency,
    temporal_constrained_hac,
)
from temporal_rep_stat_ecg.src.clustering.silhouette import (
    compute_modified_silhouette,
    temporal_silhouette_analysis,
)
from temporal_rep_stat_ecg.src.testing.mmd import (
    imq_kernel,
    gaussian_kernel,
    compute_mmd_sq,
    compute_mmd_sq_from_features,
)
from temporal_rep_stat_ecg.src.testing.block_permutation import (
    create_feature_blocks,
    permute_blocks_once,
    two_sample_block_permutation_test,
)
from temporal_rep_stat_ecg.src.testing.multiple_testing import (
    adjust_pvalues_fdr,
    pairwise_episode_testing,
)
from temporal_rep_stat_ecg.src.graph.clique_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_cliques,
)
from temporal_rep_stat_ecg.src.data.synthetic_ar import (
    build_temporal_adjacency_banded,
    generate_autoregressive_ecg_simulation,
)
from temporal_rep_stat_ecg.src.pipeline import TemporalRepSpat


def test_compute_attribute_distances_euclidean():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    D = compute_attribute_distances(X, metric="euclidean")
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert np.isclose(D[0, 1], 5.0)
    assert np.isclose(D[0, 2], 0.0)
    assert np.allclose(D, D.T)


def test_compute_attribute_distances_jaccard():
    X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
    D = compute_attribute_distances(X, metric="jaccard")
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    # Jaccard dist between [1,0,1] and [1,1,0]: intersection=1, union=3 => 1 - 1/3 = 2/3
    assert np.isclose(D[0, 1], 2.0 / 3.0)


def test_construct_temporal_adjacency():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    L = construct_temporal_adjacency(t, m_neighbors=2, symmetric=True)
    dense = L.toarray()
    assert dense.shape == (6, 6)
    assert np.all(np.diag(dense) == 0)
    assert np.all(dense == dense.T)
    # Consecutive points should be connected
    assert dense[0, 1] == 1
    assert dense[1, 2] == 1


def test_temporal_constrained_hac():
    # 2 clearly separated runs along time
    X = np.concatenate([np.ones((10, 2)) * 0.0, np.ones((10, 2)) * 10.0])
    t = np.linspace(0, 19, 20)
    labels, model, conn = temporal_constrained_hac(X, t, n_clusters=2, m_neighbors=2)
    assert len(labels) == 20
    assert len(np.unique(labels)) == 2
    # First 10 should share one label, next 10 should share another
    assert len(np.unique(labels[:10])) == 1
    assert len(np.unique(labels[10:])) == 1
    assert labels[0] != labels[10]


def test_modified_silhouette():
    X = np.concatenate([np.zeros((10, 2)), np.ones((10, 2)) * 5.0, np.zeros((10, 2))])
    t = np.linspace(0, 29, 30)
    D = compute_attribute_distances(X, metric="euclidean")
    labels = np.array([1]*10 + [2]*10 + [3]*10)
    L = construct_temporal_adjacency(t, m_neighbors=2)

    avg_sil, sils = compute_modified_silhouette(D, labels, L)
    assert -1.0 <= avg_sil <= 1.0
    assert len(sils) == 30


def test_imq_and_gaussian_kernels():
    d_sq = np.array([[0.0, 4.0], [4.0, 0.0]])
    K_imq = imq_kernel(d_sq, c_param=1.0)
    assert np.isclose(K_imq[0, 0], 1.0)
    assert np.isclose(K_imq[0, 1], 1.0 / np.sqrt(5.0))

    K_gauss = gaussian_kernel(d_sq, sigma_param=2.0)
    assert np.isclose(K_gauss[0, 0], 1.0)
    assert np.isclose(K_gauss[0, 1], np.exp(-2.0))


def test_compute_mmd_sq():
    # Identical samples should have MMD close to 0
    rng = np.random.default_rng(42)
    X1 = rng.normal(0, 1, size=(50, 4))
    X2 = rng.normal(0, 1, size=(50, 4))
    mmd_identical = compute_mmd_sq_from_features(X1, X2, kernel="IMQ", kernel_param=1.0)

    # Disparate samples should have significantly larger MMD
    X3 = rng.normal(10, 1, size=(50, 4))
    mmd_different = compute_mmd_sq_from_features(X1, X3, kernel="IMQ", kernel_param=1.0)

    assert mmd_identical < mmd_different
    assert mmd_identical >= 0.0


def test_block_permutation():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(40, 3))
    labels = np.array([1]*20 + [2]*20)
    block_ids = create_feature_blocks(X, labels, m_neighbors=4, random_state=42)
    assert len(block_ids) == 40
    assert len(np.unique(block_ids)) >= 2

    D = compute_attribute_distances(X)
    res = two_sample_block_permutation_test(
        idx_g=np.arange(20),
        idx_h=np.arange(20, 40),
        dist_matrix=D,
        block_ids=block_ids,
        n_permutations=50,
        random_state=42,
    )
    assert 0.0 <= res["p_value"] <= 1.0
    assert len(res["null_distribution"]) == 50


def test_multiple_testing_fdr():
    p_vals = np.array([0.001, 0.002, 0.04, 0.3, 0.8])
    rejected, adj_p = adjust_pvalues_fdr(p_vals, alpha=0.05)
    assert len(rejected) == 5
    assert len(adj_p) == 5
    assert np.all(adj_p >= p_vals)
    assert rejected[0] and rejected[1]


def test_clique_reassignment():
    # Test maximal clique extraction and reassignment
    # Form a 3-clique between nodes 1, 2, 3 and isolated node 4
    df = [
        {"episode_1": 1, "episode_2": 2, "obs_mmd_sq": 0.01, "adj_p": 0.8},
        {"episode_1": 2, "episode_2": 3, "obs_mmd_sq": 0.01, "adj_p": 0.8},
        {"episode_1": 1, "episode_2": 3, "obs_mmd_sq": 0.01, "adj_p": 0.8},
        {"episode_1": 1, "episode_2": 4, "obs_mmd_sq": 0.50, "adj_p": 0.001},
        {"episode_1": 2, "episode_2": 4, "obs_mmd_sq": 0.50, "adj_p": 0.001},
        {"episode_1": 3, "episode_2": 4, "obs_mmd_sq": 0.50, "adj_p": 0.001},
    ]
    import pandas as pd
    pairwise_df = pd.DataFrame(df)
    G = build_similarity_graph(pairwise_df, all_episodes=[1, 2, 3, 4], alpha=0.05)
    cliques = extract_maximal_cliques(G, min_size=3)
    assert len(cliques) == 1
    assert set(cliques[0]) == {1, 2, 3}

    initial_labels = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    reassigned, mapping = reassign_cliques(initial_labels, cliques)
    # Episodes 1, 2, 3 should now share unified label 1
    assert mapping[1] == mapping[2] == mapping[3]
    # Episode 4 should retain label 4
    assert mapping[4] == 4
    assert np.all(reassigned[:6] == 1)
    assert np.all(reassigned[6:] == 4)


def test_synthetic_ar_generator():
    sim = generate_autoregressive_ecg_simulation(
        n_beats=100, p_features=5, eta=0.4, random_state=42
    )
    assert sim["X"].shape == (100, 5)
    assert len(sim["timestamps"]) == 100
    assert len(sim["ground_truth_rtp"]) == 100
    # Must have repeated labels (at least RTP-A label 2)
    assert np.sum(sim["ground_truth_rtp"] == 2) > 0


def test_temporal_rep_spat_pipeline_end_to_end():
    sim = generate_autoregressive_ecg_simulation(
        n_beats=150, p_features=5, eta=0.3, noise_scale=0.2, random_state=42
    )
    pipeline = TemporalRepSpat(
        metric="euclidean",
        m_neighbors=4,
        n_clusters=6,
        kernel="IMQ",
        kernel_param=1.0,
        n_permutations=50,
        alpha=0.05,
        min_clique_size=3,
        random_state=42,
    )
    rtp_labels = pipeline.fit_predict(sim["X"], sim["timestamps"])
    assert len(rtp_labels) == 150
    summary = pipeline.get_summary()
    assert "initial_episodes_count" in summary
    assert "rtp_clusters_count" in summary
    assert summary["initial_episodes_count"] == 6
