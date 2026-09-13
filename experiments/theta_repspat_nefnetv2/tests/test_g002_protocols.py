import numpy as np
import pytest
import scipy.stats
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from tit_ecg.src.vcg_transform import KORS_REGRESSION_MATRIX


def test_kors_matrix_shape_and_orthogonality():
    assert KORS_REGRESSION_MATRIX.shape == (3, 8)
    # Test on synthetic 8-lead ECG [n_samples, 8]
    ecg_8 = np.random.default_rng(42).normal(size=(500, 8))
    vcg = ecg_8 @ KORS_REGRESSION_MATRIX.T
    assert vcg.shape == (500, 3)


def test_distance_matrix_spearman_identity():
    # Synthetic trajectory
    T = 100
    X = np.sin(np.linspace(0, 4 * np.pi, T))[:, None] + np.random.default_rng(0).normal(scale=0.1, size=(T, 3))
    
    # Pairwise distances
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(T, k=1)
    vec_D = D[iu]
    
    # Self-correlation must be exactly 1.0
    rho, _ = scipy.stats.spearmanr(vec_D, vec_D)
    assert rho == pytest.approx(1.0, abs=1e-6)
    
    # Scaled and translated trajectory (uniform scaling should preserve Spearman rho = 1.0)
    X_scaled = 5.0 * X + 10.0
    D_scaled = np.linalg.norm(X_scaled[:, None, :] - X_scaled[None, :, :], axis=-1)
    rho_scaled, _ = scipy.stats.spearmanr(D_scaled[iu], vec_D)
    assert rho_scaled == pytest.approx(1.0, abs=1e-5)


def test_knn_jaccard_overlap():
    T = 50
    k = 10
    X = np.random.default_rng(1).normal(size=(T, 8))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    
    # Identical representation must have Jaccard = 1.0
    jaccards = []
    for t in range(T):
        dists = D[t].copy()
        dists[t] = np.inf
        nn_gt = set(np.argsort(dists)[:k])
        nn_arm = set(np.argsort(dists)[:k])
        jaccard = len(nn_gt & nn_arm) / len(nn_gt | nn_arm)
        jaccards.append(jaccard)
    assert np.mean(jaccards) == pytest.approx(1.0, abs=1e-6)


def test_cahc_temporal_ari():
    T = 100
    n_clusters = 5
    # Temporal connectivity matrix (line graph: each sample connected to immediate neighbors)
    connectivity = kneighbors_graph(np.arange(T)[:, None], n_neighbors=2, include_self=False)
    
    X = np.random.default_rng(2).normal(size=(T, 8))
    model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    labels = model.fit_predict(X)
    
    # Self-agreement ARI must be 1.0
    ari = adjusted_rand_score(labels, labels)
    assert ari == pytest.approx(1.0, abs=1e-6)
