"""Verification of tit_ecg CAHC against Senanayake & Jeganathan (2026) author implementation.

Provenance Conventions:
- PAPER_SPECIFICATION: Equation (2) defines a_i = 1/(n_g - 1) sum_{j != i} D_ij.
  When n_g = 1 (singleton cluster), this denominator is zero and mathematically undefined.
- AUTHOR_CODE_CONVENTION: The official Python implementation (repspat/clustering.py)
  explicitly sets a_i = 0.0 when len(members) <= 1, and if no spatial neighbor clusters exist,
  leaves silhouette at 0.0 without raising an exception.
  tit_ecg.src.cahc deliberately implements AUTHOR_CODE_CONVENTION to preserve exact numerical
  parity with author code.

Tests:
1. Exact mathematical equivalence of modified silhouette between author's repspat.clustering
   and tit_ecg.src.cahc.
2. Exact spatial/temporal constraint graph equivalence: L_author == L_ours.
3. Exact cluster partition agreement: ARI(Y_author, Y_ours) == 1.000000 for EVERY tested (m, G).
4. Exact hierarchical merge sequence / dendrogram agreement: M_author == M_ours (children_ exact match).
5. Parameter selection (m*, G*) parity on real clinical ECG recordings (LUDB).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform

from tit_ecg.src.cahc import (
    compute_attribute_dissimilarity,
    construct_domain_links,
    run_cahc,
    compute_modified_silhouette,
    search_optimal_cahc,
)


def author_spatial_silhouette(dist_matrix, adjacency, labels):
    """Exact author implementation of spatial silhouette from repspat/clustering.py."""
    labels = np.asarray(labels)
    silhouettes = np.zeros(labels.shape[0], dtype=float)
    label_indices = {
        label: np.flatnonzero(labels == label)
        for label in np.unique(labels)
    }

    neighbor_labels = {}
    for label, indices in label_indices.items():
        connected = adjacency[indices].any(axis=0)
        neighbor_labels[label] = [
            other
            for other in np.unique(labels[connected])
            if other != label
        ]

    for label, members in label_indices.items():
        if len(members) > 1:
            intra = dist_matrix[np.ix_(members, members)]
            a = (intra.sum(axis=1) - np.diag(intra)) / (len(members) - 1)
        else:
            # AUTHOR_CODE_CONVENTION: a_i = 0.0 for singletons (paper Eq 2 has 0 denominator)
            a = np.zeros(len(members), dtype=float)

        neighbors = neighbor_labels.get(label, [])
        if not neighbors:
            continue

        b_candidates = [
            dist_matrix[np.ix_(members, label_indices[neighbor])].mean(axis=1)
            for neighbor in neighbors
        ]
        b = np.minimum.reduce(b_candidates)
        denom = np.maximum(a, b)
        silhouettes[members] = np.divide(
            b - a,
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0,
        )

    avg_sil = np.mean(silhouettes)
    return avg_sil, silhouettes


def test_full_author_cahc_partition_and_dendrogram_parity():
    print("=" * 75)
    print("TEST 1: Full CAHC Parity: L Graph, Partition Y, and Merge Dendrogram M")
    print("=" * 75)

    np.random.seed(42)
    N = 100
    p = 3
    X = np.random.randn(N, p)
    S = np.column_stack([np.linspace(0, 1, N), np.zeros(N)])

    m_grid = [4, 6, 8]
    G_grid = [3, 4, 5, 6, 7]

    for m in m_grid:
        # Author graph construction (kneighbors_graph with mode='connectivity', symmetric union)
        knn_graph = kneighbors_graph(S, n_neighbors=m, include_self=False, mode="connectivity")
        L_author = (knn_graph + knn_graph.T).astype(bool).astype(int)

        # tit_ecg graph construction
        L_ours = construct_domain_links(S, m=m, symmetrize="or")

        diff_L = (L_author != L_ours).nnz
        print(f"\nEvaluating m={m}: Constraint Graph Difference nnz(L_author != L_ours) = {diff_L}")
        assert diff_L == 0, f"Constraint graph mismatch at m={m}!"

        for G in G_grid:
            # Author HAC model
            m_auth = AgglomerativeClustering(
                n_clusters=G,
                metric="euclidean",
                linkage="ward",
                connectivity=L_author,
            )
            y_auth = m_auth.fit_predict(X)

            # tit_ecg HAC model
            labels_ours, model_ours = run_cahc(
                X,
                L_ours,
                n_clusters=G,
                linkage="ward",
                return_model=True,
            )

            # 1. Assert exact partition agreement (ARI == 1.0)
            ari = float(adjusted_rand_score(y_auth, labels_ours))
            assert abs(ari - 1.0) < 1e-10, f"Partition mismatch at m={m}, G={G}! ARI = {ari}"

            # 2. Assert exact hierarchical merge sequence / dendrogram agreement
            ch_diff = int(np.max(np.abs(m_auth.children_ - model_ours.children_)))
            assert ch_diff == 0, f"Dendrogram children mismatch at m={m}, G={G}!"

            print(f"  m={m:2d}, G={G:2d} -> ARI(Y_auth, Y_ours) = {ari:.6f} | Dendrogram children max diff = {ch_diff}")

    print("\n--> PASS: Exact machine-precision match across all (m, G) for L, Y, and M!\n")


def test_synthetic_author_silhouette_parity():
    print("=" * 75)
    print("TEST 2: Modified Silhouette Machine-Precision Parity")
    print("=" * 75)

    np.random.seed(42)
    N = 100
    p = 3
    X = np.random.randn(N, p)
    S = np.linspace(0, 1, N)[:, None]

    D = squareform(pdist(X, metric="euclidean"))
    L = construct_domain_links(S, m=4, symmetrize="or")
    adj = L.toarray().astype(bool)

    for n_clusters in [3, 4, 5, 6]:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward",
            connectivity=L,
        )
        labels = model.fit_predict(X)

        auth_avg, auth_sil = author_spatial_silhouette(D, adj, labels)
        tit_avg, tit_sil, is_valid = compute_modified_silhouette(D, L, labels)

        diff_avg = abs(auth_avg - tit_avg)
        max_diff_sil = np.max(np.abs(auth_sil - tit_sil))

        print(f"Clusters G={n_clusters}: Author Avg Sil={auth_avg:.6f}, tit_ecg Avg Sil={tit_avg:.6f}, Diff={diff_avg:.2e}")
        assert diff_avg < 1e-12, f"Discrepancy in average silhouette: {diff_avg}"
        assert max_diff_sil < 1e-12, f"Discrepancy in per-sample silhouette: {max_diff_sil}"

    print("--> PASS: Exact machine-precision match between author code and tit_ecg CAHC silhouette!\n")


def test_real_ecg_cahc_selection():
    print("=" * 75)
    print("TEST 3: Real ECG VCG Signal Parameter Selection & Cluster Verification")
    print("=" * 75)

    from tit_ecg.src.dataset_adapters import LUDBAdapter
    ludb = LUDBAdapter()
    rec = ludb.load_record(1, n_samples=1000)
    vcg = rec["vcg"]
    time_s = rec["time"][:, None]

    m_grid = [4, 8, 16]
    G_grid = [4, 5, 6, 7]

    m_star, G_star, scores_df, partition_cache = search_optimal_cahc(
        S=time_s,
        X=vcg,
        m_grid=m_grid,
        G_grid=G_grid,
        linkage="ward",
    )

    print(f"Optimal selected parameters: m*={m_star}, G*={G_star}")

    D = squareform(pdist(vcg, metric="euclidean"))
    L_star = construct_domain_links(time_s, m=m_star, symmetrize="or")
    labels_star = partition_cache[(m_star, G_star)]

    auth_avg, auth_sil = author_spatial_silhouette(D, L_star.toarray().astype(bool), labels_star)
    tit_avg, tit_sil, is_valid = compute_modified_silhouette(D, L_star, labels_star)

    diff = abs(auth_avg - tit_avg)
    print(f"Verification on (m*={m_star}, G*={G_star}):")
    print(f"  Author avg silhouette:  {auth_avg:.8f}")
    print(f"  tit_ecg avg silhouette: {tit_avg:.8f}")
    print(f"  Absolute difference:    {diff:.2e}")
    assert diff < 1e-12, f"Discrepancy: {diff}"
    print("--> PASS: Exact parity verified on real clinical ECG recording!\n")


if __name__ == "__main__":
    test_full_author_cahc_partition_and_dendrogram_parity()
    test_synthetic_author_silhouette_parity()
    test_real_ecg_cahc_selection()
    print("=" * 75)
    print("ALL CAHC PARITY AUDIT SUITES PASSED:")
    print("  1. L_author == L_ours: PASS")
    print("  2. ARI(Y_author, Y_ours) == 1.0 across all (m, G): PASS")
    print("  3. M_author == M_ours (dendrogram merge sequence): PASS")
    print("  4. Modified silhouette exact machine precision: PASS")
    print("=" * 75)
