import numpy as np
import pandas as pd
import pytest


def make_sample_adata():
    anndata = pytest.importorskip("anndata")

    adata = anndata.AnnData(
        X=np.array([[0.0, 0.0], [1.0, 1.0]]),
        obs=pd.DataFrame(
            {"sample_id": ["sample", "sample"], "mm": ["a", "b"]},
            index=["cell1", "cell2"],
        ),
    )
    adata.var_names = ["marker_a", "marker_b"]
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    return adata


def make_layered_adata():
    adata = make_sample_adata()
    adata.X = np.array([[0.2, 4.0], [2.0, 0.1]])
    adata.layers["binary"] = np.array([[0, 1], [1, 0]])
    return adata


def test_compute_distances_requires_feature_layer_and_updates_adata():
    from repspat import compute_distances

    adata = make_layered_adata()
    adata.X = None

    returned = compute_distances(adata, layer="binary", metric="jaccard")

    assert returned is adata
    assert adata.uns["repspat"]["layer"] == "binary"
    assert adata.uns["repspat"]["metric"] == "jaccard"
    assert adata.uns["repspat"]["distance_key"] == "repspat_distances"
    np.testing.assert_array_equal(
        adata.obsp["repspat_distances"],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )


def test_compute_distances_rejects_missing_feature_layer():
    from repspat import compute_distances

    with pytest.raises(KeyError, match="Layer 'missing' not found"):
        compute_distances(make_layered_adata(), layer="missing", metric="jaccard")


def test_compute_distances_rejects_empty_feature_layer():
    from repspat import compute_distances

    with pytest.raises(ValueError, match="layer must be provided"):
        compute_distances(make_layered_adata(), layer=None, metric="jaccard")


def make_three_cell_binary_sample():
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.array(
            [
                [0.2, 4.0],
                [2.0, 0.1],
                [3.0, 5.0],
            ]
        ),
        obs=pd.DataFrame(
            {
                "sample_id": ["sample", "sample", "sample"],
                "mm": ["a", "b", "c"],
            },
            index=["cell1", "cell2", "cell3"],
        ),
    )
    adata.var_names = ["marker_a", "marker_b"]
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    adata.layers["binary"] = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
        ]
    )
    return adata


def fake_spatial_neighbors(adata, **kwargs):
    from scipy.sparse import csr_matrix

    n_obs = adata.n_obs
    adata.obsp["spatial_connectivities"] = csr_matrix(
        np.ones((n_obs, n_obs)) - np.eye(n_obs)
    )


def test_spatial_constrained_hac_uses_selected_feature_layer_with_ward(monkeypatch, capsys):
    from repspat import compute_distances
    import repspat.clustering as clustering

    adata = compute_distances(
        make_three_cell_binary_sample(), layer="binary", metric="jaccard"
    )
    captured = {}

    class FakeAgglomerativeClustering:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["model_kwargs"] = kwargs

        def fit_predict(self, X):
            captured["X"] = np.asarray(X)
            return np.array([0, 1, 1])

    monkeypatch.setattr(clustering, "_spatial_neighbors", fake_spatial_neighbors)
    monkeypatch.setattr(clustering, "AgglomerativeClustering", FakeAgglomerativeClustering)

    clustered_adata = clustering.spatial_constrained_hac(
        adata,
        n_clusters=2,
        n_neighs=1,
        linkage="ward",
    )
    captured_output = capsys.readouterr()
    labels = clustered_adata.obs["labels"].astype(int).to_numpy()

    np.testing.assert_array_equal(
        captured["X"],
        np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ]
        ),
    )
    np.testing.assert_array_equal(labels, np.array([1, 2, 2]))
    assert clustered_adata is adata
    assert (
        'HAC labels are stored in adata.obs["labels"]'
        in captured_output.out
    )
    assert "repspat_hac_model" not in clustered_adata.uns
    assert clustered_adata.uns["repspat_hac"]["layer"] == "binary"
    assert clustered_adata.uns["repspat_hac"]["metric"] == "jaccard"
    assert clustered_adata.uns["repspat_hac"]["linkage"] == "ward"
    assert captured["model_kwargs"]["n_clusters"] == 2
    assert captured["model_kwargs"]["metric"] == "euclidean"
    assert captured["model_kwargs"]["linkage"] == "ward"


def test_spatial_silhouette_analysis_result_omits_linkage_column(monkeypatch, capsys):
    from repspat import compute_distances
    import repspat.clustering as clustering

    adata = compute_distances(
        make_three_cell_binary_sample(), layer="binary", metric="jaccard"
    )

    class FakeAgglomerativeClustering:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_predict(self, X):
            return np.array([0, 0, 1])

    monkeypatch.setattr(clustering, "_spatial_neighbors", fake_spatial_neighbors)
    monkeypatch.setattr(clustering, "AgglomerativeClustering", FakeAgglomerativeClustering)
    example_adata = adata

    returned = clustering.spatial_silhouette_analysis(
        example_adata,
        n_neighbors_list=[1],
        n_clusters_range=[2],
        linkage="ward",
    )
    results = example_adata.uns["silhouette_scores"]
    captured = capsys.readouterr()

    assert returned is example_adata
    assert (
        'Silhouette scores are stored in adata.uns["silhouette_scores"]'
        in captured.out
    )
    assert results.columns.tolist() == [
        "n_neighbors",
        "n_clusters",
        "avg_silhouette",
    ]


def test_validate_hac_linkage_rejects_unknown_linkage():
    import repspat.clustering as clustering

    with pytest.raises(ValueError, match="linkage must be one of"):
        clustering._validate_hac_linkage("auto")


def test_compute_mmd_sq_df_gaussian_identical_groups_is_zero():
    from repspat.mmd import compute_mmd_sq_df

    dist = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )

    result = compute_mmd_sq_df(["a", "b"], ["a", "b"], dist, kernel="Gaussian")

    assert result == pytest.approx(0.0)


def test_compute_mmd_sq_df_rejects_unknown_kernel():
    from repspat.mmd import compute_mmd_sq_df

    dist = pd.DataFrame([[0.0]], index=["a"], columns=["a"])

    with pytest.raises(ValueError, match="kernel must be one of"):
        compute_mmd_sq_df(["a"], ["a"], dist, kernel="linear")


def test_multiple_comparison_stores_results_and_returns_adata(monkeypatch, capsys):
    from repspat import compute_distances
    import repspat.mmd as mmd

    adata = compute_distances(
        make_three_cell_binary_sample(), layer="binary", metric="jaccard"
    )
    adata.obs["labels"] = pd.Series(
        [1, 1, 2], index=adata.obs_names
    ).astype("category")
    adata.obs["repspat_block_id"] = [1, 1, 1]

    def fake_two_sample_mmd(*args, **kwargs):
        return {
            "obs_mmd_sq": 0.25,
            "p_value": 0.5,
            "null_dist": np.array([0.1, 0.2]),
        }

    monkeypatch.setattr(mmd, "two_sample_mmd", fake_two_sample_mmd)

    returned = mmd.multiple_comparison(adata, kernel="IMQ")
    captured = capsys.readouterr()

    assert returned is adata
    assert (
        'MMD results are stored in adata.uns["repspat_mmd_results"]'
        in captured.out
    )
    results = adata.uns["repspat_mmd_results"]
    assert results.loc[0, "obs_mmd_sq"] == pytest.approx(0.25)
    assert results.loc[0, "p_value"] == pytest.approx(0.5)
    assert results.loc[0, "adj_p"] == pytest.approx(0.5)


def test_pairwise_results_to_matrix_links_non_significant_pairs():
    from repspat.visualization import pairwise_results_to_matrix

    results = pd.DataFrame(
        {
            "region_1": [1, 1, 2],
            "region_2": [2, 3, 3],
            "obs_mmd_sq": [0.25, 0.75, 0.5],
            "adj_p": [0.10, 0.01, 0.20],
        }
    )

    matrix = pairwise_results_to_matrix(results, plot=False)

    assert matrix.loc[1, 2] == pytest.approx(0.25)
    assert matrix.loc[2, 3] == pytest.approx(0.5)
    assert matrix.loc[1, 3] == pytest.approx(0.0)


def test_plot_cluster_feature_presence_creates_ranked_plot_per_cluster():
    import matplotlib.pyplot as plt

    from repspat import compute_distances, plot_cluster_feature_presence

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.array(
            [
                [1, 0, 0, 0.2],
                [1, 1, 0, 0.4],
                [0, 0, 1, 0.6],
                [0, 1, 1, 0.8],
                [1, 1, 1, 1.0],
            ]
        ),
        obs=pd.DataFrame({"labels": [1, 1, 2, 2, 2]}),
    )
    adata.var_names = ["marker_a", "marker_b", "marker_c", "continuous"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]]
    )
    adata.layers["features"] = adata.X.copy()
    adata = compute_distances(adata, layer="features", metric="euclidean")

    figures = plot_cluster_feature_presence(
        adata,
        top_n=2,
    )

    assert set(figures) == {1, 2}

    cluster_1_figure, cluster_1_axes = figures[1]
    spatial_ax, feature_ax = cluster_1_axes
    assert len(spatial_ax.collections[1].get_offsets()) == 2
    assert spatial_ax.get_legend() is None
    assert [tick.get_text() for tick in feature_ax.get_yticklabels()][-1] == "marker_a"
    assert feature_ax.patches[-1].get_width() == pytest.approx(100.0)

    for figure, _ in figures.values():
        plt.close(figure)


def test_plot_cluster_feature_presence_falls_back_to_enrichment_for_continuous_data():
    import matplotlib.pyplot as plt

    from repspat import compute_distances, plot_cluster_feature_presence

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.array(
            [
                [1000.0, 2.0, 4.0],
                [1000.0, 2.0, 5.0],
                [1100.0, 0.0, 5.0],
                [900.0, 0.2, 4.0],
                [1000.0, 0.4, 4.0],
            ]
        ),
        obs=pd.DataFrame({"labels": [1, 1, 2, 2, 2]}),
    )
    adata.var_names = ["large_scale_marker", "cluster_marker", "other_marker"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]]
    )
    adata.layers["features"] = adata.X.copy()
    adata = compute_distances(adata, layer="features", metric="euclidean")

    figures = plot_cluster_feature_presence(
        adata,
        top_n=2,
    )

    _, cluster_1_axes = figures[1]
    feature_ax = cluster_1_axes[1]
    assert feature_ax.get_xlabel() == "Cluster-vs-rest standardized enrichment"
    assert [tick.get_text() for tick in feature_ax.get_yticklabels()][-1] == "cluster_marker"
    assert feature_ax.patches[-1].get_width() > 0

    for figure, _ in figures.values():
        plt.close(figure)


def test_plot_spatial_clusters_reads_anndata_slots():
    import matplotlib.pyplot as plt

    from repspat import plot_spatial_clusters

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"labels": [1, 1, 2]}),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [1, 1], [2, 0]])

    fig, ax = plot_spatial_clusters(adata)

    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_spatial_clusters_accepts_dataframe_spatial_coords():
    import matplotlib.pyplot as plt

    from repspat import plot_spatial_clusters

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"labels": [1, 1, 2]}),
    )
    adata.obsm["spatial"] = pd.DataFrame(
        {"centroidX": [0, 1, 2], "centroidY": [0, 1, 0]},
        index=adata.obs_names,
    )

    fig, ax = plot_spatial_clusters(adata)

    assert len(ax.collections) == 2
    plt.close(fig)


def test_create_blocks_writes_block_ids_to_adata(capsys):
    from repspat import compute_distances, create_blocks

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.array([[0, 0], [0, 1], [5, 5], [5, 6]]),
        obs=pd.DataFrame({"labels": [1, 1, 2, 2]}),
    )
    adata.var_names = ["marker_a", "marker_b"]
    adata.layers["features"] = adata.X.copy()
    adata = compute_distances(adata, layer="features", metric="euclidean")

    returned = create_blocks(adata, knn=8)
    captured = capsys.readouterr()

    assert returned is adata
    assert "repspat_block_id" in adata.obs
    assert adata.obs["repspat_block_id"].tolist() == [1, 1, 1, 1]
    assert (
        'Block IDs are stored in adata.obs["repspat_block_id"]'
        in captured.out
    )


def test_pairwise_results_to_matrix_reads_anndata_results():
    from repspat.visualization import pairwise_results_to_matrix

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(X=np.ones((2, 1)))
    adata.uns["repspat_mmd_results"] = pd.DataFrame(
        {
            "region_1": [1],
            "region_2": [2],
            "obs_mmd_sq": [0.25],
            "adj_p": [0.10],
        }
    )

    matrix = pairwise_results_to_matrix(adata, plot=False)

    assert matrix.loc[1, 2] == pytest.approx(0.25)
