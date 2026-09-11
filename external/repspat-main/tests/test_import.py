def test_package_imports_core_api():
    import repspat

    assert repspat.__version__ == "0.1.0"
    assert "compute_distances" in repspat.__all__
    assert "multiple_comparison" in repspat.__all__
    assert "visualization" in repspat.__all__
    assert "plot_spatial_clusters" in repspat.__all__
    assert "plot_cluster_feature_presence" in repspat.__all__
    assert "pairwise_results_to_matrix" in repspat.__all__


def test_import_loads_visualization():
    import sys

    sys.modules.pop("repspat", None)
    sys.modules.pop("repspat.visualization", None)

    import repspat  # noqa: F401

    assert "repspat.visualization" in sys.modules
    assert repspat.visualization.plot_spatial_clusters is repspat.plot_spatial_clusters
    assert (
        repspat.visualization.pairwise_results_to_matrix
        is repspat.pairwise_results_to_matrix
    )
