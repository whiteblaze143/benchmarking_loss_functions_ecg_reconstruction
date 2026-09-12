__version__ = "0.1.0"

from .data import compute_distances
from .clustering import (
    create_blocks,
    spatial_silhouette_analysis,
    spatial_constrained_hac,
)
from .mmd import two_sample_mmd, multiple_comparison
from . import visualization
from .visualization import (
    pairwise_results_to_matrix,
    plot_cluster_feature_presence,
    plot_spatial_clusters,
)

__all__ = [
    "compute_distances",
    "create_blocks",
    "spatial_silhouette_analysis",
    "spatial_constrained_hac",
    "two_sample_mmd",
    "multiple_comparison",
    "visualization",
    "plot_spatial_clusters",
    "plot_cluster_feature_presence",
    "pairwise_results_to_matrix",
]
