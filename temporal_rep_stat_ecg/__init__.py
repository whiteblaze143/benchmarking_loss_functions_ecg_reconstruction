"""temporal_rep_stat_ecg: Nonparametric Discovery of Repeated Temporal Patterns (RTPs) in ECG Signals.

Translating repSpat (Senanayake & Jeganathan, Spatial Statistics, 2026) to 1D Temporal Biomedical Signals.
"""

__version__ = "0.1.0"

from .src.pipeline import TemporalRepSpat
from .src.clustering.distance import compute_attribute_distances
from .src.clustering.cahc import temporal_constrained_hac, construct_temporal_adjacency
from .src.clustering.silhouette import temporal_silhouette_analysis
from .src.testing.mmd import compute_mmd_sq, imq_kernel, gaussian_kernel
from .src.testing.block_permutation import two_sample_block_permutation_test
from .src.testing.multiple_testing import adjust_pvalues_fdr
from .src.graph.clique_reassignment import build_similarity_graph, extract_maximal_cliques, reassign_cliques

__all__ = [
    "TemporalRepSpat",
    "compute_attribute_distances",
    "temporal_constrained_hac",
    "construct_temporal_adjacency",
    "temporal_silhouette_analysis",
    "compute_mmd_sq",
    "imq_kernel",
    "gaussian_kernel",
    "two_sample_block_permutation_test",
    "adjust_pvalues_fdr",
    "build_similarity_graph",
    "extract_maximal_cliques",
    "reassign_cliques",
]
