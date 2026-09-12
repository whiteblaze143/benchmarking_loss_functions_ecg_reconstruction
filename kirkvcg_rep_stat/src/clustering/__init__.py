"""Clustering module for VCG repSpat."""
from .distance import compute_attribute_distances
from .vcg_adjacency import (
    construct_vcg_adjacency,
    construct_hybrid_adjacency,
)
from .cahc import vcg_constrained_hac
from .silhouette import (
    compute_vcg_modified_silhouette,
    optimize_vcg_hyperparameters,
)

__all__ = [
    "compute_attribute_distances",
    "construct_vcg_adjacency",
    "construct_hybrid_adjacency",
    "vcg_constrained_hac",
    "compute_vcg_modified_silhouette",
    "optimize_vcg_hyperparameters",
]
