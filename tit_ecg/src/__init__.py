"""Exact repSpat Reference and Minimal-Change ECG/VCG Adaptation.

Implementation of Senanayake & Jeganathan (Spatial Statistics, 2026).
"""
from __future__ import annotations

from .cahc import (
    compute_attribute_dissimilarity,
    compute_modified_silhouette,
    construct_domain_links,
    run_cahc,
    search_optimal_cahc,
)
from .config import Lineage, RepSpatConfig
from .data_loader import (
    generate_synthetic_ecg_vcg,
    list_clinical_ecg_records,
    load_clinical_ecg_record,
)
from .graph_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_by_cliques,
    reassign_by_connected_components,
)
from .mmd_test import (
    apply_kernel,
    compute_biased_mmd2_from_dist,
    create_attribute_blocks,
    run_all_pairwise_mmd_tests,
    run_block_permutation_test,
)
from .pipeline import ExactRepSpat, TemporalRepSpatECG, VCGStateRepSpatResidual
from .vcg_transform import KORS_REGRESSION_MATRIX, ecg_to_vcg_kors

__all__ = [
    "Lineage",
    "RepSpatConfig",
    "compute_attribute_dissimilarity",
    "construct_domain_links",
    "run_cahc",
    "compute_modified_silhouette",
    "search_optimal_cahc",
    "apply_kernel",
    "compute_biased_mmd2_from_dist",
    "create_attribute_blocks",
    "run_block_permutation_test",
    "run_all_pairwise_mmd_tests",
    "build_similarity_graph",
    "extract_maximal_cliques",
    "reassign_by_connected_components",
    "reassign_by_cliques",
    "KORS_REGRESSION_MATRIX",
    "ecg_to_vcg_kors",
    "load_clinical_ecg_record",
    "list_clinical_ecg_records",
    "generate_synthetic_ecg_vcg",
    "ExactRepSpat",
    "TemporalRepSpatECG",
    "VCGStateRepSpatResidual",
]
