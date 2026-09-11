"""Motifs, Spatial Domains, MMD Block-Permutation, Quotient Closure, and Representation Extractors."""
from .spatial_domains import CoordinateStandardizer, VoxelGrid3D
from .mmd import (
    compute_median_heuristic,
    compute_kernel_matrix,
    compute_mmd2_u_stat,
    compute_patient_block_perm_mmd,
    calibrate_domain_self_reproducibility,
)
from .quotient import QuotientClosure
from .tokenizer import QVCGTokenizer
from .representation import (
    extract_r0_vcg_summary,
    extract_r1_spatial_domain_features,
    extract_r2_quotient_motif_features,
    extract_r3_quotient_plus_where,
    compute_motif_occupancy_and_dwell,
    compute_transition_matrix,
)
from .microstates import (
    detect_r_peaks,
    build_beat_microstates,
    select_patient_balanced_ecgs,
)

__all__ = [
    "CoordinateStandardizer",
    "VoxelGrid3D",
    "compute_median_heuristic",
    "compute_kernel_matrix",
    "compute_mmd2_u_stat",
    "compute_patient_block_perm_mmd",
    "calibrate_domain_self_reproducibility",
    "QuotientClosure",
    "QVCGTokenizer",
    "extract_r0_vcg_summary",
    "extract_r1_spatial_domain_features",
    "extract_r2_quotient_motif_features",
    "extract_r3_quotient_plus_where",
    "compute_motif_occupancy_and_dwell",
    "compute_transition_matrix",
    "detect_r_peaks",
    "build_beat_microstates",
    "select_patient_balanced_ecgs",
]
