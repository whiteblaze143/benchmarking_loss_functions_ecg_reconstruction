"""Configuration dataclass and provenance tracking for repSpat and ECG/VCG adaptation.

Every parameter is tagged with its provenance status:
- PAPER: Explicitly defined in the paper (Senanayake & Jeganathan, 2026).
- PAPER_EXAMPLE: Used in paper examples, simulations, or application (§2.1.1, §3, §4).
- AMBIGUOUS: Omitted or underspecified in paper; deterministic completion default.
- ECG_ADAPTATION: Minimal-change adaptation to ECG/VCG temporal domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class Lineage(str, Enum):
    """Provenance category for an algorithmic parameter or decision rule."""
    PAPER = "PAPER"
    PAPER_EXAMPLE = "PAPER_EXAMPLE"
    AMBIGUOUS = "AMBIGUOUS"
    ECG_ADAPTATION = "ECG_ADAPTATION"


@dataclass
class RepSpatConfig:
    """Complete, transparent configuration for repSpat and ECG adaptation.

    Attributes
    ----------
    # --- STAGE 1: Contiguity & Dissimilarity ---
    metric : str
        Attribute dissimilarity metric ('euclidean' for continuous, 'jaccard' for binary).
        Status: [PAPER §2.1]
    m_grid : list[int]
        Grid of m-nearest neighbor values to search. Default: [2, 3, 4, 5, 6, 7, 8, 9, 10].
        Status: [PAPER_EXAMPLE §2.1.1]
    G_grid : list[int]
        Grid of cluster counts G to search. Default: [2, 3, 4, 5, 6, 7, 8, 9, 10].
        Status: [PAPER_EXAMPLE §2.1.1]
    linkage : str
        HAC linkage criterion ('ward', 'single', 'complete', 'average').
        Default: 'ward' (as used in Section 4 TNBC application).
        Status: [PAPER_EXAMPLE / Table A.7]
    symmetrize_links : str
        Symmetrization rule for directed m-NN graph ('or', 'and', 'none').
        Default: 'or' (L_ij = 1 if j in m-NN(i) or i in m-NN(j)).
        Status: [AMBIGUOUS §2.1 completion]

    # --- STAGE 2: MMD & Kernel ---
    kernel : str
        Two-sample MMD kernel ('imq' or 'gaussian').
        Default: 'imq'. Formula: k(x,y) = 1 / sqrt(||x-y||^2 + c^2).
        Status: [PAPER Table 1]
    kernel_param : float
        Kernel parameter c for IMQ or sigma for Gaussian.
        Default: 1.0 (paper application choice).
        Status: [PAPER_EXAMPLE §§2.2.1, 3.2, 4]
    biased_mmd : bool
        Whether to include diagonal terms in MMD^2 (biased empirical V-statistic, Eq. 6).
        Default: True.
        Status: [PAPER Eq. 6]

    # --- STAGE 3: Block Permutation ---
    n_permutations : int
        Number of block permutations B.
        Default: 9999 for final analysis, or 200/1000 for rapid testing.
        Status: [AMBIGUOUS §2.3 completion]
    rounding_rule : str
        Rounding rule for block count b_g = n_g / m ('nearest' = floor(n_g/m + 0.5), 'floor', 'ceil').
        Default: 'nearest'.
        Status: [AMBIGUOUS Eq. A.1 completion]
    strict_block_exceed : bool
        Whether block selection stops strictly when selected count > n_min (True),
        versus >= n_min (False). Paper Appendix A.2 says "exceed".
        Default: True.
        Status: [PAPER Appendix A.2]
    kmeans_n_init : int
        Number of random initializations for k-means attribute blocking.
        Default: 50.
        Status: [AMBIGUOUS Appendix A.2 completion]
    p_value_correction : bool
        Whether to use (R + 1) / (B + 1) upper-tail Monte-Carlo p-value formula
        to avoid invalid zero p-values (Phipson & Smyth, 2010).
        Default: True.
        Status: [AMBIGUOUS §2.3 completion]

    # --- STAGE 4: Inference & Graph Reassignment ---
    fdr_alpha : float
        Benjamini-Hochberg false discovery rate threshold.
        Default: 0.05.
        Status: [PAPER §2.3]
    clique_min_size : int
        Minimum clique size to consider for reassignment.
        Default: 2 (mathematical clique; no artificial >=3 restriction).
        Status: [PAPER §2.4 / AMBIGUOUS]
    reassignment_primary : str
        Primary reassignment interpretation ('clique' or 'connected_components').
        Default: 'clique' (per Section 2.4 and simulations). Both are saved.
        Status: [AMBIGUOUS §2.4 vs Algorithm 1 resolution]

    # --- ECG Adaptation ---
    vcg_projection : str
        Method to convert 12-lead ECG to 3D VCG ('kors', 'frank', 'bimec').
        Default: 'kors'.
        Status: [ECG_ADAPTATION]
    random_state : int
        Global random seed for reproducibility.
        Default: 42.
        Status: [AMBIGUOUS completion]
    """
    # Contiguity & Dissimilarity
    metric: str = "euclidean"
    m_grid: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10])
    G_grid: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10])
    linkage: str = "ward"
    symmetrize_links: str = "or"

    # MMD & Kernel
    kernel: str = "imq"
    kernel_param: float = 1.0
    biased_mmd: bool = True

    # Block Permutation
    n_permutations: int = 9999
    rounding_rule: str = "nearest"
    strict_block_exceed: bool = True
    kmeans_n_init: int = 50
    p_value_correction: bool = True

    # Inference & Graph Reassignment
    fdr_alpha: float = 0.05
    clique_min_size: int = 2
    reassignment_primary: str = "clique"

    # ECG Adaptation
    vcg_projection: str = "kors"
    random_state: int = 42

    @classmethod
    def fast_test_config(cls, **kwargs) -> RepSpatConfig:
        """Convenience preset for rapid unit testing and development."""
        defaults = {
            "m_grid": [2, 4],
            "G_grid": [3, 4, 5],
            "n_permutations": 100,
            "kmeans_n_init": 10,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def get_lineage_metadata(self) -> dict[str, str]:
        """Returns map of parameter names to their methodological lineage."""
        return {
            "metric": Lineage.PAPER.value,
            "m_grid": Lineage.PAPER_EXAMPLE.value,
            "G_grid": Lineage.PAPER_EXAMPLE.value,
            "linkage": Lineage.PAPER_EXAMPLE.value,
            "symmetrize_links": Lineage.AMBIGUOUS.value,
            "kernel": Lineage.PAPER.value,
            "kernel_param": Lineage.PAPER_EXAMPLE.value,
            "biased_mmd": Lineage.PAPER.value,
            "n_permutations": Lineage.AMBIGUOUS.value,
            "rounding_rule": Lineage.AMBIGUOUS.value,
            "strict_block_exceed": Lineage.PAPER.value,
            "kmeans_n_init": Lineage.AMBIGUOUS.value,
            "p_value_correction": Lineage.AMBIGUOUS.value,
            "fdr_alpha": Lineage.PAPER.value,
            "clique_min_size": Lineage.AMBIGUOUS.value,
            "reassignment_primary": Lineage.AMBIGUOUS.value,
            "vcg_projection": Lineage.ECG_ADAPTATION.value,
            "random_state": Lineage.AMBIGUOUS.value,
        }
