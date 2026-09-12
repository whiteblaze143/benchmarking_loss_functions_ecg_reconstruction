"""tit_ecg: Exact repSpat implementation and minimal-change ECG/VCG adaptation.

Implements the exact methodology from Senanayake & Jeganathan (Spatial Statistics, 2026):
1. Spatial contiguity links L & Attribute dissimilarity D
2. Constrained Agglomerative Hierarchical Clustering (CAHC) with Table A.7 Lance-Williams updates
3. Spatially-informed modified silhouette optimization for (m*, G*)
4. Pairwise biased empirical MMD^2 (Eq. 6) with IMQ kernel
5. Attribute k-means block permutation with strict '>' stopping rule
6. Benjamini-Hochberg FDR multiple testing (alpha=0.05)
7. Dual reassignment: Connected Components (Algorithm 1) & Maximal Cliques (Section 2.4)
8. Minimal-change ECG/VCG adaptation: s_t = (tau_t, 0) in R^2, X(s_t) = v(t) in R^3.
"""

from .src.config import RepSpatConfig, Lineage
from .src.pipeline import ExactRepSpat, TemporalRepSpatECG, VCGStateRepSpatResidual

__version__ = "1.0.0"
__all__ = [
    "RepSpatConfig",
    "Lineage",
    "ExactRepSpat",
    "TemporalRepSpatECG",
    "VCGStateRepSpatResidual",
]
