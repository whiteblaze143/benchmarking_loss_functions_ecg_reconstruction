# Temporal repSpat for ECG Signal Analysis (`temporal_rep_stat_ecg`)

A principled nonparametric framework for discovering **Repeated Temporal Patterns (RTPs)** in electrocardiogram (ECG) recordings.

Adapted from the **repSpat** framework (Senanayake & Jeganathan, *Spatial Statistics*, 2026, Elsevier).

---

## Overview & Architecture

Standard temporally constrained clustering algorithms enforce strict temporal contiguity. This causes them to over-segment recurring cardiac phenomena (e.g. paroxysmal arrhythmias, intermittent ST-segment depression, or periodic autonomic fluctuations) into separate labels. **Temporal repSpat** addresses this by pairing temporally constrained agglomerative hierarchical clustering with a post-clustering statistical testing stage based on the **Inverse Multiquadratic (IMQ) Maximum Mean Discrepancy ($\hat{MMD}^2$)**, **attribute-based block permutation**, and **maximal clique graph reassignment**.

```
=====================================================================================
                          Temporal repSpat PIPELINE FOR ECG
=====================================================================================

  [ STEP 1: ATTRIBUTE & TEMPORAL DISSIMILARITY ]
  ├── Attribute Dissimilarity D : Euclidean (continuous morphology) / Jaccard (binary markers)
  └── Temporal Adjacency L      : m-Nearest Neighbors along beat time axis (R¹)
                                         │
                                         ▼
  [ STEP 2: TEMPORAL CAHC & PARAMETER OPTIMIZATION ]
  ├── CAHC Merging              : Ward.D2 Lance–Williams subject to temporal links (L_ij = 1)
  └── Hyperparameter Search     : Maximize Temporally-Informed (Modified) Silhouette Score
                                  where b(i) is strictly restricted to adjacent episodes
                                         │
                                         ▼
  [ STEP 3: PAIRWISE MMD² & ATTRIBUTE BLOCK PERMUTATION ]
  ├── Empirical MMD²            : Characteristic IMQ Kernel k(x,y) = (||x-y||² + c²)⁻¹/²
  ├── Autocorrelation Null      : k-means morphological blocking b(g) = max(1, floor(n_g/m))
  └── Multiple Testing          : Benjamini–Hochberg FDR correction at alpha = 0.05
                                         │
                                         ▼
  [ STEP 4: SIMILARITY GRAPH & MAXIMAL CLIQUE REASSIGNMENT ]
  ├── Similarity Graph G_sim    : Edges between episodes with p_adj >= 0.05 (weight = MMD²)
  └── Maximal Clique Merging    : Extract maximal cliques (size >= 3) -> Unified RTP Labels
=====================================================================================
```

---

## Directory Structure

```
temporal_rep_stat_ecg/
├── __init__.py
├── README.md
├── author_repspat/          # Exact copy of the authors' original Python source code
│   ├── __init__.py
│   ├── clustering.py
│   ├── data.py
│   ├── mmd.py
│   └── visualization.py
├── repspat/                 # Importable copy of the authors' original Python code
├── src/                     # Core Temporal repSpat ECG implementation
│   ├── __init__.py
│   ├── pipeline.py          # Master TemporalRepSpat class
│   ├── visualization.py     # Timeline and graph visualizers
│   ├── clustering/
│   │   ├── distance.py      # Euclidean and Jaccard dissimilarity
│   │   ├── cahc.py          # 1D temporal adjacency and CAHC
│   │   └── silhouette.py    # Temporally-informed modified silhouette score
│   ├── testing/
│   │   ├── mmd.py           # Empirical MMD^2 with IMQ / Gaussian kernels
│   │   ├── block_permutation.py # k-means blocking and permutation resampling
│   │   └── multiple_testing.py  # Benjamini-Hochberg FDR multiple testing
│   ├── graph/
│   │   └── clique_reassignment.py # Similarity graph & maximal clique extraction (>=3)
│   └── data/
│       ├── synthetic_ar.py  # Autoregressive ECG simulation (Section 4.1)
│       └── ecg_features.py  # Multi-lead feature extraction & PTB-XL loader (Section 4.2)
├── scripts/
│   ├── run_ar_simulation_benchmark.py # Table 4 replication in 1D
│   ├── run_clinical_ecg_benchmark.py  # PTB-XL continuous vs binary benchmark
│   └── run_comprehensive_evaluation.py # Master evaluation orchestrator
├── tests/
│   └── test_temporal_rep_stat.py      # 12 automated unit tests
├── results/                 # Metrics tables (CSV, JSON)
└── figures/                 # Publication-quality figures (PNG)
```

---

## Quickstart Example

```python
from temporal_rep_stat_ecg import TemporalRepSpat
from temporal_rep_stat_ecg.src.data.synthetic_ar import generate_autoregressive_ecg_simulation

# Generate synthetic ECG beat sequence with repeated temporal patterns
sim = generate_autoregressive_ecg_simulation(n_beats=300, p_features=5, eta=0.5)
X, t = sim["X"], sim["timestamps"]

# Initialize and fit Temporal repSpat
pipeline = TemporalRepSpat(
    metric="euclidean",
    m_neighbors=[2, 4],
    n_clusters=range(3, 8),
    kernel="IMQ",
    kernel_param=1.0,
    n_permutations=200,
    alpha=0.05,
    min_clique_size=3,
)

# Predict unified RTP labels
rtp_labels = pipeline.fit_predict(X, t)
summary = pipeline.get_summary()

print("Initial episodes:", summary["initial_episodes_count"])
print("Unified RTP clusters:", summary["rtp_clusters_count"])
print("Maximal cliques extracted:", summary["maximal_cliques"])
```
