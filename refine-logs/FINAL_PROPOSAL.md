# Research Proposal: Temporal repSpat — Nonparametric Discovery of Repeated Temporal Patterns in Electrocardiograms

**Date**: 2026-09-12  
**Status**: ACTIVE PROPOSAL (`temporal_rep_stat_ecg`)  
**Target Venues**: IEEE Transactions on Biomedical Engineering / NeurIPS / Spatial Statistics / Nature Digital Medicine  
**Reference Origin**: Senanayake & Jeganathan (*Spatial Statistics*, 2026, Elsevier)  

---

## 1. Problem Anchor & Executive Summary

Standard temporally constrained clustering algorithms enforce strict temporal contiguity. When applied to long-term or multi-beat electrocardiograms (ECGs), they inevitably **over-segment recurring physiological states** (such as paroxysmal arrhythmias, intermittent ST-segment ischemic episodes, or respiratory sinus arrhythmia cycles) into separate, disconnected cluster labels. Conversely, unconstrained clustering algorithms (e.g., standard K-Means or unconstrained HAC) discard temporal adjacency entirely, arbitrarily mixing scattered, isolated beats and inflating false positive pattern associations due to unaddressed temporal autocorrelation.

We introduce **Temporal repSpat (`temporal_rep_stat_ecg`)**, a principled nonparametric framework that translates the spatial **repSpat** methodology to 1D temporal biomedical signals. By pairing **Constrained Agglomerative Hierarchical Clustering (CAHC)** with a post-clustering statistical testing stage based on the **Inverse Multiquadratic (IMQ) Maximum Mean Discrepancy ($\hat{MMD}^2$)**, **attribute-based block permutation**, and **maximal clique graph reassignment**, Temporal repSpat discovers **Repeated Temporal Patterns (RTPs)** while rigorously controlling the False Discovery Rate under temporal autocorrelation.

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

## 2. Mathematical Architecture Contract

### 2.1 Domain Mapping
- Locations $s \in \mathbb{R}^2$ $\longrightarrow$ Beat timestamps $t \in \mathbb{R}^1$.
- Sampling points $\{s_1, \dots, s_n\}$ $\longrightarrow$ R-peak timestamps $\{t_1, \dots, t_n\}$.
- $p$-dimensional attribute vector $X(t) \in \mathbb{R}^p$ across $n$ beats:
  - Continuous morphology: pre/post RR intervals, QRS duration, QRS peak-to-peak amplitude, ST deviation, T-wave amplitude.
  - Binary markers: ST elevation, ST depression, T inversion, pathological Q, QRS prolongation, premature beat.

### 2.2 Temporally-Informed Silhouette Separation
Unlike global silhouette scores, the between-cluster separation $b(i)$ evaluates only temporally adjacent episodes:
$$b(i) = \min_{\mathcal{C}(\ell) \neq \mathcal{C}(g) : \ell_{g\ell} = 1} \left( \frac{1}{n_\ell} \sum_{j \in \mathcal{C}(\ell)} d_{ij} \right)$$
ensuring that segmentation preserves sharp local temporal contrast.

### 2.3 Dependence-Preserving Block Permutation
Because consecutive heartbeats exhibit significant autonomic and physiological autocorrelation, standard i.i.d. beat permutation violates $H_0$ and inflates Type I error. We partition beats into $b(g) = \max(1, \lfloor n_g/m \rfloor)$ morphological blocks via $k$-means in attribute space and permute entire blocks without replacement, preserving local dependence structure under the null hypothesis.

### 2.4 Maximal Clique Criterion ($\text{size} \ge 3$)
A recurring clinical pattern requires mutual distributional equivalence across at least three temporally separated episodes:
$$\text{clique}(G_{\text{sim}}) = \{ \mathcal{C}(g_1), \dots, \mathcal{C}(g_k) \} \quad \text{with } k \ge 3, \quad \forall u, v \in \text{clique}, \; p_{\text{adj}}(u, v) \ge 0.05$$
preventing isolated, spurious pairwise matches from contaminating global episode taxonomy.

---

## 3. Dominant Contribution & Rejected Complexity

- **Dominant Contribution**: The exact, provably valid translation of repSpat to 1D temporal biomedical time series, establishing the first nonparametric statistical discovery framework for Repeated Temporal Patterns in ECG.
- **Explicitly Rejected Complexity**:
  - *No heavy neural black-box clustering*: The framework remains fully non-parametric, interpretable, and reproducible without stochastic gradient descent or latent training instability.
  - *No arbitrary heuristic thresholding*: Hypotheses are judged strictly by MMD with exact permutation null distributions and Benjamini-Hochberg FDR control.
