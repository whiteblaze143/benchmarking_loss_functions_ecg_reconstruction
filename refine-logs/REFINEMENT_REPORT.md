# Refinement Report: Temporal repSpat for ECG Signal Analysis

**Date**: 2026-09-12  
**Document**: `refine-logs/REFINEMENT_REPORT.md`  
**Package Root**: `temporal_rep_stat_ecg/`  

---

## 1. Problem Framing & Line-by-Line Translation

| repSpat Spatial Object | Temporal repSpat ECG Analogue | Mathematical Specification |
|---|---|---|
| Sampling location $s \in \mathbb{R}^2$ | Beat timestamp $t \in \mathbb{R}^1$ | Detected R-peak time along continuous ECG recording |
| Attribute vector $X(s) \in \mathbb{R}^p$ | Beat feature vector $X(t) \in \mathbb{R}^p$ | Continuous morphology (RR, QRS, ST, T) or binary clinical markers |
| Spatial adjacency $L$ ($m$-NN in $\mathbb{R}^2$) | Temporal adjacency $L$ ($m$-NN in $\mathbb{R}^1$) | $L_{ij} = 1$ if beat $j$ is among the $m$ closest temporal neighbors of beat $i$ |
| Spatial CAHC | Temporal CAHC | Ward.D2 agglomeration subject to temporal contiguity ($L_{ij}=1$) |
| Modified spatial silhouette | Modified temporal silhouette | $b(i)$ evaluates only temporally adjacent episodes ($\ell_{g\ell}=1$) |
| IMQ Kernel on spatial features | IMQ Kernel on beat features | $k(x, y) = (\|x-y\|_2^2 + c^2)^{-1/2}$ ($c \ge 1$) |
| Spatial $k$-means blocking | Temporal $k$-means blocking | $b(g) = \max(1, \lfloor n_g/m \rfloor)$ morphological feature blocks |
| Block permutation resampling | Block permutation resampling | Shuffling whole blocks across episodes ($B=200$) preserving autocorrelation |
| Benjamini-Hochberg FDR | Benjamini-Hochberg FDR | Adjusted p-values across all $\binom{G}{2}$ pairwise tests at $\alpha = 0.05$ |
| Similarity graph $G_{\text{sim}}$ | Similarity graph $G_{\text{sim}}$ | Undirected edge where $p_{\text{adj}} \ge 0.05$ weighted by $\hat{MMD}^2$ |
| Maximal cliques of size $\ge 3$ | Maximal cliques of size $\ge 3$ | Reassignment into unified Repeated Temporal Patterns (RTPs) |
| Repeated Spatial Patterns (RSPs) | Repeated Temporal Patterns (RTPs) | Unified recurring clinical / physiological rhythm states |

---

## 2. Refinement Evolution & Risk Mitigations

1. **Risk: False discoveries from temporal autocorrelation**:
   - *Mitigation*: Attribute-based block permutation preserves the within-block temporal autocorrelation of consecutive heartbeats. Beats within a block share identical physiological states.
2. **Risk: Over-segmentation from strict temporal contiguity**:
   - *Mitigation*: Step 4 clique reassignment merges mutually distributionally equivalent episodes across the entire time recording into unified RTP labels.
3. **Risk: Under-segmentation from unconstrained clustering**:
   - *Mitigation*: Step 2 CAHC enforces strict temporal contiguity during initial episode formation, preventing noise from fusing distant, unrelated beats prematurely.
