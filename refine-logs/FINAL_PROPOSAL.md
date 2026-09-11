# Research Proposal: GRAIL-ECG v2 — Representation-Theoretic Qualification of a Clinical ECG State Space

**Date**: 2026-09-11  
**Status**: ACTIVE CHAMPION PROPOSAL (v2 Superseding Reconstruction Framework)  
**Target Venues**: NeurIPS / ICLR / ICML / Nature Machine Intelligence  

---

## 1. Executive Summary & Problem Anchor

The fundamental object of investigation in electrocardiographic machine learning is **not waveform reconstruction and not narrow multi-label classification accuracy**. It is the learned latent representation itself:

$$
E: \mathcal{X} \longrightarrow \mathcal{Z}, \qquad \mathcal{Z} \in \mathbb{R}^d \quad (d=96)
$$

The central scientific question is whether $\mathcal{Z}$ is a **valid, compact, and sufficient representation of the clinically relevant information** contained in a standard 12-lead ECG.

Waveform reconstruction (synthesizing 60,000 raw voltage samples) forces a model to expend representational capacity on high-frequency acquisition noise, electrode skin-impedance drift, and idiosyncratic baseline wander. Conversely, training end-to-end task classifiers entangles clinical knowledge into arbitrary non-linear decision boundaries that fail to transfer to unseen cardiac pathologies.

We propose **GRAIL-ECG v2**, a representation-theoretic framework divided into two decoupled, verifiable stages:

$$
\boxed{\text{\bf Stage A: Construct & Qualify the Full-ECG Representation } \mathcal{Z}^\star = T_\theta(X_{\text{full}})}
$$
$$\Downarrow \quad (\text{only if qualified across multi-dimensional criteria})$$
$$
\boxed{\text{\bf Stage B: Map the Complete 4,095-Subset Information Lattice } Z_S = G_\phi(X_S, S)}
$$

---

## 2. Theoretical Formulation & Architecture Contracts

### 2.1 The Representation Target
Let complete standard ECG measurements be $X = \{x_I, x_{II}, x_{III}, x_{aVR}, x_{aVL}, x_{aVF}, x_{V1}, \ldots, x_{V6}\} \in \mathbb{R}^{12 \times 5000}$.
We seek a teacher encoder $T_\theta(X) = Z^\star \in \mathbb{R}^{96}$ such that $Z^\star$ serves as a **task-family-sufficient statistic**:

$$
Y \perp X \mid Z^\star
$$

for all physiological variables $Y$ spanning cardiac rhythm, conduction kinetics, chamber morphology, and repolarization dynamics.

### 2.2 Mathematical Architecture Contract: Lead-Order Permutation Invariance
The ordering of leads in a tensor is arbitrary. The symmetric group $S_k$ acts on the order of observed lead channels. Global representation $Z$ must carry the **trivial representation** of $S_k$:

$$
G(\pi \cdot X_S) = G(X_S) \qquad \forall \pi \in S_k
$$

Using set cross-attention over spatio-temporal tokens where keys and values are permuted by permutation matrix $P$:

$$
\operatorname{softmax}\left(Q(PK)^T\right) (PV) = \operatorname{softmax}\left(QK^T P^T\right) PV = \operatorname{softmax}\left(QK^T\right) V
$$

Because no cross-lead positional embedding is injected, permutation invariance holds analytically. Consequently, the $1.3 \times 10^9$ ordered sequences collapse into exactly **4,095 unique subset equivalence classes**.

### 2.3 Frontal Lead Geometry & Algebraic Rank
Displayed lead count $k = |S|$ does not equal electrical information rank. Under Einthoven's Law ($III = II - I$) and Goldberger's equations, the frontal limb leads have algebraic rank $r_{\text{limb}} \le 2$. Every subset $S$ is characterized by independent rank $r(S) \in \{1..8\}$ relative to basis $\{I, II, V1..V6\}$.

---

## 3. The 10-Model Factorial Architecture Matrix

To isolate the independent contributions of physical inductive bias and structured latent representations, we define a binary $2^3$ factorial design on factors $(G, S, V)$:
- $G \in \{0, 1\}$: Spherical geometry harmonics ($\theta, \phi$).
- $S \in \{0, 1\}$: Structured clinical slots (6 slots $\times$ 16D = 96D: Rhythm, Conduction, Morphology, ST-T, Residual 1, Residual 2).
- $V \in \{0, 1\}$: View reconstruction auxiliary decoder.

| Model ID | Geometry ($G$) | Structured Slots ($S$) | View Aux ($V$) | SSL (VICReg) | Clinical BCE |
|---|:---:|:---:|:---:|:---:|:---:|
| **UB** | — | — | — | — | Fully Supervised Upper Bound |
| **B0** | 0 | 0 | 0 | ✓ | — (True Plain SSL Baseline) |
| **B1** | 0 | 0 | 0 | ✓ | ✓ |
| **B3_geom** | 1 | 0 | 0 | ✓ | ✓ |
| **B2_slots** | 0 | 1 | 0 | ✓ | ✓ |
| **Model_001**| 0 | 0 | 1 | ✓ | ✓ |
| **Model_110**| 1 | 1 | 0 | ✓ | ✓ |
| **Model_101**| 1 | 0 | 1 | ✓ | ✓ |
| **Model_011**| 0 | 1 | 1 | ✓ | ✓ |
| **Model_M**  | 1 | 1 | 1 | ✓ | ✓ (Full GRAIL Architecture) |

This factorial structure computes main effects $\Delta_G, \Delta_S, \Delta_V$ and all second- and third-order interactions.

---

## 4. Multi-Dimensional Representation Qualification Suite

A representation does not pass by single-number AUROC. It must pass an exhaustive qualification profile:

1. **Linear Sufficiency**: Logistic probes on frozen $Z$ for Anchor concepts ($N=25$), Tier P1 (directly related, $N=11$), Tier P2 (within-domain transfer, $N=15$), and Tier P3 (pure ontology-distinct, $N=3$: `LVOLT`, `NORM`, `PACE`).
2. **Compactness & Intrinsic Geometry**: Singular value spectrum, effective rank $r_{\text{eff}} = \exp(-\sum p_i \log p_i)$, participation ratio $PR$, condition number $\kappa$, and TwoNN intrinsic dimension $d_{\text{TwoNN}}$.
3. **Geometric Separability**: Fisher separation criterion $J_c$, centroid distance $d_c$, and within-versus-between distance ratio $R_c$.
4. **Local Semantic Geometry**: Latent nearest-neighbor retrieval precision $P@k$, $Recall@k$, and $nDCG@k$ for $k \in \{1, 5, 10, 20, 50\}$ under multi-label Jaccard overlap.
5. **Functional Disentanglement**: $6 \times C$ Slot $\times$ Concept probe matrix, slot selectivity gap, and intervention specificity (selectively zeroing/ablating slot $z_j$ to verify domain-specific prediction drop).
6. **Residual Slot Challenge**: Verifying that residual discovery slots ($Z_{\text{residual}} \in \mathbb{R}^{32}$) encode meaningful non-trivial information beyond structured slots ($Z_{\text{structured}} \in \mathbb{R}^{64}$).
7. **Nuisance Accessibility**: Probing frozen $Z$ for patient Age ($R^2$ and MAE via Ridge regression) and Sex (AUROC).

---

## 5. Integration of LVCG Multi-Benchmark Probing Suite

To ensure direct reproducibility and benchmark comparability against frontier ECG foundation models (such as LVCG, Zhan et al.), we integrate the complete LVCG downstream linear probing battery:
- **Standard Downstream Benchmarks**:
  1. `ptbxl_super_class` (5 superclasses: NORM, MI, STTC, CD, HYP)
  2. `ptbxl_sub_class` (23 diagnostic subclasses)
  3. `ptbxl_form` (19 ECG form/morphology diagnostic classes)
  4. `ptbxl_rhythm` (12 rhythm statement classes)
  5. `icbeb` (ICBEB 2018 9-class arrhythmia challenge)
  6. `chapman` (Chapman CSN 4-class arrhythmia challenge)
- **Protocol**:
  - Backbone frozen; feature extraction precomputed in $O(N \times 12)$ steps.
  - Multi-seed linear probing with Adam optimizer (lr=1e-3, batch_size=128/256).
  - Class-wise optimal decision threshold search on validation set via F1 grid sweep.
  - Multi-metric reporting: Macro AUROC, Macro F1, Accuracy, Precision, Recall.
  - Low-shot sample efficiency sweeps across label ratios $\{0.01, 0.05, 0.10, 0.25, 0.50, 1.00\}$.

---

## 6. Stage B: Lead Information Theory across 4,095 Subsets

Once the teacher encoder $Z^\star = T(X_{\text{full}})$ passes Stage A qualification, the generalist arbitrary-lead model $Z_S = G(X_S, S)$ is trained to align with $Z^\star$.

We exhaustively evaluate all **4,095 non-empty subsets** using fast spatio-temporal token caching, yielding:
1. **Coordinate-Preserving vs Reprobed Recovery**:
   - $P_{\text{fixed}}(S) = \operatorname{Perf}(W^\star Z_S)$ vs $P_{\text{reprobe}}(S) = \operatorname{Perf}(W_S Z_S)$.
   - Coordinate drift gap: $D_{\text{coordinate}}(S) = P_{\text{reprobe}}(S) - P_{\text{fixed}}(S)$.
2. **Exact Lead Shapley Values**:
   $$\phi_i^{(d)} = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(12-|S|-1)!}{12!} [v_d(S \cup \{i\}) - v_d(S)]$$
   Computed for latent similarity $\phi_i^{(Z)}$ and clinical diagnostic accuracy $\phi_i^{(d)}$.
3. **Empirical Lead Synergy & Redundancy Graph**:
   Pairwise Harsanyi/Möbius 2nd-order interaction matrix $I_{ij}^{(d)}$ identifying genuine clinical cooperation ($I_{ij} > 0$) vs redundant coverage ($I_{ij} < 0$).
4. **Pareto Information Frontiers**:
   Information retention bounds $\{P_{\min}, P_{\text{med}}, P_{\max}\}$ plotted against displayed lead count $k$ and independent rank $r(S)$.
5. **Disease-Specific Minimal Sufficient Lead Sets**:
   Minimal cardinality subsets $k_d^\star$ satisfying $P_d(S) \ge P_d(\text{full}) - 0.02$.
