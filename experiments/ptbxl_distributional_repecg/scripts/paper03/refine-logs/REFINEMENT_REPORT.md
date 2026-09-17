# Refinement Report: Paper 03
# Evolution from Naive Path Signatures to Phase-Path-Signature

## 1. Executive Summary of Changes
This report documents the structural refinements made to Paper 03 to eliminate mathematical contradictions, narrow the novelty boundary against *PathFusion-Net*, and establish a defensible claim-identification matrix.

| Dimension | Initial Conception | Refined Phase-Path-Signature | Rationale |
| :--- | :--- | :--- | :--- |
| **Novelty Claim** | "First path-signature ECG representation" | "Cardiac-phase-indexed field of trajectory signature distributions" | PathFusion-Net (2024) is direct prior art for 1D windowed rough paths. |
| **Descriptor Form** | $[\gamma(0), \gamma(1), \bar{\gamma}, \operatorname{LogSig}_3]$ ($228$D) | $d^{\text{sig}} = [\gamma(0), \gamma(1), \operatorname{LogSig}_3]$ ($220$D) | $\bar{\gamma}$ is not reparameterization invariant under non-linear time warping. |
| **Morphology Breakdown** | Undifferentiated 228D vector | Absolute morphology ($16$D) + Translation-invariant geometry ($204$D) | Signatures depend purely on $d\gamma$; boundary levels restore baseline voltage. |
| **Terminology** | "Vectorcardiographic loop" | "Multilead lead-space trajectory in $(V_1, V_5)$ coordinates" | Precordial leads are scalar projections, not calibrated orthogonal VCGs. |
| **Chirality Claim** | "Conduction chirality" | "Trajectory orientation/chirality" | Lévy area measures path orientation in lead space, not tissue-level conduction. |
| **KME Ablation** | None (only compared to Paper 02) | Mandatory `phase_mean_signature` ($\bar{s}_g$) | Isolates whether beat distribution matters vs simple signature averaging. |
| **Signature Depth** | Fixed $m=3$ default | Empirical hypothesis: depth 1 vs depth 2 vs depth 3 | Tests whether signed area ($m=2$) or higher brackets ($m=3$) drive clinical gain. |
| **Synthetic Tests** | 4 basic shape checks | 6 rigorous mathematical proof worlds with null-relative gates | Verifies Chen's identity, monotonicity, exact multiset scrambling, and tree-like cancellation. |

---

## 2. Detailed Mathematical Rectifications

### Rectification A: Decoupling $d^{\text{sig}}$ and $d^{\text{aug}}$
Let $\phi: [0, 1] \to [0, 1]$ be a smooth, strictly increasing reparameterization with $\phi(0)=0, \phi(1)=1$.
For any continuous path $\gamma$, by integration by substitution:
$$\int_{0 < t_1 < \dots < t_k < 1} d(\gamma \circ \phi)^{i_1}(t_1) \dots d(\gamma \circ \phi)^{i_k}(t_k) = \int_{0 < u_1 < \dots < u_k < 1} d\gamma^{i_1}(u_1) \dots d\gamma^{i_k}(u_k)$$
Hence $S(\gamma \circ \phi) = S(\gamma)$ and $\operatorname{LogSig}(\gamma \circ \phi) = \operatorname{LogSig}(\gamma)$.
Furthermore, the boundary values are invariant: $(\gamma \circ \phi)(0) = \gamma(0)$ and $(\gamma \circ \phi)(1) = \gamma(1)$.
Therefore, the pure descriptor:
$$d^{\text{sig}} = [\gamma(0), \gamma(1), \operatorname{LogSig}_3(\gamma)]$$
satisfies exact reparameterization invariance:
$$d^{\text{sig}}(\gamma \circ \phi) = d^{\text{sig}}(\gamma)$$
However, for the temporal mean:
$$\bar{\gamma}_{\phi} = \int_0^1 (\gamma \circ \phi)(t) dt = \int_0^1 \gamma(u) (\phi^{-1})'(u) du \neq \int_0^1 \gamma(u) du = \bar{\gamma}$$
unless $\phi(t) = t$ is strictly linear. Including $\bar{\gamma}$ breaks the mathematical invariance guarantee.

### Rectification B: Exact Multiset Order Destruction
In Synthetic World 3, the sample points $\{x_1, \dots, x_N\}$ are permuted internally ($x_{\pi(1)}, \dots, x_{\pi(N)}$) while fixing endpoints.
Because the underlying multiset of points is identical:
$$\bar{x}_{\text{perm}} = \frac{1}{N}\sum_{i=1}^N x_{\pi(i)} = \bar{x}_{\text{orig}}, \quad \Sigma_{\text{perm}} = \Sigma_{\text{orig}}$$
Any unordered distribution model (such as Paper 02's `unordered_kme`) has distance $0$:
$$\text{MMD}^2(P_{\text{orig}}, P_{\text{perm}}) = 0$$
However, the path signature changes drastically:
$$\|\operatorname{LogSig}(\gamma_{\text{orig}}) - \operatorname{LogSig}(\gamma_{\text{perm}})\|_2 > 0.5 \|\operatorname{LogSig}(\gamma_{\text{orig}})\|_2$$
This provides unambiguous proof of the path-ordering inductive bias.
