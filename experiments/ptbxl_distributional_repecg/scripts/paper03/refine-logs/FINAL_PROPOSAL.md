# Final Proposal: Paper 03
# Phase-Path-Signature: Distributional Path Signatures on Cardiac Phase Trajectories

## Problem Anchor & Physiological Motivation
In 12-lead electrocardiography (ECG), the heart's electrical activation sequence physically traces out a continuous geometric trajectory in multilead space:
$$\gamma(t) = \begin{pmatrix} X_{\text{I}}(t) \\ X_{\text{II}}(t) \\ X_{\text{V1}}(t) \\ \vdots \\ X_{\text{V6}}(t) \end{pmatrix} \in \mathbb{R}^8$$
While Paper 02 models the *static marginal distribution* of voltages $P(X \mid \theta) \in \mathcal{P}(\mathbb{R}^8)$ within normalized cardiac phase cells, it deliberately discards the sequential ordering of sample points, treating the cardiac phase cell as an unordered point cloud. Consequently, Paper 02 cannot distinguish trajectories that have identical pointwise marginals but distinct dynamical ordering, loop curvature, or orientation.

Conversely, prior rough path approaches to ECG (most notably *PathFusion-Net*, PubMed 41196785) extract path signatures over unaligned, global, or sliding-window 1D time-series signals and feed them directly into generic CNNs or LSTMs:
$$\text{PathFusion-Net:} \quad \gamma \longrightarrow \operatorname{Sig}(\gamma) \longrightarrow \text{deep classifier}$$
This misses the fundamental quasi-periodic structure of the cardiac cycle, fails to anchor signatures to specific electrophysiological phases (depolarization vs. repolarization), and discards beat-to-beat distributional variability across the cardiac cycle.

## The Core Defensible Object
Paper 03 formalizes ECG as a **cardiac-phase-indexed field of probability distributions over local multilead trajectory signatures**:
$$\boxed{ \theta \longmapsto \mu_{P(\operatorname{LogSig}(\gamma_{b, \theta}))} \in \mathcal{H}_{k_{\text{sig}}} }$$
The complete pipeline proceeds through a strict structural hierarchy:
$$\text{Paper 03:} \quad \gamma_{b,g} \longrightarrow \operatorname{LogSig}_3(\gamma_{b,g}) \longrightarrow P_g^{\text{sig}} \longrightarrow \mu_{P_g^{\text{sig}}} \longrightarrow C_{16}\text{-field}$$

### 1. Continuous Trajectory Representation in Cardiac Phase Cells
Each heartbeat $b \in \{1, \dots, B\}$ is partitioned into $G=16$ normalized phase cells $g \in \mathbb{Z}_{16} \cong C_{16}$ anchored to ventricular depolarization ($g=0$ at R-peak). Within cell $g$, the multilead lead-space trajectory is a continuous path parameterized on the unit interval:
$$\gamma_{b,g}: [0, 1] \longrightarrow \mathbb{R}^8$$

### 2. Truncated Log-Signature & Lie Algebra Projection
The continuous path signature $S(\gamma) = (1, S^1, S^2, S^3, \dots) \in T((\mathbb{R}^8))$ is defined by iterated path integrals:
$$S^{i_1, \dots, i_k}(\gamma) = \int_{0 < t_1 < \dots < t_k < 1} d\gamma^{i_1}(t_1) \dots d\gamma^{i_k}(t_k)$$
By Chen's Identity and rough path theory, $S(\gamma)$ is strictly invariant to any orientation-preserving, continuous, non-decreasing reparameterization $\phi: [0, 1] \to [0, 1]$:
$$S(\gamma \circ \phi) = S(\gamma)$$
To eliminate tensor shuffle redundancies and obtain a minimal geometric coordinate system, we project onto the free Lie algebra $\mathfrak{g}^{(3)}(\mathbb{R}^8)$ via the depth-3 log-signature:
$$\operatorname{LogSig}_3(\gamma) = \log S(\gamma) = \sum_{k=1}^3 \operatorname{logsig}^{(k)}(\gamma) \in \mathbb{R}^{204}$$
The dimensions decompose strictly as:
- **Level 1 ($m=1$)**: Net lead-space displacement $\Delta \gamma^i = \gamma^i(1) - \gamma^i(0)$ (dimension $8$).
- **Level 2 ($m=2$)**: Antisymmetric Lévy area $A^{i,j} = \frac{1}{2}(S^{i,j} - S^{j,i})$ measuring signed planar loop area across lead pairs (dimension $\binom{8}{2} = 28$).
- **Level 3 ($m=3$)**: Non-commutative Lie brackets $[[e_i, e_j], e_k]$ capturing 3D trajectory non-flatness, torsion, and higher-order curvature (dimension $\frac{1}{3}(8^3 - 8) - \binom{8}{2} = 168$).
Total log-signature dimension: $8 + 28 + 168 = 204$.

### 3. Decoupling Absolute Morphology from Translation-Invariant Path Geometry
Because iterated integrals depend exclusively on differentials $d\gamma$, the pure signature is strictly translation-invariant: $S(\gamma + c) = S(\gamma)$.
In electrocardiography, baseline voltage level at phase boundaries contains physiological information (e.g. ST-segment elevation/depression). We explicitly decouple the descriptor into:
$$\boxed{ d_{b,g}^{\text{sig}} = [\gamma_{b,g}(0), \gamma_{b,g}(1), \operatorname{LogSig}_3(\gamma_{b,g})] \in \mathbb{R}^{220} }$$
representing **absolute boundary morphology** ($16$ dims) plus **translation-invariant path geometry** ($204$ dims).
We formally reject including the temporal mean $\bar{\gamma} = \int_0^1 \gamma(t) dt$ in the primary invariant descriptor because $\bar{\gamma}$ is **not** reparameterization invariant under non-linear time warping. We retain $d^{\text{aug}} = [d^{\text{sig}}, \bar{\gamma}]$ ($228$ dims) strictly as an ablation (`signature_plus_mean`).

### 4. Distributional Kernel Mean Embedding across Beats
Heartbeats exhibit respiratory sinus arrhythmia, rate-dependent conduction changes, and morphological dispersion. Rather than collapsing beats by simple averaging $\bar{s}_g = \frac{1}{B}\sum_b s_{b,g}$, we embed the empirical distribution of beat signatures $P_g^{\text{sig}} = \frac{1}{B}\sum_{b=1}^B \delta_{s_{b,g}}$ into an RKHS $\mathcal{H}_{k_{\text{sig}}}$.
Descriptors are first whitened via Truncated PCA ($W_{\text{pca}}: \mathbb{R}^{228} \to \mathbb{R}^{64}$) and mapped via an Inverse Multiquadric (IMQ) kernel Nyström feature map $\psi(z) \in \mathbb{R}^{128}$ ($M=128$ landmarks):
$$\hat{\mu}_g = \frac{1}{B} \sum_{b=1}^B \psi(z_{b,g}) \in \mathbb{R}^{128}$$
The complete multibeat ECG record is represented as:
$$U = [\hat{\mu}_0, \dots, \hat{\mu}_{15}] \in \mathbb{R}^{16 \times 128}$$

### 5. $C_{16}$-Equivariant PhaseCNN
$U$ forms a closed periodic loop over the discrete cyclic group $C_{16} \cong \mathbb{Z}_{16}$, processed by `PhaseCNN` with circular residual convolutions (`CircularResidualBlock`), guaranteeing discrete shift equivariance:
$$F(T_k U) = T_k F(U), \quad \forall k \in \mathbb{Z}_{16}$$

---

## Inductive Biases & Assumptions Embedded
1. **Local Trajectory Ordering**: The sequence of sample points within each phase cell matters; replacing the path with an unordered point cloud destroys diagnostic loop geometry.
2. **Reparameterization Invariance**: Non-linear stretching or compression of intra-phase time (conduction slowing, rate variation) should leave the underlying geometric path signature invariant.
3. **Trajectory Chirality & Orientation**: Reversing the trajectory time arrow inverts the sign of the Lévy area ($A^{i,j} \mapsto -A^{i,j}$), directly capturing trajectory orientation.
4. **Distributional Dispersion**: Intra-patient beat-to-beat variability in trajectory shape carries diagnostic signal beyond the beat-averaged mean signature.
5. **Periodic Cyclic Topology**: Cardiac phases form a closed periodic circle $C_{16}$, requiring circular boundary conditions.

---

## Formal Rejected Complexity
1. **Generic Rough-Path Claims**: We explicitly reject claiming the "first rough path ECG model", as PathFusion-Net (2024) is direct prior art for unaligned/windowed 1D signatures.
2. **VCG Terminology**: We reject calling lead-space projections in $(V_1, V_5)$ coordinates "vectorcardiograms", reserving VCG for calibrated orthogonal transformations (e.g. Frank XYZ).
3. **Conduction Direction Claims**: We reject describing Lévy area sign changes as "conduction chirality" in cardiac tissue, formalizing it strictly as "trajectory orientation/chirality in multilead space".
4. **Non-Invariant Mean Inclusion**: We reject appending temporal mean $\bar{\gamma}$ to the formal invariant descriptor because it breaks Chen/reparameterization invariance.
