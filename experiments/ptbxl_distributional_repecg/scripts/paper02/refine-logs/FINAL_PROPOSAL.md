# Final Proposal: Paper 02
# Phase-KME: Phase-Conditioned Kernel Mean Embeddings for Distribution-Valued ECG Representation

## Problem Anchor & Physiological Motivation
Standard electrocardiographic (ECG) representation methods collapse the ensemble of cardiac cycles into a single point-estimate waveform, most commonly via signal-averaged electrocardiography (SAECG):
$$\bar{x}(t) = \frac{1}{B} \sum_{b=1}^B x_b(t) = \mathbb{E}[X(t)] \in \mathbb{R}^8$$
While SAECG suppresses zero-mean uncorrelated noise, it fundamentally discards intra-patient beat-to-beat distributional dispersion. In clinical electrophysiology, beat-to-beat variability (e.g., microvolt T-wave alternans, conduction delay dispersion in bundle branch blocks, respiratory vectorcardiographic rotation, and localized repolarization turbulence) is not mere measurement noise, but carries direct diagnostic markers of arrhythmogenic risk and structural remodeling.

However, existing distribution-based ECG methods treat the entire recording as a single undifferentiated point cloud in an RKHS (e.g., *Robust screening of atrial fibrillation with distribution classification*, Scientific Reports 2025), discarding the temporal and anatomical choreography of the cardiac cycle. Conversely, contemporary phase-equivariant representations (Winder, arXiv:2608.21147, Aug 2026) operate on deterministic point trajectories without modeling beat ensembles as distributions.

## Core Thesis & Mathematical Formulation
We formalize a multi-beat ECG recording as a **phase-indexed field of local probability distributions** mapped into a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_k$:
$$\mu: C_{16} \longrightarrow \mathcal{H}_k, \quad g \mapsto \mu_{P_g}$$
where the cardiac cycle is discretized into $G=16$ normalized phase cells $g \in \mathbb{Z}_{16} \cong C_{16}$ anchored to ventricular depolarization ($g=0$ at R-peak). Within each phase cell $g$, the collection of observed 8-lead voltage vectors across all beats $b=1,\dots,B$ forms an empirical distribution $P_g \in \mathcal{P}(\mathbb{R}^8)$.

### 1. Characteristic Kernel Mean Embedding
For each phase cell $g$, the distribution $P_g$ is nonparametrically embedded via the Inverse Multiquadric (IMQ) kernel:
$$k_c(x, y) = \frac{1}{\sqrt{\|x - y\|_2^2 + c^2}}, \quad x, y \in \mathbb{R}^8, \; c^2 = 1.0$$
The population kernel mean embedding is:
$$\mu_{P_g} = \mathbb{E}_{X \sim P_g}[k_c(X, \cdot)] = \int_{\mathbb{R}^8} k_c(x, \cdot) \, dP_g(x) \in \mathcal{H}_k$$
Because $k_c$ is strictly characteristic and $c_0$-universal on $\mathbb{R}^8$ (Sriperumbudur et al., 2011), the map $P_g \mapsto \mu_{P_g}$ is injective:
$$\mu_{P} = \mu_{Q} \iff P = Q$$
guaranteeing that $\mu_{P_g}$ uniquely distinguishes distributions beyond any fixed finite collection of polynomial moments (e.g., distinguishing skewed, heavy-tailed, or bimodal beat distributions from Gaussian controls with identical mean and covariance).

### 2. Nyström Feature Map at Operating Point
To evaluate downstream models efficiently, we construct a finite $M$-dimensional Nyström feature map $\psi(x) \in \mathbb{R}^M$ ($M=256$ landmarks $Z \in \mathbb{R}^{M \times 8}$ selected via MiniBatchKMeans on whitened training data):
$$\psi(x) = k(x, Z) (K_{ZZ} + \lambda I)^{-1/2} \in \mathbb{R}^{256}$$
where $\lambda = 10^{-6} \operatorname{Tr}(K_{ZZ})/M$ ensures stable invertibility. The empirical phase embedding is:
$$\hat{\mu}_{P_g} = \frac{1}{N_g} \sum_{i=1}^{N_g} \psi(x_{g,i}) \in \mathbb{R}^{256}$$
The complete distribution-valued cardiac representation is the sequence:
$$U = \left[ \hat{\mu}_{P_0}, \hat{\mu}_{P_1}, \dots, \hat{\mu}_{P_{15}} \right] \in \mathbb{R}^{16 \times 256}$$

### 3. $C_{16}$-Equivariant Phase Field Processing
The 16 phase cells form the cyclic group $C_{16}$. We process $U$ with `PhaseCNN`:
- Pointwise input 1D conv: $\mathbb{R}^{256} \to \mathbb{R}^{128}$.
- 3 Residual Blocks with **Circular Padding** ($C_{16}$ boundary handling):
  $$\tilde{x} = \operatorname{pad}_{C_{16}}(x, (1, 1)) \implies \tilde{x}[0] = x[15], \; \tilde{x}[17] = x[0]$$
  guaranteeing exact discrete shift equivariance:
  $$F(T_k U) = T_k F(U), \quad \forall k \in \mathbb{Z}_{16}$$
- Readout Head: We compare Global Average Pooling (GAP, enforcing shift invariance) against a Phase-Aware Readout (flattened linear head preserving anchored depolarization coordinates).

## Inductive Biases & Assumptions Embedded
1. **Distributional Completeness**: Higher-order moments of beat-to-beat voltage distributions (skewness, dispersion, multimodality) contain diagnostic signal that is destroyed by signal averaging ($\mathbb{E}[X]$) and moment-matching (Gaussian surrogate).
2. **Phase Locality**: Beat-to-beat variability must be conditioned on cardiac phase ($P(X \mid \theta)$). Global pooling of beats across phases destroys the electrophysiological meaning of localized conduction events.
3. **Cyclic Topology ($C_{16}$)**: The cardiac cycle is periodic and closed. Boundary conditions connecting phase 15 (diastole) to phase 0 (atrial systole) must be modeled cyclically without zero-padded boundary truncation.
4. **Depolarization Phase Anchoring**: Because beats are synchronized by R-peak detection, absolute phase $\theta=0$ carries anatomical meaning (ventricular activation). A phase-aware readout preserves this coordinate system while the convolutional encoder maintains local cyclic equivariance.

## Formal Rejected Complexity
1. **Continuous $S^1$ Lie Group Equivariance**: We explicitly reject claiming continuous $S^1$ Lie group equivariance or continuous harmonic representation for 16-bin discretized data. The discrete group $C_{16} \cong \mathbb{Z}_{16}$ is the exact mathematical symmetry of 16-bin circular convolution.
2. **Arbitrary Neural Set Embeddings without RKHS Guarantees**: We reject end-to-end black-box set encoders (e.g., DeepSets/SetTransformers) as the primary representation, as they lack injectivity proofs and exhibit severe sample-size instability on small beat ensembles ($B \sim 8$). We retain DeepSets strictly as a generic baseline.
3. **Ad-hoc Distance Heuristics**: We reject heuristic moment distance metrics (Wasserstein approximations, sample kurtosis ratios) in favor of the closed-form characteristic IMQ RKHS embedding.
