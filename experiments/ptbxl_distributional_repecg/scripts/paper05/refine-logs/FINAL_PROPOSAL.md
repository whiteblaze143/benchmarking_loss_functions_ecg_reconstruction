# Final Research Proposal: Paper 05
## The Koopman Operator in Lifted RKHS: Dual Phase-Progression and Beat-to-Beat Transition Dynamics

### 1. Abstract & Defensible Novelty Boundary
Understanding cardiac electrophysiology requires capturing both the rapid intra-cycle sequence of myocardial depolarization/repolarization and the beat-to-beat variability that signals autonomic or arrhythmic pathology. While recent work has applied Koopman and Extended Dynamic Mode Decomposition (EDMD) to raw ECG waveforms (HAVOK, Brunton et al., 2017; Ghosh & Monfared, March 2026 arXiv:2603.08339 with tuned RBF dictionaries and Transformers), Paper 05 establishes a distinct, defensible contribution:

$$\boxed{ \text{A record-specific EDMD approximation on KME-represented distribution-valued ECG states, separating intra-cycle phase dynamics } (K_{\text{phase}}) \text{ from beat-to-beat cycle dynamics } (K_{\text{cycle}}). }$$

Crucially, Paper 05 does **not** claim novelty from generic $\text{ECG} + \text{EDMD} + \text{RBF observables} + \text{classification}$. Instead, it addresses a foundational scientific question:
**Do transition laws between distribution-valued cardiac states provide diagnostic information beyond which states are statically occupied?**

### 2. Precise Mathematical Ontology: Koopman on Observables of Distribution-Valued States
In classical dynamical systems:
- Perron-Frobenius transfer operators act on probability distributions and densities.
- Koopman operators act on *observables* of state spaces (Klus, Schuster, & Muandet, 2020).

In Paper 05:
1. The underlying state at beat $b \in \{1, \dots, B\}$ and phase sector $g \in \{1, \dots, 16\}$ is an empirical probability distribution over voltage samples: $x_t^\star = P_{b, g}$.
2. Its Kernel Mean Embedding (KME) in an RKHS $\mathcal{H}_k$ is $\mu(P_{b, g}) \in \mathcal{H}_k$, approximated in finite dimensions by $\hat{\mu}_{b, g} \in \mathbb{R}^{128}$ via Nyström landmarks.
3. Observables $\psi: \mathbb{R}^{128} \to \Delta^{31} \subset \mathbb{R}^{32}$ are constructed via soft Radial Basis Functions (RBF) centered at global cluster anchors $\{\mathbf{c}_k\}_{k=1}^{32}$:
   $$\psi_k(\hat{\mu}) = \frac{\exp(-\|\hat{\mu} - \mathbf{c}_k\|_2^2 / \tau)}{\sum_{j=1}^{32} \exp(-\|\hat{\mu} - \mathbf{c}_j\|_2^2 / \tau)}$$
   satisfying $\mathbf{1}^\top \psi(\hat{\mu}) = 1$ and $\psi_k(\hat{\mu}) \ge 0$.
4. EDMD computes regularized linear predictors on these observable coordinates, yielding record-specific transition operators.

### 3. Resolution of the Flattened Dynamics Conflation (Dual Operators)
A major conceptual flaw in early formulations was flattening the $B \times 16$ state array into a single sequence $\mu_1, \dots, \mu_{16B}$, asking a single operator $K$ to simultaneously explain intra-beat phase progression $(b, g) \to (b, g+1)$ ($15B$ transitions) and inter-beat transitions $(b, 16) \to (b+1, 1)$ (only $B-1$ transitions). 

Paper 05 rigorously decouples this into two distinct physiological operators:

#### 3.1 Phase-Progression Operator ($K_{\text{phase}}$)
Advances the distribution-valued state within the cardiac cycle across phases $g \to g+1$:
$$Z_{\text{phase}}^- = [\psi(\hat{\mu}_{b, g})]_{b=1\dots B, g=1\dots 15} \in \mathbb{R}^{32 \times 15B}$$
$$Z_{\text{phase}}^+ = [\psi(\hat{\mu}_{b, g+1})]_{b=1\dots B, g=1\dots 15} \in \mathbb{R}^{32 \times 15B}$$
For a typical 10-beat record, $M_{\text{phase}} = 150$ transitions.

#### 3.2 Cycle-to-Cycle Operator ($K_{\text{cycle}}$)
Advances the state from beat to beat at matched cardiac phase $\psi(\hat{\mu}_{b, g}) \to \psi(\hat{\mu}_{b+1, g})$:
$$Z_{\text{cycle}}^- = [\psi(\hat{\mu}_{b, g})]_{b=1\dots B-1, g=1\dots 16} \in \mathbb{R}^{32 \times 16(B-1)}$$
$$Z_{\text{cycle}}^+ = [\psi(\hat{\mu}_{b+1, g})]_{b=1\dots B-1, g=1\dots 16} \in \mathbb{R}^{32 \times 16(B-1)}$$
For a 10-beat record, $M_{\text{cycle}} = 144$ transitions.

### 4. Normalized Sample Moments & Scale-Relative Regularization
Unnormalized ridge estimators $(Z_- Z_-^\top + \lambda I)^{-1}$ scale with sample size $M$. Paper 05 solves with normalized sample moments:
$$G = \frac{1}{M} Z_- Z_-^\top \in \mathbb{R}^{32 \times 32}, \quad A = \frac{1}{M} Z_+ Z_-^\top \in \mathbb{R}^{32 \times 32}$$
$$K = A (G + \lambda I_{32})^{-1}$$
where $\lambda$ is scale-relative with a strictly positive floor:
$$\lambda = \max\left( \alpha \frac{\operatorname{Tr}(G)}{d}, \lambda_{\min} \right), \quad \alpha = 10^{-3}, \quad \lambda_{\min} = 10^{-6}$$
Guaranteeing consistent regularized behavior across records with varying beat counts.

### 5. Analytical Independence Null & Excess Operator
Under temporal independence (scrambled chronology), the cross-covariance satisfies $\mathbb{E}[\psi_{t+1} \psi_t^\top] \approx \bar{\psi} \bar{\psi}^\top$.
The analytical independence operator is:
$$K_{\text{iid}} = \bar{\psi} \bar{\psi}^\top (G + \lambda I)^{-1}$$
The excess operator isolates transition laws existing strictly beyond static occupancy:
$$K_{\text{excess}} = K_{\text{observed}} - K_{\text{iid}}$$

### 6. Representation Vector & Diagnostic Invariants
The unified patient representation is:
$$x_i = [\bar{\psi}_i, \Phi(K_{\text{phase}, i}), \Phi(K_{\text{cycle}, i})]$$
where $\Phi(K)$ comprises:
1. Dimensionality-reduced PCA coefficients of $\operatorname{vec}(K)$.
2. Permutation-invariant spectral and structural scalars:
   - Spectral radius $\rho(K) = \max_j |\lambda_j|$
   - Frobenius and spectral norms: $\|K\|_F, \|K\|_2$
   - Matrix traces: $\operatorname{Tr}(K), \operatorname{Tr}(K^2), \operatorname{Tr}(K^3), \operatorname{Tr}(K^4)$
   - Non-normality measure: $\|K^\top K - K K^\top\|_F$
   - Mass conservation defect: $\epsilon_{\text{mass}} = \frac{\|\mathbf{1}^\top K - \mathbf{1}^\top\|_2}{\sqrt{32}}$
   - Negative entry fraction: $f_{\text{neg}} = \frac{1}{1024} \sum_{i,j} \mathbb{I}(K_{ij} < 0)$
   - Cross-validated split-half rollout residual $e_{\text{cross}}$ and operator reproducibility $S_K$.\n