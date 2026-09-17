# Peer Review Summary: Paper 05
## Critical Methodological Review & Author Rebuttals

### Round 1 Review: Literature Prior Art & Boundary Realignment
- **Reviewer Critique**: Ghosh & Monfared (March 2026, *arXiv:2603.08339*) recently presented EDMD Koopman feature extraction with tuned RBF dictionaries and Transformers for ECG classification. HAVOK (Brunton et al., 2017) previously applied Koopman delay modeling to ECG. Paper 05 cannot claim novelty for EDMD on ECG or RBF observable lifting.
- **Author Response**: Acknowledged and adopted. The novelty boundary is formally narrowed and fortified: Paper 05 does not claim generic EDMD on raw ECG. Its contribution is specifically **record-specific EDMD on empirical probability-distribution embeddings (Phase-KMEs), decoupling intra-cycle phase dynamics ($K_{\text{phase}}$) from beat-to-beat cycle dynamics ($K_{\text{cycle}}$)**.

### Round 2 Review: Mathematical Ontology (Koopman vs Perron-Frobenius)
- **Reviewer Critique**: The text states the Koopman operator acts "on probability distributions". By definition, Koopman acts on observables of states; transfer operators acting on densities/distributions are Perron-Frobenius (Klus et al., 2020).
- **Author Response**: Corrected throughout. The state space is distribution-valued ($x_t^\star = P_{b,g}$), represented by KME coordinates $\hat{\mu} \in \mathbb{R}^{128}$, and the Koopman operator governs the evolution of scalar observables $\psi_k(\hat{\mu}) \in \Delta^{31}$ on this state space.

### Round 3 Review: Flattened Dynamics Conflation
- **Reviewer Critique**: Flattening $B$ beats and 16 phases into a single sequence mixes $15B$ intra-beat phase steps with $B-1$ inter-beat boundaries. 94% of the operator estimation is driven by intra-beat phase progression, invalidating claims of beat-to-beat cycle modeling.
- **Author Response**: Resolved by splitting into two distinct operators:
  1. $K_{\text{phase}}$: transitions within beats ($g \to g+1$, $M=15B$).
  2. $K_{\text{cycle}}$: transitions across successive beats at matched phase ($b \to b+1$, $M=16(B-1)$).

### Round 4 Review: Regularization & Estimator Normalization
- **Reviewer Critique**: The estimator $(Z_- Z_-^\top + 0.01 I)^{-1}$ uses an unnormalized Gram matrix. Because $Z_- Z_-^\top$ scales with $M$, a 6-beat record receives significantly different effective regularization than a 15-beat record.
- **Author Response**: Switched to sample moments $G = \frac{1}{M} Z_- Z_-^\top, A = \frac{1}{M} Z_+ Z_-^\top$ and scale-relative ridge $\lambda = \max(\alpha \frac{\operatorname{Tr}(G)}{d}, 10^{-6})$.

### Round 5 Review: Missing Baselines & Mechanism Controls
- **Reviewer Critique**: (1) Why RBF-EDMD over linear DMD on KME states? (2) Why Koopman inverse-covariance over empirical Markov transition counts? (3) A single chronology shuffle is insufficient to test order dependence.
- **Author Response**: Added:
  1. `linear_state_dmd`: linear DMD directly on PCA-projected KME states $z_{t+1} \approx A z_t$.
  2. `soft_markov`: empirical column-normalized transition matrix $T_{ij} = C_{ij} / \sum_r C_{rj}$.
  3. Three targeted destructive controls (`phase_order_shuffled`, `beat_order_shuffled`, `global_shuffled`) evaluated across $S=5$ independent seeds.
  4. Analytical independence null $K_{\text{iid}} = \bar{\psi} \bar{\psi}^\top (G + \lambda I)^{-1}$.\n