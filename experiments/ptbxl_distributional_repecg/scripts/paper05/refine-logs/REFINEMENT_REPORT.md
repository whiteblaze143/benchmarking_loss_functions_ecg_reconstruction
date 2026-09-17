# Refinement Report: Paper 05
## Systematic Modifications from Initial Blueprint to Production-Locked Specification

| Component | Initial Specification | Refined Specification | Mathematical Justification |
| :--- | :--- | :--- | :--- |
| **Novelty Claim** | Generic Koopman operator on ECG distributions | Dual-operator EDMD on KME states separating intra-cycle and cycle dynamics | Avoids collision with Ghosh & Monfared (2026) and HAVOK (2017) |
| **Ontological Domain** | "Koopman operator on distribution space" | EDMD on observables of distribution-valued states in KME coordinates | Complies with transfer operator theory (Klus et al., 2020) |
| **Dynamical Trajectory** | Flattened $16B$ single sequence | Decoupled $K_{\text{phase}}$ ($g \to g+1$) and $K_{\text{cycle}}$ ($b \to b+1$) | Eliminates 15:1 intra-beat to inter-beat transition conflation |
| **Moment Estimation** | Unnormalized sum $Z_- Z_-^\top + 0.01 I$ | Sample moments $G = \frac{1}{M} Z_- Z_-^\top, A = \frac{1}{M} Z_+ Z_-^\top$ | Invariant to variable record length and beat count $B$ |
| **Ridge Regularization** | Fixed $\lambda = 0.01$ | Scale-relative $\lambda = \max(\alpha \frac{\operatorname{Tr}(G)}{d}, 10^{-6})$ | Guarantees well-conditioned inversion across spectral scales |
| **Linear Baseline** | Absent | `linear_state_dmd` ($z_t = W_{\text{PCA}} \hat{\mu}_t$) | Proves necessity of nonlinear RBF lifting |
| **Markov Baseline** | Absent | `soft_markov` ($T_{ij} = C_{ij}/\sum_r C_{rj}$) | Proves necessity of Koopman inverse-covariance solve |
| **Chronology Null** | Single shuffle seed | 3-tier destruction + $S=5$ seeds + analytical null $K_{\text{iid}}$ | Eliminates spurious single-permutation artifacts |
| **Spectral Features** | Sorted $|\lambda_i|, \operatorname{Arg}(\lambda_i)$ | Permutation-invariant moments $\rho(K), \|K\|_F, \operatorname{Tr}(K^p), \|K^\top K - K K^\top\|_F$ | Eliminates branch-cut and sorting instability |
| **Eligibility Gate** | Count-only $M \ge d$ | Tiered: $M \ge d$, $r_{\text{eff}} = \operatorname{rank}_\epsilon(Z_-)$, $r_{\text{entropy}}$, $\kappa(G+\lambda I)$ | Prevents certifying rank-deficient degenerate trajectories |\n