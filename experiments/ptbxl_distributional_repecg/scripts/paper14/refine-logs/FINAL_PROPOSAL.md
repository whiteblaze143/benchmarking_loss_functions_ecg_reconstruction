# Final Proposal: Paper 14 (V2.2)

## Problem Anchor
Kernelized Maximum Mean Discrepancy (MMD) representation alignment across prespecified acquisition environments.

## Core Thesis
**Kernel MMD regularization encourages alignment of learned representation distributions across prespecified acquisition perturbations holding patient and diagnosis constant.**

## Precise Claim Boundary
1. **Acquisition-Perturbation Representation Alignment**: Biased IMQ $\text{MMD}^2$ penalty $L_{\text{MMD}}$ acts on normalized latent representations $Z = \operatorname{LayerNorm}(f_\theta(X), \text{elementwise\_affine}=\text{False})$ to penalize differences in marginal representation distributions across controlled acquisition perturbations ($T_e$).
2. **Explicitly Disclaimed Scopes**:
   - *Not Hospital Generalization / Domain Generalization*: Evaluated solely across controlled same-record perturbations ($X_i^{(e)} = T_e(X_i)$), not across unobserved hospital sites.
   - *Not Invariant Mechanism Discovery*: Marginal MMD does not identify causal invariant mechanisms. When marginal distributions match while label relationships reverse ($P(S \mid E=0) \approx P(S \mid E=1)$ but $S \approx Y$ in $E_0$ and $S \approx -Y$ in $E_1$), MMD is blind to the reversal (demonstrated by Synthetic Gate G2).
   - *Not Invariant to Label-Mixture Shifts*: Marginal MMD penalizes differing class prevalences $P(Y \mid E)$ even when class-conditionals $P(X \mid Y, E)$ are identical (demonstrated by Synthetic Gate G6). Real-data acquisition environments hold diagnosis prevalence invariant by construction ($\Delta_{e, e', c} = 0$).
3. **Scale-Inflation Escape Resolution (G5 & G5b)**:
   - Fixed-bandwidth biased IMQ MMD under unconstrained scale inflation $z \mapsto az$ approaches its finite-sample diagonal floor $\lim_{a \to \infty} \widehat{\text{MMD}}_b^2 = \frac{1}{n} + \frac{1}{m}$ without distributional alignment.
   - Structurally resolved by enforcing $Z = \operatorname{LayerNorm}(H, \text{elementwise\_affine}=\text{False})$, eliminating trainable post-normalization scale parameters $\gamma$.
4. **Finite-Sample Null Calibration (G0)**:
   - The biased estimator null is distribution-dependent, not a universal constant. Real-data null calibration is performed per frozen model via matched permutation of environment labels ($E^{(b)} = \pi_b(E)$).
5. **Intervention Survival (G7b)**:
   - Verified that acquisition perturbations ($0.8\times, 1.2\times$ gain, 20 dB noise, resampling) survive preprocessing and feature encoding ($D_{\text{raw}}, D_{\text{pre}}, D_{\text{input}} > 0$).
6. **Geometry Contract & Triad Diagnostic (G8)**:
   - Monitors the triad $(\text{MMD} \downarrow, \text{AUC}_E \downarrow, r_{\text{rel}} \ge 0.10)$ with relative effective rank $r_{\text{rel}} = \frac{r_{\text{eff}}}{d - 1}$, preventing low-rank collapse.

## Operator Specification
- **Biased IMQ Kernel**: $k(x, y) = \frac{c^2}{c^2 + \|x - y\|^2}$ with frozen $c^2 = 1.0$.
- **Constrained Representation**: $Z = \operatorname{LayerNorm}(f_\theta(X), \text{elementwise\_affine}=\text{False})$.
- **Exact Multi-Environment MMD²**: Mean pairwise biased MMD² across distinct environment groups with environment-balanced batch sampling ($E_{\text{batch}} \ge 2$, $n_e = \lfloor B / E \rfloor$, failing closed if $<2$ environments).
- **Objective**:
  $$L = L_{\text{BCE}} + \lambda_{\text{MMD}} L_{\text{MMD}}$$
  Logged separately: `loss_total`, `loss_bce`, `loss_mmd`, `weighted_mmd`, `var_z` ($V_Z = \frac{1}{d}\sum_j \text{Var}(Z_j)$), and validation AUROC.
