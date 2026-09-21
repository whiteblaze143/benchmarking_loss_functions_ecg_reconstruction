# Experiment Plan: Paper 14 (V2.2)

## Objective
Validate environmental representation alignment via biased IMQ MMD² regularization, structurally eliminate latent scale inflation via `LayerNorm(elementwise_affine=False)`, and audit claim boundaries across synthetic and admissibility gates before real-data training.

## Registered Variants
- **erm**: Standard Empirical Risk Minimization ($L = L_{\text{BCE}}$, `use_mmd=False`).
- **mmd**: Joint Supervised + Representation Alignment ($L = L_{\text{BCE}} + \lambda L_{\text{MMD}}$, `use_mmd=True`).
- **full**: Backward compatibility alias mapping to `mmd` (`use_mmd=True`).
- **linear_probe**: Backward compatibility alias mapping to linear ERM (`use_mmd=False`).
- **mmd_penalty_only**: Collapse diagnostic only ($L = L_{\text{MMD}}$), retained exclusively to demonstrate trivial $\text{Var}(Z) \to 0$ collapse.

## Verification Gate Stack (G0 to G8)
1. **G0 (Numerical MMD contract & synthetic null)**:
   - Identical samples produce $MMD^2 = 0$.
   - Shifted samples produce $MMD^2 > 0.05$.
   - Symmetry $MMD^2(X, Y) = MMD^2(Y, X)$.
   - Gradients are non-zero, finite, and backpropagate cleanly.
   - Batch environment count $<2$ fails closed.
   - Finite-sample null calibration for synthetic setup recorded ($Q_{50}=0.0151, Q_{99}=0.0156$). Real-data calibration uses matched within-representation permutation null ($E^{(b)} = \pi_b(E)$).
2. **G1 (Positive alignment world with environment probe)**:
   - Input contains signal $S$ ($Y = \mathbf{1}[S > 0]$) and shifting nuisance $N_e = U + \delta_e$.
   - MMD variant suppresses nuisance and aligns representations ($MMD_{\text{aligned}} < 0.65 \cdot MMD_{\text{ERM}}$) while preserving classification accuracy ($\text{Acc} \ge 85\%$).
   - Linear environment probe decodability drops: $\text{AUC}_E(Z_{\text{MMD}}) < \text{AUC}_E(Z_{\text{ERM}})$ (dropping from 0.765 to near-chance 0.538).
3. **G2 (Matched-marginal failure world - Claim Boundary PASS)**:
   - Environments have symmetric $Y \in \{-1, +1\}$ and $S = Y + \epsilon$ in $E_0$, $S = -Y + \epsilon$ in $E_1$.
   - Marginals match ($P(S|E_0) \approx P(S|E_1)$), yielding $MMD^2 < 0.005$ despite mechanism reversal ($\rho > 0.9$ in $E_0$ vs $\rho < -0.9$ in $E_1$).
   - Formally establishes that marginal MMD does not identify causal mechanisms.
4. **G3 (Collapse world & geometry audit)**:
   - Unconstrained MMD-only drives $\text{Var}(Z) \to 0$ ($< 0.01$) and $r_{\text{eff}} \le 1.05$.
   - Joint BCE + MMD preserves representation variance ($> 0.1$, $\text{Var} \approx 1.0$), effective rank ($\ge 1.0$), and task accuracy ($\ge 85\%$).
5. **G4 (Objective execution)**:
   - Tests $\|\nabla L_{\text{ERM}} - \nabla L_{\text{MMD}}\|_2 > 10^{-4}$ on identical weights and heterogeneous batch, preventing silent ERM fallback.
6. **G5 (Scale escape audit to finite-sample diagonal floor)**:
   - Proves unconstrained biased IMQ MMD under scale inflation $z \mapsto az$ converges to its diagonal floor $\lim_{a \to \infty} \widehat{\text{MMD}}_b^2 = \frac{1}{n} + \frac{1}{m} = 0.020$ without domain alignment.
   - Proves LayerNorm eliminates global scale escape ($M(a) = \text{const}$).
7. **G5b (Trainable-normalizer escape audit)**:
   - Proves `InvariantMechanismDiscoveryModel.norm` contains 0 parameters (`elementwise_affine=False`).
   - Proves $H \mapsto aH + b\mathbf{1}$ leaves $Z$ invariant for $a \ge 0.5$.
   - Proves end-to-end training contains no trainable post-norm scale parameters.
8. **G6 (Label-shift boundary world - Claim Boundary PASS)**:
   - Invariant class-conditionals $P(X \mid Y, E)$, but unequal label prevalence ($P(Y=1 \mid E_0)=0.9$ vs $P(Y=1 \mid E_1)=0.1$).
   - Marginal MMD evaluates to positive ($> 0.5$, observed 0.980) despite zero domain nuisance.
   - Formally establishes that marginal MMD cannot distinguish domain shift from label-mixture shift.
9. **G7 (Real-data environment provenance)**:
   - Freezes controlled same-record acquisition perturbation bank (`clean`, `gain_0.8`, `gain_1.2`, `noise_20db`, `resample_250hz`), guaranteeing $\Delta_{e, e', c} = 0$.
   - Disclaims hospital and unobserved domain generalization.
10. **G7b (Intervention-survival audit)**:
    - Verifies $D_{\text{raw}}, D_{\text{pre}}, D_{\text{input}} > 0.01$ across all acquisition transforms.
    - Confirms that global dataset-level normalization does not erase amplitude gain ($0.8\times, 1.2\times$).
11. **G8 (Representation geometry contract & comparative relative-rank triad)**:
    - Reports relative effective rank $r_{\text{rel}} = \frac{r_{\text{eff}}}{d - 1}$.
    - Audits triad: $\text{MMD} \downarrow, \text{AUC}_E \downarrow, r_{\text{rel}} \ge 0.10$.
