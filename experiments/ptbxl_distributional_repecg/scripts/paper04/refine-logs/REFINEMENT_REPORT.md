# Refinement Report: Paper 04
## Comprehensive Architectural Diff & Evolution

### 1. From Raw Spectral Descriptors to Differentiable Delay Dynamics

| Dimension | Historical Legacy Approach | Modern Refined Paper 04 |
| :--- | :--- | :--- |
| **Input Signal** | Raw 32-sample lead voltage segments | Non-parametric Phase-KME $U \in \mathbb{R}^{16 \times 128}$ |
| **Representation** | 30D SVD/DMD eigenvalues + singular values | End-to-end differentiable latent projections $x \in \mathbb{R}^{16 \times 4}$ |
| **Embedding Engine** | Heuristic Nyström IMQ Kernel across beats | Differentiable cyclic delay operator solve $A \in \mathbb{R}^{16 \times 16}$ |
| **Fidelity Gate** | Spearman $\rho < 0.68$, error $> 0.81$ (FAILED) | Analytical solve error $< 10^{-12}$, exact gradient backprop |
| **Underlying Topology**| Discontinuous branch cuts, sorting jumps | Smooth $C_{16}$ cyclic group wrap-around ($16 \to 1$) |
| **Operator Dimension**| Heuristic singular values | $D_H = d L = 16$, perfectly matched to $T=16$ transitions |
| **Primary Baseline** | Linear probe on mean-pooled features | Paper 02 `PhaseCNN` on identical frozen $U$ |

### 2. Codebase Refinements Summary

1. **`src/repecg/common/models.py` (`HankelDynamicsModel`)**:
   - Implemented default $d_{\text{proj}}=4, L=4$ ($D_H=16$).
   - Implemented cyclic roll-based delay stacking with wrap-around $16 \to 1$.
   - Fixed matrix solve orientation: $A = \operatorname{solve}(C_{11}^{\text{reg}}, C_{21}^\top)^\top$.
   - Added dual solve $A = Y (X^\top X + \lambda I)^{-1} X^\top$ when $D_H > M$.
   - Implemented scale-relative ridge with absolute floor $\lambda = \max(10^{-4} \frac{\operatorname{Tr}(C_{11})}{D}, 10^{-6})$.
   - Implemented `operator_summary_probe` calculating 8 smooth invariants.
   - Implemented baseline dispatches (`paper02_phasecnn`, `flat_phase_mlp`, `phase_aware_linear_probe`, `linear_probe`).

2. **`src/repecg/common/variants.py` (`PAPER04_VARIANTS`)**:
   - Registered 12 variants covering proposed model, primary comparator, lag memory sweeps, topological controls, order permutations, and regularization sensitivity.

3. **`scripts/paper04/run_paper04_grid.sh`**:
   - Reconciled historical gate check, logging historical negative finding while running modern cyclic delay operators on Phase-KME representations.
