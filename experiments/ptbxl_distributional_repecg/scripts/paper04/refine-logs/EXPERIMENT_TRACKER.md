# Experiment Tracker: Paper 04
## Verification Status & Protocol Log

### 1. 9 Synthetic Recovery Proof Worlds

| World | Test Name | Target Property | Success Threshold | Status |
| :---: | :--- | :--- | :--- | :---: |
| **W1** | `test_world_1_linear_system_recovery` | Linear system recovery & prediction error | $\|\hat{A} - A^*\|_F < 10^{-5}$, test residual $< 0.05$ | PENDING |
| **W2** | `test_world_2_finite_memory_delay_augmentation` | Memory order recovery on AR(2) oscillator | $e_{L=2} \ll e_{L=1}$, $e_{L=4} \approx e_{L=2}$ | PENDING |
| **W3** | `test_world_3_ridge_floor_and_degeneracy` | Scale-relative ridge floor on all-zero input | $\kappa < 10^5$, non-NaN/Inf forward/backward | PENDING |
| **W4** | `test_world_4_out_of_sample_order_sensitivity` | Out-of-sample temporal ordering ($T=500$) | $e_{\text{shuffle}} > 2 e_{\text{ordered}}$ | PENDING |
| **W5** | `test_world_5_historical_ineligibility_preservation` | Historical Nyström gate fail-closed audit | Status ineligible, $\rho < 0.70$, err $> 0.80$ | PENDING |
| **W6** | `test_world_6_cyclic_closure_advantage` | Cyclic vs open-chain boundary error on orbit | $e_{\text{cyclic}} < 0.1 \cdot e_{\text{open}}$ | PENDING |
| **W7** | `test_world_7_regularization_path_tradeoff` | Ridge bias-variance trade-off path | Minimum held-out error at intermediate $\lambda$ | PENDING |
| **W8** | `test_world_8_solve_orientation_and_gradients` | Float64 unit test & end-to-end backprop | Error $< 10^{-10}$, non-zero $\nabla W_{\text{proj}}$ | PENDING |
| **W9** | `test_world_9_pydmd_oracle_equivalence` | External PyDMD numerical cross-validation | Eigenvalues match PyDMD to $< 10^{-10}$ | PENDING |

### 2. Implementation Checklist
- [x] Correct batch matrix solve orientation ($A = \operatorname{solve}(C_{11}^{\text{reg}}, C_{21}^\top)^\top$)
- [x] Implement scale-relative ridge with floor $\lambda = \max(10^{-4} \frac{\operatorname{Tr}(C_{11})}{D}, 10^{-6})$
- [x] Resolve underidentification ($d=4, L=4 \implies D_H = 16$)
- [x] Implement cyclic $C_{16}$ delay state with wrap-around
- [x] Implement dual ridge formulation for $D_H > T$
- [x] Implement `operator_summary_probe` scalar dynamics head
- [x] Expand `PAPER04_VARIANTS` with all baseline comparators
- [x] Verify forward and backward pass across all variants in PyTorch
- [ ] Implement and verify all 9 synthetic recovery tests in `tests/test_paper04_synthetic_recovery.py`
- [ ] Run 1-epoch GPU verification on PTB-XL development representations
