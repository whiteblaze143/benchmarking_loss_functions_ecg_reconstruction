# Experiment Tracker: Paper 01 (Distributional Recurrence Operator)

## Comprehensive 15-Variant Status Matrix

| # | Variant Identifier | Role | Status | Synthetic Verified | Grid Status | Target Metric |
|---|---|---|---|---|---|---|
| 1 | `mmd_recurrence_full` | Proposed Method | **DESIGNED** | Pending | Queued (9 cells) | Non-inferiority to SOTA |
| 2 | `euclidean_phase_recurrence` | Prior-Art Baseline | **SMOKE PASSED** | Verified | Queued (9 cells) | Benchmark reference |
| 3 | `mean_recurrence` | Centroid Baseline | **SMOKE PASSED** | Verified | Queued (9 cells) | First-moment check |
| 4 | `linear_kernel_recurrence` | Linear Kernel | **DESIGNED** | Pending | Queued (9 cells) | Linear RKHS check |
| 5 | `imq_mmd_recurrence` | Heavy-Tailed Kernel | **DESIGNED** | Pending | Queued (9 cells) | Robustness to outliers |
| 6 | `linear_probe` | Sufficiency Probe | **SMOKE PASSED** | Verified | Queued (9 cells) | Linear separability |
| 7 | `mlp_probe` | Non-Linear Probe | **DESIGNED** | Pending | Queued (9 cells) | Spatial inductive bias |
| 8 | `raw_phase_cnn` | No-Recurrence Control | **DESIGNED** | Pending | Queued (9 cells) | Value of recurrence |
| 9 | `fixed_phase_permutation` | Coordinate Permutation | **DESIGNED** | Pending | Queued (9 cells) | Relabeling invariance |
| 10 | `recordwise_phase_permutation` | Semantic Falsification | **DESIGNED** | Pending | Queued (9 cells) | Must collapse AUROC |
| 11 | `cyclic_shift_no_aug` | Baseline Equivariance | **DESIGNED** | Pending | Queued (9 cells) | Measure anchor jitter |
| 12 | `cyclic_shift_aug` | Augmented Invariance | **DESIGNED** | Pending | Queued (9 cells) | Achievable invariance |
| 13 | `amplitude_removed` | Shape-Only Baseline | **DESIGNED** | Pending | Queued (9 cells) | Quantify voltage loss |
| 14 | `amplitude_appended` | Scale Disentangled | **DESIGNED** | Pending | Queued (9 cells) | $[R_{\text{shape}}, a]$ synergy |
| 15 | `label_permutation` | Negative Control | **DESIGNED** | Pending | Queued (9 cells) | Empirical null (0.50) |

---

## Pre-Production Milestone Checklist
- [ ] **Synthetic World Verification**:
  - [ ] Synthetic A: Distributional separation under identical means.
  - [ ] Synthetic B: Finite-sample MMD convergence rate.
  - [ ] Synthetic C: Cyclic shift equivariance test ($\Delta_{\text{cyclic}} \le 0.01$).
  - [ ] Synthetic D: Scale invariance and energy preservation.
- [ ] **Representation Stability Analysis**:
  - [ ] Odd-even split-half stability (NFS score).
  - [ ] Eigenvalue spectral correlation ($\rho_{\text{spectral}}$).
- [ ] **Prior-Art Baseline Implementation**:
  - [ ] Classical thresholded RP + RQA features (XGBoost).
  - [ ] Deep RP ResNet-18 replication.
- [ ] **Sparse-Lead Battery**:
  - [ ] Evaluation across 12, 6, 3, 2, 1, and ICM configurations.
  - [ ] Degradation curve computation ($k = 1, \dots, 12$).
- [ ] **Production Grid Execution**:
  - [ ] Target: Macro-AUROC $\ge 0.920$ on PTB-XL diagnostic superclasses (Folds 1–8 train, Fold 9 val, Fold 10 test).
