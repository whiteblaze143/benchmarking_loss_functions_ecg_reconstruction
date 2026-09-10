# Geometry-specificity oracle result

- Decision: **LOW_RANK_ONLY**
- Cohort: PTB-XL fold 9 only (2 ECGs; 2 patients)
- Limb algebra audit: **PASS**, max absolute residual 0.0015
- Training PCA rank-2 residual energy: 89.1958%

| System | Mean independent r | p05 independent r | Independent MSE | SemiSeg mIoU | QRS MAE ms |
|---|---:|---:|---:|---:|---:|
| G0_fixed_vcg | 0.876636 | 0.841879 | 0.02406624 | 0.753338 | 11.000 |
| P0_train_pca | 0.964683 | 0.957354 | 0.01079318 | 0.773888 | 9.000 |
| T0_patch10 | 0.849972 | 0.814108 | 0.02249946 | 0.762380 | 6.000 |

VCG − PCA paired patient bootstrap: -0.088047 "
f"[-0.134810, -0.041283].

The fixed VCG mean-correlation result is at the 100.0th percentile of the 100 frozen random rank-2 bases. The training PCA result is at the 100.0th percentile.

Neural action: Proceed with B1/C1 for the low-rank claim; M1 is negative-control only.
