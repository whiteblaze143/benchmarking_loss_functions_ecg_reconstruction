# Geometry-specificity oracle result

- Decision: **FAIL**
- Cohort: PTB-XL fold 9 only (8 ECGs; 3 patients)
- Limb algebra audit: **FAIL**, max absolute residual 0.0015
- Training PCA rank-2 residual energy: 82.3224%

| System | Mean independent r | p05 independent r | Independent MSE | SemiSeg mIoU | QRS MAE ms |
|---|---:|---:|---:|---:|---:|
| G0_fixed_vcg | 0.884626 | 0.844276 | 0.03335782 | 0.665406 | 8.222 |
| P0_train_pca | 0.965584 | 0.957407 | 0.01026078 | 0.650047 | 9.889 |
| T0_patch10 | 0.829501 | 0.790716 | 0.03982792 | 0.638461 | 7.111 |

VCG − PCA paired patient bootstrap: -0.080958 "
f"[-0.130058, -0.040966].

The fixed VCG mean-correlation result is at the 100.0th percentile of the 100 frozen random rank-2 bases. The training PCA result is at the 100.0th percentile.

Neural action: Stop; source limb-algebra audit exceeded the frozen threshold.
