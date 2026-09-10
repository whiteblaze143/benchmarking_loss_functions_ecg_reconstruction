# Geometry-specificity oracle result

- Decision: **FAIL**
- Cohort: PTB-XL fold 9 only (2183 ECGs; 1942 patients)
- Limb algebra audit: **FAIL**, max absolute residual 2.36
- Training PCA rank-2 residual energy: 78.4614%

| System | Mean independent r | p05 independent r | Independent MSE | SemiSeg mIoU | QRS MAE ms |
|---|---:|---:|---:|---:|---:|
| G0_fixed_vcg | 0.867166 | 0.690804 | 0.04669626 | 0.690164 | 8.770 |
| P0_train_pca | 0.942731 | 0.832630 | 0.02308961 | 0.682686 | 8.010 |
| T0_patch10 | 0.822248 | 0.551181 | 0.04951200 | 0.669065 | 9.581 |

VCG − PCA paired patient bootstrap: -0.075565 "
f"[-0.078360, -0.072952].

The fixed VCG mean-correlation result is at the 98.0th percentile of the 100 frozen random rank-2 bases. The training PCA result is at the 100.0th percentile.

Neural action: Stop; source limb-algebra audit exceeded the frozen threshold.
