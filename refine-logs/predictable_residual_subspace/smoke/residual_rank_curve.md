# Residual PCA rank characterization

`FIXED_VCG_SPECIFICITY = NOT_SUPPORTED`

| Rank | Training energy | Energy/rank | Oracle mean r | Marginal gain | Oracle p05 r | Independent MSE | SemiSeg mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.493098 | 0.493098 | 0.884114 | 0.309721 | 0.866134 | 0.04146715 | 0.544632 |
| 2 | 0.855958 | 0.427979 | 0.960626 | 0.076512 | 0.954029 | 0.01789370 | 0.578627 |
| 3 | 0.972047 | 0.324016 | 0.984307 | 0.023681 | 0.976796 | 0.00609718 | 0.756076 |
| 4 | 0.985958 | 0.246490 | 0.986782 | 0.002475 | 0.978827 | 0.00508728 | 0.763339 |
| 5 | 0.994629 | 0.198926 | 0.992815 | 0.006033 | 0.989280 | 0.00414340 | 0.767136 |
| 6 | 0.998846 | 0.166474 | 0.999527 | 0.006712 | 0.999330 | 0.00107967 | 0.886483 |

```text
rank_1_energy = 0.493097777
rank_2_energy = 0.855958000
rank_3_energy = 0.972047440
rank_4_energy = 0.985958314
rank_5_energy = 0.994629411
rank_6_energy = 0.998846433

rank_1_oracle_mean_r = 0.884113975
rank_2_oracle_mean_r = 0.960626121
rank_3_oracle_mean_r = 0.984307194
rank_4_oracle_mean_r = 0.986782356
rank_5_oracle_mean_r = 0.992815261
rank_6_oracle_mean_r = 0.999527175

K_STAR = 3
K_STAR_SELECTION_REASON = smallest rank with >=90% training residual energy and G_4<0.005
```

The rank gate uses all 2,183 fold-9 ECGs for independent-lead endpoints. The frozen dependent-limb QC rule (`max_abs_residual > 0.01 mV`) identifies 0 records; only dependent-limb sensitivity metrics exclude them.
