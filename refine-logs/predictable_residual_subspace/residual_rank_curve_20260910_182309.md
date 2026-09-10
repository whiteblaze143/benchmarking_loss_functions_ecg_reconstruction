# Residual PCA rank characterization

`FIXED_VCG_SPECIFICITY = NOT_SUPPORTED`

| Rank | Training energy | Energy/rank | Oracle mean r | Marginal gain | Oracle p05 r | Independent MSE | SemiSeg mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.516660 | 0.516660 | 0.838629 | 0.273486 | 0.646504 | 0.04107087 | 0.646720 |
| 2 | 0.784614 | 0.392307 | 0.942731 | 0.104102 | 0.832630 | 0.02308961 | 0.682686 |
| 3 | 0.845925 | 0.281975 | 0.962113 | 0.019382 | 0.885456 | 0.01750973 | 0.678935 |
| 4 | 0.897727 | 0.224432 | 0.977669 | 0.015556 | 0.936412 | 0.01372294 | 0.682265 |
| 5 | 0.936810 | 0.187362 | 0.986054 | 0.008385 | 0.958022 | 0.00880077 | 0.684412 |
| 6 | 0.971619 | 0.161937 | 0.997643 | 0.011590 | 0.993678 | 0.00322705 | 0.684634 |

```text
rank_1_energy = 0.516659502
rank_2_energy = 0.784613603
rank_3_energy = 0.845925056
rank_4_energy = 0.897726972
rank_5_energy = 0.936809851
rank_6_energy = 0.971619258

rank_1_oracle_mean_r = 0.838628995
rank_2_oracle_mean_r = 0.942730856
rank_3_oracle_mean_r = 0.962112685
rank_4_oracle_mean_r = 0.977668888
rank_5_oracle_mean_r = 0.986053604
rank_6_oracle_mean_r = 0.997643276

K_STAR = 5
K_STAR_SELECTION_REASON = smallest rank reaching 90% training residual energy; oracle curve had not met the saturation condition
```

The rank gate uses all 2,183 fold-9 ECGs for independent-lead endpoints. The frozen dependent-limb QC rule (`max_abs_residual > 0.01 mV`) identifies 2 records; only dependent-limb sensitivity metrics exclude them.
