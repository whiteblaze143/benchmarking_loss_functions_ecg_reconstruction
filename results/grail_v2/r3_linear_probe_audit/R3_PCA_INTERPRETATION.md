# R3 PCA Interpretation

Estimand: **fixed-readout PCA clinical accessibility**. PC coordinates are re-standardized before the L2 logistic probe, so d95 is not a coordinate-invariant estimate of intrinsic information dimensionality. Sustained d95 is the conservative accessibility-complexity measure. Every median must be read with its n_reached/n_stable fields in the tier summary. P3 is exploratory (n=2).

Endpoint QC: median |delta|=0.008291, p95=0.066982, max=0.177962; 198/234 exceed 1e-3.

```text
    model   tier  n_concepts  n_stable  full_auroc_mean  d95_median  d95_sustained_median  auroc_k8_mean  auroc_k16_mean  auroc_k24_mean  auroc_k32_mean
       b1     P1           8         8         0.907609        20.0                  32.0       0.837608        0.881955        0.889933        0.892791
       b1     P2           4         4         0.962824        20.0                  20.0       0.882344        0.932742        0.941398        0.950241
       b1     P3           2         2         0.950074        12.0                  20.0       0.917929        0.926016        0.945105        0.944668
       b1 anchor          25        25         0.895133        24.0                  24.0       0.829993        0.872733        0.884729        0.890291
model_001     P1           8         8         0.903466        36.0                  48.0       0.706108        0.853341        0.877056        0.880130
model_001     P2           4         4         0.945007        24.0                  24.0       0.779921        0.910119        0.933947        0.936830
model_001     P3           2         2         0.947715        20.0                  20.0       0.885990        0.925871        0.939558        0.941029
model_001 anchor          25        25         0.876480        24.0                  24.0       0.732377        0.849148        0.869928        0.870921
model_101     P1           8         8         0.925609        24.0                  24.0       0.863568        0.897330        0.911772        0.912059
model_101     P2           4         4         0.948301        20.0                  32.0       0.870985        0.920599        0.921357        0.938148
model_101     P3           2         2         0.948455        12.0                  12.0       0.930035        0.941182        0.943959        0.945856
model_101 anchor          25        25         0.911123        24.0                  24.0       0.829738        0.872284        0.898018        0.902497
  model_m     P1           8         8         0.907567        20.0                  64.0       0.866101        0.872491        0.874734        0.884553
  model_m     P2           4         4         0.906855        32.0                  56.0       0.810527        0.831408        0.839267        0.850502
  model_m     P3           2         2         0.935791        16.0                  16.0       0.908877        0.922647        0.926979        0.926225
  model_m anchor          25        25         0.889510        16.0                  48.0       0.842987        0.859691        0.871069        0.875262
```

## model_101-b1

```text
        full_auroc_delta  d95_clinical_delta  d95_sustained_delta   8_delta  16_delta  24_delta  32_delta
tier                                                                                                     
P1              0.018000                2.25             0.571429  0.025960  0.015375  0.021838  0.019268
P2             -0.014523               -2.00             0.000000 -0.011359 -0.012143 -0.020041 -0.012092
P3             -0.001620                0.00            -8.000000  0.012106  0.015166 -0.001146  0.001188
anchor          0.015990               -1.56            -3.150000 -0.000255 -0.000448  0.013289  0.012206
```

## model_001-b1

```text
        full_auroc_delta  d95_clinical_delta  d95_sustained_delta   8_delta  16_delta  24_delta  32_delta
tier                                                                                                     
P1             -0.004142               11.25             7.200000 -0.131500 -0.028614 -0.012877 -0.012661
P2             -0.017818                2.00             0.000000 -0.102423 -0.022623 -0.007451 -0.013411
P3             -0.002359                8.00             0.000000 -0.031939 -0.000145 -0.005547 -0.003639
anchor         -0.018653               -0.64            -0.941176 -0.097616 -0.023585 -0.014801 -0.019370
```

## model_m-b1

```text
        full_auroc_delta  d95_clinical_delta  d95_sustained_delta   8_delta  16_delta  24_delta  32_delta
tier                                                                                                     
P1             -0.000042                5.50            24.666667  0.028492 -0.009464 -0.015199 -0.008238
P2             -0.055969               16.00            28.000000 -0.071817 -0.101334 -0.102132 -0.099739
P3             -0.014284                4.00            -4.000000 -0.009052 -0.003369 -0.018127 -0.018443
anchor         -0.005623                4.32            23.800000  0.012994 -0.013042 -0.013660 -0.015029
```

## model_101-model_m

```text
        full_auroc_delta  d95_clinical_delta  d95_sustained_delta   8_delta  16_delta  24_delta  32_delta
tier                                                                                                     
P1              0.018042               -3.25           -30.666667 -0.002533  0.024839  0.037038  0.027506
P2              0.041446              -18.00           -10.666667  0.060458  0.089191  0.082091  0.087647
P3              0.012664               -4.00            -4.000000  0.021158  0.018535  0.016980  0.019631
anchor          0.021613               -5.88           -23.227273 -0.013249  0.012593  0.026949  0.027235
```
