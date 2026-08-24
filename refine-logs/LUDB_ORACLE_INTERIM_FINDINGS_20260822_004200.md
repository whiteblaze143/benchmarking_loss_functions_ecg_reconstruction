# LUDB Oracle Interim Findings: MMD-only Slice

Updated: 2026-08-22 00:42 America/Toronto

These findings are provisional. Five of 160 ECG-AIM masks have completed the
fixed-clock LUDB evaluation, all at seed 42. The completed slice holds MSE on,
all other binary losses off, and varies the MMD kernel.

| Mask | MMD kernel | Pearson p05 ↑ | MSE p95 ↓ | Landmark error p95, mV ↓ | QRS RMSE p95, mV ↓ | QRS area error p95, mV·ms ↓ | T RMSE p95, mV ↓ | T area error p95, mV·ms ↓ | J error p95, mV ↓ | Validation Pearson ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000000 | none | 0.4232 | 0.01619 | 0.3574 | 0.2341 | 15.34 | 0.1055 | 14.56 | 0.0691 | 0.8191 |
| 1000001 | global RBF | 0.0848 | 0.02164 | 0.5864 | 0.3084 | 21.54 | 0.1388 | 15.11 | 0.0945 | 0.5212 |
| 1000002 | anatomical Laplacian | 0.3805 | 0.01846 | 0.3801 | 0.2554 | 12.81 | 0.1078 | 12.76 | 0.0875 | 0.8404 |
| 1000003 | anatomical multi-IMQ | 0.4359 | 0.01607 | 0.3299 | 0.2525 | 12.83 | 0.1138 | 14.58 | 0.0829 | 0.8260 |
| 1000004 | temporal K-means multi-IMQ | 0.4229 | 0.01981 | 0.4842 | 0.2776 | 15.41 | 0.1393 | 20.75 | 0.0902 | 0.8492 |

## Harsh interpretation

1. Reject global-RBF MMD (`1000001`). It is worse than MSE-only on every
   oracle endpoint and validation Pearson. Seven of eight paired record-level
   oracle bootstrap intervals are wholly negative; the T-area interval also
   trends worse but crosses zero.
2. Do not choose temporal K-means MMD (`1000004`) from validation Pearson. It
   is oracle-dominated by MSE-only and is significantly worse in MSE,
   landmark, QRS-window, T-window, T-area, and J-point tails.
3. Anatomical Laplacian (`1000002`) improves the point estimates for QRS and T
   area tails, but both paired 95% intervals cross zero. Its J-point tail is
   significantly worse than MSE-only.
4. Anatomical multi-IMQ (`1000003`) has the best point estimates for p05
   correlation, signal MSE, landmark voltage error, and QRS area within this
   slice. Those apparent improvements are not stable under paired record
   bootstrap, while its J-point degradation is stable.
5. For an extreme-robustness objective, MSE-only (`1000000`) is the current
   leader. This is not yet the loss configuration to develop: correlation,
   derivative, VCG, energy-distance, lead-consistency, and their interactions
   remain unevaluated.

Validation composite loss is not used for cross-mask selection because each
mask changes the composite objective and scale. Validation missing-lead
Pearson is shown as supporting evidence. LUDB oracle morphology remains the
primary decision panel when validation and oracle behavior conflict.
