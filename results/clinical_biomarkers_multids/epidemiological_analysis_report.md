# Comprehensive Epidemiological & Multivariable Statistical Analysis Report
*Methodology directly adapted from Ansari et al. (Circulation, 2025/2026)*

**Total Records Evaluated**: 32 database entries across 1 unique models.

## Module 1: Per-Target & Biomarker-Specific Aggregation Summary
| target                                        |   n_models |   mae_mean |   mae_sd |   pearson_mean |   r2_mean |   auroc_mean |   auprc_mean |   bias_mean |
|:----------------------------------------------|-----------:|-----------:|---------:|---------------:|----------:|-------------:|-------------:|------------:|
| ECGFounder_ANTERIOR_INFARCT                   |          1 |        nan |      nan |            nan |       nan |     0.581401 |    0.0195661 |         nan |
| ECGFounder_ANTEROLATERAL_INFARCT              |          1 |        nan |      nan |            nan |       nan |     0.801269 |    0.150704  |         nan |
| ECGFounder_ANTEROLATERAL_LEADS                |          1 |        nan |      nan |            nan |       nan |     0.806022 |    0.112735  |         nan |
| ECGFounder_ANTEROSEPTAL_INFARCT               |          1 |        nan |      nan |            nan |       nan |     0.921097 |    0.690384  |         nan |
| ECGFounder_ATRIAL_FIBRILLATION                |          1 |        nan |      nan |            nan |       nan |     0.977562 |    0.898259  |         nan |
| ECGFounder_ATRIAL_FLUTTER                     |          1 |        nan |      nan |            nan |       nan |     0.96766  |    0.422412  |         nan |
| ECGFounder_ELECTRONIC_ATRIAL_PACEMAKER        |          1 |        nan |      nan |            nan |       nan |     0.782604 |    0.0912366 |         nan |
| ECGFounder_INFERIOR_INFARCT                   |          1 |        nan |      nan |            nan |       nan |     0.739424 |    0.28411   |         nan |
| ECGFounder_LATERAL_INFARCT                    |          1 |        nan |      nan |            nan |       nan |     0.794857 |    0.215175  |         nan |
| ECGFounder_LEFT_ANTERIOR_FASCICULAR_BLOCK     |          1 |        nan |      nan |            nan |       nan |     0.91965  |    0.607646  |         nan |
| ECGFounder_LEFT_ATRIAL_ENLARGEMENT            |          1 |        nan |      nan |            nan |       nan |     0.639478 |    0.0342982 |         nan |
| ECGFounder_LEFT_BUNDLE_BRANCH_BLOCK           |          1 |        nan |      nan |            nan |       nan |     0.982134 |    0.917594  |         nan |
| ECGFounder_LEFT_POSTERIOR_FASCICULAR_BLOCK    |          1 |        nan |      nan |            nan |       nan |     0.806014 |    0.166932  |         nan |
| ECGFounder_LEFT_VENTRICULAR_HYPERTROPHY       |          1 |        nan |      nan |            nan |       nan |     0.760692 |    0.327105  |         nan |
| ECGFounder_LOW_VOLTAGE_QRS                    |          1 |        nan |      nan |            nan |       nan |     0.654059 |    0.251762  |         nan |
| ECGFounder_Macro_150                          |          1 |        nan |      nan |            nan |       nan |     0.849176 |    0.399546  |         nan |
| ECGFounder_NONSPECIFIC_INTRAVENTRICULAR_BLOCK |          1 |        nan |      nan |            nan |       nan |     0.680856 |    0.0830419 |         nan |
| ECGFounder_NORMAL_ECG                         |          1 |        nan |      nan |            nan |       nan |     0.817096 |    0.755645  |         nan |
| ECGFounder_PREMATURE_VENTRICULAR_COMPLEXES    |          1 |        nan |      nan |            nan |       nan |     0.97674  |    0.751539  |         nan |
| ECGFounder_QT_HAS_LENGTHENED                  |          1 |        nan |      nan |            nan |       nan |     0.945172 |    0.141945  |         nan |
| ECGFounder_RIGHT_ATRIAL_ENLARGEMENT           |          1 |        nan |      nan |            nan |       nan |     0.934049 |    0.232404  |         nan |
| ECGFounder_RIGHT_BUNDLE_BRANCH_BLOCK          |          1 |        nan |      nan |            nan |       nan |     0.91408  |    0.62124   |         nan |
| ECGFounder_RIGHT_VENTRICULAR_HYPERTROPHY      |          1 |        nan |      nan |            nan |       nan |     0.902638 |    0.137088  |         nan |
| ECGFounder_SEPTAL_INFARCT                     |          1 |        nan |      nan |            nan |       nan |     0.952848 |    0.742137  |         nan |
| ECGFounder_SINUS_BRADYCARDIA                  |          1 |        nan |      nan |            nan |       nan |     0.93996  |    0.303095  |         nan |
| ECGFounder_SINUS_RHYTHM                       |          1 |        nan |      nan |            nan |       nan |     0.806719 |    0.921641  |         nan |
| ECGFounder_SINUS_TACHYCARDIA                  |          1 |        nan |      nan |            nan |       nan |     0.990156 |    0.792137  |         nan |
| ECGFounder_SUPRAVENTRICULAR_TACHYCARDIA       |          1 |        nan |      nan |            nan |       nan |     0.99772  |    0.400093  |         nan |
| ECGFounder_VENTRICULAR_TACHYCARDIA            |          1 |        nan |      nan |            nan |       nan |     0.989877 |    0.271976  |         nan |
| ECGFounder_WITH_1ST_DEGREE_AV_BLOCK           |          1 |        nan |      nan |            nan |       nan |     0.904576 |    0.366462  |         nan |

---

## Module 2: Architecture-Stratified Analysis (UNet vs MS-VAE vs ECG-AIM)
| target                                        | architecture   |   mae_mean |   pearson_mean |   r2_mean |   auroc_mean |   auprc_mean |
|:----------------------------------------------|:---------------|-----------:|---------------:|----------:|-------------:|-------------:|
| ECGFounder_ANTERIOR_INFARCT                   | unet           |        nan |            nan |       nan |     0.581401 |    0.0195661 |
| ECGFounder_ANTEROLATERAL_INFARCT              | unet           |        nan |            nan |       nan |     0.801269 |    0.150704  |
| ECGFounder_ANTEROLATERAL_LEADS                | unet           |        nan |            nan |       nan |     0.806022 |    0.112735  |
| ECGFounder_ANTEROSEPTAL_INFARCT               | unet           |        nan |            nan |       nan |     0.921097 |    0.690384  |
| ECGFounder_ATRIAL_FIBRILLATION                | unet           |        nan |            nan |       nan |     0.977562 |    0.898259  |
| ECGFounder_ATRIAL_FLUTTER                     | unet           |        nan |            nan |       nan |     0.96766  |    0.422412  |
| ECGFounder_ELECTRONIC_ATRIAL_PACEMAKER        | unet           |        nan |            nan |       nan |     0.782604 |    0.0912366 |
| ECGFounder_INFERIOR_INFARCT                   | unet           |        nan |            nan |       nan |     0.739424 |    0.28411   |
| ECGFounder_LATERAL_INFARCT                    | unet           |        nan |            nan |       nan |     0.794857 |    0.215175  |
| ECGFounder_LEFT_ANTERIOR_FASCICULAR_BLOCK     | unet           |        nan |            nan |       nan |     0.91965  |    0.607646  |
| ECGFounder_LEFT_ATRIAL_ENLARGEMENT            | unet           |        nan |            nan |       nan |     0.639478 |    0.0342982 |
| ECGFounder_LEFT_BUNDLE_BRANCH_BLOCK           | unet           |        nan |            nan |       nan |     0.982134 |    0.917594  |
| ECGFounder_LEFT_POSTERIOR_FASCICULAR_BLOCK    | unet           |        nan |            nan |       nan |     0.806014 |    0.166932  |
| ECGFounder_LEFT_VENTRICULAR_HYPERTROPHY       | unet           |        nan |            nan |       nan |     0.760692 |    0.327105  |
| ECGFounder_LOW_VOLTAGE_QRS                    | unet           |        nan |            nan |       nan |     0.654059 |    0.251762  |
| ECGFounder_Macro_150                          | unet           |        nan |            nan |       nan |     0.849176 |    0.399546  |
| ECGFounder_NONSPECIFIC_INTRAVENTRICULAR_BLOCK | unet           |        nan |            nan |       nan |     0.680856 |    0.0830419 |
| ECGFounder_NORMAL_ECG                         | unet           |        nan |            nan |       nan |     0.817096 |    0.755645  |
| ECGFounder_PREMATURE_VENTRICULAR_COMPLEXES    | unet           |        nan |            nan |       nan |     0.97674  |    0.751539  |
| ECGFounder_QT_HAS_LENGTHENED                  | unet           |        nan |            nan |       nan |     0.945172 |    0.141945  |
| ECGFounder_RIGHT_ATRIAL_ENLARGEMENT           | unet           |        nan |            nan |       nan |     0.934049 |    0.232404  |
| ECGFounder_RIGHT_BUNDLE_BRANCH_BLOCK          | unet           |        nan |            nan |       nan |     0.91408  |    0.62124   |
| ECGFounder_RIGHT_VENTRICULAR_HYPERTROPHY      | unet           |        nan |            nan |       nan |     0.902638 |    0.137088  |
| ECGFounder_SEPTAL_INFARCT                     | unet           |        nan |            nan |       nan |     0.952848 |    0.742137  |
| ECGFounder_SINUS_BRADYCARDIA                  | unet           |        nan |            nan |       nan |     0.93996  |    0.303095  |
| ECGFounder_SINUS_RHYTHM                       | unet           |        nan |            nan |       nan |     0.806719 |    0.921641  |
| ECGFounder_SINUS_TACHYCARDIA                  | unet           |        nan |            nan |       nan |     0.990156 |    0.792137  |
| ECGFounder_SUPRAVENTRICULAR_TACHYCARDIA       | unet           |        nan |            nan |       nan |     0.99772  |    0.400093  |
| ECGFounder_VENTRICULAR_TACHYCARDIA            | unet           |        nan |            nan |       nan |     0.989877 |    0.271976  |
| ECGFounder_WITH_1ST_DEGREE_AV_BLOCK           | unet           |        nan |            nan |       nan |     0.904576 |    0.366462  |

---

## Module 3: Loss Function Main Effects & Interaction Synergies (Deriv * VCG)
Tests formula: `MAE ~ L_mse + L_deriv + L_vcg + L_st + L_phase + (L_deriv * L_vcg) + (L_st * L_phase)`

_Interaction models pending evaluation data._

---

## Module 4: Mixed-Effects Repeated Measures (MMRM) Models
_MMRM models pending evaluation completion._

---

## Module 5: Nonparametric Clustered Bootstrapped 95% CIs (BCa Method)
_Bootstrapping pending metric entries._

---

## Module 6 & 7: Bland-Altman Agreement & Multivariable Logistic Adjusted Odds Ratios (aOR)
_Bland-Altman & aOR summary pending evaluations._