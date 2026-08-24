# Quarto Book Mechanical Audit

Generated: `2026-08-24T05:23:46.594676+00:00`

> This audit verifies structure and deployed artifacts, not scientific truth.

## Release gates

- Configured chapters: **19**
- Missing source chapters: **0**
- Missing rendered chapters: **0**
- Incomplete HTML files: **0**
- Source-newer-than-HTML chapters: **0**
- Missing deployed local resources: **0**
- Unlabeled executable blocks: **7**
- Duplicate executable labels: **0**

## Chapter inventory

| Chapter | Headings | Python blocks | Unlabeled | HTML | Stale | Missing resources |
|---|---:|---:|---:|---|---|---:|
| `index.qmd` | 6 | 0 | 0 | complete | False | 0 |
| `15_live_results_observatory.qmd` | 4 | 2 | 1 | complete | False | 0 |
| `16_three_lead_ecgaim_live.qmd` | 30 | 23 | 0 | complete | False | 0 |
| `17_one_lead_wavelet_ssl_live.qmd` | 9 | 9 | 0 | complete | False | 0 |
| `09_real_data_eda_and_diagnostics.qmd` | 24 | 18 | 0 | complete | False | 0 |
| `11_dataset_atlas_ludb_isp_sunnybrook.qmd` | 20 | 13 | 0 | complete | False | 0 |
| `05_zhejiang_delineation_tutorial.qmd` | 7 | 5 | 0 | complete | False | 0 |
| `01_dataset_and_preprocessing.qmd` | 7 | 5 | 0 | complete | False | 0 |
| `02_network_architectures.qmd` | 6 | 2 | 0 | complete | False | 0 |
| `03_loss_function_formulations.qmd` | 13 | 2 | 0 | complete | False | 0 |
| `05_clinical_metrics_and_delineation.qmd` | 22 | 3 | 2 | complete | False | 0 |
| `04_regression_to_mean.qmd` | 17 | 2 | 2 | complete | False | 0 |
| `06_fairness_robustness_sqi.qmd` | 17 | 2 | 1 | complete | False | 0 |
| `08_factorial_loss_matrix_benchmarks.qmd` | 32 | 12 | 1 | complete | False | 0 |
| `07_smartwatch_reconstruction_case_study.qmd` | 10 | 6 | 0 | complete | False | 0 |
| `10_echonext_classifier_external_validation.qmd` | 30 | 12 | 0 | complete | False | 0 |
| `12_exhaustive_model_analysis.qmd` | 7 | 2 | 0 | complete | False | 0 |
| `13_engineering_ml_visual_atlas.qmd` | 12 | 28 | 0 | complete | False | 0 |
| `14_cath_lab_clinical_trust_atlas.qmd` | 11 | 0 | 0 | complete | False | 0 |

## Exact defects

```json
{
  "missing_sources": [],
  "duplicate_labels": [],
  "chapters_with_unlabeled_blocks": {
    "15_live_results_observatory.qmd": [
      74
    ],
    "05_clinical_metrics_and_delineation.qmd": [
      101,
      165
    ],
    "04_regression_to_mean.qmd": [
      105,
      175
    ],
    "06_fairness_robustness_sqi.qmd": [
      155
    ],
    "08_factorial_loss_matrix_benchmarks.qmd": [
      1293
    ]
  },
  "chapters_with_missing_resources": {}
}
```
