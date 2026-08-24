# Quarto Book Mechanical Audit

Generated: `2026-08-24T15:50:40.570078+00:00`

> This audit verifies structure and deployed artifacts, not scientific truth.

## Release gates

- Configured chapters: **19**
- Missing source chapters: **0**
- Missing rendered chapters: **0**
- Incomplete HTML files: **0**
- Source-newer-than-HTML chapters: **0**
- Missing deployed local resources: **0**
- Unlabeled executable blocks: **0**
- Duplicate executable labels: **0**

- Wrong renderer identity: **0**
- Executable chapters with zero outputs: **0**
- Distinct external runtime resources: **12**

## Chapter inventory

| Chapter | Headings | Python blocks | Outputs | Renderer | HTML | Stale | Missing resources |
|---|---:|---:|---:|---|---|---|---:|
| `index.qmd` | 6 | 0 | 0 | quarto-1.10.18 | complete | False | 0 |
| `15_live_results_observatory.qmd` | 4 | 2 | 2 | quarto-1.10.18 | complete | False | 0 |
| `16_three_lead_ecgaim_live.qmd` | 30 | 23 | 41 | quarto-1.10.18 | complete | False | 0 |
| `17_one_lead_wavelet_ssl_live.qmd` | 9 | 9 | 12 | quarto-1.10.18 | complete | False | 0 |
| `09_real_data_eda_and_diagnostics.qmd` | 24 | 18 | 32 | quarto-1.10.18 | complete | False | 0 |
| `11_dataset_atlas_ludb_isp_sunnybrook.qmd` | 20 | 13 | 12 | quarto-1.10.18 | complete | False | 0 |
| `05_zhejiang_delineation_tutorial.qmd` | 7 | 5 | 9 | quarto-1.10.18 | complete | False | 0 |
| `01_dataset_and_preprocessing.qmd` | 7 | 5 | 6 | quarto-1.10.18 | complete | False | 0 |
| `02_network_architectures.qmd` | 6 | 2 | 3 | quarto-1.10.18 | complete | False | 0 |
| `03_loss_function_formulations.qmd` | 13 | 2 | 5 | quarto-1.10.18 | complete | False | 0 |
| `05_clinical_metrics_and_delineation.qmd` | 22 | 3 | 3 | quarto-1.10.18 | complete | False | 0 |
| `04_regression_to_mean.qmd` | 17 | 2 | 2 | quarto-1.10.18 | complete | False | 0 |
| `06_fairness_robustness_sqi.qmd` | 17 | 2 | 2 | quarto-1.10.18 | complete | False | 0 |
| `08_factorial_loss_matrix_benchmarks.qmd` | 11 | 9 | 9 | quarto-1.10.18 | complete | False | 0 |
| `07_smartwatch_reconstruction_case_study.qmd` | 7 | 6 | 6 | quarto-1.10.18 | complete | False | 0 |
| `10_echonext_classifier_external_validation.qmd` | 9 | 6 | 7 | quarto-1.10.18 | complete | False | 0 |
| `12_exhaustive_model_analysis.qmd` | 7 | 2 | 3 | quarto-1.10.18 | complete | False | 0 |
| `13_engineering_ml_visual_atlas.qmd` | 9 | 7 | 7 | quarto-1.10.18 | complete | False | 0 |
| `14_cath_lab_clinical_trust_atlas.qmd` | 11 | 0 | 0 | quarto-1.10.18 | complete | False | 0 |

## Exact defects

```json
{
  "missing_sources": [],
  "duplicate_labels": [],
  "chapters_with_unlabeled_blocks": {},
  "chapters_with_missing_resources": {},
  "wrong_renderer": {},
  "zero_output_chapters": [],
  "external_resources": [
    "https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js",
    "https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js",
    "https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js",
    "https://cdn.plot.ly/plotly-3.3.1.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG",
    "https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js?features=es6",
    "https://doi.org/10.1161/CIR.0000000000001193",
    "https://doi.org/10.1161/CIRCULATIONAHA.125.077494",
    "https://doi.org/10.13026/2j37-ba56",
    "https://doi.org/10.13026/eegm-h675",
    "https://physionet.org/content/echonext/",
    "https://physionet.org/content/ludb/1.0.1/"
  ]
}
```
