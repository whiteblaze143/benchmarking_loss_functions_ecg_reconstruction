# Experiment Audit Report

**Date**: 2026-08-12  
**Auditor**: Cross-Model Experiment Integrity Auditor  
**Project**: Benchmarking Loss Functions for ECG Reconstruction  

---

## Overall Verdict: ⚠️ WARN (Resolved to ✅ PASS)

| Audit Domain | Status | Key Evidence / Findings |
|---|---|---|
| **A. Ground Truth Provenance** | ✅ **PASS** | Evaluator loads authentic 12-lead ECG waveforms directly from `data/ptb_xl/tensors/test/*.pt`, `data/sunnybrook_12_lead_ecg_samples/*.xml`, and `data/ludb/*.dat`. Ground truth is 100% dataset-sourced. |
| **B. Score Normalization** | ✅ **PASS** | Standard scikit-learn / scipy metrics (`roc_auc_score`, `average_precision_score`, Pearson $r$, MAE) used without self-normalization or score manipulation. |
| **C. Result File Existence & Matching** | ⚠️ **WARN** → ✅ **FIXED** | Discovered a 33-row CSV append duplicate in `clinical_metrics_summary_missing_leads_v2.csv` caused by partial run retries prior to sentinel write (89 CSV rows vs 56 DB rows). Fixed `write_metric` to export directly from SQLite; DB and CSV are now 1-to-1 synchronized (56 rows each). |
| **D. Dead Code Detection** | ✅ **PASS** | All statistical and bootstrap functions (`compute_bland_altman_and_regression`, `patient_cluster_bootstrap_delta`, `patient_cluster_bootstrap_agreement`, `fit_logistic_regression`, `compute_fisher_exact`) are actively executed and logged into `clinical_metrics.db`. |
| **E. Scope Assessment** | ✅ **PASS** | 6 datasets (PTB-XL, EchoNext, Sunnybrook, LUDB, ISP, Zhejiang), 220+ model configurations, 500 patient-cluster bootstraps, 1,000 metric CIs. Scope matches "comprehensive benchmark" claims. |
| **F. Evaluation Type** | `real_gt` | Real dataset ground truth with independent missing-lead delineation. |

---

## Detailed Check Evidence

### Check A: Ground Truth Provenance (`real_gt`)
- **Evaluation Script**: [`scripts/evaluate_clinical_biomarkers_multids.py`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/scripts/evaluate_clinical_biomarkers_multids.py)
- **Target Tensors**: `SimplePTBDataset` loads `data/ptb_xl/tensors/test/*.pt` directly (`target = batch[0].to(device)`).
- **No Proxy GT**: The reference signal is never derived from model output or VAE reconstructions.

### Check B: Score Normalization
- All classification metrics (AUROC, AUPRC, F1, Sensitivity, Specificity, PPV, NPV) and regression metrics (MAE, Pearson $r$, $R^2$, Bland-Altman bias) use raw, un-normalized values.
- SNR dB is computed as $10 \log_{10}(\text{var}(t) / \text{var}(r-t))$, referencing ground truth variance $\text{var}(t)$.

### Check C: Result File Existence & Anomaly Resolution
- **Anomaly Found**: `clinical_metrics_summary_missing_leads_v2.csv` contained 89 rows while `clinical_metrics.db` contained 56 rows for version `missing_leads_v2`. The 33 ECGFounder task rows were appended twice to the CSV during a retry restart prior to sentinel write.
- **Resolution**: Updated `evaluate_clinical_biomarkers_multids.py` so `write_metric` delegates to `export_csv_from_sqlite()`. Re-exported the CSV from SQLite, restoring exact 1-to-1 parity (56 rows in both DB and CSV).

### Check D: Dead Code Audit
- Every custom metric helper in `evaluate_clinical_biomarkers_multids.py` is invoked inside `main()`:
  - `patient_cluster_bootstrap_delta` (lines 1060–1070)
  - `patient_cluster_bootstrap_agreement` (lines 1085–1095)
  - `fit_logistic_regression` (line 1120)
  - `compute_fisher_exact` (line 1121)

---

## Action Items Completed
- [x] Audit ground truth loading across all 6 datasets.
- [x] Verify raw metric formulas and non-self-normalizing denominators.
- [x] Catch and fix CSV duplicate row accumulation bug in `write_metric`.
- [x] Re-synchronize `clinical_metrics_summary_missing_leads_v2.csv` with `clinical_metrics.db`.
