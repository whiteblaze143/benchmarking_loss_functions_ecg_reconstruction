# Quarantined Synthetic / Stale Results

This directory contains files that were quarantined because they contained synthetic simulations, heuristic formulas, or non-empirical data:

1. **`deep_clinical_endpoints_summary.csv` & `COMPREHENSIVE_EXPANDED_CLINICAL_ENDPOINTS_REPORT.md`**:
   - Quarantined because it contained synthetic/approximated endpoints (e.g. formula-based QTc Bazett/Fridericia, Cornell voltage, and individual EchoNext task AUROCs computed with arbitrary offsets rather than genuine inference).

2. **`clinical_mmrm_results.csv` & `CLINICAL_MMRM_AND_GEE_REPORT.md`**:
   - Quarantined because it simulated patient frailty and continuous residuals using `np.random.normal` and `hash(patient_id)` rather than actual patient-level inference tensors.

3. **`multivariable_clinical_associations.csv` & `MULTIVARIABLE_CLINICAL_ASSOCIATION_REPORT.md`**:
   - Quarantined because it generated synthetic predictor scores (`pred_arr_score = arr_auc * y + noise * np.random.normal(0, 1, N)`) rather than using genuine per-recording model predictions.

**Authoritative Empirical Replacement:**
All real, strictly empirical evaluations are stored in:
- `results/clinical_biomarkers_multids/clinical_metrics.db` (evaluation_version: `1lead_clinical_v1`)
- `results/clinical_biomarkers_multids/FINAL_CLINICAL_BENCHMARK_METRICS_MASTER.csv` (100% empirical metrics)
- `results/clinical_biomarkers_multids/empirical_clinical_endpoints_cheung_taxonomy.csv` (100% empirical metrics)
- `results/clinical_biomarkers_multids/clinical_metrics_summary.csv` and `clinical_metrics_summary_missing_leads_v2.csv`
