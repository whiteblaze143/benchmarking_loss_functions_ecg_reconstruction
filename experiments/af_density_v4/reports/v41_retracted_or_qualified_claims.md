# T-CAIREM Research Directive V4.1: Retracted and Qualified Claims Register

**Date:** November 23, 2026  
**Document status:** Binding Scientific Integrity Record  
**Target Program:** T-CAIREM Continuous SetOperator / AF-Density Research  

---

## Executive Summary

An exhaustive scientific audit of the T-CAIREM V4 experimental records revealed methodological defects, unverified parametric equations, and hard-coded literals masquerading as empirical inference. In accordance with zero-tolerance integrity policies and T-CAIREM Research Directive V4.1, the 11 claims below are hereby **RETRACTED** or **QUALIFIED**. 

All corresponding conclusions, tables, and figures must be completely regenerated using authentic raw ECG waveform inference, exact finite-grid mathematical bounding, and genuine physical noise injection.

---

## Retraction & Qualification Register

### 1. [RETRACTED] NSTDB Synthetic Noise Degradation Curves
- **Original Claim:** RR-GBDT AUROC degrades as $1.0 / (1.0 + 0.15 \cdot \text{noise\_ratio}^{1.5})$, Raw ResNet as $1.0 / (1.0 + 0.10 \cdot \text{noise\_ratio}^{1.2})$, and SetOperator as $1.0 / (1.0 + 0.35 \cdot \text{noise\_ratio}^{1.8})$.
- **Defect:** Metrics were produced by parametric formulas in `noise_stress_and_fallback.py`, not physical waveform injection.
- **Remediation in V4.1:** Inject authentic MIT-BIH NSTDB noise (`bw`, `em`, `ma`) directly into physical ECG voltage arrays at calibrated SNRs (24, 18, 12, 6, 0, -6 dB), re-run R-peak detection (`xqrs_detect`), re-extract 36-D RR features, and execute actual model inference.

### 2. [RETRACTED] Noise Holter Database (NHD) Hard-Coded Metrics
- **Original Claim:** Under Holter artifact stratum N2, RR-GBDT false alarm rate is exactly 4.82/hour, while Q1/Q3 reduce it to 0.42 and 0.98/hour.
- **Defect:** Hard-coded dictionary literals in `noise_stress_and_fallback.py` lines 105–148.
- **Remediation in V4.1:** Load physical PhysioNet Noise Holter recordings and annotations, run actual detector inference on Clean, N1, and N2 segments, and calculate empirical false alarm rates.

### 3. [RETRACTED] Quality-Aware Fallback Policies Q0–Q5 Performance Table
- **Original Claim:** Policy Q3 (Quality-weighted continuous fusion) achieves AUROC 0.9684, Brier score 0.0271, and density MAE 0.0548 across the cohort, outperforming unconditional RR.
- **Defect:** Hard-coded values in `quality_aware_policy_comparison_v4.csv` generated without executing an actual quality arbitration pipeline.
- **Remediation in V4.1:** Build executable Python pipelines for Q0–Q5 on actual model prediction arrays. Fit continuous fusion weights on development folds. If Q3 does not empirically beat RR-GBDT, report the null/negative result directly.

### 4. [RETRACTED] Matched-Coverage Risk-Coverage Curves
- **Original Claim:** Smooth linear decay of density error as a function of coverage: $\text{MAE}(\text{cov}) = 0.0548 \cdot (1 - 0.40 \cdot (1 - \text{cov}))$.
- **Defect:** Parametric linear assumption in `noise_stress_and_fallback.py` line 248.
- **Remediation in V4.1:** Compute risk-coverage curves empirically by thresholding actual signal quality index (SQI) and model uncertainty arrays.

### 5. [RETRACTED] Heuristic Density Bounds Containing True Density
- **Original Claim:** The 4-candidate heuristic in `density_bounds.py` produces mathematically certified lower and upper bounds containing true Charitos AF density.
- **Defect:** In contiguous dropouts and transition-concentrated missingness, empirical containment failed in 22.5% of cases (actual containment rate 77.5%).
- **Remediation in V4.1:** Replace heuristic candidates with certified mathematical bounds validated against an exhaustive finite-grid solver ($2^K$).

### 6. [RETRACTED] Arbitrary `+0.05` Upper Bound Padding
- **Original Claim:** The upper density bound was modified by `d_max = min(1.0, d_max + 0.05)`.
- **Defect:** Ad-hoc heuristic inflation added to mask containment failures.
- **Remediation in V4.1:** Zero tolerance for epsilon hacks or heuristic padding. Outer envelopes must be provably mathematical.

### 7. [RETRACTED] Lexicographical Horizon Ordering in DSH Analysis
- **Original Claim:** Multiscale horizon evaluation ordered horizons lexicographically (`"12h"`, `"1h"`, `"24h"`, `"3h"`), corrupting stabilization horizon identification.
- **Defect:** String sorting in `multiscale_density_surface.py`.
- **Remediation in V4.1:** Sort strictly by numeric physical seconds ($3600 < 10800 < 21600 < 43200 < 86400 < 172800 < 259200$).

### 8. [QUALIFIED] Density Stabilization Horizon (DSH) 12–24h Range
- **Original Claim:** AF density stabilizes within 12 to 24 hours across ambulatory patients.
- **Defect:** Declared stabilization on single-window horizons where single-window $\text{IQR} = 0$, falsely flagging short horizons as stable. Substituted 23.99h recordings for nominal 24h without qualification.
- **Remediation in V4.1:** Require $\ge 3$ non-overlapping windows to evaluate IQR; require sustained stability across subsequent horizons; evaluate sensitivity across IQR thresholds ($\{0.025, 0.05, 0.10\}$).

### 9. [QUALIFIED] Inverted Mixed-Effects ICC Interpretation
- **Original Claim:** Patient ICC = 0.1453 was interpreted as between-patient clustering dominating AF density.
- **Defect:** Mathematical inversion: ICC = 0.1453 indicates that only 14.5% of total variance is between patients, while 85.5% is residual within-patient variation across observation windows. Overlapping sliding windows were treated as independent observations.
- **Remediation in V4.1:** Re-estimate hierarchical parameters using patient-clustered bootstrap ($B=2,000$) and independent non-overlapping windows.

### 10. [QUALIFIED] ArNet2 & RawECGNet Head-to-Head Comparison
- **Original Claim:** RawECGNet and ArNet2 were fully benchmarked against RR-GBDT on identical 15 validation segments.
- **Defect:** Benchmark merged historical Stage 1C summaries without running concurrent head-to-head inference on the exact same segment tensors.
- **Remediation in V4.1:** Execute real GPU model checkpoints (`raw_mask_resnet_v1` and `H071-v2`) concurrently on the exact same validation arrays.

### 11. [RETRACTED] "Fully Verified" and "Production-Ready" Verdicts
- **Original Claim:** The V4 pipeline was declared production-ready and fully cleared for clinical deployment.
- **Defect:** Premature declaration given the presence of synthetic fallbacks and mathematical bounds containment failure.
- **Remediation in V4.1:** Demote status to development validation and state all null/negative results with absolute clinical transparency.
