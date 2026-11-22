# Stage 1D Updated Statistical Analysis Plan (SAP)
**Protocol Identifier:** AF-DENSITY-SETOPERATOR-SAP-V1.4  
**Date:** November 22, 2026  
**Status:** BINDING SCIENTIFIC GOVERNANCE  

---

## 1. Study Rationale & Clinical Objectives
Long-term non-invasive monitoring for recurrent atrial fibrillation (AF) requires accurate quantification of both **AF Burden** (total fraction of time in AF) and **AF Density** (temporal dispersion vs aggregation of AF episodes, formulated by Charitos et al.). The clinical objective is to reconstruct episode timelines from multi-channel ECG recordings subject to intermittent channel dropout and noise.

This study benchmarks continuous neural operator architectures (Continuous SetOperator, Phase-KME) against convolutional waveform models (1D ResNet) and gradient-boosted decision trees (RR-GBDT) using authentic ECG recordings from the IRIDIA-AF cohort.

---

## 2. Cohort Partitioning & Custody Restrictions
The IRIDIA-AF cohort is partitioned at the patient level to prevent data leakage:

| Partition | Patients | Recordings | 30s Windows | AF Prevalence | Access Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Partition** | 90 | 181 | 504,777 | 24.3% | Unrestricted development, 5-fold patient-grouped CV |
| **Full Validation Partition** | 30 | 60 | 166,047 | 17.8% | Exploratory model validation, calibration tuning |
| **Informative Validation Cohort** | 11 | 15 | 42,431 | 16.5% | Paired multi-episode timeline & density benchmark |
| **Held-Out Test Partition** | 32 | 48 | 131,667 | Quarantined | **STRICT QUARANTINE: ZERO ACCESS AUTHORIZED** |

> [!CAUTION]
> Under no circumstances shall `test_windows.npz` or any test feature shards be accessed, read, or evaluated until all Stage 1D gates are independently audited and cleared.

---

## 3. Mathematical Endpoints & Metric Definitions

### 3.1 Primary Classification Endpoint: Window-Level AF Discrimination
- **Area Under Receiver Operating Characteristic (AUROC):** Calculated both as pooled AUROC across the cohort and mean within-patient AUROC.
- **Balanced Accuracy:** Calculated at threshold $\tau = 0.50$ (or optimal Platt-calibrated operating point):
  $$\text{BAcc} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right)$$
- **Brier Skill Score (BSS):** Assesses calibration and probabilistic accuracy relative to the non-informative training prevalence baseline ($b_0 = 0.1511$):
  $$\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\text{BS}_{\text{prevalence}}}$$
  where $\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$. A model with $\text{BSS} \le 0$ demonstrates zero predictive skill over predicting constant prevalence.

### 3.2 Primary Clinical Endpoint: Episode Detection & Boundary Resolution
- **Episode Sensitivity:** Proportion of reference AF episodes ($\ge 30\text{s}$) detected by the decoder.
- **Episode Positive Predictive Value (PPV):** Proportion of decoded AF episodes that overlap true reference AF episodes.
- **Mean Boundary Displacement:** Mean absolute difference (in seconds) between reference episode transition timestamps and decoded transition timestamps.
- **False Episode Rate:** Decoded false episodes per monitoring hour during confirmed sinus rhythm.

### 3.3 Secondary Endpoint: Charitos Continuous AF Density ($D_{\text{AF}}$)
AF density measures the spatial clustering of AF episodes across a normalized observation window $T$:
$$D_{\text{AF}} = \frac{2}{1 - b} \int_0^1 [m - L(m)] \, dm$$
where $b$ is the patient's global AF burden, $m \in [0, 1]$ is the normalized window size, and $L(m)$ is the lower envelope of the cumulative AF duration over all intervals of length $m$.
- **Translation-Invariance Requirement:** The metric must be provably invariant to arbitrary shifts in time ($D_{\text{AF}}(A(t + \Delta t)) = D_{\text{AF}}(A(t))$). The naive cumulative area formula $\frac{2}{T^2}\int_0^T |A(t) - bt|\, dt$ is strictly rejected due to location-dependent distortion.
- **Density Error Metric:** Mean Absolute Error (MAE) between predicted and reference continuous density:
  $$\text{MAE}_D = \frac{1}{K} \sum_{k=1}^K |D_{\text{pred}, k} - D_{\text{ref}, k}|$$
  evaluated strictly across segments where reference density is defined ($N_{\text{episodes}} \ge 1$ and $0 < b < 1$). Undefined density segments must be explicitly enumerated in the denominator.

---

## 4. Benchmark Architectures & Pre-Registered Baselines
All models must be evaluated against the following pre-registered baselines:

1. **B1: RR-GBDT (Primary Clinical Benchmark):** HistGradientBoostingClassifier trained on 36-D local RR interval statistics (WFDB `xqrs_detect`, 30s local windows, zero look-ahead).
2. **B2: RR-MLP:** 2-layer neural baseline (64-32 GELU) on identical 36-D inputs.
3. **B3: RR-Sequence:** Temporal median-smoothed GBDT predictions (5-window rolling median).
4. **B4: RR-GBDT Minimal:** Parsimonious 10-feature subset (CV, RMSSD, Histogram Entropy per channel + flags).
5. **B5: Constant Prevalence Baseline:** Constant prediction $p_i = 0.1511$ establishing zero-skill threshold.
6. **B6: 1D Convolutional ResNet (H071-v2):** End-to-end 1D ResNet trained on authentic 6000-sample raw ECG.
7. **B7: Continuous SetOperator (A001):** Canonical continuous neural operator on Phase-KME response representations.
8. **B8: Multimodal Hybrid (G061):** Cross-attention transformer fusing Phase-KME with 36-D RR features.
9. **B9: Residual Waveform Fusion (I082-v2):** Authentic ResNet + Continuous SetOperator with zero-initialized residual connection.

---

## 5. Statistical Hypotheses & Non-Inferiority Margins

### Hypothesis H1 (Rhythm Dominance)
Short-window RR interval statistics account for the primary discriminative signal in 30s ECG monitoring.
- **Falsification Criterion:** Continuous SetOperator morphology (A001) or Raw ResNet (H071-v2) outperforms RR-GBDT by $\ge 0.02$ AUROC.

### Hypothesis H2 (Morphological Signal Contribution)
In multimodal architectures (G061, I082-v2), Phase-KME morphology contributes negligible incremental discrimination over standalone rhythm or raw waveform branches.
- **Falsification Criterion:** Shuffling Phase-KME features in G061 causes a statistically significant degradation ($\Delta \text{AUROC} > 0.05, p < 0.01$).

### Hypothesis H3 (Lead-Dropout Robustness)
Unimodal rhythm models maintain balanced discrimination across single-lead dropout, whereas naive multimodal models experience asymmetric channel collapse.
- **Evaluation:** Evaluated across Pattern 0 (Lead 0 only), Pattern 1 (Lead 1 only), and Pattern 2 (Both leads) with genuine feature masking.

---

## 6. Pre-Registered Decision Gate Matrix
Advancement to Stage 2 (Architecture Scaling) or Stage 3 (Ablation Campaign) is contingent on clearing the following quantitative criteria:

| Gate | Description | Threshold | Action on Failure |
| :--- | :--- | :--- | :--- |
| **G1: Feature Lineage & Leakage** | Parity of raw voltage reconstruction & zero lookahead | Zero discrepancies, Scramble AUC $\approx 0.50$ | Quarantine feature pipeline |
| **G2: Calibration & BSS** | Positive Brier Skill Score against prevalence baseline | $\text{BSS} > 0.0$ | Reject uncalibrated pooling |
| **G3: Decoder Parity** | D2 HSMM passes 16 synthetic stress scenarios | Zero infinite loops, 0 false AF in sinus | Refactor emission strength |
| **G4: Charitos Invariance** | Mathematical translation-invariance of density metric | Difference $< 10^{-6}$ across shifts | Enforce lower-envelope $L(m)$ |
| **G5: Single-Lead Realism** | Single-lead evaluation with genuine channel masking | Both Lead 0 and Lead 1 evaluated | Reject unmasked 36-D inputs |
| **G6: Baseline Non-Inferiority** | Novel candidate beats or matches RR-GBDT | $\text{AUROC} \ge \text{AUROC}_{\text{GBDT}} - 0.02$ | Reframe around rhythm-first |
