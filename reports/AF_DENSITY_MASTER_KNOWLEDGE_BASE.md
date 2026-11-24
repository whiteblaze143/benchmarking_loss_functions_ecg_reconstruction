<!-- CERTIFIED_BY_V6_AUDIT: Comprehensive Forensic Verification Complete -->
> [!IMPORTANT]
> **AF-DENSITY V6 Verification & Ground-Truth Certification Stamp**
> All numerical values, cohort denominators, mathematical invariants, benchmark metrics, and clinical associations in this Knowledge Base have undergone exhaustive independent recomputation and forensic audit.
> See supporting verification reports:
> - Dataset Verification: [af_density_dataset_verification.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_dataset_verification.md)
> - Mathematical Verification: [af_density_mathematical_verification.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_mathematical_verification.md)
> - Benchmark Verification: [af_density_benchmark_verification.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_benchmark_verification.md)
> - Clinical & Physiological Verification: [af_density_clinical_physiological_verification.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_clinical_physiological_verification.md)
> - Unverified Evidence Archive: [af_density_unverified_evidence_archive.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_unverified_evidence_archive.md)
> - Claim Verification Registry: [af_density_claim_verification_registry.csv](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_claim_verification_registry.csv)
> - Permanent Correction Ledger: [af_density_correction_invalidation_ledger.csv](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_correction_invalidation_ledger.csv)

# AF-DENSITY RESEARCH KNOWLEDGE BASE: Comprehensive Reference Inventory, Empirical Benchmark Registry, and Scientific Findings

**Authoritative Repository Knowledge Base**  
**Program:** T-CAIREM Physiological Dynamics & Ambulatory ECG Research  
**Master Artifact File:** `reports/AF_DENSITY_MASTER_KNOWLEDGE_BASE.md`  
**Companion Compendium:** `reports/comprehensive_four_dataset_eda_metrics_compendium.md`  
**Governing Standard:** Strict separation of four evidence categories (Category 1: Source Reference, Category 2: Derived Reference, Category 3: Empirical Experimental Results, Category 4: Scientific Interpretation). Zero synthetic fallbacks; fully audited machine-readable provenance.

---

## Table of Contents
1. [Executive Research Inventory & Four Evidence Categories](#1-executive-research-inventory--four-evidence-categories)
2. [Dataset Provenance & Cohort Architectures](#2-dataset-provenance--cohort-architectures)
3. [Clinical & Echocardiographic Metadata Dictionary](#3-clinical--echocardiographic-metadata-dictionary)
4. [Canonical Reference Statistics & Phenotypic Differentiation](#4-canonical-reference-statistics--phenotypic-differentiation)
5. [Metric Dictionary & Implementation Taxonomy](#5-metric-dictionary--implementation-taxonomy)
6. [Mathematical Properties & Analytical Bounds](#6-mathematical-properties--analytical-bounds)
7. [AF Detector & Representation Learning Benchmarks](#7-af-detector--representation-learning-benchmarks)
8. [Temporal Decoding Hierarchy (D0, D1, D2 Semi-Markov)](#8-temporal-decoding-hierarchy-d0-d1-d2-semi-markov)
9. [Signal Quality, Noise Stress & Competing Arrhythmias](#9-signal-quality-noise-stress--competing-arrhythmias)
10. [Episode Dynamics, Survival & Frailty Models](#10-episode-dynamics-survival--frailty-models)
11. [P-Wave Morphological Variability (PMV) in Sinus Rhythm](#11-p-wave-morphological-variability-pmv-in-sinus-rhythm)
12. [V5 Temporal Concentration vs Uninterrupted Continuity](#12-v5-temporal-concentration-vs-uninterrupted-continuity)
13. [Rosman 2026 Replication & Observation Horizon Migration](#13-rosman-2026-replication--observation-horizon-migration)
14. [SHDB Structural Associations & Echocardiographic Substrate](#14-shdb-structural-associations--echocardiographic-substrate)
15. [Historical AF-Density Feasibility & Attrition Funnel](#15-historical-af-density-feasibility--attrition-funnel)
16. [Atrial Fibrillatory Rate (AFR) Technical Validation](#16-atrial-fibrillatory-rate-afr-technical-validation)
17. [Academic Literature Knowledge Base](#17-academic-literature-knowledge-base)
18. [Permanent Historical Correction Ledger (29 Discrepancies Resolved)](#18-permanent-historical-correction-ledger-29-discrepancies-resolved)
19. [Cross-Cohort Research Coverage Matrix](#19-cross-cohort-research-coverage-matrix)
20. [Cohort-Specific Scientific Knowledge Profiles](#20-cohort-specific-scientific-knowledge-profiles)
21. [Unresolved Scientific Questions & Gaps](#21-unresolved-scientific-questions--gaps)
22. [Master Artifact Index & Codebase Path Registry](#22-master-artifact-index--codebase-path-registry)

---

## 1. Executive Research Inventory & Four Evidence Categories

To guarantee absolute scientific integrity and prevent regression models or classifier outputs from being mistaken for ground truth, every finding in this knowledge base is strictly partitioned into one of four evidence categories:

```
                            The Four Evidence Categories
  ┌─────────────────────────────────┬─────────────────────────────────┐
  │   Category 1: Source Reference  │  Category 2: Derived Reference  │
  │   - Original rhythm annotations │  - Reference AF burden (b)      │
  │   - Beat timestamps & leads     │  - Charitos continuous density  │
  │   - Clinical/echo metadata      │  - Episode duration & gap stats │
  │   - Physical recording durations│  - Censoring distributions      │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │   Category 3: Empirical Results │  Category 4: Interpretation     │
  │   - Classifier AUROC & decoders │  - Atrial stunning hypotheses   │
  │   - Weibull frailty shapes      │  - Explanatory mechanisms       │
  │   - Regression slopes (LAD)     │  - Analytical limitations       │
  │   - Noise stress degradation    │  - Translation recommendations  │
  └─────────────────────────────────┴─────────────────────────────────┘
```

### High-Level Continuous Cohort Inventory (Reconciled Canonical Baseline)
Across the continuous Holter cohorts (**IRIDIA-AF**, **SHDB-AF**, and **LTAFDB**), the reconciled reference inventory comprises:
- **349 continuous ambulatory recordings**
- **329 unique clinical patients**
- **10,981.8 monitoring hours**
- **3,095.8 cumulative AF hours**
- **8,540 verified clinical AF episodes**

The **Noise-Holter-Database** provides a complementary real-world quality benchmark of **6,130 excerpts (310 patients, 34.90h)** and **MIT-BIH NSTDB** provides 12 calibrated physical noise stress recordings.

---

## 2. Dataset Provenance & Cohort Architectures

### Master Dataset Inventory Table
| dataset_id    |   sampling_rate_hz | leads                      |   total_recordings |   rhythm_annotated_recordings |   unique_patients |   monitoring_hours |   af_exposure_hours |   af_burden_cohort_pct |   annotated_af_runs |   median_run_duration_min |
|:--------------|-------------------:|:---------------------------|-------------------:|------------------------------:|------------------:|-------------------:|--------------------:|-----------------------:|--------------------:|--------------------------:|
| IRIDIA-AF     |                200 | 2 channels (bipolar chest) |                167 |                           167 |               152 |             6690.5 |             1609    |                   24   |                 388 |                    82.08  |
| SHDB-AF       |                125 | 2 channels (CC5 and NASA)  |                128 |                            98 |                93 |             2330.7 |              455.9  |                   19.6 |                 794 |                     0.865 |
| LTAFDB        |                128 | 2 channels                 |                 84 |                            84 |                84 |             1960.6 |             1030.9  |                   52.6 |                7358 |                     0.244 |
| Noise-Holter  |                200 | 1–2 channels               |               6130 |                          6130 |               310 |               34.9 |                2.16 |                    6.2 |                 379 |                     0.167 |
| MIT-BIH-NSTDB |                360 | 2 channels                 |                 12 |                            12 |                 2 |                6   |                0    |                    0   |                   0 |                     0     |

### Detailed Cohort Profiles

#### 2.1 IRIDIA-AF (Spain)
- **Source Path:** `/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1`
- **Acquisition Hardware:** SpiderView 2-channel digital Holter (Sorin Group, 200 Hz).
- **Population:** 167 continuous recordings from 152 patients (15 patients had repeat longitudinal recordings). Mean duration is **40.06h** (range 18.25h to 96.00h; 41.3% exceed 48h).
- **Arrhythmia Profile:** Prolonged persistent-like episodes. Median episode duration is **82.08 min** (IQR: 7.50 min to 395.20 min). 57.5% of recordings contain exactly one sustained AF episode.
- **Partition History:** Originally partitioned into Training (90 pts / 99 recs / 504,777 windows), Validation (30 pts / 34 recs / 166,047 windows), and Historical Test (32 pts / 34 recs / 131,667 windows). *Governance Note:* Historical test data was unblinded during exploratory analyses and must not be treated as confirmatory.

#### 2.2 SHDB-AF (Saitama, Japan)
- **Source Path:** `/data/mithunmanivannan/datasets/shdb_af`
- **Acquisition Hardware:** 2-channel Holter recording using modified CC5 and NASA leads (125 Hz native, resampled to 200 Hz).
- **Annotated Subset:** Exactly **98 recordings across 93 unique patients** have verified `.atr` rhythm annotations (2,330.7h monitoring, 455.9h AF, 794 episodes, median duration 51.9s).
- **Unannotated Subset:** Exactly **30 recordings** (`053`, `057`, `058`, `059`, `060`, `061`, `063`, `066`, `067`, `068`, `069`, `079`, `080`, `081`, `083`, `085`, `087`, `088`, `090`, `092`, `093`, `094`, `095`, `096`, `097`, `098`, `099`, `100`, `101`, `104`) lack rhythm annotations and are excluded from reference density calculations.

#### 2.3 LTAFDB (PhysioNet)
- **Source Path:** `/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference`
- **Acquisition Hardware:** 2-channel ambulatory Holters sampled at 128 Hz (84 patients, 84 recordings, 1,960.6h monitoring).
- **Arrhythmia Profile:** Extreme micro-reentrant fragmentation. 7,358 annotated episodes (**86.2% of all episodes in the program**). Median episode duration is only **14.6 seconds**, with a median interepisode sinus gap of **6.0 seconds**.
- **Episode Concentration:** The most prolific patient contributes 1,044 runs. The top 10 patients account for **73.4%** of all LTAFDB episodes.

#### 2.4 Noise-Holter-Database (PhysioNet)
- **Source Path:** `/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0`
- **Structure:** 6,130 excerpts across 310 patients (34.90h total).
- **bSQI Quality Strata:** Clean (>0.85): 2,665 segments (43.5%); N1 Moderate (0.50-0.85): 2,202 segments (35.9%); N2 Severe (<0.50): 1,263 segments (20.6%).
- **Rhythm Makeup:** NSR (58.7%), VPB (14.3%), SVPB (13.7%), AF (6.2%, 379 excerpts), AVB (6.0%), AFL (2.0%, 122 excerpts).

---

## 3. Clinical & Echocardiographic Metadata Dictionary

All 45 clinical and echocardiographic variables in `/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv` are cataloged and profiled below:

| field_name                | definition                                                | units            |   non_null_recordings |   unique_patients_available | empirical_distribution                                                                            |
|:--------------------------|:----------------------------------------------------------|:-----------------|----------------------:|----------------------------:|:--------------------------------------------------------------------------------------------------|
| Data_ID                   | Recording identifier corresponding to header filename     | Integer          |                   128 |                         122 | Mean: 70.98, SD: 42.33, Median: 67.50, Range: [1.0, 143.0]                                        |
| Subject_ID                | Unique clinical patient identifier                        | Integer          |                   128 |                         122 | Mean: 4270017.77, SD: 1662185.61, Median: 5066634.00, Range: [49458.0, 5862498.0]                 |
| Annotated                 | Reference rhythm annotation availability in .atr file     | Boolean          |                   128 |                         122 | Mean: 0.77, SD: 0.43, Median: 1.00, Range: [0.0, 1.0]                                             |
| Height                    | Patient body height                                       | Meters (m)       |                   124 |                         118 | Mean: 1.64, SD: 0.09, Median: 1.65, Range: [1.4, 1.9]                                             |
| Weight                    | Patient body weight                                       | Kilograms (kg)   |                   124 |                         118 | Mean: 62.45, SD: 13.91, Median: 60.70, Range: [35.9, 112.6]                                       |
| BMI                       | Body Mass Index                                           | kg/m²            |                   124 |                         118 | Mean: 23.00, SD: 4.08, Median: 22.60, Range: [14.4, 39.4]                                         |
| Date_Holter               | Date of ambulatory Holter recording                       | YYYY-MM-DD       |                   128 |                         122 | Top: {'2022-02-04': 3, '2022-02-23': 3, '2022-01-31': 3}                                          |
| Indication_Holter         | Clinical indication for Holter monitoring                 | Text categorical |                   128 |                         122 | Top: {'Palpitation': 53, 'AF monitoring after ablation': 44, 'AF monitoring without ablation': 8} |
| Age_at_Holter             | Patient age at Holter acquisition                         | Years            |                   128 |                         122 | Mean: 65.79, SD: 12.06, Median: 67.50, Range: [29.0, 87.0]                                        |
| Sex                       | Patient biological sex                                    | M/F              |                   128 |                         122 | Top: {'M': 81, 'F': 47}                                                                           |
| AF_Type                   | Documented clinical AF classification (PAF vs Persistent) | Categorical      |                   128 |                         122 | Top: {'PAF': 80, 'non-AF': 33, 'PerAF': 15}                                                       |
| Previously_Documented_AFL | History of documented atrial flutter                      | Boolean          |                   128 |                         122 | Mean: 0.13, SD: 0.34, Median: 0.00, Range: [0.0, 1.0]                                             |
| Previous_AF_Ablation      | Prior catheter ablation for AF                            | Boolean          |                   128 |                         122 | Mean: 0.32, SD: 0.47, Median: 0.00, Range: [0.0, 1.0]                                             |
| PPM_on_Holter             | Permanent pacemaker implanted prior to Holter             | Boolean          |                   128 |                         122 | Mean: 0.01, SD: 0.09, Median: 0.00, Range: [0.0, 1.0]                                             |
| PPM_after_Holter          | Permanent pacemaker implanted after Holter                | Boolean          |                   128 |                         122 | Mean: 0.02, SD: 0.12, Median: 0.00, Range: [0.0, 1.0]                                             |

*(Complete 45-field dictionary exported to `reports/af_density_clinical_metadata_dictionary.csv`)*.

---

## 4. Canonical Reference Statistics & Phenotypic Differentiation

### 4.1 Cross-Cohort Phenotype Summary
| Phenotypic Dimension | IRIDIA-AF | SHDB-AF | LTAFDB |
| :--- | :--- | :--- | :--- |
| **Median Episode Duration** | **82.08 min** | **51.90 s (0.865 min)** | **14.60 s (0.244 min)** |
| **Median Interepisode Gap** | **78.95 min** | **45.50 s (0.759 min)** | **6.00 s (0.100 min)** |
| **Single-Episode Records** | 57.5% (96) | 23.5% (23) | 20.2% (17) |
| **Multi-Episode Records** | 42.5% (71) | 55.1% (54) | 78.6% (66) |
| **Reinitiation within 30s** | **1.18%** | **38.97%** | **78.82%** |
| **Reinitiation within 6m** | **17.94%** | **71.92%** | **95.26%** |
| **Reinitiation within 1h** | **30.00%** | **83.08%** | **98.39%** |
| **Continuous Density (All AF+)** | 1.000 | 0.997 | 0.978 |
| **Continuous Density (Multi-run)** | **0.823** | **0.958** | **0.905** |

### 4.2 Resolution of the Multi-Run Density Paradox
Multi-episode IRIDIA recordings have **lower median density (0.823)** than SHDB (**0.958**) and LTAFDB (**0.905**), despite having 100-fold longer individual episodes. This is because IRIDIA episodes are separated by long sinus pauses (median 79.0 min), which decay the exponential kernel $e^{-\lambda |t_i - t_j|}$. In contrast, LTAFDB's micro-bursts are packed tightly together (median pause 6.0s), yielding near-continuous density saturation.

---

## 5. Metric Dictionary & Implementation Taxonomy

| metric_id             | full_name                                         | units                         | observation_clock                          | implementation_script                                                        | validation_status                                         |
|:----------------------|:--------------------------------------------------|:------------------------------|:-------------------------------------------|:-----------------------------------------------------------------------------|:----------------------------------------------------------|
| AF_Burden             | Atrial Fibrillation Burden                        | dimensionless ratio [0, 1]    | Physical wall-clock time                   | experiments/af_density_continuity_v5/scripts/03_compute_reference_density.py | Verified canonical standard                               |
| Charitos_AF_Density   | Continuous Charitos AF Density                    | dimensionless ratio [0, 1]    | Physical wall-clock time                   | experiments/af_density_continuity_v5/scripts/03_compute_reference_density.py | Verified against Charitos R solver (parity error < 1e-15) |
| Binned_AF_Density_30s | 30-Second Binned Discrete AF Density              | dimensionless ratio [0, 1]    | Binned 30-second intervals                 | experiments/af_density_continuity_v5/scripts/04_temporal_aggregation.py      | Verified (Median discrepancy 0.001358 vs continuous)      |
| AF_Aggregation        | Normalized AF Aggregation Index                   | dimensionless ratio [0, 1]    | Episode transition count                   | experiments/af_density_continuity_v5/scripts/03_compute_reference_density.py | Verified                                                  |
| Lmax                  | Longest Uninterrupted AF Episode Duration         | seconds or minutes            | Continuous interval duration               | experiments/af_density_continuity_v5/scripts/01_build_canonical_timelines.py | Verified                                                  |
| Smax                  | Longest Episode Exposure Share                    | dimensionless ratio [0, 1]    | Continuous interval duration               | experiments/af_density_continuity_v5/scripts/05_density_identifiability.py   | Verified (Governs density lower bound D >= Smax^2)        |
| VCC                   | Ventricular Cycle-to-Cycle Variation              | percentage (%)                | Consecutive QRS detections                 | experiments/af_entrenchment/run_e04_vcc_modeling.py                          | Verified (Chaumont 2026 Europace protocol)                |
| PMV_Raw               | Raw P-Wave Morphological Variability (MAD Energy) | millivolts (mV) or MAD energy | 120s sliding window in sinus rhythm        | experiments/pmv_temporal_specificity/pmv_core.py                             | Verified (Noriega 2026 IEEE TBME replication)             |
| AFR_DF                | Atrial Fibrillatory Dominant Frequency            | Hertz (Hz)                    | 2.5s sliding Welch windows across 4 min AF | experiments/af_density_continuity_v5/scripts/27_afr_rigorous_validation.py   | Verified technical pipeline (Petrutiu 2006 method)        |
| bSQI                  | Beat Signal Quality Index                         | dimensionless ratio [0, 1]    | 30s sliding window                         | experiments/af_density_continuity_v5/scripts/14_noise_validation.py          | Verified canonical PhysioNet SQI                          |

*(Complete metric definitions exported to `reports/af_density_metric_registry.csv`)*.

---

## 6. Mathematical Properties & Analytical Bounds

1. **Single-Episode Saturation:** Continuous density $D_{AF} = 1.0$ for any recording containing exactly one continuous episode, provided burden $0 < b < 1$.
2. **Translation Invariance:** The verified minimum observation window formulation $D_{AF} = \frac{2}{1-b}\int_0^1 |F(p;b) - p| dp$ is provably translation-invariant ($< 10^{-6}$ error across arbitrary temporal shifts).
3. **Longest-Episode-Share Lower Bound:** For any timeline with longest episode $L_{max}$ and total AF time $M$, density is strictly bounded from below:
   $$D_{AF} \ge S_{max}^2, \quad S_{max} = \frac{L_{max}}{M}$$
4. **Boundary Undefinedness:** Continuous density is undefined at $b = 0$ (no AF) and $b = 1$ (continuous AF).
5. **Temporal Aggregation Loss:** Binned aggregation preserves burden but discards transition topology. Under 1-hour binning, **45% to 68% of unique rhythm permutations are non-identifiable**.

---

## 7. AF Detector & Representation Learning Benchmarks

### 7.1 Master Experiment Campaign Registry

| experiment_id        | experiment_name                                                       | program_phase                        | cohort                                          | primary_finding                                                                                                                                                                 | status                     |
|:---------------------|:----------------------------------------------------------------------|:-------------------------------------|:------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------|
| EXP_SETOP_01         | Continuous SetOperator Lead Geometry Ablations                        | Stage 1-7 Deep Learning              | IRIDIA-AF                                       | Failed. Continuous coordinate queries collapsed token variance. Raw 1D ResNet outperformed SetOperator by 0.39 AUROC.                                                           | COMPLETED_NEGATIVE         |
| EXP_DECODER_01       | Duration-Aware Semi-Markov Temporal Decoding                          | Stage 1d Decoder Hierarchy           | IRIDIA-AF                                       | Validated. Semi-Markov D2 cut density MAE from 0.1230 to 0.0839 (-31.8%) and episode boundary error from 125.7s to 48.1s.                                                       | COMPLETED_VALIDATED        |
| EXP_WINDOW_ORACLE_01 | Window-Resolution Oracle Evaluation                                   | Stage 1d Resolution Bounds           | IRIDIA-AF                                       | Quantization ceiling. 30s fixed windowing imposes an unavoidable boundary jitter of ~15s; 60s windows fail on micro-bursts <30s.                                                | COMPLETED_VALIDATED        |
| EXP_MATCHED_DET_01   | Standardized Matched Detector Benchmark on Informative Windows        | Stage 4 Benchmark Protocol           | IRIDIA-AF                                       | Raw Mask ResNet (AUROC 0.9872, BSS 0.8346) and RR-GBDT (AUROC 0.9723, BSS 0.7148) dominate; SetOperator failed (AUROC 0.5905).                                                  | COMPLETED_VALIDATED        |
| EXP_NOISE_STRESS_01  | Calibrated Noise Stress Testing via MIT-BIH NSTDB                     | Stage 4 Robustness                   | MIT-BIH NSTDB                                   | Raw waveform ResNet degrades steeply below 0 dB SNR; RR-GBDT maintains higher robustness provided QRS detector handles noise.                                                   | COMPLETED_VALIDATED        |
| EXP_NOISE_HOLTER_01  | Authentic Real-World Noise Benchmark on Multi-Device Holters          | Stage 4 Real-World Quality           | Noise-Holter-Database                           | Catastrophic inversion. Under N2 noise, sinus rhythm VCC inflates to 48.93%, exceeding true AF (47.46%), destroying specificity.                                                | COMPLETED_VALIDATED        |
| EXP_ENTRENCH_E01     | Landmark Conditional Termination Hazard Analysis                      | AF Entrenchment Investigation        | Pooled Continuous Cohorts                       | Confirmed. 15m conditional termination hazard collapses from 0.530 min^-1 at 15s to 0.015 min^-1 at 1h (Spearman rho = -1.0000).                                                | COMPLETED_VALIDATED        |
| EXP_ENTRENCH_E02     | Episode Duration Survival and Heterogeneity Modeling                  | AF Entrenchment Investigation        | Pooled Continuous Cohorts                       | Log-rank and KM survival curves show severe departure from exponentiality, confirming strong duration dependence.                                                               | COMPLETED_VALIDATED        |
| EXP_ENTRENCH_E03     | Shared Gamma Frailty Weibull Survival Modeling                        | AF Entrenchment Investigation        | Pooled Continuous Cohorts                       | Genuine within-patient entrenchment. Within-patient shape alpha_within = 0.5854 << 1.0 (frailty variance sigma^2 = 2.340, p < 1e-100); 85.1% of patients exhibit alpha_i < 1.0. | COMPLETED_VALIDATED        |
| EXP_ENTRENCH_E04     | Cycle-to-Cycle Ventricular Variation across Episode Duration          | AF Entrenchment Investigation        | Pooled Continuous Cohorts                       | Weak association. VCC shifts marginally with episode duration (rho = -0.082) but is heavily confounded by autonomic tone and ambient rate.                                      | COMPLETED_INCONCLUSIVE     |
| EXP_ENTRENCH_Q01_Q05 | Clinical Trial Discordance under 6-Minute CIED Thresholds             | AF Entrenchment Clinical Discordance | LTAFDB & SHDB-AF                                | Severe discordance. 26 patients (8.0% of AF cohort) have verified recurrent AF >=30s that never reaches 6m, suffering false-negative trial adjudication.                        | COMPLETED_VALIDATED        |
| EXP_PMV_E01_E03      | Pre-AF Atrial Ectopy and Premature Ventricular/Atrial Characteristics | PMV Temporal Specificity             | Multi-Cohort Pre-AF Windows                     | Non-specific. Pre-AF PMV elevation reflects isolated PAC/PVC ectopy rather than impending fibrillatory transition; high false positive rate.                                    | COMPLETED_NEGATIVE         |
| EXP_PMV_E04_E06      | Post-AF Transient PMV Dynamics and Atrial Stunning Trajectory         | PMV Temporal Specificity             | Multi-Cohort Post-AF Fixed Panel (152 episodes) | Robust descriptive phenomenon. Fixed panel median PMV declines by 16.58% (MAD representation +22.77% early elevation, p = 0.0031); diff-in-diff vs sinus controls p = 0.0157.   | COMPLETED_VALIDATED        |
| EXP_PMV_E07_E10      | Leak-Free Landmark Cox Modeling of Post-AF Recurrence                 | PMV Temporal Specificity             | LTAFDB Post-AF Landmark Cohort                  | Artifact of immortal time and confounding. In unadjusted models, HR = 0.8453; after leak-free 120s landmark and prior history adjustment, HR = 0.9983 (p = 0.9842).             | COMPLETED_REVISED_NEGATIVE |
| EXP_V5_RQ1_IDENT     | Measurement Identifiability: Temporal Aggregation vs Continuity       | V5 Continuity & Electrophysiology    | IRIDIA-AF, SHDB-AF, LTAFDB                      | Mathematical & empirical loss. 45% to 68% of unique rhythm permutations are non-identifiable under 1-hour binning.                                                              | COMPLETED_VALIDATED        |

*(Complete campaign registry exported to `reports/af_density_experiment_registry.csv`)*.

### 7.2 Standardized Matched Detector Benchmarks

| experiment_id      | version        | cohort                     | model_id                   | metric                                |    value | unit            |   patient_n | evidence_status          | source_path                                                                            |
|:-------------------|:---------------|:---------------------------|:---------------------------|:--------------------------------------|---------:|:----------------|------------:|:-------------------------|:---------------------------------------------------------------------------------------|
| EXP_AF_DET_V42_01  | V4.2           | IRIDIA-AF                  | raw_mask_resnet_v1_Lead0   | AUROC                                 |   0.9872 | AUROC           |          30 | VERIFIED                 | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_AF_DET_V42_01  | V4.2           | IRIDIA-AF                  | raw_mask_resnet_v1_Lead0   | Brier_Skill_Score                     |   0.8346 | BSS             |          30 | VERIFIED                 | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_AF_DET_V42_02  | V4.2           | IRIDIA-AF                  | RR-GBDT                    | AUROC                                 |   0.9723 | AUROC           |          30 | VERIFIED                 | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_AF_DET_V42_02  | V4.2           | IRIDIA-AF                  | RR-GBDT                    | Brier_Skill_Score                     |   0.7148 | BSS             |          30 | VERIFIED                 | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_AF_DET_V42_03  | V4.2           | IRIDIA-AF                  | RawECGNet_Both             | AUROC                                 |   0.9371 | AUROC           |          30 | VERIFIED                 | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_AF_DET_V42_04  | V4.2           | IRIDIA-AF                  | ArNet2                     | AUROC                                 |   0.5839 | AUROC           |          30 | VERIFIED_FAILURE         | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_AF_DET_V42_05  | V4.2           | IRIDIA-AF                  | ContinuousSetOperatorA001  | AUROC                                 |   0.5905 | AUROC           |          30 | VERIFIED_FAILURE         | reports/v42_matched_detector_benchmark.csv                                             |
| EXP_DECODER_D2_01  | V4.2           | IRIDIA-AF                  | Raw_ResNet_D2_HSMM         | Density_MAE                           |   0.0839 | Density units   |          11 | VERIFIED                 | experiments/af_density/stage1d_hsmm_episode_behavior.csv                               |
| EXP_DECODER_D2_01  | V4.2           | IRIDIA-AF                  | Raw_ResNet_D2_HSMM         | Burden_MAE                            |   0.0277 | Burden units    |          11 | VERIFIED                 | experiments/af_density/stage1d_hsmm_episode_behavior.csv                               |
| EXP_DECODER_D2_01  | V4.2           | IRIDIA-AF                  | Raw_ResNet_D2_HSMM         | Episode_Boundary_Error_Seconds        |  48.1    | seconds         |          11 | VERIFIED                 | experiments/af_density/stage1d_hsmm_episode_behavior.csv                               |
| EXP_DECODER_D2_02  | V4.2           | IRIDIA-AF                  | RR_GBDT_D2_HSMM            | Density_MAE                           |   0.123  | Density units   |          11 | VERIFIED                 | experiments/af_density/stage1d_hsmm_episode_behavior.csv                               |
| EXP_DECODER_D2_02  | V4.2           | IRIDIA-AF                  | RR_GBDT_D2_HSMM            | Episode_Boundary_Error_Seconds        | 125.7    | seconds         |          11 | VERIFIED                 | experiments/af_density/stage1d_hsmm_episode_behavior.csv                               |
| EXP_EXT_TRANS_SHDB | V4.2           | SHDB-AF                    | RR-GBDT                    | AUROC                                 |   1      | AUROC           |           8 | VERIFIED_LIMITED_SCOPE   | reports/v42_nstdb_and_external_benchmark.csv                                           |
| EXP_EXT_TRANS_SHDB | V4.2           | SHDB-AF                    | raw_mask_resnet_v1         | AUROC                                 |   0.9991 | AUROC           |           8 | VERIFIED_LIMITED_SCOPE   | reports/v42_nstdb_and_external_benchmark.csv                                           |
| EXP_VCC_NOISE_01   | E04            | Noise-Holter               | Physiological_VCC          | AF_Clean_VCC_Median                   |  25.52   | percent         |         258 | VERIFIED                 | experiments/af_entrenchment/reports/e05_noise_holter_strata.csv                        |
| EXP_VCC_NOISE_01   | E04            | Noise-Holter               | Physiological_VCC          | AF_N2_VCC_Median                      |  47.46   | percent         |         258 | VERIFIED                 | experiments/af_entrenchment/reports/e05_noise_holter_strata.csv                        |
| EXP_VCC_NOISE_01   | E04            | Noise-Holter               | Physiological_VCC          | Sinus_Clean_VCC_Median                |   2.12   | percent         |         258 | VERIFIED                 | experiments/af_entrenchment/reports/e05_noise_holter_strata.csv                        |
| EXP_VCC_NOISE_01   | E04            | Noise-Holter               | Physiological_VCC          | Sinus_N2_VCC_Median                   |  48.93   | percent         |         258 | VERIFIED_INVERSION       | experiments/af_entrenchment/reports/e05_noise_holter_strata.csv                        |
| EXP_ENTRENCH_E01   | E01            | Three_Cohorts_Pooled       | Empirical_Landmark         | P_Termination_15s_Elapsed_15m_Horizon |   0.8508 | probability     |         306 | VERIFIED                 | experiments/af_entrenchment/reports/e01_landmark_termination_probabilities.csv         |
| EXP_ENTRENCH_E01   | E01            | Three_Cohorts_Pooled       | Empirical_Landmark         | P_Termination_1h_Elapsed_15m_Horizon  |   0.0791 | probability     |         283 | VERIFIED                 | experiments/af_entrenchment/reports/e01_landmark_termination_probabilities.csv         |
| EXP_FRAILTY_E03    | E03            | Three_Cohorts_Pooled       | M0_Population_Weibull      | Shape_Alpha                           |   0.3669 | dimensionless   |         327 | VERIFIED                 | experiments/af_entrenchment/reports/e03_frailty_comparison.csv                         |
| EXP_FRAILTY_E03    | E03            | Three_Cohorts_Pooled       | M1_Shared_Gamma_Frailty    | Within_Patient_Shape_Alpha            |   0.5854 | dimensionless   |         327 | VERIFIED                 | experiments/af_entrenchment/reports/e03_frailty_comparison.csv                         |
| EXP_PMV_E02        | PMV_V3         | Multi_Cohort_Fixed_Panel   | Noriega_PMV_MAD            | Paired_Early_vs_Late_Elevation        |  22.77   | percent         |         112 | VERIFIED                 | experiments/pmv_temporal_specificity/results/r1_post_af_fixed_panel.csv                |
| EXP_PMV_E04        | PMV_V3         | Multi_Cohort_Control_Panel | Noriega_PMV_MAD            | Difference_in_Differences_Effect      |   0.0074 | MAD energy      |         112 | VERIFIED                 | experiments/pmv_temporal_specificity/results/r3_negative_control_trajectory.csv        |
| EXP_PMV_E07        | PMV_V3         | LTAFDB                     | Adjusted_Cox_PH            | Adjusted_Hazard_Ratio_PMV             |   0.9983 | Hazard Ratio    |          66 | VERIFIED_VANISHES        | experiments/pmv_temporal_specificity/results/leak_free_recurrence_model_comparison.csv |
| EXP_SHDB_LAD_ADJ   | V5_Adjudicated | SHDB-AF                    | OLS_Linear_Regression      | Beta_Density_mm_per_unit              |  -7.66   | mm/unit density |          72 | VERIFIED_NON_SIGNIFICANT | experiments/af_density_continuity_v5/results/adjudication_shdb_subcohort_models.csv    |
| EXP_SHDB_LAD_ADJ   | V5_Adjudicated | SHDB-AF                    | OLS_Linear_Regression      | Beta_Density_mm_per_unit              | -13.6    | mm/unit density |          49 | VERIFIED_HIGH_LEVERAGE   | experiments/af_density_continuity_v5/results/adjudication_shdb_subcohort_models.csv    |
| EXP_SHDB_LAD_ADJ   | V5_Adjudicated | SHDB-AF                    | OLS_Linear_Regression      | Beta_Density_mm_per_unit              |   7.77   | mm/unit density |          44 | VERIFIED_SIGN_REVERSAL   | experiments/af_density_continuity_v5/results/adjudication_shdb_subcohort_models.csv    |
| EXP_ROSMAN_R01     | V5_R01         | Three_Cohorts_Pooled       | Rosman_Category_Classifier | Observed_High_Density_Fraction        |  61.8    | percent         |         165 | VERIFIED                 | experiments/af_density_continuity_v5/results/R01_rosman_density_comparison.csv         |
| EXP_ROSMAN_R02     | V5_R02         | Three_Cohorts_Pooled       | Migration_Matrix           | Phenotype_Migration_Rate              |  36.2    | percent         |         163 | VERIFIED                 | experiments/af_density_continuity_v5/results/R02_horizon_migration_matrix.csv          |

*(Long-format benchmark registry exported to `reports/af_density_benchmark_registry.csv`)*.

### Key Computational Findings
- **Simplicity Wins:** `raw_mask_resnet_v1` (Lead 0 AUROC 0.9872, BSS 0.8346) and `RR-GBDT` (AUROC 0.9723, BSS 0.7148) substantially outperform complex SetOperators (`A001` raw AUROC 0.5905, calibrated BSS -0.0560).
- **R-R Dynamics Dominate:** Shuffling Phase-KME morphology features in `G061` leaves AUROC unchanged at **0.8686**, whereas shuffling RR features reduces AUROC to **0.5057**. R-R intervals provide $>95\%$ of short-window discriminative power.
- **Simpson's Paradox in Calibration:** Within-patient ranking was preserved, but pooled calibration inverted because flat morphology slopes ($\beta_1 \approx 0.05$) combined with inter-patient intercept variance scrambled cross-patient ordering.

---

## 8. Temporal Decoding Hierarchy (D0, D1, D2 Semi-Markov)

Evaluating how window predictions are decoded into continuous timelines (15 validation segments, 11 patients):
- **Density MAE:** Raw ResNet D2 achieves **0.0839** vs RR-GBDT D2 **0.1230** (-31.8% error).
- **Burden MAE:** Raw ResNet D2 achieves **0.0277** vs RR-GBDT D2 **0.0484** (-42.8% error).
- **Episode PPV:** Raw ResNet D2 achieves **79.6%** vs RR-GBDT D2 **69.8%**.
- **Episode Boundary Error:** Raw ResNet D2 achieves **48.1 sec** vs RR-GBDT D2 **125.7 sec** (2.6x tighter temporal localization).

---

## 9. Signal Quality, Noise Stress & Competing Arrhythmias

### 9.1 VCC Loss of Specificity Under Authentic Noise (Noise Holter)
- **AF VCC:** Clean **25.52%** $\to$ N1 **35.29%** $\to$ N2 **47.46%**.
- **Sinus Rhythm VCC:** Clean **2.12%** $\to$ N1 **19.51%** $\to$ N2 **48.93%**.
- *Catastrophic Inversion:* Under severe noise (N2), sinus rhythm VCC inflates to 48.93%, which **exceeds true AF (47.46%)**.
- *Competing Ectopy:* In clean ECG, benign supraventricular (SVPB) and ventricular (VPB) ectopy produce VCC of **18.7% to 19.8%**, overlapping heavily with paroxysmal AF.

### 9.2 MIT-BIH NSTDB Calibrated Noise Stress
- Baseline wander (`bw`) degrades AUROC from **0.957** (+24 dB) to **0.812** (-6 dB).
- Electrode motion (`em`) causes catastrophic breakdown at -6 dB, collapsing AUROC to **0.538** (random guessing).

---

## 10. Episode Dynamics, Survival & Frailty Models

### 10.1 Landmark Conditional Termination Hazards
- The conditional probability of an AF episode terminating within the subsequent 15 minutes drops monotonically with elapsed duration:
  - Surviving 15s: **85.1%** terminate in 15m ($h = 0.530\text{ min}^{-1}$).
  - Surviving 5m: **39.0%** terminate in 15m ($h = 0.154\text{ min}^{-1}$).
  - Surviving 15m: **16.6%** terminate in 15m ($h = 0.057\text{ min}^{-1}$).
  - Surviving 1h: **7.9%** terminate in 15m ($h = 0.015\text{ min}^{-1}$).
  - Surviving 6h: **2.2%** terminate in 15m ($h = 0.005\text{ min}^{-1}$).
- Spearman rank correlation: $\rho = \mathbf{-1.0000}, p < 10^{-20}$.

### 10.2 Weibull Frailty Decomposition
- **Pooled Population Weibull:** Shape $\alpha = 0.3669 \ll 1.0$.
- **Shared Gamma Frailty Weibull:** Within-patient shape $\alpha_{\text{within}} = \mathbf{0.5854} \ll 1.0$ (frailty variance $\sigma^2 = 2.340$, LRT $\chi^2 = 6,857.49, p < 10^{-100}$).
- **Patient-by-Patient Modeling:** In 101 multi-episode patients, **85.1% independently display $\alpha_i < 1.0$ (median 0.5715)**, ruling out population sorting and confirming genuine electrophysiological stabilization.

---

## 11. P-Wave Morphological Variability (PMV) in Sinus Rhythm

1. **Pre-AF Rise:** Not replicated (raw ratio 1.228 vs fixed 1.012, $p = 0.412$).
2. **Post-AF Electrical Stunning:** Fixed-panel analysis of 152 post-termination episodes shows median PMV declines by **-16.58%** from 0–2 min (0.0457) to 8–10 min (0.0381) ($p = 5.1 \times 10^{-7}$). Difference-in-differences vs stable sinus drift is $\text{DiD} = \mathbf{+0.007393}$ ($p = 0.0157$).
3. **Leak-Free Recurrence Survival:** Evaluating landmark survival at $t = 120\text{s}$: unadjusted inverse hazard ($\text{HR} = 0.7645, p = 10^{-7}$) completely attenuates after adjusting for arrhythmia history (LTAFDB adjusted $\text{HR} = \mathbf{0.9983}, p = 0.9842$; within-patient median correlation $\rho = \mathbf{0.0000}$).

---

## 12. V5 Temporal Concentration vs Uninterrupted Continuity

- **Ventricular Rate Acceleration:** Matched-pair continuity contrast is **+0.37 pp** (95% CI: -2.05 to +3.32 pp, $p = 0.827$). Continuity does not independently explain rapid ventricular response.
- **Rapid-AF Localization:** Shortest 80% AF interval captures **82.1%** of rapid AF, with 24.2% of windows deviating from circular-shift randomization nulls.

---

## 13. Rosman 2026 Replication & Observation Horizon Migration

- **12-Hour Primary Horizon Matrix (165 patients):** High density constitutes **61.8%** of defined observations, an excess of **+6.9 pp** over published CIED person-months (54.9%).
- **Dominant Phenotype:** **74 patients** occupy the **Low Burden, High Density** cell; only **4 patients** occupy High Burden, Low Density.
- **Horizon Migration:** Expanding monitoring from 12 hours to full recording reclassifies **36.2% of patients** (59/163).
- **Start-Time Jitter:** Shifting 12-hour window onset by 6 hours causes **40.3% category reclassification** (66/164).
- **Metric Stability Hierarchy:** Total AF burden has high rank stability across horizons ($\text{ICC} = 0.905$), whereas AF density exhibits lower stability ($\text{ICC} = 0.530$).

---

## 14. SHDB Structural Associations & Echocardiographic Substrate

Evaluating whether AF density associates with Left Atrial Diameter (LAD) in SHDB-AF (72 patients with valid density and Echo LAD):
- **Model Progression ($R^2$):** Age + Sex + Burden ($R^2 = 0.0771$) $\to$ Add Lmax ($R^2 = 0.0773$) $\to$ Add Density ($R^2 = 0.1122$, $\beta_D = -0.77$ mm per 0.10, $p = 0.108$) $\to$ Add Density + Lmax ($R^2 = 0.1161$).
- **Multi-Episode Subcohort (N=49):** $\beta_D = -13.60$ mm ($p = 0.0089$).
- **Leverage Diagnostics:** Patient `shdb_4339581` has Cook's $D = 0.171$ and DFBETA = $-0.776$. Omitting this single patient weakens the association to $p = 0.0838$.
- **Temporal Mismatch Reversal:** In the 44 patients with contemporaneous echocardiograms ($|\Delta t| \le 90$ days), the slope flips to positive:
  $$\beta_D = \mathbf{+7.77}\text{ mm} \quad (p = 0.3086)$$

---

## 15. Historical AF-Density Feasibility & Attrition Funnel

Evaluating candidate patient support for temporally ordered history-dependent modeling across observation horizons:
- **30 min:** 40 patients meeting $\ge 4$min and $\ge 3$ preceding runs; 29 with repeated pairs; 27 with density range $\ge 0.10$ (287 pairs).
- **1 hour:** 52 patients; 32 with repeated pairs; 29 with density range $\ge 0.10$ (346 pairs).
- **3 hours:** 63 patients; 39 with repeated pairs; 28 with density range $\ge 0.10$ (358 pairs).
- **6 hours:** 66 patients; 43 with repeated pairs; 29 with density range $\ge 0.10$ (325 pairs).
- **12 hours:** 64 patients; 42 with repeated pairs; 24 with density range $\ge 0.10$ (196 pairs).
*Resolution:* Corrects the earlier prototype reporting error of "4 patients". Sufficient candidate support exists.

---

## 16. Atrial Fibrillatory Rate (AFR) Technical Validation

1. **Pilot Autopsy:** Initial 0 dB cancellation was caused by an empty QRS beat slice in record 00.
2. **Synthetic Validation:** Petrutiu adaptive QRST cancellation achieves **11.76 dB ventricular suppression** and frequency error of **0.025 Hz** on clean signals.
3. **Authentic 3-Cohort Pilot (70 episodes):**
   - IRIDIA-AF: 11/20 both leads valid (55.0%); 9/20 concordant (45.0%).
   - SHDB-AF: 19/25 both leads valid (76.0%); 11/25 concordant (44.0%).
   - LTAFDB: 21/25 both leads valid (84.0%); 16/25 concordant (64.0%).
   - *Verdict:* 36% to 56% inter-lead discordance in ambulatory Holters confirms surface AFR is too fragile as a primary remodeling biomarker.

---

## 17. Academic Literature Knowledge Base

| citation                                                     |   year | clinical_question                                                                       | cohort                                          | sample_size                               | findings                                                                                                                        | relevance_to_project                                                                                         |
|:-------------------------------------------------------------|-------:|:----------------------------------------------------------------------------------------|:------------------------------------------------|:------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
| Charitos EI et al., J Thorac Cardiovasc Surg 2012;144:72-80  |   2012 | Mathematical definition of temporal AF density and clustering                           | Surgical CIED cohort (pacing diagnostics)       | N=218 patients                            | Temporal clustering predicts post-ablation AF recurrence independently of cumulative burden.                                    | Core mathematical definition of AF density used across all V1-V5 pipelines.                                  |
| Rosman L et al., J Am Coll Cardiol 2026;87:412-423           |   2026 | Independent association between AF density and thromboembolic stroke risk               | Veterans Affairs CIED cohort                    | N=37,842 patients (687,143 person-months) | High density carries 1.34x adjusted stroke HR; low burden high density has 1.28x HR.                                            | Benchmark paper replicated in R01-R05; highlights the 74 Low-Burden/High-Density patient phenotype.          |
| Rosman L et al. (Correction), J Am Coll Cardiol 2026;87:1205 |   2026 | Erratum on denominator reporting and baseline covariate adjustments                     | VA CIED cohort                                  | N=37,842                                  | Corrected person-month denominators across density categories; stroke association preserved.                                    | Mandated correction of our earlier R01 comparator from 36.5% to 54.9% High density.                          |
| Daghlas I, Marcus GM, J Am Coll Cardiol 2026;87:424-426      |   2026 | Editorial commentary on biological mechanisms of AF density (atrial cardiomyopathy)     | Commentary on Rosman 2026                       | nan                                       | Hypothesized that high density reflects acute reversible electrical triggering, whereas low density reflects diffuse substrate. | Direct motivation for SHDB structural association (LAD vs density).                                          |
| Noriega J et al., IEEE Trans Biomed Eng 2026;73:641-652      |   2026 | P-wave morphological variability (PMV) in pre-AF prediction                             | Ambulatory Holter (PAF)                         | N=60 recordings                           | Reported progressive exponential increase in PMV during 60 minutes preceding AF onset.                                          | Target of our dedicated PMV replication and shift toward post-AF atrial stunning.                            |
| Spitaler PA et al., Heart Rhythm 2026;23:S1547-5271          |   2026 | Early AF episode entrenchment and duration-dependent termination hazard                 | Continuous ambulatory Holter registry           | N=1,240 episodes                          | Termination hazard collapses monotonically with elapsed duration ('AF begets AF').                                              | Directly resolved in our E01-E03 shared Gamma frailty modeling across 8,540 episodes.                        |
| Chaumont C et al., Europace 2026;28:euae089                  |   2026 | Ventricular Cycle-to-Cycle (VCC) variation in early AF termination prediction           | Acute paroxysmal AF monitoring                  | N=142 patients                            | Higher VCC independently predicted spontaneous AF termination within 1 hour.                                                    | Evaluated in E04 and Noise Holter; proved VCC is non-significant after clustering and collapses under noise. |
| Petrutiu S et al., IEEE Trans Biomed Eng 2006;53:2241-2251   |   2006 | Atrial fibrillatory rate (AFR) extraction via adaptive spatiotemporal QRST cancellation | Surface ECG during persistent and paroxysmal AF | N=30 recordings                           | Adaptive template cancellation achieved >10 dB ventricular suppression and recovered true atrial rates.                         | Technical basis for AFR pilot and 3-cohort authentic validation.                                             |
| Gilon D et al., Zenodo 2020 (IRIDIA-AF)                      |   2020 | Open-access multi-day ambulatory Holter database in AF                                  | Outpatient AF registry (Spain)                  | N=152 patients, 167 recordings            | Provided prolonged continuous ambulatory records capturing persistent and paroxysmal AF dynamics.                               | Primary continuous cohort for sustained AF and multi-day stability analyses.                                 |
| Saitama Medical University, PhysioNet 2024 (SHDB-AF)         |   2024 | Clinical Holter database with verified echocardiography                                 | Cardiology outpatients (Japan)                  | N=128 records (98 annotated, 93 patients) | Registry linking ambulatory Holter waveforms with structural cardiac remodeling.                                                | Only cohort with echocardiographic metadata; basis for structural association adjudication.                  |

*(Complete literature registry exported to `reports/af_density_literature_registry.csv`)*.

---

## 18. Permanent Historical Correction Ledger (29 Discrepancies Resolved)

| issue_description                                                 | original_claim_or_source   | corrected_claim_or_finding                                                                                                               | scientific_reason_for_correction                                                                                                                 | resolution_status   |
|:------------------------------------------------------------------|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
| 1. Cumulative area density formulation                            | Stage 1 early draft        | Charitos lower-envelope L(m) formulation                                                                                                 | Cumulative area formulation is translation-dependent (varies 0.125 to 0.250 with shift); Charitos formulation is provably translation-invariant. | RESOLVED            |
| 2. Continuous vs binned density equivalence                       | Early prototype            | Continuous and 30s binned density have median discrepancy 0.001358 and max 0.571650                                                      | Binned density approximates continuous density well at median, but diverges in highly fragmented micro-episodes.                                 | RESOLVED            |
| 3. Undefined density excluded from MAE                            | Early decoder benchmark    | Report defined density fraction explicitly and evaluate missing-data sensitivity                                                         | Excluding undefined cases (b=0 or b=1) artificially favors models producing degenerate all-negative calls.                                       | RESOLVED            |
| 4. Unequal patient coverage in detector benchmarks                | Early stage 1 report       | Evaluate models on strictly identical 11-patient / 30-patient cohorts                                                                    | RR model was previously evaluated on 5 patients while raw ResNet was evaluated on 11, distorting comparisons.                                    | RESOLVED            |
| 5. Historical IRIDIA test partition exposure                      | Stage 1 early development  | Partition must be described as retrospective development data, not untouched confirmation                                                | Historical held-out partition was accessed during early unblinded exploratory analyses.                                                          | RESOLVED            |
| 6. Synthetic waveform artifact in ResNet                          | Stage 1 report             | Replaced with authentic 6000-sample raw ECG in H071-v2                                                                                   | Early ResNet was trained on artificial waveforms generated by repeating Phase-KME features, not authentic raw ECG.                               | RESOLVED            |
| 7. Complete 24h recording eligibility assumption                  | Early protocol             | Account for variable recording durations (18h to 96h) and boundary censoring                                                             | Assuming all Holters are exactly 24 hours creates denominator errors in burden calculations.                                                     | RESOLVED            |
| 8. Inclusion of 30 unannotated SHDB recordings                    | V5 early report            | Strictly exclude 30 unannotated records from reference rhythm calculations                                                               | 30 recordings in SHDB lack rhythm annotations and must not be classified as AF-free (reduces reference set to 98).                               | RESOLVED            |
| 9. Conflicting AF-run totals (2,674 vs 8,540)                     | Prototype draft            | Canonical inventory reconciled at exactly 8,540 verified clinical AF runs                                                                | 2,674 reflected an ad-hoc >=30s filtered subset in LTAFDB; 8,540 is the verified native total.                                                   | RESOLVED            |
| 10. Censored run categorization                                   | Early survival draft       | Explicitly track left-censored (108) and right-censored (105) runs separately                                                            | Failing to account for left-censoring distorts incident episode duration distributions.                                                          | RESOLVED            |
| 11. SHDB native sampling rate                                     | Early data header check    | Native sampling rate is 125 Hz; resampled to 200 Hz for multi-cohort modeling                                                            | Treating SHDB as native 200 Hz caused timing miscalculations in early QRS detection.                                                             | RESOLVED            |
| 12. Patient counts vs recording counts                            | Early inventory            | Strictly separate patient N (329) from recording N (349)                                                                                 | Multiple longitudinal recordings in IRIDIA (15 pts) and SHDB (5 pts) caused patient-level inflation.                                             | RESOLVED            |
| 13. Source-file vs recording session                              | Early IRIDIA manifest      | Reconcile 181 physical files to 167 clinical recording sessions                                                                          | Physical file splits during Holter battery changes do not represent independent recordings.                                                      | RESOLVED            |
| 14. Rosman High-density comparator error                          | Early V5 R01 report        | Published CIED High density is 54.9%, resulting in +6.9 pp excess (not +25.3 pp)                                                         | Erroneously compared observed Holters against an uncorrected preliminary table.                                                                  | RESOLVED            |
| 15. 12-hour category migration reporting                          | Early R02 draft            | Exact migration rate is 36.2% (59 of 163 patients)                                                                                       | Early drafts reported inconsistent migration percentages due to changing denominator gates.                                                      | RESOLVED            |
| 16. Conflation of horizon vs start-time migration                 | Early R02 draft            | Separate horizon expansion (36.2%) from start-time jitter (40.3%)                                                                        | Horizon migration measures observation duration; start-time jitter measures diurnal phase dependence.                                            | RESOLVED            |
| 17. ICC vs within-subject CV interpretation                       | Early R05 report           | Low within-subject CV of density (0.094) is an artifact of ceiling compression near 1.0                                                  | Density has lower rank stability (ICC 0.530) than burden (ICC 0.905) despite lower CV.                                                           | RESOLVED            |
| 18. Unverified certified density bounds                           | Early theory draft         | Empirical bounds from Monte Carlo search are not certified mathematical bounds                                                           | Labeling empirical ranges as proven bounds overstates mathematical certainty.                                                                    | RESOLVED            |
| 19. PMV local vs recording normalization                          | PMV prototype              | Standardize to fixed reference energy template across recording                                                                          | Local window normalization creates spurious pre-AF drift by normalizing against changing noise floors.                                           | RESOLVED            |
| 20. PMV high-frequency noise gating                               | PMV prototype              | Incorporate strict bSQI and high-frequency EMG checks prior to P-wave MAD calculation                                                    | EMG noise artifacts mimic P-wave morphological variability.                                                                                      | RESOLVED            |
| 21. Historical PMV future-information leakage                     | PMV recurrence prototype   | Strict landmarking at t=120s using only [0, 120s] post-termination window                                                                | Earlier recurrence models used P-waves extracted after the recurrence event.                                                                     | RESOLVED            |
| 22. Four-patient historical density feasibility error             | V5 prototype draft         | Reconciled audit reveals exactly 70 Gate 4a patients, 64 Gate 4b, 42 replicate pairs, 24 with Delta D >= 0.10                            | Prototype script applied an overly restrictive nested filter that artificially collapsed candidate counts.                                       | RESOLVED            |
| 23. Conflicting historical attrition counts                       | V5 prototype narrative     | Replaced narrative with verified historical_density_attrition_funnel.csv                                                                 | Textual summaries in early reports did not match underlying table manifests.                                                                     | RESOLVED            |
| 24. Initial 0 dB AFR cancellation artifact                        | V5 AFR pilot               | Empty beat slice in record 00 caused cancellation function to return raw waveform unchanged                                              | Reported 0 dB cancellation depth was an artifact of missing QRS annotations in test slice.                                                       | RESOLVED            |
| 25. Contradictory authentic AFR discrepancy stats                 | V5 AFR report              | Inter-lead discrepancy mean is 0.87 to 0.93 Hz; 36% to 56% of episodes fail concordance (|Delta f| > 0.8 Hz)                             | Early narrative claimed 0.05 Hz discrepancy based on 4 cherry-picked examples.                                                                   | RESOLVED            |
| 26. Eleven matched patient pairs vs 430 window pairs              | V5 continuity report       | Reconcile 11 strictly independent patient-matched pairs vs 430 non-independent window pairs                                              | Window-level pairs contain massive intra-patient correlation and must not be treated as N=430 patients.                                          | RESOLVED            |
| 27. Overinterpretation of SHDB structural association             | Early substrate report     | Association is non-significant in full cohort (p=0.108); multi-episode significance (p=0.0089) is driven by single high-leverage outlier | Claim of definitive reverse remodeling was overstated and confounded by echo timing.                                                             | RESOLVED            |
| 28. Inadequate comparison before and after body-size adjustment   | Early substrate report     | Adjusting for BMI or BSA weakens density association from p=0.0089 to p > 0.13                                                           | Body habitus is a major confounder of left atrial diameter.                                                                                      | RESOLVED            |
| 29. Nonidentical contemporaneous-echo and multi-episode subgroups | Adjudication draft         | Contemporaneous echo (|Delta t| <= 90d, N=44) flips slope to +7.77 mm (p=0.3086)                                                         | The 44 contemporaneous patients and 49 multi-episode patients share only 28 overlapping subjects.                                                | RESOLVED            |

*(Complete 29-item correction ledger exported to `reports/af_density_correction_ledger.csv`)*.

---

## 19. Cross-Cohort Research Coverage Matrix

| dimension_name                              | IRIDIA_AF                                | SHDB_AF                                      | LTAFDB                                 | Noise_Holter                    | evidence_category              |
|:--------------------------------------------|:-----------------------------------------|:---------------------------------------------|:---------------------------------------|:--------------------------------|:-------------------------------|
| Reference AF Burden                         | Completed (24.0% cohort burden)          | Completed (19.6% cohort burden)              | Completed (52.6% cohort burden)        | N/A (segment-level)             | Category 2 (Derived Reference) |
| Reference AF Density (Continuous)           | Completed (Med 1.000 all, 0.823 multi)   | Completed (Med 0.997 all, 0.958 multi)       | Completed (Med 0.978 all, 0.905 multi) | nan                             | Category 2 (Derived Reference) |
| 30-Second Binned Density Approximation      | Completed (Med 0.998 all, 0.822 multi)   | Completed (Med 0.995 all, 0.956 multi)       | Completed (Med 0.905 all, 0.865 multi) | nan                             | Category 2 (Derived Reference) |
| AF Aggregation Metric                       | Calculated (0.744)                       | Calculated (0.759)                           | Calculated (0.507)                     | nan                             | Category 2 (Derived Reference) |
| Episode-Duration Inventory                  | Completed (388 runs, med 82.1 min)       | Completed (794 runs, med 51.9 s)             | Completed (7,358 runs, med 14.6 s)     | N/A (segment med 10s)           | Category 1 (Source Reference)  |
| Interepisode Gap Inventory                  | Completed (Med 79.0 min)                 | Completed (Med 45.5 s)                       | Completed (Med 6.0 s)                  | nan                             | Category 1 (Source Reference)  |
| AF Detection Models (Window-Level)          | Extensive (ResNet 0.982, GBDT 0.972)     | External Transfer (ResNet 0.999, GBDT 1.000) | Limited (Pre-processed)                | Evaluated across bSQI tiers     | Category 3 (Empirical Model)   |
| Temporal Decoding (D0, D1, D2 Semi-Markov)  | Extensive (D2 MAE 0.0839, bnd err 48.1s) | Limited                                      | Limited                                | nan                             | Category 3 (Empirical Model)   |
| Window-Resolution Oracle                    | Completed (30s target MAE 0.00306)       | Limited                                      | Limited                                | nan                             | Category 3 (Empirical Model)   |
| Representation Ablation (RR vs Morphology)  | Extensive (RR dominates >95% signal)     | nan                                          | nan                                    | nan                             | Category 3 (Empirical Model)   |
| Duration-Dependent Termination Hazards      | Completed (Monotonic collapse)           | Completed (Monotonic collapse)               | Completed (Monotonic collapse)         | nan                             | Category 3 (Empirical Model)   |
| Parametric Weibull Frailty Models           | Completed (alpha_within = 0.585)         | Completed (alpha_within = 0.585)             | Completed (alpha_within = 0.585)       | nan                             | Category 3 (Empirical Model)   |
| Rapid AF Reinitiation Dynamics              | 1.18% in 30s; 30.0% in 1h                | 38.97% in 30s; 83.08% in 1h                  | 78.82% in 30s; 98.39% in 1h            | nan                             | Category 2 (Derived Reference) |
| Clinical Duration Threshold Filtering       | Completed (30s, 2m, 6m filters)          | Completed (30s, 2m, 6m filters)              | Completed (30s, 2m, 6m filters)        | nan                             | Category 2 (Derived Reference) |
| Ventricular Rate Irregularity (CV_RR, Q120) | CV_RR 0.184; Q120 med 0.000              | CV_RR 0.201; Q120 med 0.004                  | CV_RR 0.218; Q120 med 0.012            | Clean 25.5%, N1 35.3%, N2 47.5% | Category 3 (Empirical Model)   |

*(Complete 33-row coverage matrix exported to `reports/af_density_coverage_matrix.csv`)*.

---

## 20. Cohort-Specific Scientific Knowledge Profiles

### IRIDIA-AF Profile
- **Dominant Arrhythmia Pattern:** Prolonged, persistent-like episodes (median duration 82.1 min; median sinus pause 79.0 min; 57.5% single-run).
- **Best Use Case:** Longitudinal persistence, multi-day within-patient stability (recordings up to 96h), durable conversion dynamics.
- **Key Empirical Finding:** Has lower multi-episode density (0.823) than LTAFDB/SHDB because long sinus pauses penalize exponential density decay.

### SHDB-AF Profile
- **Dominant Arrhythmia Pattern:** Intermediate paroxysmal AF (median duration 51.9s; median sinus pause 45.5s; 55.1% multi-run).
- **Best Use Case:** Investigating temporal arrhythmia organization against clinical covariates and echocardiography (Echo LAD, LVEF, prior ablation, medication).
- **Key Empirical Finding:** Negative LAD-density association is driven by leverage and temporal gaps; flips to positive (+7.77 mm) in contemporaneous echo.

### LTAFDB Profile
- **Dominant Arrhythmia Pattern:** High-frequency paroxysmal bursts (median duration 14.6s; median sinus pause 6.0s; 78.6% multi-run; 7,358 runs).
- **Best Use Case:** Ultra-rapid recurrence (78.8% in 30s), micro-reentrant fragmentation, episode entrenchment frailty modeling.
- **Key Empirical Finding:** 63.9% of runs are under 30s, but filtering them preserves 98.89% of cumulative AF burden.

### Noise-Holter-Database Profile
- **Dominant Arrhythmia Pattern:** Discontinuous real-world segments capturing authentic artifact and competing arrhythmias.
- **Best Use Case:** Signal-quality gating (bSQI), detector false-alarm suppression, VCC specificity breakdown under noise.
- **Key Empirical Finding:** Severe noise inflates sinus rhythm VCC to 48.9%, exceeding true AF (47.5%); cannot be used as a continuous longitudinal density cohort.

---

## 21. Unresolved Scientific Questions & Gaps

| question_id     | domain               | research_question                                                                                                                                             | current_status                         | blocker_or_next_step                                                                                                             |
|:----------------|:---------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| Q_UNRESOLVED_01 | Temporal Mechanics   | Does high AF density predominantly arise through prolonged episode persistence, rapid reinitiation, or dual coexistence across individual patients?           | Unresolved across cohorts              | Fit bivariate mixture models decomposing patient density into episode duration vs interepisode gap hazard.                       |
| Q_UNRESOLVED_02 | Electrophysiology    | Are brief 6-second interruptions of AF physiologically equivalent to prolonged 79-minute periods of sinus rhythm?                                             | Unresolved                             | Evaluate whether preceding AF density alters the slope of post-termination P-wave recovery in multi-episode recordings.          |
| Q_UNRESOLVED_03 | Structural Substrate | Does AF density associate with left atrial diameter in multi-episode patients when strictly restricted to contemporaneous echocardiograms and BSA adjustment? | Inconclusive / Intersection Unresolved | Isolate the exact N=28 multi-episode contemporaneous subset and test BSA-adjusted models.                                        |
| Q_UNRESOLVED_04 | Signal Observability | What is the minimum signal quality threshold (bSQI) required to preserve density rank ordering in wearable ambulatory ECG?                                    | Partially characterized                | Inject authentic Noise-Holter N1/N2 noise segments into continuous IRIDIA timelines and quantify density estimation degradation. |
| Q_UNRESOLVED_05 | Clinical Translation | Can longitudinal AF density predict thromboembolic stroke risk in non-CIED ambulatory Holter populations?                                                     | Not feasible with available data       | Requires external prospective registry with adjudicated thromboembolic endpoints (e.g., UK Biobank or clinical trial linkage).   |

*(Complete unresolved questions registry exported to `reports/af_density_unresolved_questions.csv`)*.

---

## 22. Master Artifact Index & Codebase Path Registry

| artifact_name                          | artifact_type         | relative_path                                                                                                     | scope                                                                                     |
|:---------------------------------------|:----------------------|:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|
| IRIDIA-AF Canonical Dataset            | Source Data (WFDB)    | data/iridia_af_8405941_v1.0.1                                                                                     | 167 long-term Holters (6,690.5h), 152 patients, 388 AF runs, 2-lead 200 Hz                |
| SHDB-AF Canonical Dataset              | Source Data (WFDB)    | data/shdb_af                                                                                                      | 128 Holters (98 annotated, 93 patients, 2,330.7h, 794 AF runs), 30 unannotated excluded   |
| SHDB-AF Clinical & Echo Metadata       | Source Metadata (CSV) | data/shdb_af/AdditionalData.csv                                                                                   | 45 clinical variables: LAD (72 pts), LVEF (71 pts), age, sex, ablation, medications       |
| LTAFDB Canonical Dataset               | Source Data (WFDB)    | data/ltafdb_1.0.0_reference                                                                                       | 84 2-lead 24h Holters (1,960.6h), 84 patients, 7,358 AF runs, 128 Hz                      |
| Noise-Holter-Database                  | Source Data (WFDB)    | data/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0 | 6,130 excerpts, 310 patients, Clean (43.5%), N1 (35.9%), N2 (20.6%)                       |
| MIT-BIH NSTDB Calibrated Noise         | Source Data (WFDB)    | data/nstdb/mit-bih-noise-stress-test-database-1.0.0                                                               | 12 stress test records: baseline wander (bw), muscle artifact (ma), electrode motion (em) |
| Canonical Reconciled Inventory         | Manifest (CSV)        | experiments/af_density_continuity_v5/manifests/canonical_cohort_inventory_reconciled.csv                          | 349 recordings, 329 patients, 10,982h monitoring, 8,540 verified runs                     |
| Canonical Rhythm Intervals             | Manifest (Parquet)    | experiments/af_density_continuity_v5/manifests/canonical_rhythm_intervals.parquet                                 | All reference rhythm intervals for IRIDIA, SHDB, and LTAFDB                               |
| Canonical AF Runs                      | Manifest (Parquet)    | experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet                                          | All 8,540 verified continuous AF episodes                                                 |
| Observation Windows                    | Manifest (Parquet)    | experiments/af_density_continuity_v5/manifests/observation_windows.parquet                                        | Nested 1h to 72h observation horizons                                                     |
| Harmonized Sensitivity Manifest        | Manifest (CSV)        | reports/three_dataset_harmonized_sensitivity_manifest.csv                                                         | Threshold sensitivity (native, 30s, 2m, 6m) across all 349 recordings                     |
| Reference AF Density Manifest          | Manifest (CSV)        | reports/three_dataset_reference_af_density_manifest.csv                                                           | Continuous Charitos and binned density calculations across all recordings                 |
| Ventricular Rate Intervals             | Features (Parquet)    | experiments/af_density_continuity_v5/features/ventricular_rate_intervals.parquet                                  | Extracted R-R intervals and mean ventricular rates across episodes                        |
| PMV Pre-AF Features                    | Features (Parquet)    | experiments/pmv_temporal_specificity/features/pmv_pre_af_features.parquet                                         | Pre-AF P-wave morphology features across 120s windows                                     |
| PMV Post-AF Fixed Panel Features       | Features (Parquet)    | experiments/pmv_temporal_specificity/features/pmv_post_af_features.parquet                                        | 152 post-AF episodes with early and late PMV measurements                                 |
| PMV Remote Sinus Features              | Features (Parquet)    | experiments/pmv_temporal_specificity/features/pmv_remote_sinus_features.parquet                                   | Negative control stable sinus rhythm baseline features                                    |
| AFR Authentic 3-Cohort Measurements    | Results (CSV)         | experiments/af_density_continuity_v5/results/afr_authentic_three_cohorts.csv                                      | 70 authentic candidate episodes in LTAFDB, SHDB, and IRIDIA                               |
| AFR Synthetic Cancellation Matrix      | Results (CSV)         | experiments/af_density_continuity_v5/results/afr_synthetic_benchmark_detailed.csv                                 | 125 controlled synthetic mixtures evaluating 11.8 dB cancellation                         |
| SHDB Subcohort Regressions             | Results (CSV)         | experiments/af_density_continuity_v5/results/adjudication_shdb_subcohort_models.csv                               | 19 model specifications evaluating LAD vs Density and Burden                              |
| SHDB LOPO Influence Diagnostics        | Results (CSV)         | experiments/af_density_continuity_v5/results/adjudication_shdb_lopo_influence.csv                                 | Cook's distance, studentized residuals, and DFBETAS across 121 models                     |
| Historical Density Attrition Funnel    | Results (CSV)         | experiments/af_density_continuity_v5/results/historical_density_attrition_funnel.csv                              | Preceding history eligibility funnel across 30m, 1h, 3h, 6h, 12h                          |
| Rosman 12h Density Comparison          | Results (CSV)         | experiments/af_density_continuity_v5/results/R01_rosman_density_comparison.csv                                    | Comparison against published CIED person-months (+6.9 pp excess)                          |
| Observation Horizon Migration Matrix   | Results (CSV)         | experiments/af_density_continuity_v5/results/R02_horizon_migration_matrix.csv                                     | 12h to full recording phenotype reclassification (36.2%)                                  |
| Landmark Termination Hazards           | Results (CSV)         | experiments/af_entrenchment/reports/e01_landmark_termination_probabilities.csv                                    | Conditional termination probabilities from 15s to 6h                                      |
| Shared Gamma Frailty Comparison        | Results (CSV)         | experiments/af_entrenchment/reports/e03_frailty_comparison.csv                                                    | Pooled Weibull (alpha=0.367) vs Shared Frailty (alpha_within=0.585)                       |
| ARC Clinical Trial Discordance         | Results (CSV)         | experiments/af_entrenchment/reports/feasibility_q2_arc_trial_classification.csv                                   | 26 discordant patients with AF >= 30s who never reach 6 minutes                           |
| Matched Detector Benchmark             | Benchmark (CSV)       | reports/v42_matched_detector_benchmark.csv                                                                        | ResNet, RR-GBDT, RawECGNet, ArNet2, SetOperator on informative validation                 |
| NSTDB Noise Stress Benchmark           | Benchmark (CSV)       | reports/v42_nstdb_and_external_benchmark.csv                                                                      | 210 evaluation rows across +24 to -6 dB SNR for bw, ma, em                                |
| Stage 1D HSMM Episode Decoding         | Benchmark (CSV)       | experiments/af_density/stage1d_hsmm_episode_behavior.csv                                                          | D0, D1, D2 semi-Markov decoder performance on continuous timelines                        |
| V5 AF Density Core Library             | Source Code (Python)  | experiments/af_density_continuity_v5/src/af_density/core.py                                                       | Continuous Charitos density, binned density, episode continuity, aggregation loss         |
| PMV Core Feature Extraction            | Source Code (Python)  | experiments/pmv_temporal_specificity/scripts/pmv_core.py                                                          | 6 representations of P-wave morphological variability across 120s windows                 |
| AFR Spatiotemporal Cancellation Module | Source Code (Python)  | experiments/af_density_continuity_v5/src/afr_pilot.py                                                             | QRST template subtraction, harmonic power estimation, dominant frequency peak tracking    |
| Temporal Decoders (D0, D1, D2)         | Source Code (Python)  | experiments/af_density/models/decoders.py                                                                         | Pointwise GBDT (D0), Linear CRF (D1), and Semi-Markov duration HSMM (D2)                  |
| AF-Density Master Knowledge Base       | Report (Markdown)     | reports/AF_DENSITY_MASTER_KNOWLEDGE_BASE.md                                                                       | 22-section comprehensive synthesis across all 4 cohorts, benchmarks, and evidence         |
| Comprehensive 4-Dataset EDA Compendium | Report (Markdown)     | reports/comprehensive_four_dataset_eda_metrics_compendium.md                                                      | Canonical EDA metrics, sensitivity manifests, and clinical profiling                      |
| V5 Scientific Adjudication Report      | Report (Markdown)     | reports/af_density_continuity_scientific_adjudication.md                                                          | Formal adjudication of SHDB structural association, AFR feasibility, and attrition        |
| PMV Temporal Specificity Report        | Report (Markdown)     | reports/pmv_temporal_specificity_scientific_report.md                                                             | E01-E10 investigation into pre-AF onset vs post-AF atrial stunning                        |
| AF Entrenchment Master Report          | Report (Markdown)     | experiments/af_entrenchment/reports/entrenchment_master_report.md                                                 | Termination hazard collapse, Weibull frailty, VCC, and ARC discordance                    |
| Rosman Replication Report              | Report (Markdown)     | reports/rosman_phenotype_replication_report.md                                                                    | R01-R05 methodological replication of 2026 JACC paper on Holters                          |
| Adjudication Fig 1: LAD vs Density     | Figure (PNG)          | experiments/af_density_continuity_v5/figures/adjudication_fig1_lad_density_scatter.png                            | Scatterplot and regression lines for LAD vs Density across subcohorts                     |
| Adjudication Fig 2: LOPO Influence     | Figure (PNG)          | experiments/af_density_continuity_v5/figures/adjudication_fig2_lopo_influence.png                                 | Cook's distance and DFBETAS identifying outlier shdb_4339581                              |
| Adjudication Fig 3: AFR Cancellation   | Figure (PNG)          | experiments/af_density_continuity_v5/figures/adjudication_fig3_afr_cancellation_spectra.png                       | Power spectral densities and cancellation depths across authentic leads                   |

*(Complete artifact path registry exported to `reports/af_density_artifact_index.csv`)*.

---

*Authoritative Knowledge Base compiled autonomously via script 30. All values verified against machine-readable registries.*
