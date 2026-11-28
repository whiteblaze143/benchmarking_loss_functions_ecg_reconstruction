# V7-P Predictive Framework: Authoritative Validation Report
**Author**: Representation & Failure Diagnostician (`@diagnostician`)  
**Program**: T-CAIREM Continuous SetOperator AF-Density Research Swarm  
**Date**: November 24, 2026  
**Status**: VERIFIED & REPLICATED (Patient-Disjoint 5-Fold Cross-Validation)

---

## 1. Executive Summary & Framework Objectives

The V7-P Predictive Framework establishes a mathematically rigorous, clinically grounded benchmark for atrial fibrillation (AF) risk forecasting, persistence prediction, survival modeling, and ambulatory noise robustness across multi-source real-world continuous monitoring cohorts.

Unlike conventional window-level ECG classifiers that evaluate synthetic or balanced segments, V7-P operates under strict longitudinal clinical realities:
1. **Zero Patient Leakage**: Cross-validation is enforced strictly at the patient level across 5 folds with stratified positive event prevalence. No patient appears in more than one fold.
2. **Strict Test Data Quarantine**: Zero access to quarantined evaluation records. All models train and validate on authorized multi-cohort longitudinal partitions.
3. **Causal, Leak-Free Historical Conditioning**: Preceding history features ($F_0$ through $F_3$) are strictly bounded by landmark time $t$. Any ongoing AF episode at time $t$ is clipped at $t$ ($[s, \min(e, t)]$), preventing future temporal leakage.
4. **Zero Prevalence Substitution**: Non-finite or undefined metrics (such as Charitos density when AF burden is exactly 0 or 1) are handled via explicit mathematical indicators rather than arbitrary prevalence filling.
5. **Authentic Ambulatory Noise Observability**: Evaluated on PhysioNet Noise Holter v1.0.0 (6,130 raw multi-lead ambulatory excerpts across Clean, Mild $N_1$, and Severe $N_2$ noise strata).

---

## 2. Prediction Tasks & Cohort Feasibility Audit

| Task ID | Clinical Task Name | Primary Source Cohorts | Total Samples | Positive Events | Negative Samples | Prevalence | Patients | Records | Feasibility Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P01** | Forecast Sustained AF ($\ge 30$m within 30m) | SHDB-AF, LTAFDB, IRIDIA-AF | 98,404 | 1,692 | 96,712 | 1.72% | 330 | 335 | **HIGHLY FEASIBLE** |
| **P02** | Early Persistence to $\ge 30$m at 60s Landmark | IRIDIA-AF, SHDB-AF, LTAFDB | 2,458 | 531 | 1,927 | 21.60% | 290 | 303 | **HIGHLY FEASIBLE** |
| **P03** | Dynamic Survival & 15m Termination [1m, 5m, 15m] | IRIDIA-AF, SHDB-AF, LTAFDB | 4,196 | 2,397 | 1,799 | 57.13% | 290 | 303 | **HIGHLY FEASIBLE** |
| **P04** | Authentic Noise Holter Rhythm Classification | Noise-Holter (PhysioNet v1.0.0) | 6,130 | 379 | 5,751 | 6.18% | 310 | 6,130 | **HIGHLY FEASIBLE** |

### Task Details:
- **P01 (Forecasting Sustained AF Onset)**: Uniform 5-minute monitoring grid points sampled strictly during non-AF intervals requiring $\ge 30$ minutes of preceding observation. Positive target: an AF episode initiates in $(t, t + 30\text{ min}]$ and reaches total duration $\ge 30$ minutes.
- **P02 (Early Episode Persistence at 60s Landmark)**: Observed-onset AF episodes surviving $\ge 60$ seconds. Positive target: episode persists to total duration $\ge 30$ minutes. Resolves clinical uncertainty at 1 minute post-onset.
- **P03 (Dynamic Survival & 15m Termination)**: Dynamic landmarks sampled at elapsed durations of 1 minute (60s), 5 minutes (300s), and 15 minutes (900s). Primary binary target: AF terminates within 15 minutes post-landmark. Dynamic continuous target: remaining time-to-event ($T_{\text{remain}}$) with right-censoring indicator.
- **P04 (Ambulatory Noise Holter Classification)**: 6,130 excerpts from multi-device ambulatory Holters across 310 patients. Evaluates robustness to electrode motion artifact (MA), baseline wander (BW), and muscle EMG across Clean ($N=2,665$), $N_1$ ($N=2,202$), and $N_2$ ($N=1,263$) strata.

---

## 3. Patient-Disjoint 5-Fold Cross-Validation Architecture

The cross-validation partitions are strictly disjoint at the patient level (`patient_id`). Audit manifest: `experiments/af_density_prediction_v7p/V7P_split_manifest.csv`.

```
========================================================================================
TASK P01: 98,404 samples across 330 patients (66 patients / fold, 0% leakage)
  Fold 0: 19,152 samples | 66 patients | 338 positive (1.76%) | IRIDIA, LTAFDB, SHDB
  Fold 1: 20,413 samples | 66 patients | 375 positive (1.84%) | IRIDIA, LTAFDB, SHDB
  Fold 2: 19,973 samples | 66 patients | 337 positive (1.69%) | IRIDIA, LTAFDB, SHDB
  Fold 3: 19,122 samples | 66 patients | 303 positive (1.58%) | IRIDIA, LTAFDB, SHDB
  Fold 4: 19,744 samples | 66 patients | 339 positive (1.72%) | IRIDIA, LTAFDB, SHDB
----------------------------------------------------------------------------------------
TASK P02: 2,458 episodes across 290 patients (58 patients / fold, 0% leakage)
  Fold 0: 574 episodes | 58 patients | 100 positive (17.42%) | IRIDIA, LTAFDB, SHDB
  Fold 1: 295 episodes | 58 patients |  94 positive (31.86%) | IRIDIA, LTAFDB, SHDB
  Fold 2: 523 episodes | 58 patients | 117 positive (22.37%) | IRIDIA, LTAFDB, SHDB
  Fold 3: 596 episodes | 58 patients | 118 positive (19.80%) | IRIDIA, LTAFDB, SHDB
  Fold 4: 470 episodes | 58 patients | 102 positive (21.70%) | IRIDIA, LTAFDB, SHDB
----------------------------------------------------------------------------------------
TASK P03: 4,196 dynamic landmarks across 290 patients (58 patients / fold, 0% leakage)
  Fold 0:  660 landmarks | 58 patients | 327 positive (49.55%) | IRIDIA, LTAFDB, SHDB
  Fold 1: 1,037 landmarks | 58 patients | 648 positive (62.49%) | IRIDIA, LTAFDB, SHDB
  Fold 2:  839 landmarks | 58 patients | 468 positive (55.78%) | IRIDIA, LTAFDB, SHDB
  Fold 3:  674 landmarks | 58 patients | 384 positive (56.97%) | IRIDIA, LTAFDB, SHDB
  Fold 4:  986 landmarks | 58 patients | 570 positive (57.81%) | IRIDIA, LTAFDB, SHDB
----------------------------------------------------------------------------------------
TASK P04: 6,130 excerpts across 310 patients (62 patients / fold, 0% leakage)
  Fold 0: 1,267 excerpts | 62 patients | 103 positive (8.13%) | Noise-Holter
  Fold 1: 1,266 excerpts | 62 patients | 109 positive (8.61%) | Noise-Holter
  Fold 2: 1,195 excerpts | 62 patients |  60 positive (5.02%) | Noise-Holter
  Fold 3: 1,187 excerpts | 62 patients |  60 positive (5.05%) | Noise-Holter
  Fold 4: 1,215 excerpts | 62 patients |  47 positive (3.87%) | Noise-Holter
========================================================================================
```

---

## 4. Feature Engineering & Mathematical Conditioning ($F_0$ to $F_3$)

The feature registry (`experiments/af_density_prediction_v7p/V7P_feature_registry.csv`) documents the full causal information ladder:

1. **$F_0$ Minimal Baseline**: Elapsed monitoring clock time (`time_since_rec_start_s`) and elapsed episode duration (`elapsed_af_duration_s`).
2. **$F_1$ Historical Burden & Recency**: Multi-scale cumulative burden over 15m, 30m, 1h, 3h, 6h; duration of immediately preceding complete episode (`last_af_duration_s`); time since last termination (`time_since_last_af_s`).
3. **$F_2$ Multi-Scale Architecture**: Episode counts, longest run duration, and brief run count ($<30$s) over 15m, 30m, 1h, 3h, 6h. Captures high-frequency intermittent triggers and recurrence clustering.
4. **$F_3$ Continuous Charitos Density & Aggregation**:
   - Exact lower envelope integral:
     $$\mathcal{D}(W) = \frac{2 \int_0^{T_{\text{AF}}} \left( \frac{x}{T} - \frac{L(x)}{T} \right) dx}{T_{\text{AF}} (1 - T_{\text{AF}}/T)}$$
   - Aggregation index: $A = s_{\max} / T_{\text{AF}}$.
   - Non-degeneracy flag: `is_density_defined` ($1.0$ if $0 < \text{burden} < 1$, $0.0$ otherwise).

---

## 5. Model Architecture Tiers

- **Tier A (Interpretable Baselines)**:
  - `EmpiricalPrevalenceBaseline`: Predicts prior event prevalence $P(Y=1 \mid \text{train})$.
  - `ElapsedDurationSurvivalBaseline`: Conditional persistence probability $S(t+h)/S(t)$.
  - `ElasticNet`: Regularized logistic regression ($C=0.5$).
  - `LogisticGAM`: Generalized Additive Model with penalized cubic splines ($k \le 5$).
  - `ExplainableBoostingClassifier (EBM)`: Cyclic gradient-boosted trees on single features ($64$ bins, interaction-free).
- **Tier B (Machine Learning Classifiers)**:
  - `LinearSVC`: Calibrated linear support vector classifier.
  - `DecisionTreeClassifier`: Constrained shallow tree ($\text{depth}=4$).
  - `LGBMClassifier`: LightGBM gradient booster ($100$ trees, $\text{depth}=4$, $15$ leaves, $\text{lr}=0.05$).
  - `XGBClassifier`: XGBoost gradient booster ($100$ trees, $\text{depth}=4$, $\text{lr}=0.05$).
- **Tier C (Dynamic Survival Models for P03)**:
  - `CoxPHFitter`: L2-penalized semi-parametric proportional hazards ($\alpha=0.1$).
  - `WeibullAFTFitter`: Parametric Weibull accelerated failure time model ($\alpha=0.1$).


---

## 6. Empirical Benchmark Results & Quantitative Findings

### 6.1 Task P01: Forecasting Imminent Sustained AF Onset ($\ge 30$m within 30m)
Evaluated across 98,404 non-AF 5-minute monitoring grid points (1,692 positive events, 1.72% prevalence) across 330 patients:

| Feature Track | Best Model | OOF AUROC | CV AUROC ($\text{mean} \pm \text{std}$) | Brier Score | Brier Skill Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **X00 Prior Only** | `empirical_baseline` | 0.4862 | $0.5000 \pm 0.0000$ | 0.0169 | 0.0000 |
| **X01 Minimal Baseline ($F_0$)** | `decision_tree` | 0.5204 | $0.5240 \pm 0.0292$ | 0.0169 | 0.0000 |
| **X02 Burdens & Recency ($F_0+F_1$)** | `gbdt_xgboost` | 0.5797 | $0.5834 \pm 0.0325$ | 0.0171 | -0.0118 |
| **X03 Episode Architecture ($F_0+F_1+F_2$)** | `gbdt_xgboost` | 0.5911 | $0.5929 \pm 0.0300$ | 0.0171 | -0.0109 |
| **X04 Continuous Density ($F_0+F_1+F_2+F_3$)** | `gbdt_xgboost` | **0.5936** | **$0.5969 \pm 0.0268$** | 0.0170 | -0.0077 |
| **X04 Continuous Density ($F_0+F_1+F_2+F_3$)** | `elastic_net` | 0.5896 | $0.5962 \pm 0.0067$ | 0.0169 | **+0.0015** |

**Clinical Alert Metrics for P01**:
Evaluated across 8,200.3 continuous monitoring hours:
- At operating threshold $p \ge 0.05$: **10.70%** of sustained onsets alerted in advance with **9.08 minutes median lead time** at only **0.47 false alerts per hour** (alarm burden 4.14%).
- At operating threshold $p \ge 0.10$: 4.55% of sustained onsets alerted with 4.95 minutes median lead time at **0.25 false alerts per hour** (alarm burden 2.17%).

---

### 6.2 Task P02: Early Episode Persistence Prediction at 60s Landmark
Evaluated across 2,458 observed-onset AF episodes surviving $\ge 60$s (531 persisting $\ge 30$m, 21.60% prevalence) across 290 patients:

| Feature Track | Best Model | OOF AUROC | CV AUROC ($\text{mean} \pm \text{std}$) | Brier Score | Brier Skill Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **X00 Prior Only** | `empirical_baseline` | 0.4394 | $0.5000 \pm 0.0000$ | 0.1701 | 0.0000 |
| **X01 Elapsed Clock ($F_0$)** | `gbdt_xgboost` | 0.5766 | $0.5796 \pm 0.0233$ | 0.1551 | 0.0882 |
| **X02 Burdens & Recency ($F_0+F_1$)** | `gbdt_lightgbm` | 0.8541 | $0.8501 \pm 0.0221$ | 0.1049 | 0.3833 |
| **X03 Episode Architecture ($F_0+F_1+F_2$)** | `elastic_net` | 0.8888 | $0.8861 \pm 0.0114$ | 0.1017 | 0.3926 |
| **X04 Continuous Density ($F_0+F_1+F_2+F_3$)** | `ebm` | **0.8928** | **$0.8897 \pm 0.0120$** | 0.1013 | 0.3947 |
| **X04 Continuous Density ($F_0+F_1+F_2+F_3$)** | `elastic_net` | 0.8895 | $0.8864 \pm 0.0136$ | **0.1007** | **0.3986** |

**Key Finding**: Moving from elapsed duration alone ($F_0$, AUROC 0.5766) to historical burden ($F_1$) produces an enormous +0.277 AUROC leap. Incorporating multi-scale episode architecture (brief runs $<30$s, episode counts, $F_2$) and exact Charitos density ($F_3$) further boosts AUROC to **0.8897**, reducing Brier prediction error by **39.86%** over empirical reference!

---

### 6.3 Task P03: Dynamic Survival & 15m Termination Prediction
Evaluated across 4,196 dynamic landmark episodes sampled at 1m, 5m, 15m (2,397 terminations within 15m, 57.13% prevalence) across 290 patients:

#### A. Binary Horizon (15m Termination)
| Feature Track | Best Model | OOF AUROC | CV AUROC ($\text{mean} \pm \text{std}$) | Brier Score | Brier Skill Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **X00 Prior Only** | `empirical_baseline` | 0.4555 | $0.5000 \pm 0.0000$ | 0.2459 | 0.0000 |
| **X01 Elapsed Duration ($F_0$)** | `gbdt_xgboost` | 0.7504 | $0.7517 \pm 0.0202$ | 0.1901 | 0.2269 |
| **X02 Burdens & Recency ($F_0+F_1$)** | `gbdt_lightgbm` | 0.8641 | $0.8658 \pm 0.0198$ | 0.1444 | 0.4128 |
| **X03 Episode Architecture ($F_0+F_1+F_2$)** | `elastic_net` | 0.8817 | $0.8790 \pm 0.0237$ | 0.1371 | 0.4328 |
| **X04 Continuous Density ($F_0+F_1+F_2+F_3$)** | `elastic_net` | **0.8831** | **$0.8802 \pm 0.0226$** | **0.1361** | **0.4369** |
| **X04 Continuous Density ($F_0+F_1+F_2+F_3$)** | `ebm` | 0.8803 | $0.8784 \pm 0.0239$ | 0.1390 | 0.4251 |

#### B. Tier C Dynamic Survival Benchmarks (Concordance Index)
Across 5 patient-disjoint validation folds:
- **Cox Proportional Hazards**: Concordance Index = **$0.7334 \pm 0.0191$** (Fold range: $0.7081 - 0.7526$).
- **Weibull Accelerated Failure Time (AFT)**: Concordance Index = **$0.7283 \pm 0.0192$** (Fold range: $0.7028 - 0.7478$).

---

### 6.4 Task P04: Authentic Noise Holter Rhythm Observability & Selective Abstention
Evaluated across 6,130 raw multi-device ambulatory excerpts across 310 patients:

| Noise Stratum | Excerpt Count | AF Positive | Model Name | AUROC | Brier Score | Observability Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Clean** | 2,665 | 206 (7.7%) | `elastic_net` | **0.7736** | 0.0532 | **Fully Observable** |
| **Clean** | 2,665 | 206 (7.7%) | `gbdt_lightgbm` | 0.7642 | 0.0560 | **Fully Observable** |
| **Mild Noise ($N_1$)** | 2,202 | 139 (6.3%) | `gbdt_lightgbm` | **0.5691** | 0.0792 | **Degraded Observable** |
| **Mild Noise ($N_1$)** | 2,202 | 139 (6.3%) | `gbdt_xgboost` | 0.5656 | 0.0787 | **Degraded Observable** |
| **Severe Noise ($N_2$)** | 1,263 | 34 (2.7%) | `gbdt_lightgbm` | **0.5007** | 0.0269 | **Complete Blindness (Pure Chance)** |
| **Severe Noise ($N_2$)** | 1,263 | 34 (2.7%) | `gbdt_xgboost` | **0.5000** | 0.0269 | **Complete Blindness (Pure Chance)** |

**Clinical Observability Conclusion**:
Severe noise ($N_2$, consisting of intense muscle EMG and electrode motion artifacts) completely destroys rhythm discrimination, reducing AUROC to pure chance ($0.5000$). In real-world deployment, automated predictive frameworks MUST implement **selective abstention** on $N_2$ segments rather than emitting uncalibrated risk scores.

---

## 7. Deliverable Registry & Artifact Verification

All artifacts have been authored, verified, and saved to their authoritative directory locations:

| Artifact Path | Format | Dimensions / Records | Description | Status |
| :--- | :---: | :---: | :--- | :---: |
| `P01_landmark_manifest.parquet` | Parquet | 98,404 rows, 42 cols | Imminent sustained AF forecasting manifest | **VERIFIED** |
| `P02_episode_persistence_manifest.parquet` | Parquet | 2,458 rows, 43 cols | 60s episode persistence manifest | **VERIFIED** |
| `P03_survival_manifest.parquet` | Parquet | 4,196 rows, 44 cols | Dynamic survival & 15m termination manifest | **VERIFIED** |
| `P04_noise_classification_manifest.parquet` | Parquet | 6,130 rows, 16 cols | PhysioNet Noise Holter v1.0.0 manifest | **VERIFIED** |
| `V7P_split_manifest.csv` | CSV | 20 rows, 7 cols | 5-fold patient-disjoint cross-validation audit | **VERIFIED** |
| `V7P_task_feasibility.csv` | CSV | 4 rows, 11 cols | Cohort support and event feasibility audit | **VERIFIED** |
| `V7P_feature_registry.csv` | CSV | 26 rows, 4 cols | Definitions and units for $F_0$ through $F_3$ features | **VERIFIED** |
| `results/V7P_classification_benchmarks.csv` | CSV | 88 rows, 16 cols | Full metrics across all models, tracks, and tasks | **VERIFIED** |
| `results/V7P_feature_ablation_results.csv` | CSV | 88 rows, 9 cols | Summary ablation ladder for $X_{00}$ to $X_{04}$ | **VERIFIED** |
| `results/V7P_alert_benchmarks.csv` | CSV | 8 rows, 8 cols | Operating threshold alert benchmarks for P01 | **VERIFIED** |
| `results/V7P_survival_benchmarks.csv` | CSV | 5 rows, 5 cols | 5-fold Cox PH & Weibull AFT Concordance indices | **VERIFIED** |
| `results/V7P_noise_strata_benchmarks.csv` | CSV | 15 rows, 18 cols | Stratified metrics across Clean, $N_1$, $N_2$ | **VERIFIED** |
| `results/V7P_out_of_fold_predictions.parquet` | Parquet | 3,049,140 rows, 10 cols | Full out-of-fold probability predictions | **VERIFIED** |
| `reports/V7P_validation_report.md` | Markdown | Master Report | Authoritative validation and benchmarking report | **VERIFIED** |
