# External Validation Plan
## T-CAIREM Continuous SetOperator AF-Density Research Program

**Document Version**: 1.0 (Final Frozen Protocol)  
**Date**: 2026-11-22  
**Author**: Statistical Governance & Refinement Planner  
**Target Repository**: `experiments/af_density/`  
**Storage Destination**: `external_validation_plan.md`, `refine-logs/external_validation_plan.md`, and `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/external_validation_plan.md`  

---

## 1. Objectives & Strict Zero-Tuning Governance

The primary objective of this external validation plan is to evaluate the generalizability, lead-dropout invariance, and episode reconstruction fidelity of the Continuous $q$-conditioned SetOperator on two independent clinical cohorts without any test-guided hyperparameter tuning.

### Strict Governance Rules (The Zero-Tuning Charter):
1. **Zero Parameter Adaptation**: All neural network weights, continuous operator MLP parameters, projection vectors, and residual fusion layers must be frozen from the primary IRIDIA-AF training run. Not a single gradient update or fine-tuning step is permitted on external cohorts.
2. **Frozen Decision Threshold ($\tau$)**: The decision threshold $\tau \in (0, 1)$ must be prospectively imported from the IRIDIA validation cohort selection policy (fixed balanced-accuracy criterion). Running an ROC curve or sweeping thresholds on external test data to find an "optimal" operating point is strictly prohibited as data snooping.
3. **Frozen Temporal HSMM Priors**: The D2 Hidden Semi-Markov Model duration distributions (censored lognormal parameters fitted on IRIDIA training data) and state transition matrices must be held strictly constant.
4. **Zero Test-Label Access During Inference**: The forward prediction script must operate strictly in blind inference mode. Loading test annotations or reference interval arrays into memory during feature extraction or model forward passes is an integrity violation.
5. **Pre-Registered Subgroup Stratification**: Performance must be reported across pre-registered cohorts: informative multi-interval subjects, persistent AF subjects, and zero-AF negative controls.

---

## 2. External Dataset Profiles & Cohort Inventories

```
                      ┌───────────────────────────────────────────────┐
                      │    Primary Training & Internal Benchmark:     │
                      │                   IRIDIA-AF                   │
                      │  (152 pts, 504k train / 166k val / 131k test) │
                      └───────────────────────┬───────────────────────┘
                                              │ (Strict Zero-Tuning Transfer)
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
      ┌───────────────────────────────┐               ┌───────────────────────────────┐
      │     External Cohort 1:        │               │      External Cohort 2:       │
      │           SHDB-AF             │               │            LTAFDB             │
      │ 98 annotated records, 93 pts  │               │ 84 long-term records, 84 pts  │
      │ 279,686 windows @ 200 Hz      │               │ 232,233 windows @ 128 Hz      │
      │ 794 AF intervals, 49 multi-ep │               │ 7,358 AF intervals, 66 multi  │
      └───────────────────────────────┘               └───────────────────────────────┘
```

### 2.1 St. Petersburg Institute of Cardiological Technics 12-lead Arrhythmia Database (SHDB)
- **Clinical Nature**: 12-lead ambulatory ECG recordings (modified Frank leads) from patients undergoing cardiac evaluation for arrhythmias and ischemic episodes.
- **Cohort Inventory**:
  - Total local records: 128 records (122 distinct subjects).
  - Official PhysioNet v1.0.1 SHA-256 verification completed for all 227 metadata/header/annotation files.
  - Waveform sample rate: 200 Hz (identical to IRIDIA native rate).
  - Total windows extracted: 279,686 native-clock 30-second windows across 98 annotated records (93 distinct subjects).
  - Reference AF intervals: 794 verified episodes.
  - Multi-interval informative cohort: 49 subjects exhibit $\ge 2$ disjoint AF intervals (admissible for continuous density benchmarking).
  - Negative control cohort: 34 subjects with zero documented AF episodes (critical for patient specificity and false burden audits).

### 2.2 Long-Term Atrial Fibrillation Database (LTAFDB)
- **Clinical Nature**: Extended 2-lead ambulatory Holter recordings (up to 24–48 hours per subject), capturing paroxysmal and persistent episodes.
- **Cohort Inventory**:
  - Total records: 84 records (84 distinct subjects).
  - All 84 waveform publisher SHA-256 hashes verified against official PhysioNet archive.
  - Waveform sample rate: 128 Hz (heterogeneous from IRIDIA 200 Hz).
  - Total windows extracted: 232,233 native-clock 30-second windows.
  - Reference AF intervals: 7,358 verified episodes.
  - Multi-interval informative cohort: 66 records exhibit $\ge 2$ disjoint AF intervals.
  - Known Data Anomaly: Record 20 contains an officially flagged header sample count discrepancy ($28,799,360$ vs $28,798,976$ samples). This discrepancy is isolated and handled via explicit valid sample truncations.

---

## 3. Continuous Lead Geometry & Cross-Dataset Adaptation

A critical challenge in external validation of ECG deep learning models is lead configuration heterogeneity. The Continuous $q$-conditioned SetOperator resolves this via geometric projection.

### 3.1 Mapping to Canonical 8-Lead Clinical Basis ($q \in \mathbb{R}^8$)
All input channels are mapped into the canonical clinical vector basis $\mathcal{B} = [I, II, V_1, V_2, V_3, V_4, V_5, V_6]^T$:

1. **IRIDIA (2 Channels)**:
   - Channel 0: Nominal Lead I $\to q_{C0} = [1, 0, 0, 0, 0, 0, 0, 0]^T$
   - Channel 1: Nominal Lead II $\to q_{C1} = [0, 1, 0, 0, 0, 0, 0, 0]^T$
2. **SHDB (12 Leads Available, 2-Lead Benchmark Projection)**:
   - To match the 2-channel deployment scenario, SHDB evaluates Leads I and II directly using the identical nominal projections $q_{C0}, q_{C1}$.
   - Secondary robustness tests evaluate arbitrary lead pairs (e.g., Lead $V_1 = [0, 0, 1, 0, 0, 0, 0, 0]^T$ and Lead $V_5 = [0, 0, 0, 0, 0, 1, 0, 0]^T$) using their corresponding canonical basis vectors.
3. **LTAFDB (Modified Bipolar Torso Leads)**:
   - LTAF records two modified limb leads (typically modified $V_5$ and modified $V_1$ or Lead II).
   - Channels are mapped to nominal basis projections with documented electrode uncertainty.

### 3.2 Handling Sampling Rate Discrepancy (128 Hz vs 200 Hz)
- The raw ResNet baseline operates on fixed-length sample arrays (6,000 samples @ 200 Hz), requiring digital sinc/spline resampling of LTAFDB from 128 Hz to 200 Hz.
- The **Phase-KME SetOperator** is inherently sampling-rate agnostic:
  1. R-peaks are detected on the native clock.
  2. Each detected cardiac cycle $[R_k, R_{k+1}]$ is normalized into 256 phase-angle points $\theta \in [0, 2\pi)$ via Piecewise Cubic Hermite Interpolating Polynomials (PCHIP).
  3. The resulting phase-domain representation is independent of physical sampling frequency, eliminating interpolation artifacts.

---

## 4. Custody, Integrity, and Tamper-Evident Workflow

To prevent confirmation bias, data leakage, and accidental test access, the external validation workflow executes under cryptographic custody:

```
Step 1: Hash Verification
  ├── SHA-256 of Frozen Model Checkpoint (e.g., best_continuous_setoperator_a001.pt)
  ├── SHA-256 of Frozen HSMM Duration PMFs (waveform_v1/duration_model_v1/)
  └── SHA-256 of Pre-Extracted Phase Features (shdb_phase_cache, ltaf_phase_cache)
       │
       ▼
Step 2: Blind Model Inference (Zero Test Labels in RAM)
  ├── Input: Features & Coordinate Vectors only
  ├── Forward Pass: Predict Window Logits s_t
  ├── Probability Calibration: p_t = σ(s_t / T_cal) (using frozen T_cal)
  └── Commit Predictions: external_raw_predictions.pt (cryptographically sealed)
       │
       ▼
Step 3: Temporal Hierarchy & Timeline Reconstruction
  ├── Apply D0: Calibrated Thresholding (frozen τ)
  ├── Apply D1: Median Filter (60s non-AF gap merge within observed blocks)
  └── Apply D2: Viterbi HSMM (censored lognormal duration priors)
       │
       ▼
Step 4: Independent Evaluation Script
  ├── Load Sealed Predictions & Independent Reference Annotations
  ├── Evaluate Patient-Clustered Bootstrap Metrics (B = 2,000)
  └── Output: Immutable JSON Reconciliation & Benchmark Tables
```

---

## 5. External Benchmark Endpoints & Reporting Protocols

### 5.1 Primary Comparative Endpoints
The external cohorts must be evaluated across the identical metrics defined in the Statistical Analysis Plan (`statistical_analysis_plan.md`):

| Metric | Clinical & Algorithmic Meaning | Target Bar (SHDB / LTAF) |
|---|---|---|
| **Patient Specificity (Zero-AF)** | Proportion of negative control subjects with zero false episodes | $\ge 85.0\%$ |
| **False Burden (%)** | Proportion of total recording time erroneously labeled as AF in negative controls | $\le 1.50\%$ |
| **Charitos Continuous Density MAE** | Mean absolute error of integrated continuous density on informative cohort ($K \ge 2$) | $\le 0.100$ |
| **Charitos Binned Density MAE** | Mean absolute error of author 30s-binned density on informative cohort | $\le 0.080$ |
| **Continuous AF Burden MAE** | Mean absolute error of total cumulative AF time fraction | $\le 12.0\%$ |
| **Episode Sensitivity ($\text{Se}_{\text{ep}}$)** | Proportion of reference episodes detected with positive overlap | $\ge 80.0\%$ |
| **Episode PPV ($\text{PPV}_{\text{ep}}$)** | Proportion of predicted episodes corresponding to true AF | $\ge 75.0\%$ |
| **Burden Correlation ($r, \rho$)** | Pearson $r$ and Spearman $\rho$ between predicted and true burden | $r \ge 0.75, \, \rho \ge 0.80$ |
| **Episode Boundary Error (s)** | Mean absolute onset/offset error on isolated 1-to-1 matched episodes | $\le 10.0$ seconds |

### 5.2 Mandatory Stratification Tables

#### Table E1: Negative Control Audit (Specificity & False Alarms)
- **Population**: 34 SHDB subjects with zero reference AF.
- **Metrics Reported**: Total false episodes, mean false episode duration (seconds), cumulative false burden (%), patient-level specificity (zero episodes detected).
- **Comparison**: Continuous SetOperator (A001/I082) vs Categorical SetOperator (A002) vs Raw Mask-ResNet vs Observed-RR Baseline.

#### Table E2: Informative Multi-Interval Benchmark
- **Population**: 49 SHDB multi-interval subjects; 66 LTAF multi-interval records.
- **Metrics Reported**: Continuous Density MAE, Binned Density MAE, Burden MAE, Episode Sensitivity, PPV, Boundary Error (all with 95% Patient-Clustered Bootstrap CIs).
- **Comparison**: Continuous SetOperator vs Baselines across D0, D1, and D2 HSMM decoders.

#### Table E3: Multi-Lead Invariance & Permutation Audit on External Data
- **Population**: All 98 SHDB annotated records.
- **Protocol**: Swap Channel 0 and Channel 1; evaluate single-lead dropout (Channel 0 alone vs Channel 1 alone vs Both).
- **Metrics Reported**: Maximum logit discrepancy, burden divergence, coverage within Manski bounds $[Burden_{\min}, Burden_{\max}]$.

---

## 6. Failure Modes, Diagnostic Gates, and Stopping Rules

The external validation run must immediately halt and document an incident if any of the following gates fail:

1. **Non-Finite Output Gate**: Any forward pass generating `NaN` or `Inf` logits on valid feature inputs triggers immediate execution halt.
2. **Channel Permutation Leakage Gate**: Swapping input channels and their associated $q$ vectors must produce a logit difference $< 1 \times 10^{-5}$. Any divergence $\ge 1 \times 10^{-4}$ indicates broken permutation symmetry in the set encoder.
3. **Score Degeneracy Gate**: If the standard deviation of predicted scores across the first 10,000 external windows falls below $1 \times 10^{-4}$, the run is aborted for representation collapse.
4. **Hardware Custody Failure**: Any mismatch in input feature SHA-256 hashes or checkpoint checksums blocks inference.

---

## 7. Delivery & Output Manifest
All execution artifacts, raw prediction tensors, evaluation locks, and markdown summary tables must be sealed and committed to:
- `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/external_validation/`
- Local mirror: `experiments/af_density/external_validation/`
- Audit receipts: `shdb_external_reconciliation.json`, `ltaf_external_reconciliation.json`.
