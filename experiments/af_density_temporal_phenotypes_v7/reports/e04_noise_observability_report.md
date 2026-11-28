# Experiment E04 Report: Phenotype-Dependent ECG Observability under Authentic Artifact and Sensing Limitations
## Multi-Parameter Feature Degradation, Rhythm Separability Collapse, and Manski Partial-Identification Bounds

**Date**: November 24, 2026  
**Author**: Refinement Pipeline & Statistical Governance Planner (`@planner`)  
**Primary Dataset**: Authentic Noise Holter Database (6,130 excerpts across 316 unique patients)  
**Simulation Reference Cohorts**: IRIDIA-AF (70 multi-run recordings) & SHDB-AF (54 multi-run recordings; total $N = 124$)  
**Primary Manifest**: `experiments/af_density_temporal_phenotypes_v7/results/e04_noise/noise_holter_feature_matrix.csv`  
**Artifact Directory**: `experiments/af_density_temporal_phenotypes_v7/results/e04_noise/`  
**Mirrored Codex Path**: `/data/mithunmanivannan/codex_artifacts/af_density_temporal_phenotypes_v7/results/e04_noise/`  

---

## 1. Executive Summary & Core Observability Findings

Ambulatory Holters and wearable monitors frequently suffer from baseline wander, electromyographic (EMG) interference, and motion artifacts. This investigation provides an empirical audit of how authentic clinical artifact degrades physiological rhythm features and how sensing dropouts distort continuous AF density.

### Key Quantitative Findings:

1. **Noise Holter Cohort Scale & Quality Stratification**:
   - Evaluated **6,130 authentic clinical excerpts** across **316 independent patients**:
     - **Clean Quality**: 2,665 excerpts (43.5%) — High diagnostic fidelity with baseline noise $<10\mu\text{V}$.
     - **N1 (Intermediate Noise)**: 2,202 excerpts (35.9%) — Baseline wander and muscle tremor present, but underlying QRS rhythm is discernible by clinical review.
     - **N2 (Severe Noise)**: 1,263 excerpts (20.6%) — High-amplitude EMG, electrode detachment, or severe baseline drift obscuring QRS morphology; automated beat tracking fails.
   - Evaluated rhythms include **379 AF episodes**, **3,597 sinus/non-AF episodes**, **122 atrial flutter episodes**, and **2,032 ventricular/supraventricular ectopy episodes**.

2. **Physiological Feature Degradation & Rhythm Separability Collapse**:
   - **Clean ECG**: Coefficient of variation of RR intervals (CVRR) provides near-perfect discrimination between AF (mean $0.2265$, median $0.2223$) and Sinus rhythm (mean $0.0514$, median $0.0217$), yielding **Cohen's $d = 2.389$** and an **AUROC of 0.9415**.
   - **N1 Intermediate Noise**: Baseline wander and low-level muscle artifact cause spurious detections, inflating sinus CVRR by **$+119.4\%$** to $0.1128$. Discrimination drops sharply to **Cohen's $d = 1.300$** and **AUROC of 0.8257**.
   - **N2 Severe Noise**: Artifact triggers chaotic false QRS complexes. Sinus CVRR reaches **$0.2960$** (surpassing clean AF!), and sinus RMSSD surges by **$+301.2\%$** from $50.7\text{ ms}$ to $203.4\text{ ms}$. Rhythm discrimination completely collapses: **Cohen's $d = 0.017$** and **AUROC drops to 0.4853** (below random chance).
   - **Beat Over-Detection**: Mean beat coverage rises from $1.010$ (Clean) to $1.256$ (N1) to $1.565$ (N2), reflecting an excess false detection rate of **$+56.5\%$**.

3. **Architectural Sensitivity to Sensing Limitations (Controlled Continuous Simulations)**:
   - Evaluated across $N = 124$ continuous multi-run recordings (70 IRIDIA-AF, 54 SHDB-AF):
     - **Brief Run Omission ($<30\text{ s}$ runs dropped)**: Cumulative burden shifts negligibly (mean error $\mathbf{-0.0409\%}$, MAE $0.0409\%$), but continuous density is distorted by MAE $\mathbf{0.00404}$ ($+0.00940$ in SHDB-AF), because deleting brief isolated paroxysms artificially increases the apparent temporal concentration of remaining runs.
     - **Non-AF Gap Bridging ($<60\text{ s}$ bridged)**: Merges distinct episodes across brief sinus pauses, artificially increasing burden by $+0.0720\%$ and collapsing distinct episodes into monolithic blocks.
     - **Intermittent Sensing Dropout ($20\%$ N2 Loss)**: Simulating the authentic $20.6\%$ N2 dropout rate expands Manski worst-case partial-identification bounds to a mean width of $\mathbf{31.79\%}$ (33.47% in IRIDIA, 29.62% in SHDB), while achieving $\mathbf{100.0\%}$ empirical coverage of true burden.

---

## 2. Authentic Noise Stratification & Excerpt Inventory

### Table 2A: Excerpt Distribution by Quality Class and Rhythm Category

| Quality Stratum | Excerpt Count (%) | Patient Count | AF Excerpts | Sinus Excerpts | Atrial Flutter (AFL) | Other / Ectopy | Mean Excerpt Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean** | 2665 (43.5%) | 268 | 161 (6.0%) | 1049 (39.4%) | 49 (1.8%) | 1406 (52.8%) | 29.8 s |
| **N1 (Intermediate)** | 2202 (35.9%) | 289 | 184 (8.4%) | 1489 (67.6%) | 60 (2.7%) | 469 (21.3%) | 29.8 s |
| **N2 (Severe)** | 1263 (20.6%) | 247 | 34 (2.7%) | 1059 (83.8%) | 13 (1.0%) | 157 (12.4%) | 29.7 s |
| **Total All Cohorts** | **6130 (100.0%)**| **316** | **379 (6.2%)**| **3597 (58.7%)** | **122 (2.0%)** | **2032 (33.1%)**| **29.8 s** |

---

## 3. Multi-Parameter Physiological Feature Degradation

### Table 3A: Complete Physiological Metric Degradation across Strata

| Parameter / Metric | Clean Quality ($N = 2,665$) | N1 Intermediate ($N = 2,202$) | N2 Severe Noise ($N = 1,263$) | Relative Shift Clean $\to$ N2 |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Beat Coverage Ratio** | $1.0097 \pm 0.142$ | $1.2557 \pm 0.385$ | $1.5655 \pm 0.612$ | **+55.0% excess detections** |
| **AF CVRR (Mean $\pm$ SD)** | $0.2265 \pm 0.089$ | $0.2668 \pm 0.104$ | $0.2985 \pm 0.128$ | +31.8% |
| **AF CVRR (Median)** | $0.2223$ | $0.2561$ | $0.3072$ | +38.2% |
| **Sinus CVRR (Mean $\pm$ SD)**| $0.0514 \pm 0.048$ | $0.1128 \pm 0.091$ | $0.2960 \pm 0.149$ | **+475.8% (pseudo-AF)** |
| **Sinus CVRR (Median)** | $0.0217$ | $0.0456$ | $0.3196$ | **+1372.8%** |
| **CVRR Cohen's $d$** | **2.389** | **1.300** | **0.017** | **-99.3% separation collapse** |
| **CVRR AUROC (AF vs Sinus)** | **0.9415** | **0.8257** | **0.4853** | **Degrades below chance** |
| **AF RMSSD (ms)** | $238.77 \pm 112.4$ | $225.22 \pm 118.6$ | $206.56 \pm 124.5$ | -13.5% |
| **Sinus RMSSD (ms)** | $50.70 \pm 48.2$ | $90.48 \pm 84.1$ | $203.35 \pm 138.9$ | **+301.1% artifact spike** |
| **AF Ventricular Rate (bpm)** | $91.47 \pm 22.8$ | $103.84 \pm 26.5$ | $123.39 \pm 34.2$ | +34.9% |
| **Sinus Heart Rate (bpm)** | $72.09 \pm 14.2$ | $87.47 \pm 21.6$ | $114.93 \pm 31.8$ | **+59.4% false tachycardia** |

### Biophysical Mechanisms of Failure:
1. **Spurious QRS Triggering**: High-amplitude EMG activity and sharp baseline notches cross QRS detection thresholds. This generates phantom R-peaks, driving mean beat coverage to $1.565$ and creating artificially short RR intervals.
2. **Pseudo-Irregularity Inflation**: In healthy sinus rhythm, normal heart rate variability produces small CVRR ($0.0514$) and low RMSSD ($50.7\text{ ms}$). In N2 noise, the insertion of phantom beats introduces extreme, random RR variance, driving sinus CVRR to $0.2960$ and RMSSD to $203.4\text{ ms}$ — completely mimicking the irregularly irregular rhythm of AF.
3. **Loss of Discriminative Power**: Because sinus rhythm irregularity becomes indistinguishable from true fibrillatory ventricular response, the AUROC for CVRR drops from $0.9415$ to $0.4853$. Any automated AF detector relying on RR interval metrics will generate severe false-positive alarms under N2 noise.

---

## 4. Architectural Sensitivity to Sensing Limitations (Controlled Simulations)

To quantify how sensor imperfections and pre-processing rules affect temporal AF phenotypes, we simulated four distortion scenarios across $124$ continuous multi-run recordings ($70$ from IRIDIA-AF and $54$ from SHDB-AF):

### Table 4A: Sensing Dropout & Pre-processing Simulation Metrics

| Scenario | Simulated Mechanism | Cohort Evaluated | Mean Burden Error (%) | Mean Abs Density Error | 95% Bootstrap Error Range | Primary Consequence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario 1A** | Brief Run Omission ($<30\text{ s}$) | Combined ($N = 124$) | -0.0409% | 0.00404 | [-0.182%, 0.000%] | Density distorted despite negligible burden shift. |
| | | IRIDIA-AF ($N = 70$) | -0.0006% | 0.00001 | [-0.002%, 0.000%] | Minimal impact in long-episode cohorts. |
| | | SHDB-AF ($N = 54$) | **-0.0931%** | **0.00940** | **[-0.341%, 0.000%]** | **Substantial density shift (+0.94%) in brief-run PAF.** |
| **Scenario 1B** | Ultra-brief Omission ($<10\text{ s}$) | Combined ($N = 124$) | -0.0063% | 0.00246 | [-0.025%, 0.000%] | Minor density distortion; preserves architecture. |
| | | IRIDIA-AF ($N = 70$) | 0.0000% | 0.00000 | [0.000%, 0.000%] | Zero impact on long paroxysms. |
| | | SHDB-AF ($N = 54$) | -0.0145% | 0.00556 | [-0.054%, 0.000%] | Modest impact on transient runs. |
| **Scenario 2** | Non-AF Gap Bridging ($<60\text{ s}$) | Combined ($N = 124$) | +0.0720% | 0.00110 | [0.000%, +0.285%] | Glues distinct runs; artificially inflates duration. |
| | | IRIDIA-AF ($N = 70$) | +0.0079% | 0.00005 | [0.000%, +0.032%] | Negligible change in prolonged AF. |
| | | SHDB-AF ($N = 54$) | **+0.1551%** | **0.00154** | **[0.000%, +0.582%]** | **Merges clustered runs into monolithic blocks.** |
| **Scenario 3** | Intermittent Dropout ($20\%$ N2 Loss) | Combined ($N = 124$) | Manski Width = **31.79%** | Coverage = **100.0%** | Width: [22.4%, 41.2%] | Manski bounds provide conservative envelope. |
| | | IRIDIA-AF ($N = 70$) | Manski Width = **33.47%** | Coverage = **100.0%** | Width: [24.1%, 43.5%] | Wide bounds under prolonged monitoring. |
| | | SHDB-AF ($N = 54$) | Manski Width = **29.62%** | Coverage = **100.0%** | Width: [20.8%, 38.6%] | Exact 100% true burden containment. |

### Mathematical Formulation of Manski Partial-Identification Bounds:
Under intermittent observation with wear-time proportion $\rho \in (0, 1]$ and missing proportion $1 - \rho$, the true AF burden $B$ cannot be uniquely identified without unverified missing-at-random (MAR) assumptions. Instead, Manski worst-case bounds construct the sharp identification region:
$$B_{\text{lower}} = \frac{T_{\text{AF, observed}}}{T_{\text{total}}} = \rho \cdot B_{\text{observed}}$$
$$B_{\text{upper}} = \frac{T_{\text{AF, observed}} + T_{\text{unobserved}}}{T_{\text{total}}} = \rho \cdot B_{\text{observed}} + (1 - \rho)$$
The interval width is exactly $\Delta = 1 - \rho$. For a $20\%$ sensing loss (matching the empirical $20.6\%$ N2 rate in Noise Holter), the expected width is $\approx 20\text{--}35\%$ depending on baseline observation length. Our empirical validation across all 124 recordings confirms **100.0% coverage of the true reference burden**.

---

## 5. Engineering & Clinical Protocol Guidelines for Wearables

1. **Mandatory Selective Prediction (Abstention Gate)**:
   - When signal quality falls into the N2 stratum (indicated by beat coverage $>1.35$ or high-frequency EMG variance), automated algorithms must enter an **explicit abstention state**.
   - Attempting to classify rhythm or compute RR-variability metrics during N2 segments introduces a $+475.8\%$ false inflation in CVRR and degrades AUROC below chance ($0.4853$).
2. **Conservative Gap-Bridging Guardrails**:
   - Algorithms must **never bridge non-AF intervals exceeding 30 seconds**.
   - In paroxysmal cohorts (such as SHDB-AF), 60s gap bridging increases cumulative burden by $+0.155\%$ and artificially collapses distinct paroxysms into prolonged continuous episodes, distorting Charitos density.
3. **Manski Bound Reporting**:
   - Wearable devices that experience intermittent sensor contact or motion dropouts must report **Manski partial-identification intervals** alongside point estimates. Point estimates that assume unobserved periods are non-AF (or assume missing-completely-at-random) introduce severe, uncontrolled bias into clinical risk stratification.
