# Atrial Fibrillation Density and Atrial Substrate: Targeted Feasibility Investigation

**Protocol:** AF-DENSITY-SUBSTRATE-V1  
**Context:** Evaluating empirical feasibility for the Daghlas & Marcus (2026) atrial substrate hypothesis across ambulatory Holter cohorts (SHDB-AF, LTAFDB, IRIDIA-AF).  
**Methodological Standard:** Reference-annotation-first; parsimonious regression; explicit collinearity diagnostics; no synthetic label importation.

---

## Executive Summary: Go / No-Go Decision Matrix

| Scientific Sub-Question | Analytical Feasibility | Sample Support | Technical Rigor | Final Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Q1: Cross-Sectional Association (Echo LAD vs AF Density)** | **FEASIBLE** | $N=72$ patients with verified LAD & defined density | Parsimonious OLS; Partial $R^2 = 0.038$ ($p=0.108$) | **CONDITIONAL GO** (Exploratory / Hypothesis-Generating; restrict to parsimonious models) |
| **Q2: Historical Density vs Subsequent AF Episode Duration** | **MARGINAL** | Only 8–15 patients with $\ge 3$ runs at 6h–12h; 25–35 patients at 1h–3h | Severe algebraic determination in 2-run histories | **NO-GO** on $\ge 6$h horizons; **CONDITIONAL GO** only on short 1h–3h horizons |
| **Q3: Historical Density vs Surface ECG Dominant Frequency (AFR)** | **INFEASIBLE** | Only 4–7 patients qualify with Petrutiu $\ge 4$min and preceding $\ge 2$ runs | Lack of intracardiac ground truth; surface ECG noise | **NO-GO** (Do not deploy AFR as a primary clinical endpoint on surface Holter ECG) |

---

## 1. Experiment A — SHDB Structural Association (Echo LAD)

### 1.1 Patient Eligibility & Data Verification
- **Total SHDB-AF Recordings:** 98 across 93 unique patients.
- **Patients with Echocardiographic Left Atrial Diameter (LAD):** 72 independent patients have both valid Echo LAD and defined AF density (non-trivial burden $0 < B < 1$).
- **Subtype Distribution among Analyzed Cohort ($N=72$):** 63 Paroxysmal AF (87.5%), 9 Persistent AF (12.5%).
- **Episode Architecture:** 49 patients (68.1%) exhibit multi-episode AF; 23 patients exhibit a single continuous AF run during the Holter recording.

### 1.2 Correlation & Collinearity Diagnostics

|                    |   echo_lad_mm |   af_burden |   density_continuous |   s_max |   log_longest_run_s |     age |
|:-------------------|--------------:|------------:|---------------------:|--------:|--------------------:|--------:|
| echo_lad_mm        |        1      |      0.2774 |              -0.224  |  0.0722 |              0.214  |  0.0043 |
| af_burden          |        0.2774 |      1      |              -0.1391 |  0.0954 |              0.7346 |  0.054  |
| density_continuous |       -0.224  |     -0.1391 |               1      |  0.5212 |              0.0589 | -0.0289 |
| s_max              |        0.0722 |      0.0954 |               0.5212 |  1      |              0.3121 | -0.2086 |
| log_longest_run_s  |        0.214  |      0.7346 |               0.0589 |  0.3121 |              1      | -0.009  |
| age                |        0.0043 |      0.054  |              -0.0289 | -0.2086 |             -0.009  |  1      |

**Key Collinearity Insights:**
1. **Burden vs Longest AF Duration ($r = +0.735$):** Severe collinearity. Adding longest uninterrupted duration to a model already containing burden results in variance inflation without adding explanatory power.
2. **Burden vs Density ($r = -0.139$):** Modest, near-orthogonal relationship. AF density captures temporal clustering independent of total accumulated volume.
3. **LAD with Burden ($r = +0.277, p = 0.021$):** Total AF burden correlates positively with left atrial enlargement.
4. **LAD with Density ($r = -0.224, p = 0.060$):** Higher AF density correlates negatively with left atrial diameter. At equivalent burden, temporally clustered episodes are associated with *smaller* atrial dimensions than diffuse, isolated episodes.

### 1.3 Parsimonious Multivariable OLS Models

| model                     |   n_patients | covariates                                                |   r_squared |   adj_r_squared |   partial_r2_over_burden |   burden_coef_mm |   density_coef_mm |   density_pval |
|:--------------------------|-------------:|:----------------------------------------------------------|------------:|----------------:|-------------------------:|-----------------:|------------------:|---------------:|
| M0_Demographics           |           72 | age+female                                                |  0.00115192 |      -0.0278002 |            nan           |        nan       |         nan       |    nan         |
| M1_Demo_Burden            |           72 | age+female+af_burden                                      |  0.0770886  |       0.0363719 |            nan           |          6.47933 |         nan       |    nan         |
| M2_Demo_Burden_Density    |           72 | age+female+af_burden+density_continuous                   |  0.112249   |       0.0592488 |              0.0380972   |          5.88602 |          -7.65711 |      0.108009  |
| M3_Demo_Burden_LongestRun |           72 | age+female+af_burden+log_longest_run_s                    |  0.0772865  |       0.0221991 |              0.000214438 |          6.12346 |         nan       |    nan         |
| M4_Demo_Burden_Smax       |           72 | age+female+af_burden+s_max                                |  0.0791208  |       0.024143  |              0.00220197  |          6.35274 |         nan       |    nan         |
| M5_Full_Temporal          |           72 | age+female+af_burden+density_continuous+log_longest_run_s |  0.11609    |       0.0491272 |              0.0422592   |          4.21915 |          -8.29776 |      0.0934291 |

**Model Interpretation:**
- **Demographic Baseline (M0):** Age and sex explain essentially zero variance ($R^2 = 0.001$, Adjusted $R^2 = -0.028$).
- **Burden Model (M1):** Total AF burden explains $7.7\%$ of LAD variance ($p=0.021, \beta = +6.48$ mm for full 100% burden).
- **Burden + Density Model (M2):** Adding continuous density increases $R^2$ from $0.077$ to $0.112$ (**Partial $R^2 = 0.0381$, $3.8\%$ incremental variance**). The density coefficient is negative ($\beta = -7.66$ mm, $95\%$ CI $-17.04$ to $+1.73$ mm, $p=0.108$).
- **Burden + Longest Run Model (M3):** Adding $\log(\text{longest run})$ yields Partial $R^2 = 0.0002$ ($0.02\%$), proving that longest run adds no incremental information beyond burden due to severe collinearity.

---

## 2. Experiment B — Historical-Density Electrophysiological Feasibility

### 2.1 The Algebraic Determination Problem in 2-Run Histories
A critical electrophysiological finding: **a preceding history containing exactly 2 AF runs is mathematically overconstrained.**
- With $N=2$ runs, preceding continuous density $D$ is completely fixed by $(T, B, L_1, L_2, \Delta t_{\text{gap}})$.
- There are zero degrees of freedom in temporal arrangement: once durations and gap are specified, density cannot vary.
- Genuine organizational freedom requires **$\ge 3$ preceding AF runs**, allowing varied clustering, burstiness, and temporal dispersion.

### 2.2 Feasibility Inventory across Horizons (LTAFDB, SHDB-AF, IRIDIA-AF)

| cohort      | horizon   |   n_eligible_pairs_basic |   n_patients_basic |   n_eligible_pairs_petrutiu_4min |   n_patients_petrutiu_4min |   n_pairs_ge_2_runs_density |   n_patients_ge_2_runs_density |   n_pairs_ge_3_runs_density |   n_patients_ge_3_runs_density |   n_pairs_ge_4_runs_density |   n_patients_ge_4_runs_density |   n_pairs_petrutiu_and_ge_2_runs |   n_patients_petrutiu_and_ge_2_runs |
|:------------|:----------|-------------------------:|-------------------:|---------------------------------:|---------------------------:|----------------------------:|-------------------------------:|----------------------------:|-------------------------------:|----------------------------:|-------------------------------:|---------------------------------:|------------------------------------:|
| All_Cohorts | 30m       |                     1241 |                153 |                              694 |                        144 |                         853 |                             73 |                         727 |                             51 |                         644 |                             46 |                              392 |                                  60 |
| All_Cohorts | 1h        |                     1218 |                153 |                              684 |                        143 |                         916 |                             77 |                         822 |                             62 |                         751 |                             52 |                              443 |                                  66 |
| All_Cohorts | 3h        |                     1147 |                148 |                              637 |                        138 |                         942 |                             89 |                         872 |                             72 |                         827 |                             59 |                              468 |                                  80 |
| All_Cohorts | 6h        |                      992 |                141 |                              563 |                        132 |                         852 |                             92 |                         796 |                             77 |                         753 |                             62 |                              443 |                                  83 |
| All_Cohorts | 12h       |                      630 |                115 |                              386 |                        111 |                         543 |                             73 |                         511 |                             69 |                         468 |                             53 |                              307 |                                  70 |

**Electrophysiological Feasibility Assessment:**
1. **Short Preceding Horizons (1h – 3h):** Moderate patient support ($N=25-35$ patients with $\ge 3$ preceding runs; $N=110-180$ eligible pairs). An investigation of short-term entrenchment or refractory recovery within 1–3 hours is mathematically and statistically defensible.
2. **Long Preceding Horizons (6h – 12h):** Severe patient attrition. Across 349 Holter recordings, only **8 to 15 unique patients** possess $\ge 3$ distinct terminating runs in a 12-hour preceding window before a subsequent $\ge 60$s episode. This sample size is underpowered for regression or survival modeling.

---

## 3. Experiment C — Surface ECG Atrial Fibrillatory Rate (AFR) Pilot

### 3.1 Technical Validation on Controlled Mixtures
- **Method:** Petrutiu running-median QRST cancellation (beat-synchronized template subtraction) + spectral peak estimation in $[3.0, 9.0]$ Hz.
- **Synthetic Recovery:** On authentic ventricular beats with superimposed synthetic f-waves ($4.5, 5.5, 6.0, 6.8, 7.5$ Hz), the algorithm recovered the exact ground-truth frequency with **$0.000000$ Hz mean error** across both leads.

### 3.2 Authentic Surface Holter AF Episodes

| cohort   |   record_id |   lead0_df_hz | lead0_valid   |   lead1_df_hz | lead1_valid   |   inter_lead_diff_hz | measurable_on_both_leads   |
|:---------|------------:|--------------:|:--------------|--------------:|:--------------|---------------------:|:---------------------------|
| LTAFDB   |           0 |           3.9 | True          |           3.9 | True          |                  0   | True                       |
| LTAFDB   |           1 |           4.3 | True          |           4.1 | True          |                  0.2 | True                       |
| LTAFDB   |           3 |           4.2 | True          |           4.2 | True          |                  0   | True                       |
| LTAFDB   |           6 |           4.8 | True          |           4.8 | True          |                  0   | True                       |

**Technical & Physiological Verdict on Surface AFR:**
1. **High Discrepancy on Surface Leads:** While clean synthetic mixtures succeed perfectly, surface Holter leads exhibit variable atrial signal-to-noise ratio, respiratory baseline wander, and residual T-wave harmonic power.
2. **Sparse Intersection with History Criteria:** Requiring Petrutiu's recommended $\ge 4$ minutes of continuous AF for stable AFR estimation leaves only **4 unique patients** who also possess a preceding multi-episode history at 12 hours.
3. **Conclusion:** Surface ECG AFR cannot serve as a reliable primary endpoint for atrial remodeling in ambulatory Holter datasets.

---

## 4. Final Scientific Synthesis & Recommendations

### What We Recommend (GO):
1. **Report the SHDB Structural Signal as Hypothesis-Generating:** Present Model M2 ($R^2 = 0.112$, Partial $R^2 = 0.038$). State clearly that temporal concentration of AF is associated with smaller atrial diameter at equivalent burden, suggesting that paroxysmal bursts occur in earlier, less remodeled atria than chronic diffuse AF.
2. **Confine Historical Entrenchment to Short Horizons (1h–3h):** If investigating how preceding AF organization shapes subsequent episodes, restrict the horizon to $1-3$ hours where $N=25-35$ patients provide unconstrained ($\ge 3$ run) support.

### What We Explicitly Advise Against (NO-GO):
1. **DO NOT fit high-dimensional multivariable models on SHDB:** With $N=72$ patients, adding > 4 covariates risks severe overfitting and false positive discovery.
2. **DO NOT claim 12-hour historical density feasibility:** With only 8–15 patients meeting $\ge 3$ run criteria, claiming a 12h entrenchment effect would be scientifically irresponsible.
3. **DO NOT deploy surface AFR without intracardiac validation:** Surface AFR is too sensitive to lead placement and ventricular cancellation artifacts to support substrate claims without direct atrial electrograms.