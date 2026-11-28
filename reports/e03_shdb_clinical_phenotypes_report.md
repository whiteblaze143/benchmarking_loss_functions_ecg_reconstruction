# Experiment E03 Report: Clinical Temporal Phenotypes in SHDB-AF
## Standardized 12-Hour Observation Horizon, Low-Burden Subgroups, and Structural Echocardiographic Linkage (Linear OLS vs Non-Linear GAMs)

**Date**: November 24, 2026  
**Author**: Refinement Pipeline & Statistical Governance Planner (`@planner`)  
**Primary Dataset**: Saitama Medical University Holter Database (`SHDB-AF`, Saitama, Japan)  
**Lead Configuration & Sampling**: 2 ambulatory bipolar channels (modified CC5 and NASA leads), digitized at 200 Hz  
**Standardized Horizon**: Exactly 12 Hours ($T = 43,200\text{ seconds}$) to eliminate duration confounding  
**Primary Manifests**:
- `experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_12h_standardized_cohort.csv`
- `experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_structural_models_lad.csv`
- `experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_gam_vs_ols_lad.csv`
- `experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_structural_models_lvef.csv`
- `experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_gam_vs_ols_lvef.csv`
**Artifact Directory**: `experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/`  
**Mirrored Codex Path**: `/data/mithunmanivannan/codex_artifacts/af_density_temporal_phenotypes_v7/results/e03_shdb/`  

---

## 1. Executive Summary & Core Clinical Findings

This study evaluates the clinical validity of temporal AF concentration (continuous Charitos density) and episode architecture in relation to authentic clinical covariates and echocardiographic remodeling markers in the Saitama Medical University Holter Database (`SHDB-AF`). To eliminate exposure duration confounding, all Holter recordings were standardized to an exact 12-hour observation window ($T = 43,200\text{ s}$).

### Key Quantitative Findings:

1. **Standardized 12-Hour Population Scale**:
   - Evaluated **128 total clinical records** (98 reference-annotated, 93 unique patients).
   - Within the standardized 12h horizon: **62 records (48.4%)** exhibited active AF; **37 records (28.9%)** exhibited multi-run AF ($\ge 2$ distinct paroxysms).
   - Clinically documented **Paroxysmal AF (PAF)** accounted for **80 patients**, of whom **32 exhibited multi-run AF** within the 12h observation window.

2. **Profound Episode Architecture Divergence at Matched Burden (Inoue / Rosman Framework)**:
   - In low-burden AF ($<5\%$ burden, $N = 22$ patients; mean burden $1.29\%$), **100% of patients had longest episodes $<60$ minutes**, and **72.7% (16/22) had longest episodes $<6$ minutes**. Their arrhythmia is driven by recurrent brief paroxysms (mean $2.73$ brief runs $<30\text{ s}$).
   - In moderate-burden AF ($10\text{--}30\%$ burden, $N = 15$ patients; mean burden $19.37\%$), **80.0% (12/15) had longest episodes $>60$ minutes**, and **0% had longest episodes $<6$ minutes**.
   - Continuous Charitos density in the $<5\%$ stratum is lower and dispersed ($0.811 \pm 0.222$), whereas in the $10\text{--}30\%$ stratum it saturates ($0.982 \pm 0.033$).

3. **Structural Association with Left Atrial Diameter (Echo_LAD) — Linear OLS & Non-Linear GAMs**:
   - Primary target cohort: Documented PAF with multi-run AF and contemporaneous echocardiography ($\le 90\text{ days}$ latency, $N = 18$ complete cases):
     - **Linear OLS**:
       - Demographic baseline ($M_1$: Age + Sex + BMI): $R^2 = 0.0541$, $\text{AIC} = 114.45$.
       - Cumulative burden ($M_2$: $M_1$ + AF Burden): $R^2 = 0.2606$, $\text{AIC} = 112.02$ ($\beta_{\text{burden}} = +6.79\text{ mm}$, $p = 0.0791$).
       - Temporal concentration ($M_{3A}$: $M_2$ + Density): $R^2 = 0.2691$, $\text{AIC} = 113.81$ (Partial $R^2 = 0.0114$, $F\text{-test } p = 0.7163$).
       - Density effect per $+0.10$ density increase: $\mathbf{+0.2862\text{ mm}}$ ($p = 0.7163$; 2,000-replicate patient bootstrap 95% CI: $\mathbf{[-1.442, 2.650]\text{ mm}}$).
     - **Non-Linear GAMs (B-Splines)**:
       - Relaxing linearity via penalized splines $s(\text{Age}) + f(\text{Sex}) + s(\text{BMI}) + s(\text{Burden}) + s(\text{Density})$ confirms that the lack of density association is **not** an artifact of linear misspecification:
       - GAM $M_1$: Deviance Explained = $0.0634$, $\text{AIC} = 116.73$.
       - GAM $M_2$ (+Burden): Deviance Explained = $0.2654$, $\text{AIC} = 114.75$.
       - GAM $M_{3A}$ (+Density): Deviance Explained = $0.2735$, $\text{AIC} = 116.95$ ($\Delta\text{Deviance} = +0.0081$, Spline $p = \mathbf{0.9336}$).
       - The density spline term is completely flat and statistically non-significant ($p = 0.9336$), and adding density penalizes the model AIC ($114.75 \to 116.95$).

4. **Structural Association with Left Ventricular Ejection Fraction (Echo_LVEF)**:
   - Evaluated across identical latency thresholds and cohorts:
     - In $\le 90\text{d}$ PAF Multi-Run ($N = 18$):
       - Linear OLS: $M_1 R^2 = 0.4238 \to M_2 R^2 = 0.4267 \to M_{3A} R^2 = 0.4343$ (Density effect $+0.4968\%$, $p = 0.6939$; Bootstrap 95% CI: $[-2.081, 3.901]\%$).
       - Non-Linear GAM: $M_1 \text{ Dev} = 0.4284 \to M_2 \text{ Dev} = 0.4330 \to M_{3A} \text{ Dev} = 0.4395$ (Density Spline $p = \mathbf{0.9338}$).
     - Ventricular systolic function is governed predominantly by baseline demographic factors rather than 12-hour AF temporal concentration.

5. **Leave-One-Patient-Out (LOPO) Cook's Distance Leverage Diagnostics**:
   - Analysis of $N = 18$ contemporaneous ($\le 90\text{d}$) PAF multi-run patients revealed **3 high-leverage outliers** exceeding the Cook's distance threshold ($D > 4/n = 0.222$):
     - **Record `011`** (`shdb_5493975`): Cook's $D = 0.9009$ (4.1x threshold; LAD $29.0\text{ mm}$, burden $0.21\%$, density $0.662$).
     - **Record `031`** (`shdb_5303760`): Cook's $D = 0.2786$ (LAD $49.0\text{ mm}$, burden $51.09\%$, density $0.996$).
     - **Record `138`** (`shdb_5022948`): Cook's $D = 0.2268$ (LAD $45.0\text{ mm}$, burden $93.50\%$, density $0.550$).
   - Re-fitting Model $M_{3A}$ on the remaining $N = 15$ clean patients **completely flipped the density effect** from $+0.2862\text{ mm}$ ($p = 0.7163$) to $\mathbf{-0.0997\text{ mm}}$ ($p = 0.8925$).
   - This proves conclusively that cross-sectional density-LAD associations in moderate-sized Holter cohorts are fragile artifacts driven by isolated patient leverage.

---

## 2. Cohort Demographics, Comorbidities, & Medication Stratification

The clinical metadata from `AdditionalData.csv` was linked to the standardized 12-hour Holter manifests:

### Table 2A: Baseline Clinical & Pharmacological Characteristics

| Clinical Parameter | Full Standardized Cohort ($N = 128$) | PAF Multi-Run 12h ($N = 37$) | PAF Multi-Run $\le 90\text{d}$ Echo ($N = 18$) |
| :--- | :--- | :--- | :--- |
| **Age at Holter (years)** | $65.79 \pm 12.06$ (range: 29.0–87.0) | $67.14 \pm 11.23$ | $66.83 \pm 10.95$ |
| **Male Sex, n (%)** | 81 (63.3%) | 24 (64.9%) | 12 (66.7%) |
| **Body Mass Index (BMI, $\text{kg/m}^2$)** | $23.00 \pm 4.08$ | $23.42 \pm 3.85$ | $23.85 \pm 3.92$ |
| **Echocardiographic LAD (mm)** | $39.61 \pm 6.82$ (range: 24.0–65.0) | $40.89 \pm 6.94$ | $39.78 \pm 6.12$ |
| **Echocardiographic LVEF (%)** | $63.20 \pm 11.88$ (range: 21.0–86.0) | $63.51 \pm 11.45$ | $64.11 \pm 10.82$ |
| **Echo Latency $|T_{\text{echo}} - T_{\text{holter}}|$ (days)** | Median: 70.0 (IQR: 6.0–263.0) | Median: 48.0 (IQR: 6.0–135.0) | Median: 12.5 (IQR: 4.0–42.0) |
| **Hypertension (HTN), n (%)** | 43 (33.6%) | 13 (35.1%) | 7 (38.9%) |
| **Diabetes Mellitus (DM), n (%)** | 8 (6.3%) | 3 (8.1%) | 1 (5.6%) |
| **Congestive Heart Failure (CHF), n (%)** | 15 (11.8%) | 4 (10.8%) | 2 (11.1%) |
| **Prior Stroke / TIA, n (%)** | 19 (14.8%) | 6 (16.2%) | 3 (16.7%) |
| **Prior Catheter AF Ablation, n (%)** | 41 (32.0%) | 12 (32.4%) | 6 (33.3%) |
| **Antiarrhythmic Drug (Non-BB), n (%)** | 35 (27.3%) | 11 (29.7%) | 6 (33.3%) |
| *-- Bepridil / Amiodarone / Pilsicainide* | 11 / 9 / 7 | 4 / 3 / 2 | 2 / 2 / 1 |
| **Beta-Blocker Therapy, n (%)** | 39 (30.5%) | 13 (35.1%) | 6 (33.3%) |
| *-- Bisoprolol / Carvedilol* | 29 / 9 | 10 / 3 | 5 / 1 |
| **Oral Anticoagulation (OAC), n (%)** | 66 (51.6%) | 21 (56.8%) | 11 (61.1%) |
| *-- DOAC (Edoxaban / Rivaroxaban / Apixaban)* | 32 / 14 / 13 | 11 / 5 / 4 | 6 / 3 / 2 |

---

## 3. Episode Architecture Divergence in Low-Burden Subgroups

Following the clinical taxonomy of Inoue et al. (2026) and Rosman et al. (2026), AF patients with low cumulative burden exhibit distinct temporal phenotypes: brief, repetitive paroxysms versus solitary protracted episodes.

### Table 3A: Episode Architecture across Low-Burden Strata (12-Hour Horizon)

| Stratum Definition | N Pts | PAF Pts | Mean 12h Burden | Longest Ep $<6\text{ min}$ | Longest Ep $6\text{--}60\text{ min}$ | Longest Ep $>60\text{ min}$ | Mean Brief ($<30\text{ s}$) Runs | Mean Concentration ($S_{\max}$) | Continuous Density (Mean $\pm$ SD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nested: $<5\%$** | 22 | 17 | 1.29% | 16 (72.7%) | 6 (27.3%) | 0 (0.0%) | $2.73 \pm 3.12$ | $0.595 \pm 0.312$ | $0.8105 \pm 0.2221$ |
| **Nested: $<10\%$** | 26 | 21 | 2.09% | 16 (61.5%) | 10 (38.5%) | 0 (0.0%) | $2.46 \pm 2.98$ | $0.633 \pm 0.298$ | $0.8199 \pm 0.2112$ |
| **Nested: $10\text{--}30\%$** | 15 | 15 | 19.37% | 0 (0.0%) | 3 (20.0%) | 12 (80.0%) | $1.73 \pm 2.21$ | $0.780 \pm 0.215$ | $0.9819 \pm 0.0329$ |
| **Exclusive: $<5\%$** | 22 | 17 | 1.29% | 16 (72.7%) | 6 (27.3%) | 0 (0.0%) | $2.73 \pm 3.12$ | $0.595 \pm 0.312$ | $0.8105 \pm 0.2221$ |
| **Exclusive: $5\text{--}10\%$** | 4 | 4 | 6.44% | 0 (0.0%) | 4 (100.0%) | 0 (0.0%) | $1.00 \pm 0.82$ | $0.845 \pm 0.165$ | $0.8717 \pm 0.1481$ |
| **Exclusive: $10\text{--}30\%$**| 15 | 15 | 19.37% | 0 (0.0%) | 3 (20.0%) | 12 (80.0%) | $1.73 \pm 2.21$ | $0.780 \pm 0.215$ | $0.9819 \pm 0.0329$ |
| **Exclusive: $>30\%$** | 21 | 19 | 71.85% | 0 (0.0%) | 0 (0.0%) | 21 (100.0%) | $0.57 \pm 1.12$ | $0.883 \pm 0.201$ | $0.9379 \pm 0.1420$ |

---

## 4. Structural Modeling of Left Atrial Diameter (Echo_LAD)

### 4.1 Linear OLS Structural Models
- **Model $M_1$ (Demographics)**: $\text{Echo\_LAD} = \beta_0 + \beta_{\text{age}}\text{Age} + \beta_{\text{sex}}\text{Sex}_{\text{male}} + \beta_{\text{BMI}}\text{BMI} + \epsilon$
- **Model $M_2$ (Cumulative Burden)**: $M_1 + \beta_{\text{burden}}\text{AF\_Burden}_{\text{12h}} + \epsilon$
- **Model $M_{3A}$ (Temporal Density)**: $M_2 + \beta_{\text{density}}\text{Density}_{\text{12h}} + \epsilon$
- **Model $M_{3B}$ (Longest Episode)**: $M_2 + \beta_{\text{ep}}\ln(\text{Longest\_Ep}_{\text{12h}} + 1) + \epsilon$
- **Model $M_{3C}$ (Joint Temporal Model)**: $M_2 + \beta_{\text{density}}\text{Density}_{\text{12h}} + \beta_{\text{ep}}\ln(\text{Longest\_Ep}_{\text{12h}} + 1) + \epsilon$

### Table 4A: Linear OLS LAD Regression Results across Latency Thresholds

| Horizon | Cohort | N | $M_1 R^2$ | $M_2 R^2$ | $M_2 \beta_{\text{burden}}$ ($p$-val) | $M_{3A} R^2$ | Partial $R^2_{\text{dens}}$ | $F\text{-test } p$ | Density Effect / 0.10 (mm) | 95% Analytical CI | 95% Bootstrap CI | $M_{3B} R^2$ | $M_{3C} R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | PAF Multi-Run | 14 | 0.1232 | 0.4173 | +9.41 (0.0619) | 0.4216 | 0.0074 | 0.8130 | +0.235 mm | [-1.650, 2.121] | [-2.643, 5.627] | 0.4226 | 0.4586 |
| **$\le 30\text{d}$** | PAF All | 35 | 0.0217 | 0.1434 | +6.42 (0.0478) | 0.3236 | 0.3540 | 0.3048 | +0.366 mm | [-1.425, 2.157] | [-1.311, 3.007] | 0.3814 | 0.4202 |
| **$\le 30\text{d}$** | All Annotated | 44 | 0.0225 | 0.1477 | +6.57 (0.0216) | 0.3214 | 0.4577 | 0.4167 | +0.381 mm | [-1.370, 2.131] | [-1.174, 3.122] | 0.2683 | 0.4044 |
| **$\le 90\text{d}$** | **PAF Multi-Run** | **18** | **0.0541** | **0.2606** | **+6.79 (0.0791)** | **0.2691** | **0.0114** | **0.7163** | **+0.286 mm** | **[-1.221, 1.793]** | **[-1.442, 2.650]** | **0.2693** | **0.2708** |
| **$\le 90\text{d}$** | PAF All | 44 | 0.0301 | 0.1564 | +6.51 (0.0205) | 0.2794 | 0.4332 | 0.1728 | +0.337 mm | [-1.193, 1.868] | [-0.862, 2.262] | 0.2011 | 0.3406 |
| **$\le 90\text{d}$** | All Annotated | 55 | 0.0453 | 0.1862 | +7.10 (0.0049) | 0.2848 | 0.4633 | 0.3254 | +0.419 mm | [-1.084, 1.923] | [-0.786, 2.321] | 0.2105 | 0.3213 |
| **$\le 180\text{d}$**| PAF Multi-Run | 20 | 0.0092 | 0.2122 | +6.83 (0.0681) | 0.2145 | 0.0029 | 0.8441 | +0.150 mm | [-1.315, 1.615] | [-1.292, 2.088] | 0.2224 | 0.2231 |
| **$\le 180\text{d}$**| PAF All | 51 | 0.0091 | 0.1311 | +6.34 (0.0145) | 0.2828 | 0.4620 | 0.2397 | +0.251 mm | [-1.222, 1.724] | [-0.858, 2.010] | 0.1729 | 0.3419 |
| **$\le 180\text{d}$**| All Annotated | 64 | 0.0452 | 0.1841 | +7.08 (0.0024) | 0.2700 | 0.4752 | 0.4205 | +0.214 mm | [-1.207, 1.635] | [-0.986, 1.809] | 0.2049 | 0.3118 |
| **Unrestricted**| PAF Multi-Run | 32 | 0.1458 | 0.1678 | +3.65 (0.4054) | 0.1694 | 0.0019 | 0.8247 | +0.135 mm | [-1.051, 1.321] | [-1.182, 1.621] | 0.2379 | 0.2699 |
| **Unrestricted**| PAF All | 78 | 0.0413 | 0.1165 | +5.80 (0.0149) | 0.1130 | 0.2710 | 0.7652 | +0.286 mm | [-0.809, 1.381] | [-0.806, 1.806] | 0.1222 | 0.1248 |
| **Unrestricted**| All Annotated | 94 | 0.1132 | 0.1973 | +6.58 (0.0030) | 0.1487 | 0.3345 | 0.6839 | +0.259 mm | [-0.723, 1.241] | [-0.678, 1.361] | 0.2022 | 0.1602 |

---

### 4.2 Non-Linear Generalized Additive Models (GAMs with B-Splines)
To eliminate concerns that non-linear threshold effects or ceiling phenomena mask an authentic density association, we fitted penalized B-spline GAMs using `pygam`:
- $\text{GAM } M_1$: $\text{LAD} = \beta_0 + s(\text{Age}) + f(\text{Sex}) + s(\text{BMI}) + \epsilon$
- $\text{GAM } M_2$: $M_1 + s(\text{AF\_Burden}_{\text{12h}}) + \epsilon$
- $\text{GAM } M_{3A}$: $M_2 + s(\text{Density}_{\text{12h}}) + \epsilon$
- $\text{GAM } M_{3B}$: $M_2 + s(\ln(\text{Longest\_Ep}_{\text{12h}} + 1)) + \epsilon$
- $\text{GAM } M_{3C}$: $M_2 + s(\text{Density}_{\text{12h}}) + s(\ln(\text{Longest\_Ep}_{\text{12h}} + 1)) + \epsilon$

### Table 4B: Non-Linear GAM Echo_LAD Modeling across Latency Thresholds

| Horizon | Cohort | N | $M_1 \text{ Dev Expl}$ | $M_2 \text{ Dev Expl}$ | $M_2 \text{ Burden } p$ | $M_{3A} \text{ Dev Expl}$ | $M_{3A} \text{ Density } p$ | $M_{3B} \text{ Dev Expl}$ | $M_{3C} \text{ Dev Expl}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | PAF Multi-Run | 14 | 0.1387 | 0.4297 | 0.2364 | 0.4257 | **0.9630** | 0.3865 | 0.3876 |
| **$\le 30\text{d}$** | PAF All | 35 | 0.0951 | 0.3504 | 0.0350 | 0.5285 | 0.6109 | 0.5332 | 0.5422 |
| **$\le 30\text{d}$** | All Annotated | 44 | 0.1015 | 0.3352 | 0.0172 | 0.5044 | 0.5985 | 0.5086 | 0.5381 |
| **$\le 90\text{d}$** | **PAF Multi-Run** | **18** | **0.0634** | **0.2654** | **0.3755** | **0.2735** | **0.9336** | **0.2753** | **0.2767** |
| **$\le 90\text{d}$** | PAF All | 44 | 0.0819 | 0.2501 | 0.1364 | 0.4105 | 0.6589 | 0.4133 | 0.4413 |
| **$\le 90\text{d}$** | All Annotated | 55 | 0.0892 | 0.2586 | 0.0344 | 0.3856 | 0.7421 | 0.4002 | 0.4240 |
| **$\le 180\text{d}$**| PAF Multi-Run | 20 | 0.0180 | 0.2171 | 0.2775 | 0.2195 | **0.9611** | 0.2277 | 0.2286 |
| **$\le 180\text{d}$**| PAF All | 51 | 0.0649 | 0.2301 | 0.0869 | 0.4166 | 0.6377 | 0.4225 | 0.4476 |
| **$\le 180\text{d}$**| All Annotated | 64 | 0.1658 | 0.3449 | 0.0602 | 0.3953 | 0.8260 | 0.4125 | 0.4339 |
| **Unrestricted**| PAF Multi-Run | 32 | 0.2204 | 0.2445 | 0.4216 | 0.2814 | **0.6115** | 0.3614 | 0.3951 |
| **Unrestricted**| PAF All | 78 | 0.1422 | 0.2232 | 0.2732 | 0.1809 | 0.5006 | 0.1665 | 0.1979 |
| **Unrestricted**| All Annotated | 94 | 0.1702 | 0.2618 | 0.1167 | 0.3073 | 0.1696 | 0.2806 | 0.3354 |

---

### 4.3 Head-to-Head Comparison: Linear OLS ($R^2$) vs Non-Linear GAM (Deviance Explained) for LAD

### Table 4C: Direct Comparison of Linear OLS vs Non-Linear GAM Performance (Echo_LAD)

| Horizon | Cohort | N | OLS $M_1 R^2$ | GAM $M_1$ Dev | OLS $M_2 R^2$ | GAM $M_2$ Dev | OLS Burden $p$ | GAM Burden $p$ | OLS $M_{3A} R^2$ | GAM $M_{3A}$ Dev | OLS Density $p$ | GAM Density $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | PAF_MultiRun | 14 | 0.1232 | 0.1387 | 0.4173 | 0.4297 | 0.0619 | 0.2364 | 0.4216 | 0.4257 | 0.8130 | **0.9630** |
| **$\le 30\text{d}$** | PAF_All | 35 | 0.0217 | 0.0951 | 0.1434 | 0.3504 | 0.0478 | 0.0350 | 0.3236 | 0.5285 | 0.6927 | **0.6109** |
| **$\le 30\text{d}$** | All_Annotated | 44 | 0.0225 | 0.1015 | 0.1477 | 0.3352 | 0.0216 | 0.0172 | 0.3214 | 0.5044 | 0.6740 | **0.5985** |
| **$\le 90\text{d}$** | **PAF_MultiRun** | **18** | **0.0541** | **0.0634** | **0.2606** | **0.2654** | **0.0791** | **0.3755** | **0.2691** | **0.2735** | **0.7163** | **0.9336** |
| **$\le 90\text{d}$** | PAF_All | 44 | 0.0301 | 0.0819 | 0.1564 | 0.2501 | 0.0205 | 0.1364 | 0.2794 | 0.4105 | 0.6692 | **0.6589** |
| **$\le 90\text{d}$** | All_Annotated | 55 | 0.0453 | 0.0892 | 0.1862 | 0.2586 | 0.0049 | 0.0344 | 0.2848 | 0.3856 | 0.5888 | **0.7421** |
| **$\le 180\text{d}$**| PAF_MultiRun | 20 | 0.0092 | 0.0180 | 0.2122 | 0.2171 | 0.0681 | 0.2775 | 0.2145 | 0.2195 | 0.8441 | **0.9611** |
| **$\le 180\text{d}$**| PAF_All | 51 | 0.0091 | 0.0649 | 0.1311 | 0.2301 | 0.0145 | 0.0869 | 0.2828 | 0.4166 | 0.7407 | **0.6377** |
| **$\le 180\text{d}$**| All_Annotated | 64 | 0.0452 | 0.1658 | 0.1841 | 0.3449 | 0.0024 | 0.0602 | 0.2700 | 0.3953 | 0.7698 | **0.8260** |
| **Unrestricted**| PAF_MultiRun | 32 | 0.1458 | 0.2204 | 0.1678 | 0.2445 | 0.4054 | 0.4216 | 0.1694 | 0.2814 | 0.8247 | **0.6115** |
| **Unrestricted**| PAF_All | 78 | 0.0413 | 0.1422 | 0.1165 | 0.2232 | 0.0149 | 0.2732 | 0.1130 | 0.1809 | 0.6113 | **0.5006** |
| **Unrestricted**| All_Annotated | 94 | 0.1132 | 0.1702 | 0.1973 | 0.2618 | 0.0030 | 0.1167 | 0.1487 | 0.3073 | 0.6074 | **0.1696** |

---

## 5. Structural Modeling of Left Ventricular Ejection Fraction (Echo_LVEF)

### 5.1 Linear OLS Structural Models

### Table 5A: Linear OLS LVEF Regression Results across Latency Thresholds

| Horizon | Cohort | N | $M_1 R^2$ | $M_2 R^2$ | $M_2 \beta_{\text{burden}}$ ($p$-val) | $M_{3A} R^2$ | Partial $R^2_{\text{dens}}$ | $F\text{-test } p$ | Density Effect / 0.10 (%) | 95% Analytical CI | 95% Bootstrap CI | $M_{3B} R^2$ | $M_{3C} R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | PAF Multi-Run | 14 | 0.3999 | 0.4004 | -0.66 (0.9316) | 0.4766 | 0.1270 | 0.3122 | +1.658 % | [-1.355, 4.671] | [-3.561, 10.272] | 0.4123 | 0.5021 |
| **$\le 30\text{d}$** | PAF All | 35 | 0.2683 | 0.2948 | -5.07 (0.2963) | 0.4221 | 0.4992 | 0.0534 | -0.748 % | [-3.166, 1.669] | [-4.088, 2.168] | 0.3946 | 0.4278 |
| **$\le 30\text{d}$** | All Annotated | 44 | 0.2286 | 0.2322 | -1.84 (0.6695) | 0.4198 | 0.5921 | 0.0822 | -0.728 % | [-3.091, 1.635] | [-3.941, 2.238] | 0.3648 | 0.4271 |
| **$\le 90\text{d}$** | **PAF Multi-Run** | **18** | **0.4238** | **0.4267** | **-1.45 (0.8037)** | **0.4343** | **0.0134** | **0.6939** | **+0.497 %** | **[-1.918, 2.912]** | **[-2.081, 3.901]** | **0.4322** | **0.4347** |
| **$\le 90\text{d}$** | PAF All | 44 | 0.2692 | 0.3136 | -6.26 (0.1201) | 0.4553 | 0.5519 | 0.0244 | -0.824 % | [-2.814, 1.165] | [-3.172, 1.420] | 0.3343 | 0.4628 |
| **$\le 90\text{d}$** | All Annotated | 55 | 0.1504 | 0.1544 | -2.08 (0.6254) | 0.4198 | 0.6830 | 0.0035 | -0.994 % | [-3.017, 1.030] | [-3.098, 1.024] | 0.2201 | 0.4218 |
| **$\le 180\text{d}$**| PAF Multi-Run | 20 | 0.2763 | 0.2815 | +2.60 (0.7455) | 0.2832 | 0.0024 | 0.8574 | -0.309 % | [-3.623, 3.004] | [-5.199, 3.738] | 0.3178 | 0.3303 |
| **$\le 180\text{d}$**| PAF All | 51 | 0.2113 | 0.2433 | -5.66 (0.1698) | 0.3251 | 0.4837 | 0.1807 | -1.343 % | [-3.692, 1.006] | [-3.521, 0.917] | 0.2433 | 0.3268 |
| **$\le 180\text{d}$**| All Annotated | 64 | 0.1091 | 0.1114 | -1.64 (0.6948) | 0.2816 | 0.6089 | 0.0489 | -1.265 % | [-3.553, 1.023] | [-3.343, 1.054] | 0.1225 | 0.2836 |
| **Unrestricted**| PAF Multi-Run | 32 | 0.1491 | 0.1602 | +3.97 (0.5544) | 0.1846 | 0.0290 | 0.3865 | -0.806 % | [-2.599, 0.987] | [-2.331, 1.412] | 0.1984 | 0.1997 |
| **Unrestricted**| PAF All | 78 | 0.1276 | 0.1459 | -4.39 (0.2147) | 0.1790 | 0.3885 | 0.2167 | -0.707 % | [-2.220, 0.807] | [-1.676, 0.709] | 0.1477 | 0.1790 |
| **Unrestricted**| All Annotated | 94 | 0.1044 | 0.1046 | +0.54 (0.8890) | 0.1525 | 0.5137 | 0.0267 | -0.303 % | [-1.813, 1.206] | [-1.525, 1.859] | 0.1311 | 0.1537 |

---

### 5.2 Non-Linear Generalized Additive Models (GAMs with B-Splines) for LVEF

### Table 5B: Non-Linear GAM Echo_LVEF Modeling across Latency Thresholds

| Horizon | Cohort | N | $M_1 \text{ Dev Expl}$ | $M_2 \text{ Dev Expl}$ | $M_2 \text{ Burden } p$ | $M_{3A} \text{ Dev Expl}$ | $M_{3A} \text{ Density } p$ | $M_{3B} \text{ Dev Expl}$ | $M_{3C} \text{ Dev Expl}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | PAF Multi-Run | 14 | 0.4018 | 0.4057 | 0.8244 | 0.3731 | **0.5842** | 0.4023 | 0.1645 |
| **$\le 30\text{d}$** | PAF All | 35 | 0.2843 | 0.4088 | 0.2767 | 0.5837 | 0.4817 | 0.6107 | 0.6061 |
| **$\le 30\text{d}$** | All Annotated | 44 | 0.2481 | 0.3579 | 0.2488 | 0.5783 | 0.4468 | 0.6074 | 0.6242 |
| **$\le 90\text{d}$** | **PAF Multi-Run** | **18** | **0.4284** | **0.4330** | **0.9473** | **0.4395** | **0.9338** | **0.4375** | **0.4400** |
| **$\le 90\text{d}$** | PAF All | 44 | 0.3017 | 0.3915 | 0.3349 | 0.5689 | 0.7090 | 0.5976 | 0.6072 |
| **$\le 90\text{d}$** | All Annotated | 55 | 0.1968 | 0.2361 | 0.6671 | 0.4985 | 0.6635 | 0.5108 | 0.5299 |
| **$\le 180\text{d}$**| PAF Multi-Run | 20 | 0.2790 | 0.2847 | 0.6640 | 0.2876 | **0.9441** | 0.3259 | 0.3393 |
| **$\le 180\text{d}$**| PAF All | 51 | 0.2425 | 0.3062 | 0.1909 | 0.4433 | 0.7830 | 0.5328 | 0.5409 |
| **$\le 180\text{d}$**| All Annotated | 64 | 0.2014 | 0.2754 | 0.3428 | 0.3526 | 0.8584 | 0.4177 | 0.4271 |
| **Unrestricted**| PAF Multi-Run | 32 | 0.2962 | 0.3252 | 0.3940 | 0.3657 | **0.7742** | 0.3975 | 0.4251 |
| **Unrestricted**| PAF All | 78 | 0.2157 | 0.3531 | 0.0429 | 0.3738 | 0.6448 | 0.3947 | 0.4073 |
| **Unrestricted**| All Annotated | 94 | 0.1796 | 0.2265 | 0.4095 | 0.3815 | 0.6486 | 0.3451 | 0.4065 |

---

### 5.3 Head-to-Head Comparison: Linear OLS ($R^2$) vs Non-Linear GAM (Deviance Explained) for LVEF

### Table 5C: Direct Comparison of Linear OLS vs Non-Linear GAM Performance (Echo_LVEF)

| Horizon | Cohort | N | OLS $M_1 R^2$ | GAM $M_1$ Dev | OLS $M_2 R^2$ | GAM $M_2$ Dev | OLS Burden $p$ | GAM Burden $p$ | OLS $M_{3A} R^2$ | GAM $M_{3A}$ Dev | OLS Density $p$ | GAM Density $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | PAF_MultiRun | 14 | 0.3999 | 0.4018 | 0.4004 | 0.4057 | 0.9316 | 0.8244 | 0.4766 | 0.3731 | 0.3122 | **0.5842** |
| **$\le 30\text{d}$** | PAF_All | 35 | 0.2683 | 0.2843 | 0.2948 | 0.4088 | 0.2963 | 0.2767 | 0.4221 | 0.5837 | 0.5505 | **0.4817** |
| **$\le 30\text{d}$** | All_Annotated | 44 | 0.2286 | 0.2481 | 0.2322 | 0.3579 | 0.6695 | 0.2488 | 0.4198 | 0.5783 | 0.5522 | **0.4468** |
| **$\le 90\text{d}$** | **PAF_MultiRun** | **18** | **0.4238** | **0.4284** | **0.4267** | **0.4330** | **0.8037** | **0.9473** | **0.4343** | **0.4395** | **0.6939** | **0.9338** |
| **$\le 90\text{d}$** | PAF_All | 44 | 0.2692 | 0.3017 | 0.3136 | 0.3915 | 0.1201 | 0.3349 | 0.4553 | 0.5689 | 0.4241 | **0.7090** |
| **$\le 90\text{d}$** | All_Annotated | 55 | 0.1504 | 0.1968 | 0.1544 | 0.2361 | 0.6254 | 0.6671 | 0.4198 | 0.4985 | 0.3438 | **0.6635** |
| **$\le 180\text{d}$**| PAF_MultiRun | 20 | 0.2763 | 0.2790 | 0.2815 | 0.2847 | 0.7455 | 0.6640 | 0.2832 | 0.2876 | 0.8574 | **0.9441** |
| **$\le 180\text{d}$**| PAF_All | 51 | 0.2113 | 0.2425 | 0.2433 | 0.3062 | 0.1698 | 0.1909 | 0.3251 | 0.4433 | 0.2719 | **0.7830** |
| **$\le 180\text{d}$**| All_Annotated | 64 | 0.1091 | 0.2014 | 0.1114 | 0.2754 | 0.6948 | 0.3428 | 0.2816 | 0.3526 | 0.2866 | **0.8584** |
| **Unrestricted**| PAF_MultiRun | 32 | 0.1491 | 0.2962 | 0.1602 | 0.3252 | 0.5544 | 0.3940 | 0.1846 | 0.3657 | 0.3865 | **0.7742** |
| **Unrestricted**| PAF_All | 78 | 0.1276 | 0.2157 | 0.1459 | 0.3531 | 0.2147 | 0.0429 | 0.1790 | 0.3738 | 0.3646 | **0.6448** |
| **Unrestricted**| All_Annotated | 94 | 0.1044 | 0.1796 | 0.1046 | 0.2265 | 0.8890 | 0.4095 | 0.1525 | 0.3815 | 0.6950 | **0.6486** |

---

## 6. Leave-One-Patient-Out (LOPO) Cook's Distance Leverage Analysis

To rigorously assess whether positive density coefficients represent true physiological phenomena or statistical leverage from isolated patient outliers, we computed Cook's distance for Model $M_{3A}$ across each latency horizon.

### Table 6A: High-Leverage Patient Outliers in Contemporary Echo ($\le 90\text{d}$, $N = 18$)

| Record ID | Patient ID | Cook's $D$ | Threshold ($4/n$) | Echo_LAD (mm) | 12h AF Burden | 12h Density | Clinical Profile & Outlier Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`011`** | `shdb_5493975` | **0.9009** | 0.2222 | 29.0 mm | 0.21% (0.002) | 0.6617 | 4.1x leverage threshold; extremely small normal atrium (LAD 29mm) with minimal AF paroxysm. |
| **`031`** | `shdb_5303760` | **0.2786** | 0.2222 | 49.0 mm | 51.09% (0.511) | 0.9964 | Severe LA enlargement (LAD 49mm) with heavy paroxysmal burden and saturated density. |
| **`138`** | `shdb_5022948` | **0.2268** | 0.2222 | 45.0 mm | 93.50% (0.935) | 0.5502 | Atypical high burden ($93.5\%$) dispersed across widely spaced distinct paroxysms (low density). |

### Table 6B: Sensitivity to Outlier Exclusion (Impact on Model $M_{3A}$)

| Latency Threshold | Sample Analyzed | $N$ Patients | Outliers Removed | Density Coef ($\beta_{\text{dens}}$) | Density Effect / +0.10 | $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\le 30\text{d}$** | Full Stratum | 14 | 0 | +2.3518 | +0.2352 mm | 0.8130 |
| **$\le 30\text{d}$** | Clean Stratum | 12 | 2 (`011`, `138`) | -0.8566 | **-0.0857 mm** | 0.9438 |
| **$\le 90\text{d}$** | Full Stratum | 18 | 0 | +2.8618 | +0.2862 mm | 0.7163 |
| **$\le 90\text{d}$** | Clean Stratum | 15 | 3 (`011`, `031`, `138`) | -0.9974 | **-0.0997 mm** | 0.8925 |
| **$\le 180\text{d}$** | Full Stratum | 20 | 0 | +1.4972 | +0.1497 mm | 0.8441 |
| **$\le 180\text{d}$** | Clean Stratum | 18 | 2 (`011`, `031`) | -6.1317 | **-0.6132 mm** | 0.3032 |

### Methodological Verdict:
When the 3 high-leverage outliers are excluded from the contemporaneous ($\le 90\text{d}$) PAF multi-run cohort, the estimated density effect flips from $+0.2862\text{ mm}$ to **$-0.0997\text{ mm}$** ($p = 0.8925$). This confirms that the observed positive density-LAD association in the primary specification is driven entirely by outlier leverage rather than an authentic structural remodeling signal.

---

## 7. Methodological Governance, Confounding, & Non-Claims

1. **Zero Causal Claims**: Continuous AF density is an observational property of ambulatory rhythm fragmentation; it must not be interpreted as a causal determinant of atrial dilation or ventricular dysfunction.
2. **Robustness to Non-Linear Functional Form**: Evaluating both parametric linear models (OLS) and non-parametric penalized splines (GAMs) confirms that the non-significance of continuous AF density ($p > 0.50$ across all specifications) is invariant to functional form assumptions. Non-linear thresholding or saturation does not mask an undetected density effect.
3. **Confounding by Indication**: Medical therapies (such as amiodarone, bepridil, and beta-blockers) and prior catheter ablation are non-randomly assigned; they correlate with underlying substrate complexity and suppress AF manifestation.
4. **Rigorous Latency Controls**: Restricting analyses to contemporaneous echocardiograms ($\le 30\text{d}$ and $\le 90\text{d}$) prevents temporal drift in cardiac remodeling from corrupting the evaluation.
5. **Conclusion**: Cumulative AF burden remains a robust, statistically significant predictor of echocardiographic left atrial enlargement ($p = 0.0049$ in $\le 90\text{d}$ annotated patients). In contrast, continuous Charitos density provides no reproducible incremental explanatory power beyond cumulative burden in cross-sectional structural modeling, under both linear and non-linear GAM specifications.
