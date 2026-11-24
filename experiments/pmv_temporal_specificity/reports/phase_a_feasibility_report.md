# Phase A Feasibility Report: Data Provenance, Canonical Timelines, and Horizon Support

**Program:** Temporal Specificity of P-Wave Morphological Variability in Paroxysmal AF  
**Date:** November 23, 2026  
**Status:** Phase A Feasibility Gate Passed  

## 1. Executive Feasibility Verdict

The multi-cohort inventory provides robust independent patient representation for investigating P-wave morphological variability (PMV) across all planned horizons:

- **Pre-AF Observation Support:**
  - At **15 minutes** pre-AF: **524 episodes** across **239 unique patients** (IRIDIA: 143 pts, SHDB: 67 pts, LTAFDB: 29 pts).
  - At **60 minutes** (the Noriega published window): **399 episodes** across **223 unique patients** (IRIDIA: 141 pts, SHDB: 65 pts, LTAFDB: 17 pts).
  *(Note: Noriega IEEE 2026 studied 176 segments from 32 patients; our cohort provides nearly 7 times as many independent patients).*
- **Post-AF Observation Support:**
  - At **15 minutes** post-AF: **498 episodes** across **223 unique patients** (IRIDIA: 125 pts, SHDB: 65 pts, LTAFDB: 33 pts).
  - At **60 minutes** post-AF: **376 episodes** across **202 unique patients** (IRIDIA: 123 pts, SHDB: 61 pts, LTAFDB: 18 pts).
- **Paired Pre- and Post-AF Contrast Support:**
  - At **15 minutes**: **288 episodes** across **150 unique patients**.
  - At **60 minutes**: **211 episodes** across **131 unique patients**.
- **Complete Interepisode Sinus Intervals:**
  - At **15 minutes**: **282 complete intervals** across **117 unique patients**.
  - At **60 minutes**: **198 complete intervals** across **96 unique patients**.

## 2. Research Question Feasibility Status

| Research Question | Planned Contrast | Available Independent Patients | Feasibility Status | Recommended Analytical Strategy |
| :--- | :--- | :---: | :---: | :--- |
| **RQ1: Post-Termination PMV Dynamics** | Trajectory from 0 to 60 min post-AF | **202 patients** | **FEASIBLE** | Primary analysis in IRIDIA & SHDB with LTAFDB sensitivity |
| **RQ2: Dependence on Preceding AF Duration** | Post-AF PMV vs preceding AF duration | **202 patients** | **FEASIBLE** | Within-patient contrast across repeated episodes + between-patient |
| **RQ3: Temporal Specificity of Pre-AF PMV** | Impending AF vs preceding exposure | **223 patients** (at 60m), **239 pts** (at 15m) | **FEASIBLE** | Matched risk sets (M0->M1->M2->M3->M4) |
| **RQ4: Normalization Sensitivity** | Event-relative vs fixed baseline | **223 patients** | **FEASIBLE** | Test attenuation/inflation across pre-AF washout horizons |
| **RQ5: Measurement Validity & Noise** | Lead config, sampling rate, noise | All cohorts + NSTDB + Noise Holter | **FEASIBLE** | Analytical perturbations + authentic noise strata |

## 3. Cohort Specialization Matrix

1. **IRIDIA-AF (Primary Cohort):**
   - Ideal for prolonged post-AF observation (123 patients with >= 60 min clean post-AF sinus rhythm) and pre-AF replication (141 patients with >= 60 min pre-AF sinus rhythm).
2. **SHDB-AF (Clinical Substrate Cohort):**
   - Provides 61 patients with >= 60 min post-AF observation and 65 with >= 60 min pre-AF observation, coupled with 99% complete echocardiography (LAD, LVEF) and medications.
3. **LTAFDB (Beat-Level Ground Truth Cohort):**
   - Provides verified expert beat annotations for testing normal-beat selection and ectopy sensitivity.
