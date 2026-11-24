# Statistical Analysis Plan: AF-DENSITY V5

## 1. Study Objectives and Research Questions
- **RQ1 (Measurement Identifiability):** Quantify the bias, precision, and continuity ambiguity of AF density when exact continuous rhythm annotations are reduced to discrete time bins ($30$s, $1$m, $5$m, $15$m, $30$m, $60$m, $2$h, $6$h).
- **RQ2 (Electrophysiological Heterogeneity):** Determine whether uninterrupted AF continuity ($S_{\max} = L_{\max}/M$) is associated with ventricular rate exposure ($Q_{120} = R_{120}/M$) conditional on identical AF burden and AF density.
- **RQ3 (Temporal Localization):** Determine whether shortest AF concentration intervals $I_p$ capture rapid ventricular response ($r(t) > 120$ bpm) at or above proportional expectation ($p$).

## 2. Independent Sampling Unit and Hierarchical Structure
- The **patient** (`patient_id`, dataset-scoped) is the primary independent sampling unit.
- Multiple recordings or observation windows from the same patient are clustered.
- All bootstrap confidence intervals use **patient-clustered resampling** with $B = 2,000$ replicates.
- All cross-validation uses **patient-grouped splits** (5-fold, seed 42) ensuring that all windows from a patient remain strictly within the training fold or test fold.

## 3. Prespecified Primary Endpoints
1. **E01 Primary Endpoint:** Absolute density disagreement $|D_{\text{native}} - D_{\text{60m}}|$ under exact hourly AF-duration aggregation.
2. **E02 Primary Endpoint:** Adjusted absolute percentage-point difference in $Q_{120}$ associated with a $0.25$ contrast in $S_{\max}$, estimated via fractional logistic regression over common support with patient-clustered inference.
3. **E03 Primary Endpoint:** Patient-level localization contrast $\Delta C(0.80) = C_{\text{rapid}}(0.80) - 0.80$.

## 4. Multiplicity and Decision Framework
- Primary endpoints are prespecified; secondary endpoints (alternative bins, rate thresholds 110/130 bpm, sensitivity horizons 6h/24h) are explicitly treated as exploratory/sensitivity analyses.
- Scientific decision gates (Gate 1: Measurement Correctness; Gate 2: Physiological Feasibility; Gate 3: Scientific Contribution) determine workflow progression without manufacturing positive findings.
