<USER_REQUEST>
/goal I would give the coding agent a **reference-annotation-first investigation**, not another machine-learning pipeline. The primary purpose is to establish whether AF density adequately characterizes continuous AF exposure and its associated ventricular physiology.

The protocol should be built around three research questions:

| Research question                            | Primary contribution                                                                                                                                |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RQ1 — Measurement identifiability**        | What information about AF density and uninterrupted episode duration is lost when exact rhythm annotations are reduced to interval-level AF burden? |
| **RQ2 — Electrophysiological heterogeneity** | Among recordings with comparable burden and density, is uninterrupted AF duration associated with different ventricular-rate exposure?              |
| **RQ3 — Temporal localization**              | Do the intervals responsible for high AF density also contain the periods of rapid ventricular response?                                            |

I also identified a relevant 2025 JMIR Cardio paper that precedes Guo's 2026 work. It already incorporates AF density, frequency, variability and rapid ventricular response above 120 bpm. Consequently, simply combining these measurements would not be a sufficient contribution.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://cardio.jmir.org\&sz=32)

cardio.jmir.org



The following instructions deliberately distinguish our proposed investigation from existing work and incorporate the methodological lessons from the previous AF-density and PMV experiments.

# **Copy-ready master directive for the coding agent**

# AF-DENSITY V5 — Temporal Concentration, Continuity, and Electrophysiological Heterogeneity

## 1. Research mandate

Conduct an end-to-end, scientifically rigorous, multi-cohort investigation of atrial fibrillation density using reference-annotated continuous ambulatory ECG recordings.

The investigation must determine whether AF density, as currently defined in the literature, adequately characterizes the temporal organization and electrophysiological exposure of AF.

The central hypothesis is that temporal concentration and uninterrupted arrhythmia exposure are distinct properties. Recordings with comparable AF burden and density may contain substantially different episode structures, intervening rhythm durations, and ventricular responses.

Do not assume that this hypothesis is true. Design the investigation so that it can be rejected.

### Working manuscript title

*Temporal Concentration Is Not Continuity: Electrophysiological Heterogeneity at Equivalent Atrial Fibrillation Burden and Density.*

### Clinical motivation

Rosman et al. (2026) established an association between AF density and ischaemic stroke independently of AF burden in patients monitored using implanted cardiac devices.

However, their analysis used daily AF-burden measurements aggregated over 30-day periods. The physiological implications of different episode structures producing comparable density values remain incompletely understood.

Guo et al. (2026) incorporated AF density and tachycardia burden into a multidimensional AF assessment framework, but did not directly investigate ventricular-rate heterogeneity conditional on equivalent burden and density or the spatial-temporal correspondence between their respective concentration intervals.

Our datasets provide continuous, limited-lead ECG and rhythm annotations, allowing these questions to be investigated directly at substantially finer temporal resolution.

The study is an electrophysiological characterization and measurement investigation, not a stroke prediction study.

---

## 2. Mandatory literature grounding

Before executing experiments, create a structured literature-to-experiment mapping.

Review the available full articles, supplements, figures, equations, and original implementations for:

1. Charitos et al. (2012), *A Comprehensive Evaluation of Rhythm Monitoring Strategies for the Detection of Atrial Fibrillation Recurrence*. DOI: 10.1161/CIRCULATIONAHA.112.098079.
2. Charitos et al. (2013), *Atrial Fibrillation Density: A Novel Measure of Atrial Fibrillation Temporal Aggregation*.
3. Henriksson et al. (2021), *Modeling and Estimation of Temporal Episode Patterns in Paroxysmal Atrial Fibrillation*. DOI: 10.1109/TBME.2020.2995563.
4. Butkuvienė et al. (2024), *Characterization of Atrial Fibrillation Episode Patterns: A Comparative Study*. DOI: 10.1109/TBME.2023.3293252.
5. Butkuvienė et al. (2023), *Atrial Fibrillation Episode Patterns as Predictor of Clinical Outcome of Catheter Ablation*. DOI: 10.1007/s11517-022-02713-x.
6. ESC/EHRA clinical consensus on AF burden (2025). DOI: 10.1093/europace/euaf019.
7. *Photoplethysmography-Based Machine Learning Approaches for Atrial Fibrillation Burden: Algorithm Development and Validation* (2025). DOI: 10.2196/78075.
8. Guo et al. (2026), *Beyond Burden Metrics*. DOI: 10.1016/j.hroo.2025.12.020, including supplementary material.
9. Rosman et al. (2026), *Atrial Fibrillation Density as a Biomarker for Ischaemic Stroke Risk Prediction*. DOI: 10.1093/eurheartj/ehag203, including supplementary material and correction.
10. Liang (2026), DOI: 10.1093/eurheartj/ehag548.
11. Rosman, Wang and Passman (2026), DOI: 10.1093/eurheartj/ehag549.

Original Charitos implementation:\
[https://github.com/echaritos/AF-density](https://github.com/echaritos/AF-density)

Record for each publication:

- Exact AF-density formulation.
- Observation clock and temporal resolution.
- Monitoring horizon.
- Inclusion and exclusion criteria.
- Physiological and clinical variables.
- Experiments actually performed.
- Relevant equations and figures.
- Direct overlap with our proposed analyses.
- Unresolved questions.
- Full-text access status.

Do not claim novelty based only on abstracts.

Do not repeat established analyses of generic AF detection errors, monitoring shutdowns, beat-versus-time density, burden-density correlation, or conventional recurrence prediction as the primary scientific contribution.

Produce `literature_overlap_matrix.csv` and `literature_grounding.md`.

---

## 3. Dataset scope

Use the existing locally available datasets:

| Dataset                            | Primary role                                                                  |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| IRIDIA-AF                          | Prolonged episodes, longer recordings, within-patient temporal variation      |
| SHDB-AF                            | Mixed AF patterns, clinical metadata, echocardiography, medication context    |
| LTAFDB                             | Highly fragmented AF, annotated ventricular beats, rhythm-transition analysis |
| Noise Holter Database              | Authentic artifact and ventricular-rate measurement robustness                |
| MIT-BIH Noise Stress Test Database | Controlled QRS detection and rate-measurement stress testing                  |

Previously reported source inventory, to be independently reconciled:

- IRIDIA-AF: 167 recordings, approximately 152 patients, 200 Hz.
- SHDB-AF: 98 rhythm-annotated recordings, approximately 93 patients, 200 Hz.
- LTAFDB: 84 recordings, approximately 84 patients, 128 Hz.
- Approximately 329 patients across the three principal cohorts.
- Approximately 10,982 hours of recording in the previously reconciled inventory.

These are historical reference counts, not values to hardcode.

Previous reports contain discrepancies in raw AF runs, left-censored episodes, recording durations, patient counts, and annotation eligibility. Recompute these quantities from source files.

The Noise Holter Database and NSTDB must not be treated as independent continuous AF-density cohorts.

### Data governance

The historical IRIDIA protected-test partition has been exposed during prior analyses. Do not describe it as an untouched confirmatory test set.

Preserve previous split manifests and access records. Treat the present investigation as exploratory and hypothesis-driven retrospective analysis.

Use patient-grouped validation where appropriate, but do not claim that cross-validation creates a new independently protected cohort.

---

# PHASE A — Canonical physical-time dataset reconstruction

## A01. Construct a canonical rhythm timeline

Read source WFDB headers, waveform files, rhythm annotations, beat annotations, and dataset-specific metadata.

Construct a lossless physical-time interval representation:

```
dataset
patient_id
record_id
source_file_id
start_sample
end_sample
start_time_s
end_time_s
sampling_frequency
rhythm_label_native
rhythm_label_harmonized
annotation_source
is_left_censored
is_right_censored
signal_available
rhythm_known
```

Use half-open intervals `[start, end)`.

Canonical harmonized rhythm states:

```
AF
CONFIRMED_SINUS
ATRIAL_FLUTTER
OTHER_ATRIAL_ARRHYTHMIA
OTHER_RHYTHM
UNKNOWN
```

Preserve native annotation labels.

Do not assume that non-AF means sinus rhythm.

Do not convert flutter, atrial tachycardia, ventricular rhythms, or unknown intervals into sinus rhythm.

For datasets where only AF versus non-AF is supported, retain the corresponding uncertainty explicitly.

### Physical-time requirements

- Do not concatenate unrelated recording sessions.
- Do not compress missing intervals.
- Do not treat missing ECG as non-AF.
- Do not multiply monitoring time by the number of ECG channels.
- Do not count excluded calibration prefixes as observed AF-free time.
- Do not manufacture an onset for AF already present when recording begins.
- Do not manufacture a termination at the end of a recording.
- Preserve exact annotation boundaries whenever available.

Produce separate inventories of:

- All observed AF runs.
- Runs with observed onsets.
- Runs with observed terminations.
- Runs with both boundaries observed.
- Left-censored runs.
- Right-censored runs.
- Runs censored at both boundaries.

Verify inclusion-exclusion arithmetic:

```
complete_runs =
    total_runs
    - left_censored_runs
    - right_censored_runs
    + both_censored_runs
```

Generate `canonical_rhythm_intervals.parquet`, `canonical_af_runs.parquet`, and `cohort_inventory.csv`.

---

## A02. Standardize observation horizons

Density depends on the observation horizon. Do not pool values computed over different recording durations as though they represent the same estimand.

Use:

- Primary cross-cohort horizon: acquisition-anchored first 12 hours.
- Secondary horizon: strict acquisition-anchored first 24 hours.
- Sensitivity horizon: first 6 hours.
- Additional descriptive analysis: complete native recording.

For recordings longer than 24 hours, perform a secondary nonoverlapping-window analysis with patient-level weighting.

Do not center primary windows around AF onset or termination.

Do not shift windows until AF is encountered.

Report all eligibility losses, including recordings that contain no AF or continuous AF throughout a particular horizon.

The primary 12-hour horizon is a pragmatic cross-cohort comparison choice, not an established biological timescale. The 24-hour analysis is necessary to evaluate relevance to conventional Holter monitoring.

### Handling incomplete observations

Primary density analysis must use observation intervals with fully known rhythm status.

For intervals containing unknown rhythm:

- Report missingness explicitly.
- Do not silently concatenate observable pieces.
- Perform secondary uncertainty analyses only where defensible.
- Distinguish ECG availability from rhythm-annotation availability.

---

## A03. Define episode-duration representations

Preserve the native rhythm timeline as the primary source of truth.

Construct additional analytical representations for observed AF runs lasting:

- At least 30 seconds.
- At least 2 minutes.
- At least 6 minutes.
- At least 30 minutes.

For each threshold, report both:

1. Number of eligible observed runs.
2. Percentage of original AF exposure represented by eligible runs.

Do not overwrite excluded short AF runs as sinus rhythm.

Any analysis that removes short runs must be explicitly described as a duration-filtered AF representation, not a corrected rhythm annotation.

If gap-bridging sensitivity analyses are performed, keep them separate from duration filtering and record exactly which intervening rhythms are bridged.

Do not label short annotated runs as independently diagnosed clinical AF episodes without supporting adjudication.

---

# PHASE B — Exact AF-density implementation and independent verification

## B01. Implement the published metric

Let:

[\
A(t)\in{0,1}\
]

represent AF status over a fully observed interval ([0,T]).

Define total AF time:

[\
M=\int_0^T A(t),dt\
]

and burden:

[\
b=M/T.\
]

For each (p\in[0,1]), define:

[\
F(p;b)=\
\frac{1}{T}\
\min\_{s,e}\
\left{\
e-s:\
\int_s^e A(t),dt\ge pM\
\right}.\
]

The Charitos density functional is:

\frac{2}{1-b}\
\int_0^1|F(p;b)-p|,dp.\
]

Verify this formulation against the original publication, original R implementation, and Rosman supplementary equation.

Maintain a distinction between:

1. The continuous interval-based mathematical formulation.
2. The original discrete/bin-based numerical implementation.
3. Any approximation introduced for computational efficiency.

Do not assume the discrete implementation and continuous functional are numerically identical.

Do not substitute a cumulative-time deviation statistic, Gini coefficient, hourly variance, or longest-episode fraction for Charitos density.

Handle (b=0) and (b=1) explicitly. The continuous expression is undefined at these boundaries; report any author-specific software convention separately.

Do not clip unexpected density values to a nominal range without investigating the mathematical and numerical cause.

---

## B02. Implement an independent minimum-interval solver

Construct the cumulative AF exposure function:

[\
C(t)=\int_0^t A(u),du.\
]

Use exact AF interval boundaries and cumulative AF mass to calculate the shortest contiguous interval containing each requested exposure fraction.

Retain both:

- Minimum required interval length.
- Actual interval start and end coordinates.

The solver must account for intervals that begin or end partway through an AF episode.

Do not restrict candidate intervals to complete episodes or arbitrary fixed time bins.

Retain tied or near-tied minimum intervals according to a documented numerical tolerance.

Use deterministic tie-breaking that does not depend on ventricular-rate information.

Evaluate integral convergence with progressively finer exposure-fraction grids, or implement exact piecewise integration where feasible.

Report numerical convergence and processing time.

---

## B03. Unit tests and mathematical validation

Implement independent fixtures for:

| Fixture                                                | Expected behavior                                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Single contiguous AF run                               | Continuous density equals 1 for (0\<b<1)                                         |
| Single run translated in time                          | Density unchanged                                                                |
| Timeline reversal                                      | Density unchanged                                                                |
| Uniform fractional AF exposure                         | Density approaches 0 under its corresponding continuous intensity representation |
| Equal-duration episodes separated by different gaps    | Results match independently calculated interval lengths                          |
| Partial-bin AF exposure                                | AF duration preserved exactly                                                    |
| No AF                                                  | Explicit boundary status                                                         |
| All AF                                                 | Explicit boundary status                                                         |
| Missing rhythm intervals                               | No silent calculation as fully observed                                          |
| Multiple recording sessions                            | No cross-session density calculation                                             |
| Identical AF exposure with different ventricular rates | Density remains identical                                                        |

Generate at least 100 randomized small timelines and compare the optimized solver against an independently written exhaustive solver.

Run the original author R implementation on reproducible binned fixtures.

Do not merely compare two Python functions derived from the same implementation.

Target tolerances:

- Burden conservation: numerical precision appropriate to source sample resolution.
- Original R versus faithful port: agreement within documented floating-point tolerance.
- Optimized continuous solver versus exhaustive fixtures: agreement within a predetermined numerical tolerance.
- Integral approximation: report convergence error and do not use insufficiently converged estimates.

Produce:

```
density_reference_implementation.py
density_continuous.py
density_binned.py
test_density_math.py
density_validation_results.csv
density_validation_report.md
```

Do not proceed to scientific interpretation if fundamental invariants fail.

---

# PHASE C — EXPERIMENT E01: Temporal aggregation and identifiability

## C01. Scientific question

How much information about AF temporal concentration and uninterrupted episode duration is lost when a continuous rhythm timeline is represented only by the exact AF duration occurring within successive time bins?

This is not an AF-classifier robustness experiment.

All AF labels remain correct. Only temporal resolution changes.

---

## C02. Controlled temporal aggregation

Starting from the identical reference rhythm timeline, calculate exact AF duration within bins of:

```
30 seconds
1 minute
5 minutes
15 minutes
30 minutes
60 minutes
```

For recordings supporting longer observations, add 2-hour and 6-hour sensitivity analyses.

For every bin (j):

[\
m_j=\int\_{B_j}A(t),dt\
]

where (B_j) is the physical-time bin.

Require exact conservation:

[\
\sum_j m_j=M.\
]

Do not use majority voting.

Do not threshold a bin into AF or sinus rhythm.

Do not introduce false-positive or false-negative AF classifications.

For each resolution, compute:

1. Native continuous density.
2. Density from the published discrete-bin implementation.
3. Density under a uniform fractional-intensity interpretation within each bin.
4. Absolute and signed differences.
5. Rank changes across recordings.
6. Dependence of these differences on burden, episode duration, fragmentation, and cohort.

Keep observation duration constant within every comparison.

Perform an offset sensitivity analysis because the location of bin boundaries may influence the results.

---

## C03. Identifiability of episode continuity

A fixed vector of bin-level AF durations can correspond to multiple different underlying rhythm trajectories.

For each eligible recording and bin resolution, characterize this ambiguity.

Create alternative feasible within-bin arrangements:

- AF concentrated toward the beginning of each bin.
- AF concentrated toward the end of each bin.
- AF packed to join neighboring intervals.
- AF distributed into multiple short intervals.
- Randomized feasible placements preserving every bin's exact AF duration.

Every alternative must preserve:

- Total observation duration.
- Exact AF burden.
- Exact per-bin AF duration.
- Known rhythm constraints where applicable.

For each feasible realization, calculate:

- Native continuous AF density.
- Longest uninterrupted AF run.
- Episode count.
- Longest-episode share of total AF time.
- Number and duration of intervening non-AF intervals.

Explicitly distinguish mathematical feasibility from physiological plausibility.

Do not call simulated rearrangements patient observations.

### Required output

Identify real coarse burden sequences for which materially different episode trajectories are compatible with the same reported measurements.

Provide representative examples from all three cohorts.

---

## C04. Certified uncertainty bounds

Investigate whether rigorous bounds on continuous density can be obtained using only exact bin-level AF durations.

For a candidate physical-time interval (I), let:

- (w_j=|I\cap B_j|)
- (\Delta_j=|B_j|)
- (m_j) be known AF duration in bin (j).

The AF exposure inside the interval is bounded by:

[\
\max(0,m_j-(\Delta_j-w_j))\
\le\
M_j(I)\
\le\
\min(m_j,w_j).\
]

Summing these expressions provides lower and upper exposure bounds for any candidate interval.

Use these bounds to construct rigorous outer envelopes for (F(p;b)).

Propagate the envelopes through the absolute-value density functional.

If (F(p;b)\in[F_L(p),F_U(p)]), then:

[\
g_L(p)=\
\begin{cases}\
0,&p\in[F_L(p),F_U(p)]\\\
\min(|F_L(p)-p|,|F_U(p)-p|),&\text{otherwise}\
\end{cases}\
]

and

[\
g_U(p)=\max(|F_L(p)-p|,|F_U(p)-p|).\
]

Integrating these quantities provides conservative density bounds, subject to certified numerical integration.

Do not describe pointwise envelopes as globally sharp density bounds unless this has been demonstrated.

An additional validation target is the following conservative coarsening bound: if every bin has duration at most (\Delta), two AF exposure representations preserving the same bin masses can be coupled by moving exposure no farther than (\Delta). Under the stated assumptions, this implies a minimum-interval length difference of at most (2\Delta), and therefore a density difference bounded by:

[\
|D_1-D_2|\
\le\
\frac{4\Delta}{T(1-b)}.\
]

Derive and verify this result independently before using it. This may be loose, particularly at high AF burden.

For small discretized timelines, use exhaustive enumeration to verify coverage.

For large timelines, distinguish:

- Certified mathematical bounds.
- Empirical extrema found by search.
- Simulation-derived sensitivity ranges.

Do not use Monte Carlo minima and maxima as certified bounds.

### Primary E01 endpoints

- Native-versus-coarsened density absolute error.
- Width of certified density uncertainty intervals, where available.
- Range of compatible longest uninterrupted AF durations.
- Fraction of patients or recordings with substantial continuity ambiguity.

Report complete distributions rather than selecting only the most extreme examples.

---

# PHASE D — Construct independent ventricular physiology measurements

## D01. QRS and ventricular beat extraction

Use reference beat annotations wherever reliable annotations are available.

For recordings without suitable beat annotations:

- Apply a validated QRS detection method.
- Use the original ECG sampling frequency.
- Compare available leads.
- Record detector confidence or signal-quality indicators.
- Do not fill missing detections using assumed heart rates.

Use LTAFDB reference annotations to validate detector performance against observed beat timing.

Report sensitivity, positive predictive value, timing errors, and rate-estimation error across rhythm and quality strata.

Do not restrict ventricular-rate calculations to normal sinus beat labels during AF.

True ventricular ectopic and paced beats are part of the observed ventricular response, although ectopy and pacing should be separately characterized where supported.

Exclude artifact markers and invalid beat detections.

Retain annotation provenance for every measurement.

---

## D02. Ventricular rate definition

Do not use individual instantaneous (60/RR) values as the primary definition of rapid ventricular response.

A short RR interval during irregular AF does not necessarily represent sustained rapid ventricular rate.

Primary estimator:

- Use nonoverlapping 30-second intervals contained entirely within an observed AF run.
- Estimate ventricular rate using valid ventricular beat counts per unit time or a consistently defined RR-based rate estimator.
- Prespecify the estimator before analyzing its association with density.
- Assign the rate measurement to its exact physical-time interval.

Prespecified sensitivities:

- 10-second intervals.
- 60-second intervals.
- Alternative RR-based local ventricular-rate estimator.
- Short-run analysis using a separately validated variable-duration estimator.

Record AF exposure excluded because the episode is shorter than the required rate-estimation window.

Do not generalize measurements from prolonged measurable runs to all annotated AF runs.

For every recording, report total reference AF duration, measurable AF duration, and percentage of AF duration with valid ventricular-rate estimates.

---

## D03. Primary physiological endpoint

Use the literature-supported ventricular-rate threshold of 120 bpm.

Define:

\int_0^T\
A(t)\mathbf{1}[r(t)>120],dt\
]

where (r(t)) is the prespecified local ventricular-rate estimate.

Calculate two separate quantities:

[\
Q\_{120}=\frac{R\_{120}}{M}\
]

and

[\
B\_{120}=\frac{R\_{120}}{T}.\
]

Interpretation:

- (Q\_{120}): fraction of AF exposure occurring above 120 bpm.
- (B\_{120}): fraction of total observation time containing rapid AF.

The primary physiological endpoint is (Q\_{120}).

Do not conflate these denominators.

When ventricular-rate coverage is incomplete, let (U) represent AF duration with unknown ventricular rate, and (R\_{\mathrm{obs}}) represent observed rapid-AF duration.

Report:

[\
\frac{R\_{\mathrm{obs}}}{M}\
\le Q\_{120}\le\
\frac{R\_{\mathrm{obs}}+U}{M}.\
]

Also report the observed-only fraction using measurable AF time as the denominator.

Set the primary measurement-coverage requirement before analyzing outcomes, with 90% as a pragmatic starting threshold and 80% and 95% sensitivity analyses.

### Secondary physiological measurements

- Time-weighted mean ventricular rate during AF.
- Median and 90th-percentile ventricular rate.
- Fraction of AF exposure above 110 and 130 bpm.
- Longest observed contiguous high-rate AF period.
- RR coefficient of variation.
- Normalized successive RR variability.
- Ventricular ectopy fraction where annotated.
- Transition-associated ventricular pauses where reliable beat and rhythm annotations permit.

Pause analyses must distinguish verified long RR intervals from missed QRS detections.

Do not describe RR variability during AF as conventional sinus-rhythm HRV or a direct measurement of autonomic tone.

---

# PHASE E — EXPERIMENT E02: Physiological heterogeneity conditional on AF burden and density

## E01. Scientific question

Among comparable observation intervals with similar AF burden and AF density, does uninterrupted AF exposure identify distinct ventricular-response phenotypes?

This is the principal physiological experiment.

Do not simply demonstrate that ventricular rate varies between patients.

That result is already expected and would not constitute a meaningful density-related contribution.

---

## E02. Primary continuity variable

Calculate the longest observed uninterrupted AF run:

[\
L\_{\max}=\max_j L_j.\
]

Define its share of total AF exposure:

[\
S\_{\max}=\frac{L\_{\max}}{M}.\
]

Interpretation:

- (S\_{\max}=1): all observed AF exposure occurs in one contiguous run.
- Smaller values: total AF exposure is distributed across multiple runs.

This is a conventional descriptive transformation of longest episode duration, not a proposed new clinical biomarker.

Secondary continuity descriptors:

- Number of observed AF runs.
- Median run duration.
- Run-duration distribution.
- Number of observed AF-to-non-AF transitions.
- Median and distribution of observed interepisode gaps.
- Fraction of AF exposure contained in episodes lasting at least 6 or 30 minutes.

For analysis windows clipping an episode, distinguish longest observed run from true complete episode duration.

---

## E03. Outcome-blind feasibility analysis

Before examining relationships with ventricular-rate outcomes, characterize the joint distribution:

[\
(B\_{\mathrm{AF}},D\_{\mathrm{AF}},S\_{\max}).\
]

Produce:

- Burden-density scatterplots colored by longest-episode share.
- Cohort-specific support distributions.
- Density distributions for single-run and multi-run recordings.
- Patient counts within common-support regions.
- Within-patient variation where repeated nonoverlapping windows exist.
- Correlations and nonlinear dependence between temporal descriptors.

Do not force a comparison between patients whose burden-density values do not overlap.

Identify whether meaningful variation in continuity exists at comparable burden and density.

A large proportion of single-run recordings may have density equal to one by construction.

If the remaining variation is too limited to separate continuity from density, report that the proposed conditional effect is not identifiable in the available cohort.

Do not fit a regression model solely because the software permits it.

---

## E04. Descriptive matched analysis

Construct within-cohort comparisons using equivalent observation horizons.

Initial matching criteria:

```
Absolute AF burden difference ≤ 0.025
Absolute AF density difference ≤ 0.025
Same dataset
Same observation horizon
Different patients
```

Require an appreciable contrast in longest-episode share, initially:

```
Absolute Smax difference ≥ 0.30
```

These are pragmatic analytical calipers, not clinical equivalence thresholds.

Perform sensitivity analyses using tighter and broader calipers selected without inspecting physiological outcomes.

Use deterministic optimal matching or a clearly documented matching algorithm.

Avoid repeated use of the same patient in the primary matched analysis.

Report:

- Number of eligible patients.
- Number of matched pairs.
- Covariate balance.
- Unmatched population.
- Distribution of burden and density differences.
- Distribution of continuity contrasts.

Compare the primary endpoint (Q\_{120}) between the more continuous and more fragmented recording in each pair.

Estimate:

- Paired absolute percentage-point difference.
- Median paired difference.
- Patient-level uncertainty.
- Distribution of paired effects.
- Proportion of pairs with differences exceeding prespecified descriptive thresholds.

Include illustrative thresholds of 10 and 20 percentage points, without describing them as established clinical minimal important differences.

If matching produces too few independent patients or inadequate balance, retain the descriptive results and state the limitation.

---

## E05. Conditional statistical analysis

Use the primary continuous outcome (Q\_{120}), including zero rapid-AF exposure.

Fit interpretable models suitable for fractional outcomes, such as fractional logistic regression with patient-clustered inference.

Compare nested models:

```
M0:
    cohort

M1:
    cohort
    AF burden

M2:
    cohort
    AF burden
    AF density

M3:
    cohort
    AF burden
    AF density
    longest episode share
```

Allow nonlinear burden effects if the sample supports them.

Do not automatically fit high-dimensional spline interactions.

Primary estimand:

The adjusted difference in expected rapid-AF fraction associated with a prespecified contrast in longest-episode share, standardized over the observed common-support population.

Report the effect on the absolute percentage-point scale, not only regression coefficients.

Use a 0.25 change in longest-episode share as an interpretable contrast, only where both exposure values lie within supported regions.

Do not interpret this as the causal effect of artificially changing episode continuity.

Account for repeated recordings using dataset-scoped patient identifiers.

Estimate uncertainty with patient-clustered bootstrap resampling or appropriately clustered statistical methods.

Use cohort-specific estimates as the principal replication assessment. Do not allow LTAFDB's large number of short runs to dominate inference.

---

## E06. Head-to-head comparison with simpler temporal measurements

Liang explicitly recommends comparing density with simpler summaries.

Evaluate:

```
B0: Burden only
B1: Burden + density
B2: Burden + longest observed AF duration
B3: Burden + episode count
B4: Burden + longest episode share
B5: Burden + density + longest episode share
B6: Burden + published AF aggregation
```

Implement published AF aggregation only after confirming its exact definition.

This comparison is not AF forecasting or clinical risk prediction. It evaluates how well temporal descriptors explain or estimate an observed concurrent ECG phenotype.

Where sample support permits, use patient-grouped cross-validation with identical folds for all models.

Report:

- Out-of-fold MAE.
- RMSE.
- Calibration of predicted fractional exposure.
- Explained variability.
- Paired changes in model performance.
- Patient-bootstrap confidence intervals.

Do not use AUROC as the primary metric for a continuous fractional outcome.

Do not claim independence or equivalence merely because a coefficient is statistically nonsignificant.

If density adds no information beyond longest episode duration, report this explicitly.

---

## E07. Within-patient sensitivity analysis

IRIDIA-AF contains longer and repeated recordings that may permit comparisons within the same patient.

Construct nonoverlapping, acquisition-anchored windows of equal duration.

Identify patients with variation in:

- AF burden.
- AF density.
- Longest-episode share.
- Rapid-AF fraction.

Where joint support exists, estimate within-patient associations using patient fixed effects or paired analyses.

Report the actual number of informative patients, not merely the number of available windows.

This analysis is important because medication exposure, underlying atrial disease, AV nodal physiology, and other stable patient characteristics can confound between-patient comparisons.

Do not present within-patient inference if the relevant exposure has negligible within-patient variation.

---

# PHASE F — EXPERIMENT E03: Temporal localization of rapid ventricular response

## F01. Scientific question

Do the shortest physical-time intervals containing most AF exposure also contain most rapid ventricular-rate exposure?

Guo et al. measure AF density and tachycardia burden separately. This experiment investigates the temporal alignment of these quantities.

The objective is not to create another severity score.

---

## F02. Extract density-defining intervals

For each eligible recording, find the shortest contiguous interval (I_p) containing fraction (p) of total AF exposure.

Use:

```
Primary coverage: 80%
Secondary coverage: 50%, 75%, 90%
```

The selected coverage fractions are analytical choices, not validated clinical thresholds.

Use the exact interval solver from Phase B.

The interval must be selected using only AF rhythm annotations.

Do not use heart rate to resolve tied solutions.

For tied intervals:

- Apply deterministic rhythm-only tie-breaking.
- Report the range of rapid-AF capture across tied or near-tied solutions where feasible.

---

## F03. Calculate rapid-AF coverage

Define:

\frac{\
\int\_{I_p}A(t)\mathbf{1}[r(t)>120],dt\
}{\
\int_0^T A(t)\mathbf{1}[r(t)>120],dt\
}.\
]

Define the localization contrast:

[\
\Delta C(p)=C\_{\mathrm{rapid}}(p)-p.\
]

Interpretation:

- Negative values: rapid AF is underrepresented within the interval.
- Positive values: rapid AF is overrepresented within the interval.
- Values near zero: approximately proportional distribution.

Do not calculate this fraction when total rapid-AF exposure is zero.

Report zero-rapid-AF recordings separately.

Also calculate absolute minutes of rapid AF outside (I_p).

A large missed percentage corresponding to only a few seconds of exposure should not receive the same clinical interpretation as substantial missed exposure.

### Missing ventricular-rate coverage

Let (R_I) and (R_O) be observed rapid-AF durations inside and outside the concentration interval, and (U_I,U_O) be corresponding AF durations with unknown rate.

Where denominators are positive, conservative capture bounds are:

[\
C_L=\frac{R_I}{R_I+R_O+U_O}\
]

and

[\
C_U=\frac{R_I+U_I}{R_I+U_I+R_O}.\
]

Report these bounds rather than assuming missing ventricular-rate measurements are normal.

---

## F04. Conditional randomization

Determine whether observed rapid-AF localization differs from an appropriate conditional null distribution.

### Null A — Rate alignment on the cumulative AF-time clock

Hold the complete AF timeline fixed.

Represent ventricular-rate measurements along cumulative observed AF time.

Apply random circular shifts or block permutations of the rate trajectory while preserving, as far as possible:

- Total rapid-AF exposure.
- Ventricular-rate distribution.
- Local serial dependence.
- Valid-rate coverage structure.

Recalculate rapid-AF capture for each randomized alignment.

When the rate trajectory is fully observed and exchangeable under circular shifts, expected capture is (p).

If rate coverage is incomplete, derive the corresponding valid-exposure expectation rather than automatically using (p).

This null tests overall rate-rhythm alignment. It does not preserve every episode-specific physiological feature.

### Null B — Within-episode rate alignment

Preserve each episode's:

- Duration.
- Total rapid-AF exposure.
- Mean ventricular-rate distribution.

Randomize the temporal ordering of rate observations within sufficiently long episodes using valid block-based methods.

This tests whether additional localization exists beyond differences in episode-specific rate exposure.

Its expectation is episode-specific and is not necessarily (p).

Where possible, preserve episode-age strata or use episode-age-matched permutations.

Perform at least 2,000 feasible randomizations per eligible recording, or exact enumeration for small cases.

Use predetermined seeds.

Quantify Monte Carlo uncertainty.

Do not interpret rejection of either null as proof of a biological mechanism.

---

## F05. Compare alternative interval-selection methods

For each recording, compare rapid-AF capture under equal physical-time review budgets.

Candidate interval-selection strategies:

1. Density-defining shortest AF concentration interval.
2. Interval of the same duration anchored around the longest AF episode using a predefined rhythm-only rule.
3. Acquisition-anchored interval of the same duration.
4. Randomly positioned interval of the same duration.
5. Oracle interval maximizing directly measured rapid-AF exposure.

Recognize that maximum AF burden in a fixed-length interval may be mathematically equivalent to the density-defined interval selection at the corresponding length. Do not count mathematically equivalent methods as independent baselines.

The oracle is an upper benchmark, not an available rhythm-only method.

Report:

- Rapid-AF capture.
- Missed rapid-AF minutes.
- Observation duration required.
- Difference from oracle capture.
- Cohort-specific performance.
- Performance stratified by episode continuity.

Do not claim that AF density was originally intended to locate rapid AF. The scientific objective is to characterize correspondence and divergence between these two distinct exposure dimensions.

---

# PHASE G — Optional mechanistic decomposition of AF density

Proceed only if the principal experiments establish adequate multi-episode support.

## G01. Conditional episode-order randomization

Determine whether observed density is largely explained by episode and gap-duration distributions, or whether the specific ordering of those durations carries additional information.

For each fully observed eligible timeline, preserve:

- Total observation duration.
- Total AF time.
- All observed AF run durations.
- All internal non-AF gap durations.
- Longest episode duration.
- Leading and trailing non-AF intervals.

Randomize:

1. AF episode order only.
2. Internal gap order only.
3. Both episode and internal gap order.

Preserve known rhythm constraints where feasible.

Calculate density for each permissible ordering.

For records with few episodes, enumerate all distinct permutations.

For records with many episodes, use reproducible Monte Carlo sampling.

Do not treat records with one episode or effectively invariant permutations as evidence of no temporal organization.

Report:

- Observed density.
- Conditional permutation distribution.
- Observed percentile.
- Difference from conditional median.
- Conditional variability.
- Fraction of recordings for which ordering meaningfully changes density.

Do not propose a new clinical density score from these quantities.

This is a conditional decomposition of the existing statistic.

A permutation result does not prove AF self-excitation, progression, or a causal arrhythmia mechanism.

---

# PHASE H — Clinical metadata and electrophysiological interpretation

Use SHDB-AF for exploratory clinical-context analyses.

Verify source metadata and availability of:

- Age.
- Sex.
- AF subtype.
- Left atrial diameter.
- LVEF.
- Previous ablation.
- Beta-blocker use.
- Antiarrhythmic medication.
- Heart failure status.
- Relevant comorbidities.

Investigate whether continuity and ventricular-response heterogeneity persist after accounting for available clinical context.

Limit model complexity according to the actual number of independent patients.

Do not infer medication effectiveness from cross-sectional medication status.

Do not infer tachycardia-induced cardiomyopathy from LVEF alone.

Do not infer stroke propensity from AF density measured over 12 or 24 hours.

Do not claim mechanical atrial stunning, atrial fibrosis, thrombus formation, or haemodynamic recovery from limited-lead ECG.

The available physiological endpoints are ECG-observable correlates, not validated clinical outcomes.

---

# PHASE I — Authentic and synthetic measurement robustness

## I01. Noise Holter Database

Use authentic noisy Holter excerpts to evaluate the robustness and observability of ventricular-rate measurements.

Analyze available rhythm and quality strata separately.

Where supported, assess:

- Reference QRS agreement.
- Rate-measurement availability.
- Inter-lead disagreement.
- Rate-estimation error.
- False rapid-rate classification.
- False long-RR or pause detection.

Do not assume that every noise-quality label supplies reference QRS timestamps.

Where reference beats are unavailable, report measurement agreement or observability rather than ground-truth accuracy.

Account for unequal segment durations and subject-level clustering.

The Noise Holter excerpts do not form a continuous AF history. Do not calculate or interpret recording-level AF density from concatenated excerpts.

---

## I02. NSTDB

Use clean source recordings and calibrated noise conditions to quantify:

- QRS detection errors.
- Absolute ventricular-rate errors.
- Errors in duration above the selected rate thresholds.
- Spurious pause detections.

Distinguish noise-only records from ECG-plus-noise recordings.

Verify whether the source rhythm is AF before making AF-specific claims.

Where controlled noise is added to authentic AF excerpts from the principal datasets, label the experiment as a synthetic perturbation of real AF ECG.

Do not report synthetic-noise performance as real-world clinical validation.

Do not select signal-quality thresholds based on which thresholds produce desirable physiological associations.

---

# PHASE J — Statistical analysis standards

## J01. Independence and weighting

The patient is the principal independent sampling unit.

Use dataset-scoped patient identifiers.

Do not treat thousands of LTAFDB annotated runs as thousands of independent clinical patients.

Report:

- Patient-weighted estimates.
- Recording-weighted descriptive estimates.
- AF-duration-weighted descriptive estimates, where scientifically appropriate.

These are different estimands and must not be conflated.

Resample patients with all associated observations retained.

Use at least 2,000 patient-bootstrap replicates for primary uncertainty estimates.

For predictive or model-comparison summaries, all observations from a patient must remain in the same fold.

All preprocessing and model fitting must occur within training folds.

---

## J02. Statistical multiplicity

Prespecify three principal endpoints:

1. E01: absolute density disagreement under exact AF-duration temporal aggregation, using a 60-minute bin as the primary coarsening condition.
2. E02: adjusted absolute difference in rapid-AF fraction associated with longest-episode share.
3. E03: patient-level distribution of rapid-AF localization contrast at 80% AF coverage.

Report unadjusted and multiplicity-adjusted inference where applicable.

Treat other bin widths, coverage proportions, rate thresholds, rhythm-duration filters, and clinical associations as sensitivity or exploratory analyses.

Do not choose the most statistically significant combination as the primary result.

---

## J03. Confounding and common support

Explicitly consider:

- Cohort differences.
- Annotation protocols.
- Patient-level AF phenotype.
- Recording duration.
- AF burden.
- Medication exposure.
- Pacing and ectopy.
- Ventricular-rate measurement availability.
- Signal quality.
- Episode age.
- Time of day, where reliable clock metadata exist.

Do not invent clock-time information for recordings lacking it.

For conditional analyses, characterize joint exposure support before fitting models.

Do not use regression extrapolation to manufacture comparisons between temporal phenotypes that do not coexist in the observed data.

Document assumptions for every randomization test.

---

## J04. Uncertainty and interpretation

Report:

- Effect sizes.
- Absolute differences.
- Confidence intervals.
- Sample sizes at every level.
- Missingness.
- Cohort-specific heterogeneity.
- Model convergence.
- Sensitivity to individual influential patients.

A nonsignificant result is not proof of absence of association.

A statistically significant result is not automatically clinically meaningful.

Do not use post hoc power calculations to justify the findings.

Before modeling, produce precision estimates based on the number of independent informative patients and a transparent range of plausible variability.

Do not characterize this project as prospective validation or clinical deployment validation.

---

# PHASE K — Required figures

Generate publication-quality figures from actual computed data.

## Figure 1 — Cohort and analysis flow

Show dataset-specific counts:

```
Source recordings
→ Rhythm-annotated recordings
→ Eligible observation horizons
→ AF-positive windows
→ Density-defined windows
→ Multi-episode windows
→ Valid ventricular-rate windows
→ Common-support population
→ Final analytical samples
```

## Figure 2 — Native versus aggregated density

Show:

- Native versus coarsened scatterplots.
- Bland–Altman differences.
- Error versus bin duration.
- Cohort-specific distributions.
- Dependence on AF burden and episode continuity.

## Figure 3 — Identifiability of episode continuity

Show real bin-level AF-duration sequences and multiple feasible native realizations.

Display their resulting density values and longest uninterrupted episode durations.

Clearly label constructed realizations as hypothetical.

## Figure 4 — Burden-density-continuity joint support

Scatterplots or hexbin plots showing burden and density, colored by longest-episode share.

Include marginal distributions and the common-support region used for E02.

## Figure 5 — Physiological heterogeneity

Show patient-matched rapid-AF exposure comparisons.

Include paired lines, effect distributions, confidence intervals, and cohort-specific results.

## Figure 6 — Temporal descriptors versus ventricular physiology

Show nested-model effects and out-of-fold performance.

Include comparisons against longest episode duration and episode count.

## Figure 7 — Rapid-AF localization

Show representative real recording timelines with:

- AF rhythm state.
- Ventricular-rate trajectory.
- Rapid-AF intervals.
- Shortest AF concentration interval.
- Oracle rapid-AF concentration interval.

## Figure 8 — Conditional localization distributions

Show observed versus randomized rapid-AF capture and absolute missed rapid-AF minutes.

Report the number of independent patients contributing to each distribution.

## Figure 9 — Measurement robustness

Show ventricular-rate measurement coverage and error across authentic and synthetic noise conditions.

Generate optional figures for episode-order randomization only if that experiment is executed.

---

# PHASE L — Engineering structure and reproducibility

Create a new isolated experiment directory:

```
experiments/af_density_continuity_v5/
│
├── configs/
│   ├── analysis.yaml
│   ├── dataset_paths.yaml
│   └── statistical_analysis_plan.yaml
│
├── scripts/
│   ├── 01_build_canonical_timelines.py
│   ├── 02_build_observation_windows.py
│   ├── 03_compute_reference_density.py
│   ├── 04_temporal_aggregation.py
│   ├── 05_density_identifiability.py
│   ├── 06_extract_ventricular_rates.py
│   ├── 07_common_support_feasibility.py
│   ├── 08_conditional_physiology.py
│   ├── 09_within_patient_analysis.py
│   ├── 10_rapid_af_localization.py
│   ├── 11_conditional_randomization.py
│   ├── 12_episode_order_decomposition.py
│   ├── 13_shdb_clinical_context.py
│   ├── 14_noise_validation.py
│   ├── 15_generate_figures.py
│   ├── 16_build_master_report.py
│   └── run_all.py
│
├── src/
│   ├── dataset_adapters.py
│   ├── rhythm_intervals.py
│   ├── density_continuous.py
│   ├── density_binned.py
│   ├── density_bounds.py
│   ├── ventricular_rate.py
│   ├── matching.py
│   ├── randomization.py
│   └── statistics.py
│
├── tests/
├── manifests/
├── features/
├── results/
├── figures/
├── reports/
└── logs/
```

Reuse existing verified dataset loaders where appropriate.

Do not overwrite historical AF-density or PMV experiments.

Use configuration-driven thresholds and dataset locations.

Cache expensive intermediate calculations using source checksums and configuration hashes.

Parallelize by patient or recording while preserving deterministic results.

No GPU or neural network training is required.

Every result must be traceable to:

```
source dataset
patient identifier
record identifier
observation interval
annotation version
source file checksum
feature configuration
code version
random seed
```

---

# PHASE M — Required deliverables

Return the actual machine-readable artifacts, not only narrative summaries.

## M01. Core manifests

```
cohort_inventory.csv
canonical_rhythm_intervals.parquet
canonical_af_runs.parquet
observation_windows.parquet
eligibility_flow.csv
density_reference_features.parquet
ventricular_rate_intervals.parquet
patient_window_features.parquet
```

## M02. Primary experiment results

```
E01_resolution_sensitivity.csv
E01_density_bounds.parquet
E01_continuity_identifiability.csv

E02_common_support.csv
E02_matched_pairs.parquet
E02_matched_effects.csv
E02_nested_models.csv
E02_cohort_specific_results.csv
E02_within_patient_results.csv
E02_out_of_fold_predictions.parquet

E03_concentration_intervals.parquet
E03_rapid_af_capture.csv
E03_randomization_results.parquet
E03_interval_benchmark.csv
```

## M03. Additional results

```
episode_order_randomization.parquet
shdb_clinical_context.csv
ventricular_rate_validation.csv
noise_robustness.csv
sensitivity_analysis_summary.csv
```

Generate additional files only for experiments actually performed.

## M04. Reports

```
literature_grounding.md
density_validation_report.md
feasibility_report.md
statistical_analysis_plan.md
experiment_results.md
limitations_and_failure_modes.md
af_density_continuity_master_report.md
manuscript_candidate_findings.md
```

The master report must be generated programmatically from actual result tables.

No hardcoded findings.

No invented p-values.

No manually assigned successful experiment statuses.

Every numerical value in the report must map to a machine-readable result.

---

# PHASE N — Scientific decision gates

## Gate 1: Measurement correctness

Pass only if:

- Canonical rhythm and time inventories reconcile.
- Published density implementation is independently verified.
- Continuous and binned versions are distinguished.
- Missing data are handled explicitly.
- Ventricular-rate estimation has appropriate validation.
- Temporal aggregation conserves exact AF burden.

If this gate fails, stop the affected downstream analysis and report the failure.

## Gate 2: Physiological feasibility

Assess whether the data provide:

- Sufficient independent patients.
- Joint burden-density-continuity support.
- Meaningful variation in continuity.
- Adequate ventricular-rate coverage.
- Nontrivial rapid-AF exposure.

If common support is insufficient, do not force a physiological regression analysis.

## Gate 3: Scientific contribution

Evaluate results according to the following decision framework:

| Observed finding                                                                   | Interpretation                                                                                         |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Density remains accurate under coarsening, but continuity is poorly identified     | Density and continuity represent distinguishable measurements                                          |
| Density itself becomes materially uncertain under exact coarsening                 | Measurement identifiability is a candidate primary contribution                                        |
| Continuity explains ventricular physiology after accounting for burden and density | Pursue the physiological heterogeneity manuscript                                                      |
| Continuity association disappears after cohort or patient adjustment               | Do not claim an independent physiological phenotype                                                    |
| Rapid AF is systematically under- or overrepresented in concentration intervals    | Investigate temporal localization as a secondary or primary contribution depending on effect magnitude |
| Rapid-AF localization is approximately proportional                                | Report the negative result                                                                             |
| Simpler episode descriptors perform as well as density                             | Report limited incremental value of density for the measured phenotype                                 |
| No meaningful effects or ambiguity are demonstrated                                | Do not invent a new biomarker to rescue the hypothesis                                                 |

Do not define success exclusively by statistical significance.

The purpose of this investigation is to discover what the datasets support, including negative results.

---

# Final execution instructions

Execute the investigation sequentially.

Complete Phases A and B before physiological interpretation.

Complete temporal aggregation analysis and outcome-blind common-support assessment before fitting physiological models.

Use verified feasibility to determine whether optional experiments are justified.

Do not create another excessively broad experiment campaign if the central question is not identifiable.

Do not train an AF detector, ECG reconstruction model, recurrent neural network, or transformer.

Use the reference rhythm labels.

Do not attempt to predict stroke, assess anticoagulation eligibility, infer ablation success, or recommend clinical treatment.

The final report must explicitly distinguish:

1. Mathematical properties established by derivation.
2. Empirical findings from authentic ECG recordings.
3. Sensitivity findings from simulated or rearranged timelines.
4. Statistical associations.
5. Physiological hypotheses.
6. Unsupported mechanisms and unavailable clinical outcomes.

The ultimate objective is to establish whether AF density is sufficient to characterize temporal arrhythmia exposure, which information is lost through temporal summarization, and whether that lost information corresponds to measurable ventricular electrophysiological differences.

Be critical of the results. A technically successful pipeline is not evidence of a successful scientific hypothesis.

# **How the experiments fit together**

**Phase A–B: Reference data and mathematical validity**

Canonical timelines · Original Charitos density · Exact interval solver

**E01 — What information is lost?**

Perfect AF labels at progressively coarser temporal resolutions

**Feasibility gate**

Independent patients, common support, ventricular-rate coverage

**E02 — Physiological heterogeneity**

Matched burden and density, different continuity

**E03 — Temporal localization**

AF concentration versus rapid-AF concentration

**Cohort replication and measurement robustness**

Within-patient analysis · SHDB clinical context · Authentic noise

**Evidence-based manuscript decision**

# **What I particularly want to discover**

Of all these experiments, I am most interested in **E01's identifiability analysis and E02's conditional physiology analysis**.

The former has a potentially strong mathematical contribution that does not depend on discovering a statistically significant biological relationship.

The latter determines whether that mathematical distinction has measurable electrophysiological significance.

Consider this hypothetical example of six hours of monitoring, with identical AF duration in every hourly bin.

### **Identical hourly AF summaries, different underlying episodes**

Each hour contains exactly 30 minutes of AF.

Sustained patternFragmented pattern

Total AF burden

# **50%**

Longest uninterrupted episode

# **5 min**

AF duration

# **180 min**

Underlying AF runs

# **36**

Constructed mathematical example. Both timelines generate the same hourly AF-duration vector and therefore the same density when computed deterministically from that vector. Their native-resolution densities need not be identical.

This demonstrates a genuine identifiability problem. No algorithm receiving only those hourly summaries could determine which underlying rhythm trajectory occurred without introducing additional assumptions.

The empirical investigation would establish how important that ambiguity is in real AF recordings.

# **Expected scientific value of each experiment**

| Experiment                          | Potential contribution                                                             | Priority   |
| ----------------------------------- | ---------------------------------------------------------------------------------- | ---------- |
| Exact temporal aggregation          | Quantifies information loss without classifier errors                              | Critical   |
| Mathematical identifiability bounds | Establishes what can and cannot be recovered from summarized AF histories          | Critical   |
| Matched physiological heterogeneity | Establishes whether continuity contains additional ECG-observable information      | Critical   |
| Simpler-metric comparisons          | Determines whether density is necessary for the measured physiological phenotype   | High       |
| Rapid-AF localization               | Characterizes temporal correspondence between rhythm and ventricular-rate exposure | High       |
| Within-patient comparisons          | Reduces stable patient-level confounding                                           | High       |
| Episode-order randomization         | Decomposes the information captured by density                                     | Secondary  |
| SHDB clinical metadata              | Provides supporting clinical context                                               | Secondary  |
| Noise robustness                    | Establishes measurement reliability                                                | Supporting |

The literature already recognizes that AF episode duration, ventricular rates, irregularity and pauses may carry complementary clinical information. It also recognizes the distinction between prolonged and fragmented AF patterns. We should not claim these general concepts as novel.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov\&sz=32)

pmc.ncbi.nlm.nih.gov+1



**Our strongest potential contribution is the quantitative relationship between what AF density mathematically identifies, what temporal aggregation conceals, and what the underlying continuous ECG reveals.**

That gives us a coherent investigation even if the physiological association turns out to be negative.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-23T22:54:27-05:00.

The user has mentioned some items in the form @[ITEM]. Here is extra information about the items that were mentioned by the user, in the order that they appear:

/goal is a [Slash Command]:
The user has marked this task with /goal, indicating that this task is intended to run for a long time without user input, e.g. overnight. You should be extra thorough and only stop when you are confident the goal has been completely fulfilled. The system will force you to continue execution, prompting you to audit your work until completion. Once complete, include <!-- GOAL_COMPLETE --> in your response. If the user explicitly asked to stop or cancel this goal, include <!-- GOAL_CANCELLED --> in your response to cancel the goal.
</ADDITIONAL_METADATA>