<USER_REQUEST>
okay run everything that is alsso topitional or secondary , so run everything and verify ypur number scpuntsetc no ytnyeti fall backs or heyrtstitc hand wavy stuff, maximize throuhput we have a 40 gb gpu /goal /swarm The next experimental campaign should be substantially more focused than our earlier ones. We now have specific published studies that match the characteristics of our individual cohorts, rather than merely sharing the general topic of AF density.

I would organize the campaign around **four complementary investigations**, with the temporal organization of short and prolonged AF episodes as the primary investigation.

| Experiment                                              | Primary dataset | Scientific question                                                                                 |
| ------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- |
| E01 — Temporal neighbourhoods                           | LTAFDB, SHDB    | Do brief AF runs preferentially surround prolonged AF episodes?                                     |
| E02 — Historical concentration and electrical evolution | IRIDIA          | Does previous AF organization relate to ventricular dynamics during subsequent prolonged AF?        |
| E03 — Clinical temporal phenotypes                      | SHDB            | Do different episode architectures correspond to different structural and clinical characteristics? |
| E04 — Phenotype-dependent observability                 | Noise Holter    | Which rhythm characteristics remain measurable under authentic ambulatory artifact?                 |

The objective is to connect our existing density measurements to actual arrhythmia behaviour, not introduce another arbitrary biomarker or train another AF detector.

The coding agent must also distinguish **an association that is mathematically inevitable from an association that provides new physiological information**. For example, demonstrating that shorter interepisode gaps increase temporal concentration is not, by itself, a meaningful biological discovery.

# **Master instructions for the coding agent**

# AF-DENSITY V7 — Episode-Scale Temporal Architecture, Electrical Evolution, and Clinical Phenotyping in Continuous Ambulatory ECG

## 0. Scientific mandate

Conduct a focused, literature-informed investigation of AF temporal organization across IRIDIA-AF, SHDB-AF, LTAFDB and Noise Holter.

Use our existing reference annotations, waveform data, clinical metadata, feature extraction pipelines and model checkpoints.

Do not develop another AF classification architecture.

Do not introduce a new composite AF biomarker.

Do not merely reproduce population-level correlations between AF burden, density, longest episode and episode count.

The overarching scientific question is:

**Among patients undergoing continuous ambulatory ECG monitoring, does the temporal organization of AF provide information about subsequent rhythm behaviour, ventricular electrophysiology and clinical phenotype beyond cumulative AF exposure and conventional episode characteristics?**

The experiments must exploit the specific strengths of each cohort.

Do not force all four datasets into a single pooled model.

---

# PART A — Literature-driven experimental rationale

## A01. Required literature comparators

Before execution, establish the precise methodological connection between each experiment and the following publications.

### Van Vreeswijk et al., Europace, 2026

DOI: `10.1093/europace/euaf300`

Clinical question: Whether irregular atrial arrhythmias shorter than 30 seconds identify patients who subsequently exhibit conventionally defined AF.

Our distinction:

- Patients with documented or observed AF rather than simply screening populations.
- Native Holter rhythm transitions.
- Multiple AF duration classes.
- Temporal position of brief runs relative to longer episodes.
- Within-recording and within-patient comparisons.
- AF concentration and temporal architecture.

Do not describe our study as a replication of their subsequent AF risk estimates.

### Ngo et al., BMC Medical Research Methodology, 2026

DOI: `10.1186/s12874-026-02773-9`

Clinical question: Dynamic prediction of PAF onset using longitudinal RR sample entropy in IRIDIA-AF.

Our distinction:

- Repeated AF episodes rather than only the first observed episode per recording.
- Explicit preceding AF exposure.
- Historical temporal concentration.
- Subsequent episode behaviour and within-episode physiology.
- Patient-level dependence.

### Butkuvienė et al., IEEE TBME, 2024

DOI: `10.1109/TBME.2023.3293252`

Clinical question: Reliability of AF density and aggregation under imperfect episode detection.

Our distinction:

- Explicit short-versus-prolonged episode architecture.
- Temporal neighbourhoods.
- Dependence on clinical phenotype.
- Authentic rhythm-quality interaction measurements.

Retain their published density and aggregation methods as comparators.

### Henriksson et al., IEEE TBME, 2021

DOI: `10.1109/TBME.2020.2995563`

Clinical question: History-dependent modelling of alternating AF and non-AF transitions.

Our distinction:

- Patients with sparse prolonged paroxysms versus frequent short runs.
- Published concentration measures evaluated alongside history-dependent transition descriptors.
- Physiological and clinical outcomes, where measurable.

Do not confuse the exponential Hawkes excitation kernel with Charitos AF density.

### Heart Rate Variability Is Decreasing During Paroxysmal Atrial Fibrillation, 1999

DOI: `10.1111/j.1542-474X.1999.tb00053.x`

Clinical question: Evolution of ventricular RR irregularity during prolonged PAF episodes.

Our distinction:

- Explicit historical AF concentration.
- Fixed elapsed-time measurements.
- Repeated episodes within patients.
- Separation of ventricular rate from ventricular irregularity.

### Immediate Reinitiation of Atrial Tachyarrhythmias, 2003

DOI: `10.1046/j.1460-9592.2003.t01-1-00189.x`

Our distinction:

- Surface Holter observations rather than the specific implanted-device bradycardia population.
- Explicit short and prolonged AF duration classes.
- Multiple preceding episodes.
- Clinical treatment and structural characteristics where available.

### Inoue et al., Circulation Journal, 2026

DOI: `10.1253/circj.CJ-25-1126`

Clinical question: Extended Holter monitoring and characterization of low-burden AF.

Our distinction:

- Temporal architecture conditional on observed burden.
- Short versus prolonged individual episodes.
- Repeated interruptions.
- Native rhythm resolution.

### Rosman et al., European Heart Journal, 2026

DOI: `10.1093/eurheartj/ehag203`

### Daghlas and Marcus, Journal of Cardiovascular Aging, 2026

DOI: `10.20517/jca.2026.52`

Use these publications to motivate investigation of the biological and clinical meaning of AF density.

Our cohorts cannot establish incident stroke risk, anticoagulation benefit or a causal effect of density on thrombogenesis.

For every publication, distinguish full-text verification from abstract-level evidence. Do not invent methodological details when the complete paper is unavailable.

---

# PART B — Cohort-specific research design

## B01. Data sources

Use:

- IRIDIA-AF.
- SHDB-AF.
- LTAFDB.
- Noise Holter Database.
- MIT-BIH NSTDB only as a supplementary controlled technical benchmark.

Use the latest verified canonical manifests and claim verification registry.

The previously reported temporal characteristics are planning information, not values to force the new experiments to reproduce.

| Characteristic                   | IRIDIA    | SHDB     | LTAFDB   |
| -------------------------------- | --------- | -------- | -------- |
| Reference recordings             | 167       | 98       | 84       |
| Dataset-scoped patients          | 152       | 93       | 84       |
| Native sampling                  | 200 Hz    | 125 Hz   | 128 Hz   |
| Previously reported AF runs      | 388       | 794      | 7,358    |
| Median observed AF duration      | 82.08 min | 51.9 sec | 14.6 sec |
| Median observed interepisode gap | 78.95 min | 45.5 sec | 6.0 sec  |
| Multi-run recordings             | 71        | 54       | 66       |

Reconcile these values against the actual verified manifests before analysis.

## B02. Cohort roles

### IRIDIA-AF

Primary cohort for:

- Prolonged paroxysmal AF.
- Episodes lasting tens of minutes to hours.
- Within-episode ventricular dynamics.
- Multi-hour preceding AF histories.
- Longer monitoring horizons.
- Repeated prolonged episodes.

Do not describe patients as clinically persistent AF merely because their observed episodes are prolonged.

### SHDB-AF

Primary cohort for:

- Intermediate-duration recurrent AF.
- Clinical AF subtype.
- Previous ablation.
- Medication exposure.
- LAD and LVEF.
- Temporal architecture conditional on clinical characteristics.

Use only appropriately annotated recordings for reference rhythm analyses.

### LTAFDB

Primary cohort for:

- Very short annotated AF runs.
- Rapid recurrence.
- Repeated AF/non-AF transitions.
- Temporal neighbourhoods around prolonged AF.
- High episode-count histories.

Do not infer a micro-reentrant mechanism from rhythm transitions alone.

### Noise Holter

Primary cohort for:

- Authentic ECG artifact.
- Competing arrhythmias.
- Rhythm-specific feature observability.
- AF detector errors.
- Reliability of ventricular measurements.

Do not concatenate disconnected excerpts into artificial longitudinal AF histories.

---

## B03. Targeted pre-execution corrections

The current knowledge base still contains statements that must not propagate into V7:

1. Charitos density does not use an exponential decay kernel.
2. Observed non-AF gaps are not necessarily sinus rhythm.
3. Declining fitted termination hazard does not establish electrical remodeling.
4. A pooled frailty parameter is not automatically a separately validated parameter for every cohort.
5. Noise Holter expert quality labels must not be replaced by bSQI thresholds.
6. A significant structural association disappearing after patient exclusion does not establish that the original association was an artifact.
7. Difference in significance between subgroups does not establish a statistically significant subgroup difference.
8. A historical density range of 0.10 does not establish conditional exposure variation after accounting for burden and recent episode history.
9. Synthetic AFR recovery does not establish physiological accuracy on authentic ECG.
10. The previously exposed IRIDIA test partition must not be described as untouched confirmatory data.

Correct these statements where they directly affect the current investigation.

Do not launch another unrestricted forensic audit. Use focused validation gates for the dependencies required by V7.

Respect existing restrictions on protected data access.

---

# EXPERIMENT E01 — Temporal neighbourhoods of prolonged AF

## E01.1 Primary scientific question

Among patients with documented AF undergoing continuous ambulatory ECG monitoring:

**Are brief AF runs disproportionately concentrated in the temporal neighbourhood of prolonged AF episodes, and does their distribution differ before versus after those episodes?**

The objective is not simply to demonstrate that AF episodes cluster.

The objective is to determine whether episodes of different duration classes exhibit distinct temporal relationships.

## E01.2 Duration classes

Assign each native annotated AF run to:

| Class | Duration                          |
| ----- | --------------------------------- |
| S0    | Less than 10 seconds              |
| S1    | 10 to less than 30 seconds        |
| S2    | 30 seconds to less than 2 minutes |
| S3    | 2 to less than 6 minutes          |
| M1    | 6 to less than 30 minutes         |
| L1    | 30 to less than 60 minutes        |
| L2    | 1 to less than 6 hours            |
| L3    | At least 6 hours                  |

Primary brief-run definition:

`0 < duration < 30 seconds`

Primary prolonged index-episode definition:

`duration ≥ 30 minutes`

Secondary index thresholds:

- At least 6 minutes.
- At least 60 minutes.
- At least 6 hours, if supported.

Use fully observed brief runs for the primary duration-class assignment.

Left- or right-censored runs with observed duration below 30 seconds cannot be confidently classified as genuinely brief.

Preserve the native rhythm timeline. Duration classes are analytical labels, not modifications to the source annotations.

## E01.3 Patient-level feasibility inventory

Before statistical analysis, determine the number of patients in each cohort who have:

- At least one prolonged AF episode.
- At least one brief AF run.
- Both prolonged and brief AF episodes.
- At least three brief runs.
- At least five brief runs.
- At least two prolonged index episodes.
- Complete observation before index onset.
- Complete observation after index termination.
- Adequate comparable remote non-AF time.

Produce a patient-by-duration-class matrix.

Do not assume that the total number of brief episodes indicates sufficient independent patient support.

Specifically quantify how much of the eligible LTAFDB population is contributed by the most prolific patients.

## E01.4 Define temporal neighbourhoods

For each eligible prolonged index episode, define:

- Onset time (t\_{\mathrm{on}}).
- Observed termination time (t\_{\mathrm{off}}).

Construct fixed physical-time windows:

### Pre-onset

- 0–5 minutes before onset.
- 5–15 minutes before onset.
- 15–30 minutes before onset.
- 30–60 minutes before onset.

### Post-termination

- 0–5 minutes after termination.
- 5–15 minutes after termination.
- 15–30 minutes after termination.
- 30–60 minutes after termination.

Primary comparison: the first 15 minutes on each side of the index episode against suitable remote observation intervals.

Secondary comparisons: the complete 60-minute neighbourhood and individual distance bands.

The pre-onset and post-termination windows must not overlap the index episode itself.

Exclude unknown rhythm and unobserved acquisition intervals from eligible exposure.

For each neighbourhood, report:

- Physical duration.
- Known rhythm duration.
- AF duration.
- Known non-AF duration.
- Brief AF onset count.
- Complete brief AF run count.
- Intermediate AF onset count.
- Other prolonged AF onset count.
- Overlapping index-episode neighbourhoods.
- Censoring.
- Boundary completeness.

## E01.5 Handle overlapping neighbourhoods

In highly recurrent LTAFDB histories, one brief AF run may be close to multiple prolonged episodes.

For the primary analysis, assign each eligible onset to at most one index-episode neighbourhood using a deterministic nearest-boundary rule.

Document how ties are resolved.

Conduct a sensitivity analysis allowing repeated neighbourhood membership, with dependence accounted for appropriately.

Do not count one brief episode as multiple independent events.

## E01.6 Distinguish physical-time exposure from onset opportunity

A central analytical issue is that AF cannot initiate again during time already occupied by AF.

Therefore calculate two complementary rates.

### Physical-time brief-run rate

\frac{N\_{\mathrm{brief\ onsets}}}\
{T\_{\mathrm{known\ rhythm}}}\
]

This quantifies the observed concentration of brief AF events within physical time.

### Non-AF risk-time brief-run rate

\frac{N\_{\mathrm{brief\ onsets}}}\
{T\_{\mathrm{known\ nonAF}}}\
]

This quantifies the occurrence of brief AF onset per unit of available non-AF time.

Use hours as the denominator for reported rates.

Both measurements are necessary.

An apparently greater episode concentration may arise from differences in available non-AF time rather than an increase in onset propensity.

Additionally, distinguish:

- All AF onset intensity.
- Brief AF onset intensity.
- Proportion of subsequent AF onsets that remain brief.

This separates increased recurrence frequency from changes in the duration distribution of recurrent episodes.

## E01.7 Remote comparison intervals

Select comparable remote observation intervals from the same recording where possible.

Requirements:

- Same patient.
- Same physical window duration.
- Known rhythm coverage.
- No acquisition gap.
- Appropriate separation from prolonged AF boundaries.
- Similar available non-AF exposure where possible.
- Comparable time of day where reliable timestamps permit.

Primary remote exclusion: at least 60 minutes from a prolonged index-episode boundary.

Secondary: 30 and 120 minutes.

Do not select remote windows because they contain zero brief episodes.

The selection rule must be outcome-independent.

If highly concentrated histories contain no eligible remote comparison, report that limitation explicitly.

## E01.8 Primary estimands

Estimate:

1. Brief AF onset rate ratio: post-termination neighbourhood versus remote observation.
2. Brief AF onset rate ratio: pre-onset neighbourhood versus remote observation.
3. Post-termination versus pre-onset rate ratio.

Use known non-AF risk time as the primary offset.

Report physical-time rate contrasts as complementary descriptive results.

Estimate patient-level uncertainty.

Where feasible, use patient-stratified Poisson or negative-binomial models with appropriate offsets.

Inspect:

- Overdispersion.
- Excess zeros.
- Sparse event counts.
- Within-patient support.
- Patient influence.
- Cohort heterogeneity.

Do not pool all LTAFDB episodes as independent observations.

## E01.9 Conditional null experiments

A naive random shuffle of AF labels is insufficient.

Construct null histories designed to distinguish different explanations of temporal organization.

### Null A — Reposition brief runs conditional on prolonged episodes

Preserve:

- Observation duration.
- Total AF exposure.
- Duration of every prolonged episode.
- Position of every prolonged episode.
- Number of brief runs.
- Brief-run duration distribution.
- Other fixed rhythm intervals where required.

Reposition brief runs only into eligible, nonoverlapping portions of the observation interval.

Preserve minimum gap constraints where imposed.

Validate every generated history.

This null asks whether the observed brief-run neighbourhood pattern exceeds what is expected from the available placement opportunities.

### Null B — Episode-duration ordering

Preserve:

- Episode-duration multiset.
- Interepisode-gap multiset.
- Observation duration.
- Total AF exposure.
- Leading and trailing observation intervals where possible.

Permute allowable episode orderings.

This asks whether observed relationships between short and prolonged episodes differ from alternative temporal arrangements with identical duration and gap distributions.

For small histories, enumerate feasible arrangements.

For larger histories, use Monte Carlo sampling.

Document whether the sampler actually targets the intended distribution.

Do not assume that an arbitrary sequential placement algorithm samples uniformly from all feasible histories.

Use at least 2,000 valid realizations where computationally practical.

If the null is mathematically or computationally infeasible for a patient, mark the comparison unavailable.

These are conditional descriptive nulls, not proofs of a biological mechanism.

## E01.10 AF density connection

For every eligible recording calculate:

- AF burden.
- Native continuous Charitos density.
- Published AF aggregation.
- Longest observed episode.
- Longest-episode share.
- Brief-run count.
- Brief-run AF exposure.
- Intermediate-run count.
- Prolonged-run count.
- Neighbourhood brief-run rate ratios.

Then examine whether recordings with comparable burden and density exhibit different brief-run neighbourhood architectures.

Specifically quantify:

- High density with one predominant prolonged episode.
- High density with recurrent short-run clusters.
- High density with both prolonged and short-run components.

Use continuous descriptive variables rather than imposing arbitrary biological phenotype labels.

A new composite score is not required.

## E01.11 Sensitivity analyses

Repeat the primary analysis under:

- Brief runs below 10 seconds excluded.
- Brief runs below 5 seconds excluded.
- Minimum AF duration of 30 seconds for index histories where applicable.
- Bridging non-AF gaps of 3, 5, 10 and 30 seconds.
- Excluding the most prolific LTAFDB patients.
- Equal patient weighting.
- Restriction to clinical PAF in SHDB.
- Exclusion of prior ablation in SHDB, if supported.
- Alternative neighbourhood durations.
- Alternative remote comparison definitions.

Gap bridging must be recorded as a derived analytical transformation. Never overwrite the native timeline.

## E01.12 Clinical interpretation gate

If brief runs preferentially occur after prolonged AF, report a temporal association.

Do not automatically call this:

- Atrial stunning.
- Incomplete mechanical recovery.
- Atrial remodeling.
- Reentry.
- Pulmonary vein triggering.
- Increased thrombogenicity.

Those mechanisms require additional independent evidence.

---

# EXPERIMENT E02 — Historical AF concentration and subsequent prolonged-episode electrophysiology

## E02.1 Primary clinical question

Among patients with prolonged PAF episodes:

**Does the organization of preceding AF exposure relate to the subsequent ventricular electrical trajectory, beyond preceding burden, recent AF duration and the most recent non-AF interval?**

Primary cohort: IRIDIA-AF.

Secondary cohorts: SHDB-AF and LTAFDB, restricted to patients and episodes satisfying comparable eligibility.

## E02.2 Separate the rhythm and physiological populations

Create two analytical populations.

### Population A — Subsequent rhythm behaviour

Includes newly observed AF onsets with complete preceding history.

Do not require the subsequent episode to last a minimum of four minutes.

Short episodes are relevant outcomes when investigating duration and termination.

### Population B — Within-episode electrophysiology

Includes episodes that remain in AF sufficiently long to support fixed elapsed-time measurements.

The eligibility requirement depends on the physiological landmark under examination.

Do not use Population B to estimate termination probabilities for all newly occurring AF episodes.

## E02.3 Historical exposure windows

For each new AF onset at time (t_0), construct:

[\
H_1=[t_0-1h,t_0)\
]

[\
H_3=[t_0-3h,t_0)\
]

[\
H_6=[t_0-6h,t_0)\
]

Secondary:

- 30-minute history.
- 12-hour history.

Primary candidate horizon: 6 hours.

Select the primary executable horizon according to prespecified patient-support criteria before examining physiological outcome associations.

Historical data must be entirely prior to the current onset.

Exclude incomplete rhythm histories from the primary analysis.

## E02.4 Historical features

Calculate:

- Historical AF burden.
- Historical continuous density.
- Historical aggregation.
- Historical AF duration.
- Historical run count.
- Longest preceding observed run.
- Longest preceding complete run.
- Longest-episode share.
- Immediately preceding AF episode duration.
- Most recent known non-AF interval.
- Time since last prolonged AF episode.
- AF burden during the most recent 15, 30 and 60 minutes.
- Historical AF onset rate.
- Historical duration-class composition.

Distinguish no prior AF from undefined density.

Do not assign density zero to histories without AF.

For primary conditional density analyses, distinguish histories containing one run from histories containing multiple runs.

Density is structurally saturated at one in a single contiguous AF history under the stated continuous formulation.

## E02.5 Feasibility and common support

The previous historical-density pilot reported approximately 24–29 patients with meaningful raw within-patient density variation across different horizons.

Do not assume those patients satisfy the present prolonged-episode physiological requirements.

Reconstruct eligibility specifically for:

- Subsequent episodes lasting at least 5 minutes.
- At least 15 minutes.
- At least 30 minutes.
- At least 60 minutes.
- At least 120 minutes.

For each horizon and landmark, report:

- Independent patients.
- Eligible episodes.
- Repeated eligible episodes.
- Observed termination events.
- Available beats.
- Valid waveform coverage.
- Historical density range.
- Historical burden range.
- Conditional density variation after accounting for burden and recent history.
- Contribution by cohort.
- Contribution by patient.

Report correlations and joint distributions of historical variables.

Do not automatically select only patients with density range ≥0.10 for the primary population. That threshold can define a secondary informative subset.

A patient fixed-effects coefficient is not meaningful if the relevant exposure has essentially no residual within-patient variation.

## E02.6 Primary electrophysiological outcome

Use conventional RR irregularity measurements rather than inventing another feature.

Primary candidate:

[\
CV\_{RR}=\frac{SD(RR)}{\overline{RR}}\
]

Secondary:

- RMSSD.
- SDNN.
- Median absolute successive RR difference.
- Published VCC.
- Sample entropy.
- Mean ventricular rate.
- Median ventricular rate.
- Rapid ventricular response exposure.

Verify feature definitions against the existing metric registry and original publications.

These are ventricular interval dynamics measured during AF. Do not describe them as conventional sinus-rhythm autonomic HRV without qualification.

## E02.7 Fixed elapsed-time physiology

Measure physiological features in fixed 150-second windows during AF, motivated by the historical Holter variability study.

Candidate windows:

- 30–180 seconds after onset.
- 5–7.5 minutes.
- 10–12.5 minutes.
- 15–17.5 minutes.
- 30–32.5 minutes.
- 60–62.5 minutes.

The first window excludes the immediate 30-second transition region.

Perform sensitivity analyses that include the immediate onset region.

For every window require:

- Appropriate reference AF coverage.
- Reliable beat timing.
- Sufficient RR intervals.
- No unhandled acquisition discontinuity.
- Explicit artifact assessment.

Do not silently treat automated beat detections as manually verified beats.

Document whether interval selection includes ectopy, pacing or abnormal pauses.

## E02.8 Primary trajectory estimand

Estimate the association between historical density and the change in ventricular RR irregularity between the initial and a later fixed elapsed-time window.

Primary candidate comparison:

- 30–180 seconds after onset.
- 15–17.5 minutes after onset.

This contrast is conditional on the episode remaining active and measurable through the second window.

Explicitly state that conditioning on episode survival may introduce selection.

Report the characteristics of episodes terminating before the second landmark.

Do not generalize the contrast to all AF episodes.

Secondary analysis: a repeated-measures trajectory across fixed elapsed-time landmarks.

Include a separate onset-only analysis using the initial available AF window.

## E02.9 Separate rate and irregularity

Mean RR and RR variability are related.

Report:

- Changes in mean ventricular rate.
- Changes in RR variability.
- Rate-adjusted RR variability association.
- Unadjusted RR variability association.

Do not interpret rate adjustment as establishing a causal electrophysiological effect.

Investigate whether observed variability changes arise mainly from:

- Rate change.
- A small number of extreme RR intervals.
- Ectopy.
- Beat detection error.
- Broad changes in the interval distribution.

## E02.10 Statistical models

Use nested models fitted on identical analytical populations.

### M0 — Patient and elapsed-time structure

- Patient effect.
- Elapsed-time landmark.

### M1 — Recent AF history

Add:

- Immediately preceding episode duration.
- Most recent non-AF gap.

### M2 — Cumulative exposure

Add historical AF burden.

### M3 — Episode architecture

Add:

- Historical run count.
- Relevant recency descriptor.
- Longest preceding episode or share, subject to collinearity.

### M4 — Temporal concentration

Add historical density and, for trajectory models, its interaction with elapsed time.

Report:

- Association per 0.10 density increase.
- Within-patient association.
- Between-patient association separately.
- Confidence interval.
- Conditional support.
- Incremental model fit.
- Patient-level influence.
- Cohort-specific estimates.

Use patient fixed effects or an appropriate within-between decomposition where supported.

Do not fit every available historical descriptor simultaneously in a population of only a few dozen informative patients.

Use parsimonious models and document the effective number of independent patients.

## E02.11 Subsequent episode duration and termination

Using Population A, investigate whether preceding density is associated with the duration of the next AF episode.

Model the time from observed AF onset to observed termination.

Handle right-censoring explicitly.

Exclude episodes with unobserved onset from onset-based survival analyses.

Compare:

1. Recent episode history.
2. Recent history plus historical burden.
3. Recent history plus conventional temporal architecture.
4. The preceding model plus historical density.

Use appropriate clustered or within-patient survival methods.

Report the distinction between:

- Association with subsequent episode duration.
- Incremental predictive information, if evaluated.
- Causal interpretation, which is not established.

Do not require long future episode duration to enter this primary survival population.

Do not describe observed termination as spontaneous unless intervention-related termination can be excluded.

## E02.12 Published methodology comparisons

Where technically feasible, reproduce the published 1999 onset-middle-end ventricular irregularity comparison as a secondary descriptive analysis.

Do not treat middle or terminal windows defined using final episode duration as prospectively available features.

For high-event-count patients, compare established history-dependent transition descriptors from Henriksson's methodology.

Apply the published minimum event requirements.

Do not fit unstable patient-specific Hawkes models to two or three episodes merely to obtain a numerical result.

## E02.13 AFR and PMV secondary analyses

Use existing AFR and PMV pipelines only where independently established eligibility is satisfied.

### AFR

Investigate historical density versus initial dominant fibrillatory frequency only if:

- Both ventricular cancellation and peak identification are technically credible.
- Measurement repeatability is adequate relative to the expected frequency difference.
- Paired valid patient and episode support is sufficient.
- Cohort-specific selection is documented.

Do not treat inter-lead agreement as independent physiological ground truth.

### PMV

Investigate preceding temporal architecture versus post-termination P-wave variability only in windows with adequate atrial waveform observability.

Distinguish confirmed sinus rhythm from unspecified non-AF.

Do not require PMV or AFR availability for the primary rhythm-duration investigation.

---

# EXPERIMENT E03 — Clinical characterization of temporal AF phenotypes

## E03.1 Primary scientific question

Among SHDB patients with clinically documented PAF:

**At comparable observed AF burden, do different episode architectures correspond to different structural and clinical characteristics?**

This directly addresses the clinical-substrate question raised by Daghlas and Marcus, within the limitations of available metadata.

## E03.2 Primary population

SHDB reference-annotated patients with:

- Documented clinical AF subtype.
- Known observation duration.
- Defined AF burden.
- Available rhythm architecture.
- Appropriate patient-level clinical linkage.

Primary clinical subgroup:

- Clinically documented PAF.
- Multiple native observed AF runs.

Secondary:

- All PAF patients with measurable density.
- All eligible AF-positive patients.
- Persistent AF patients described separately.

Do not infer clinical AF subtype from the duration of the available Holter episode.

## E03.3 Standardize observation duration

Primary descriptive horizon: first 12 hours of continuous known rhythm, anchored to a documented acquisition boundary.

Secondary:

- First 24 hours where fully available.
- Full native recording.
- First 6 hours.

Use only eligible complete horizons for standardized comparisons.

Do not artificially extend recordings to 24 hours.

Report patients excluded because a standardized horizon is unavailable.

## E03.4 Temporal architecture

For each patient measure:

- AF burden.
- Continuous AF density.
- Longest episode.
- Longest-episode share.
- Episode frequency.
- Brief-run fraction.
- Brief-run exposure fraction.
- Median episode duration.
- Median interepisode gap.
- Rapid reinitiation frequency.
- Neighbourhood measures from E01, where supported.

Do not initially use unsupervised clustering to manufacture phenotype groups.

Begin with interpretable continuous distributions and joint plots.

For visualization, use the following descriptive architecture categories:

- Single observed episode.
- Multiple episodes with one predominant episode.
- Multiple episodes without one predominant episode.

Any threshold used to define predominance must be documented and subjected to sensitivity analysis.

These categories are descriptive, not validated clinical subtypes.

## E03.5 Low-burden AF subgroup

Motivated by Inoue et al. and the Rosman burden-density framework, characterize:

- Burden below 5%.
- Burden below 10%.
- Burden from 10% to 30%.

Within these strata, examine:

- Longest episode below 6 minutes.
- Longest episode from 6 to 60 minutes.
- Longest episode above 60 minutes.
- Brief-run frequency.
- Longest-episode share.
- AF density.

The first two burden categories are nested and must not be treated as mutually exclusive groups.

For mutually exclusive tables, additionally use:

- Below 5%.
- 5% to below 10%.
- 10% to 30%.
- Above 30%.

Quantify available support before formal inference.

Do not transfer Rosman's published stroke risk estimates to these patients.

## E03.6 Clinical characteristics

Retrieve independently linked:

- Age.
- Sex.
- Height.
- Weight.
- BMI.
- BSA.
- Clinical AF subtype.
- Prior AF ablation.
- Prior flutter ablation.
- Documented atrial flutter history.
- Beta-blocker exposure.
- Antiarrhythmic medication exposure.
- Anticoagulation.
- LAD.
- LVEF.
- Hypertension.
- Heart failure.
- Diabetes.
- Previous stroke/TIA.

Report exact field definitions and missingness.

Medication exposure must refer to a clinically relevant time, where available.

Do not treat unavailable medication information as absence of treatment.

## E03.7 Structural association

Primary structural outcome:

Measured LAD in millimetres.

Primary comparison:

Relationship between temporal architecture and LAD in clinically documented PAF patients with multiple observed episodes and contemporaneous echocardiography.

Report exact sample support for:

- Echo within 30 days.
- Echo within 90 days.
- Echo within 180 days.
- Unrestricted echo interval as an exploratory comparison.

Use 90 days as the primary candidate interval if adequate patient support exists.

Report the signed echo-to-Holter interval.

Do not assume echocardiography is contemporaneous merely because it appears in the same clinical metadata file.

Fit parsimonious models:

- Age and sex.
- Add body size.
- Add AF burden.
- Add one temporal architecture measure at a time.

Compare density against longest episode, longest-episode share, episode frequency and recurrence descriptors on identical complete-case populations.

Avoid simultaneous inclusion of deterministically related measures.

At fixed observation duration and burden, longest episode and longest-episode share encode equivalent information.

Do not interpret LAD as a comprehensive measure of atrial fibrosis or atrial cardiomyopathy.

## E03.8 Structural inference requirements

Report:

- Effect in mm per 0.10 density increase.
- Confidence interval.
- Patient count.
- Residual diagnostics.
- Partial R².
- Influence diagnostics.
- Patient-bootstrap interval where appropriate.
- Echo timing.
- Same-population body-size sensitivity.

Do not remove a valid influential patient solely because the coefficient becomes nonsignificant.

Do not conclude that prior ablation or medication causes a particular temporal phenotype.

Treatment associations are subject to confounding by indication and previous disease severity.

## E03.9 Other clinical outcomes

LVEF and prior stroke/TIA may be described as secondary cross-sectional characteristics.

Do not fit an incident stroke-risk model.

Do not use previous stroke/TIA as a surrogate for prospectively observed thromboembolism.

Avoid high-dimensional clinical prediction models in small strata.

The primary purpose of this experiment is clinical characterization, not classification accuracy.

---

# EXPERIMENT E04 — Phenotype-dependent ECG observability under authentic artifact

## E04.1 Scientific question

**Which rhythm characteristics and physiological measurements remain observable in real ambulatory ECG when artifact compromises the waveform, and does this differ according to AF duration, ventricular response and competing arrhythmia?**

Primary dataset: Noise Holter.

This experiment supports the interpretation of E01–E03.

It is not another general detector leaderboard.

## E04.2 Verify the complete local Noise Holter population

Reconcile the official release inventory with the actual local files.

Construct:

`Rhythm × Expert quality × Duration class × Lead availability`

Include every supported rhythm class.

Preserve the expert-assigned quality taxonomy.

Calculate bSQI separately.

Do not assume that Clean, N1 and N2 are equivalent to fixed bSQI thresholds.

## E04.3 Duration and model eligibility

Characterize excerpt lengths before inference.

Determine whether excerpts support:

- 5-second measurements.
- 10-second measurements.
- 20-second measurements.
- 30-second measurements.
- Minimum beat-count requirements.

A model trained on 30-second ECG cannot automatically be evaluated validly on a 10-second excerpt.

Do not use repetition, zero-padding, artificial concatenation or stretching unless the adaptation was previously validated and is explicitly reported as a separate condition.

If an existing detector cannot accept an excerpt, classify that excerpt as outside its supported input domain.

## E04.4 Existing detectors only

Use existing authorized checkpoints for:

- RR-GBDT.
- Raw Mask ResNet.
- RawECGNet.
- ArNet2.
- Other existing valid comparators where applicable.

Freeze preprocessing and thresholds.

Do not optimize detector thresholds on the same Noise Holter evaluation excerpts.

If calibration adaptation is necessary, create a separate patient-disjoint exploratory calibration protocol.

## E04.5 Rhythm-quality evaluation

For every rhythm-quality-duration cell, report:

- Independent subjects.
- Excerpts.
- Total duration.
- Eligible excerpts.
- Valid predictions.
- Rejected predictions.
- AF sensitivity.
- Non-AF specificity.
- False AF calls.
- Missed AF calls.
- Abstention rate.
- Confidence distribution.

Analyze competing arrhythmias individually.

In particular:

- Variable-conduction flutter.
- Supraventricular ectopy.
- Ventricular ectopy.
- Other available rhythms.

Do not represent an excerpt-derived false-positive window rate as a continuous-recording false episode rate.

## E04.6 Physiological feature observability

For every eligible excerpt calculate:

- Beat detection coverage.
- Mean ventricular rate.
- RR count.
- CVRR.
- RMSSD.
- SDNN.
- VCC.
- Sample entropy where adequate support exists.
- Lead agreement.

Report technical validity and measured feature distributions separately.

If reference QRS timestamps are unavailable, distinguish:

- Agreement between detectors.
- Agreement between leads.
- Physiological plausibility.
- Actual beat detection accuracy, which cannot be directly established without appropriate references.

Two beat detectors agreeing does not necessarily establish correctness.

## E04.7 Artifact versus rhythm

Investigate whether observed RR irregularity differences between AF and non-AF narrow across expert quality strata.

Use patient-aware analysis.

Do not claim that unpaired cross-sectional comparisons establish the causal effect of noise on an individual ECG.

Examine whether apparent poor discrimination is driven by:

- Genuine ventricular irregularity.
- Erroneous QRS detections.
- Missed beats.
- Duplicate beats.
- Very short excerpts.
- Competing ectopy.
- Low feature observability.

Report complete distributions rather than only medians.

## E04.8 Controlled connection to continuous cohorts

Use the continuous reference cohorts to investigate which episode architectures are vulnerable to empirically motivated sensing limitations.

Construct a separate simulation experiment in which reference timelines are subjected to plausible duration- and quality-dependent detection errors.

Do not describe these simulated effects as measured real-world longitudinal density errors.

Where applicable, evaluate:

- Omission of brief AF runs.
- Misclassification of brief non-AF gaps.
- Loss of AF onset information.
- Loss of AF termination information.
- Unobserved intervals.

Preserve source timelines and original labels.

Do not inject entire noisy clinical ECG excerpts into clean recordings and describe them as pure artifact. Authentic excerpts contain underlying cardiac activity.

Compare the resulting:

- Burden error.
- Density error.
- Aggregation error.
- Longest-episode error.
- Brief-run count error.
- E01 temporal neighbourhood error.

This establishes whether the physiological patterns studied in the continuous cohorts could be selectively obscured by monitoring limitations.

---

# PART C — Integrated cross-cohort analysis

## C01. Harmonized temporal feature table

Create a common patient-recording feature table containing:

- Dataset.
- Patient ID.
- Recording ID.
- Observation horizon.
- Known rhythm duration.
- AF exposure.
- AF burden.
- Continuous density.
- Binned density.
- Aggregation.
- Longest episode.
- Longest-episode share.
- Episode counts by duration class.
- Interepisode gap summaries.
- Rapid recurrence.
- Temporal neighbourhood statistics.
- Available ventricular features.
- Available clinical variables.

Keep source-specific annotation provenance.

Do not pool the Noise Holter excerpts into this longitudinal table.

## C02. Common-horizon comparisons

Compare IRIDIA, SHDB and LTAFDB at:

- 6 hours.
- 12 hours.
- 24 hours, where eligible.

Additionally report full native-recording summaries.

Distinguish:

- Differences caused by observation horizon.
- Differences caused by duration thresholds.
- Differences associated with cohort selection.
- Differences in annotation resolution.
- Genuine observed temporal differences.

Do not assume cohort membership is a biological variable.

## C03. Density's incremental information

For each scientific outcome, compare density with appropriate conventional temporal measurements.

The central test is:

**Does density convey information not already represented by cumulative AF exposure, longest episode, frequency and recent episode history?**

A correlation between density and its own mathematical components is not sufficient.

For example:

- Density versus longest-episode share is primarily a mathematical characterization.
- Density versus independently measured LAD is a clinical association.
- Historical density versus subsequent episode duration is a temporally ordered rhythm association.
- Historical density versus subsequent ventricular irregularity is a physiological association.
- Density versus future stroke cannot be established from these datasets.

Report these evidence classes separately.

## C04. No unrestricted hypothesis searching

Prespecify:

- Primary outcome for each experiment.
- Primary exposure.
- Primary cohort.
- Primary time horizon.
- Primary duration threshold.
- Statistical model.
- Adjustment variables.
- Minimum analytical support.
- Sensitivity analyses.

Freeze the plan before examining outcome associations.

Where feasibility requires changing a planned definition, document the reason and ensure the decision is independent of the observed effect.

Report all negative and inconclusive results.

Do not search arbitrary episode thresholds until a significant association appears.

---

# PART D — Statistical design and validity

## D01. Independent patient support

For every experiment report:

- Total observations.
- Independent patients.
- Patients with repeated observations.
- Patients supporting the relevant within-patient contrast.
- Largest patient contribution.
- Top-five patient contributions.
- Top-ten patient contributions.

Where support is sparse, prioritize descriptive distributions and uncertainty rather than unstable regression.

As a planning gate, fewer than 10 informative independent patients should generally be treated as a feasibility or case-series analysis. Larger sample sizes do not automatically establish adequate statistical power.

## D02. Patient-level inference

Use patient-aware inference appropriate to the estimand:

- Patient-level bootstrap.
- Patient-stratified models.
- Fixed-effects models.
- Appropriate mixed-effects models.
- Cluster-aware survival analysis.
- Patient-grouped validation, where prediction is evaluated.

Resample entire patient histories rather than isolated episodes.

Preserve temporal dependence and overlapping historical windows.

Use at least 2,000 patient-bootstrap replicates where appropriate.

Report failed replicates and numerical convergence.

## D03. Cohort heterogeneity

Report each cohort independently before any pooled analysis.

Pooling is permitted only where:

- Exposure definitions match.
- Outcome definitions match.
- Eligibility is comparable.
- Relevant covariates have overlapping support.
- Annotation differences are addressed.
- The pooled estimand is clinically interpretable.

Do not claim that similar coefficient directions establish independent clinical replication.

## D04. Multiplicity

Declare three principal neighbourhood contrasts in E01 and apply an appropriate family-wise or false-discovery procedure.

For E02, specify one primary physiological trajectory outcome and one separately identified rhythm-duration outcome.

For E03, specify one primary structural outcome.

E04 should primarily report stratified measurement performance with uncertainty.

Clearly label all secondary, sensitivity and exploratory analyses.

## D05. Negative controls

Where supported, use negative-control analyses such as:

- Random index times matched on observation opportunity.
- Temporally displaced AF boundaries.
- Alternative duration assignments under valid conditional nulls.
- Remote within-patient intervals.
- Patient-label permutation for clinical associations, respecting the analytical structure.

Negative controls must preserve the data features needed for a fair comparison.

Do not use a null that destroys all AF clustering and then interpret rejection as evidence of a particular physiological mechanism.

---

# PART E — Required figures

Generate figures directly from verified machine-readable results.

## Figure 1 — Cohort episode-duration architecture

- Logarithmic AF duration distributions.
- Interepisode gap distributions.
- Independent patient counts.
- Episode-weighted versus patient-weighted distributions.

## Figure 2 — Temporal neighbourhood of prolonged AF

- Brief-run onset rates before index onset.
- Brief-run onset rates after index termination.
- Remote comparison rates.
- Physical-time and non-AF-risk-time denominators.
- Patient-level confidence intervals.

## Figure 3 — Short-run concentration and density

- Burden-density scatterplots.
- Colour or marker encoding longest-episode share.
- Brief-run neighbourhood rates.
- Cohort-specific panels.

## Figure 4 — Conditional null comparisons

- Observed neighbourhood statistic.
- Patient-level null distribution.
- Cohort-level deviations.
- Representative actual reference timelines.

## Figure 5 — Historical density and subsequent physiology

- Fixed elapsed-time CVRR trajectories.
- Historical density contrasts.
- Mean ventricular rate trajectories.
- Independent patient support at every landmark.

## Figure 6 — Historical density and subsequent episode duration

- Censoring-aware survival curves.
- Cohort-specific comparisons.
- Within-patient association estimates.
- Conditional support.

## Figure 7 — SHDB clinical characterization

- LAD versus temporal architecture.
- Echo timing.
- Clinical subtype.
- Prior ablation.
- Body-size-adjusted estimates.
- Influence sensitivity.

## Figure 8 — Authentic observability

- Rhythm-by-quality performance matrix.
- Valid feature coverage.
- Ventricular irregularity distributions.
- AF sensitivity and false AF calls.
- Excerpt-duration support.

## Figure 9 — Integrated evidence summary

Clearly separate:

- Reference rhythm evidence.
- Observed physiological evidence.
- Clinical associations.
- Technical observability findings.
- Unresolved hypotheses.

Do not use illustrative plots as substitutes for experimental data.

---

# PART F — Implementation structure

Create:

`experiments/af_density_temporal_phenotypes_v7/`

Suggested structure:

- `configs/analysis.yaml`
- `configs/statistical_analysis_plan.yaml`
- `src/cohort_adapters.py`
- `src/episode_classes.py`
- `src/neighbourhood_analysis.py`
- `src/risk_time.py`
- `src/conditional_randomization.py`
- `src/historical_exposure.py`
- `src/ventricular_trajectories.py`
- `src/clinical_phenotypes.py`
- `src/noise_observability.py`
- `src/statistics.py`
- `tests/`
- `scripts/`
- `results/`
- `figures/`
- `reports/`

Execution sequence:

1. `01_validate_dependencies.py`
2. `02_build_duration_class_inventory.py`
3. `03_build_index_episode_neighbourhoods.py`
4. `04_calculate_neighbourhood_risk_time.py`
5. `05_evaluate_neighbourhood_associations.py`
6. `06_run_conditional_null_experiments.py`
7. `07_link_neighbourhoods_to_density.py`
8. `08_build_historical_exposure_manifest.py`
9. `09_extract_fixed_elapsed_rr_features.py`
10. `10_model_ventricular_trajectories.py`
11. `11_model_subsequent_episode_duration.py`
12. `12_build_shdb_clinical_phenotypes.py`
13. `13_analyze_shdb_structural_associations.py`
14. `14_analyze_noise_holter_observability.py`
15. `15_evaluate_temporal_measurement_sensitivity.py`
16. `16_generate_integrated_figures.py`
17. `17_generate_master_report.py`

Use existing validated code rather than duplicating the entire project.

Create independent tests specifically for new estimands and new interval constructions.

---

# PART G — Required machine-readable outputs

Create:

## E01 — Temporal neighbourhoods

- `E01_duration_class_inventory.csv`
- `E01_patient_feasibility.csv`
- `E01_index_episode_manifest.parquet`
- `E01_neighbourhood_intervals.parquet`
- `E01_neighbourhood_event_assignments.parquet`
- `E01_risk_time_summary.csv`
- `E01_remote_comparison_manifest.parquet`
- `E01_neighbourhood_rate_ratios.csv`
- `E01_patient_level_effects.csv`
- `E01_conditional_null_results.parquet`
- `E01_density_architecture_results.csv`
- `E01_sensitivity_analysis.csv`

## E02 — Historical exposure and physiology

- `E02_historical_exposure_manifest.parquet`
- `E02_landmark_feasibility.csv`
- `E02_conditional_support.csv`
- `E02_fixed_elapsed_rr_features.parquet`
- `E02_ventricular_trajectory_models.csv`
- `E02_within_patient_effects.csv`
- `E02_between_patient_effects.csv`
- `E02_survival_risk_sets.csv`
- `E02_subsequent_episode_models.csv`
- `E02_nested_model_comparisons.csv`

## E03 — SHDB clinical phenotype

- `E03_clinical_population_manifest.parquet`
- `E03_temporal_architecture_features.parquet`
- `E03_low_burden_phenotypes.csv`
- `E03_clinical_characteristics.csv`
- `E03_echo_timing_inventory.csv`
- `E03_structural_models.csv`
- `E03_same_population_sensitivity.csv`
- `E03_patient_influence.csv`

## E04 — ECG observability

- `E04_noise_holter_inventory.csv`
- `E04_model_input_eligibility.csv`
- `E04_rhythm_quality_predictions.parquet`
- `E04_rhythm_quality_performance.csv`
- `E04_physiological_feature_observability.csv`
- `E04_ventricular_feature_distributions.parquet`
- `E04_temporal_measurement_sensitivity.csv`

## Integrated outputs

- `V7_cross_cohort_feature_manifest.parquet`
- `V7_primary_results.csv`
- `V7_all_statistical_results.csv`
- `V7_negative_results.csv`
- `V7_feasibility_limitations.csv`
- `V7_literature_comparison.csv`

Every result must include provenance, cohort, analytical population, observation horizon, unit of analysis and applicable patient denominator.

---

# PART H — Validation and execution gates

Before interpreting results, establish:

### Gate 1 — Reference timeline validity

All relevant AF runs, non-AF gaps and acquisition boundaries are correctly represented.

### Gate 2 — Neighbourhood construction

No unintended overlap, duplicate event assignment, denominator error or inclusion of index-episode duration.

### Gate 3 — Risk-time validity

AF onset counts and eligible non-AF observation time are correctly matched.

### Gate 4 — Historical exposure validity

No future rhythm information enters historical predictors.

### Gate 5 — Physiological landmark validity

Only eligible AF intervals contribute to fixed elapsed-time measurements.

### Gate 6 — Statistical support

Adequate independent patient counts and conditional exposure variation are established.

### Gate 7 — Clinical linkage

SHDB clinical variables and echo dates are correctly linked to patients and recordings.

### Gate 8 — Authentic observability

Noise Holter quality labels, input durations and feature availability are correctly represented.

### Gate 9 — Conditional null validity

All simulated histories preserve their explicitly required constraints.

### Gate 10 — Scientific interpretation

Conclusions do not exceed what the data and estimands support.

Use PASS, PARTIAL, FAIL or BLOCKED.

If an experiment fails its necessary feasibility gate, report the failure rather than substitute an unrelated hypothesis.

---

# PART I — Final master report

Create:

`reports/af_density_temporal_phenotypes_v7_master_report.md`

Required structure:

1. Scientific rationale.
2. Published literature comparators.
3. Verified dataset characteristics.
4. Prespecified experimental definitions.
5. Patient-level feasibility.
6. E01 temporal neighbourhood findings.
7. E02 historical concentration and physiological evolution.
8. E03 clinical phenotype findings.
9. E04 authentic observability findings.
10. Cross-cohort integration.
11. Negative and inconclusive results.
12. Statistical limitations.
13. Measurement limitations.
14. Exact artifact and code paths.

Update:

`reports/AF_DENSITY_MASTER_KNOWLEDGE_BASE.md`

Add all V7 results, including unsuccessful experiments.

Do not overwrite earlier findings without documenting their relationship to the new analyses.

---

# FINAL INSTRUCTION

The objective is to learn something scientifically meaningful about AF temporal organization using the actual populations represented in our datasets.

Do not discard a research question merely because another paper investigated related AF physiology.

Compare:

- Patient population.
- Acquisition modality.
- Individual episode duration.
- Interepisode gap duration.
- Observation horizon.
- Clinical treatment exposure.
- Physiological measurement.
- Statistical estimand.

Equally, do not claim novelty merely because the dataset names or numerical thresholds differ.

The primary scientific contribution must come from an informative distinction in the question being investigated.

Do not select a manuscript thesis based on the smallest p-value.

Return all valid findings, all failures and the actual patient-level support so that we can evaluate the scientific contribution ourselves.

# **How these experiments fit together**

## **Temporal organization of atrial fibrillation**

Burden · Density · Episode duration · Interepisode intervals

**E01 — Short-run neighbourhoods**

LTAFDB + SHDB

Determine how brief episodes relate temporally to prolonged AF.

**E02 — Electrical evolution**

IRIDIA-AF

Test whether previous concentration relates to subsequent AF behaviour.

**E03 — Clinical phenotype**

SHDB-AF

Connect rhythm organization to clinical and echocardiographic measurements.

**E04 — Signal observability**

Noise Holter

Establish which physiological distinctions survive authentic measurement limitations.

Schematic research relationships, not observed experimental results.

**Integrated scientific assessment**

Does AF density characterize something meaningful beyond cumulative exposure?

## **What would constitute a genuinely interesting finding?**

The results should be evaluated against concrete scientific possibilities rather than significance alone.

| Potential finding                                                                                           | Scientific significance                                                                |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Brief AF onset rates are elevated following prolonged AF, even after accounting for available non-AF time   | Evidence of duration-specific temporal organization                                    |
| Brief runs preferentially precede rather than follow prolonged AF                                           | Suggests asymmetric temporal behaviour deserving physiological investigation           |
| Patients with similar burden and density have substantially different short-run neighbourhoods              | Demonstrates that concentration does not uniquely characterize episode architecture    |
| Historical density relates to subsequent RR variability trajectories after accounting for recent AF history | Supports a relationship between prior organization and subsequent electrical behaviour |
| Historical density adds no information beyond the immediately preceding episode and gap                     | Suggests simpler historical measurements may capture the relevant dynamics             |
| SHDB temporal patterns associate differently with LAD or prior treatment                                    | Provides exploratory clinical characterization                                         |
| Authentic artifact preferentially obscures brief or relatively regular AF                                   | Identifies a specific measurement limitation relevant to longitudinal characterization |

## **My preferred execution order**

1

**E01 — Temporal neighbourhoods**

Strongest immediate use of the SHDB and LTAFDB short-run inventories. Does not depend on uncertain waveform-derived atrial biomarkers.

2

**E02 — Historical exposure and subsequent physiology**

Potentially more biologically informative, particularly in IRIDIA, but requires genuine within-patient exposure variation and adequate prolonged-episode support.

3

**E03 — SHDB clinical characterization**

Adds independent clinical context to the temporal findings without relying exclusively on ECG-derived surrogates.

4

**E04 — Authentic measurement limitations**

Determines whether the temporal and physiological distinctions identified in the other experiments can actually be observed under realistic sensing conditions.

The experiment I am most interested in is E01 because it exploits something distinctive in our data: **the coexistence of AF runs lasting seconds, minutes and hours within continuous rhythm histories.**

We have already measured how much AF patients experience, how concentrated it is, how long their episodes last and how frequently they recur. What we have not adequately established is how episodes of different durations relate to one another within the same patient's history.

That is the specific scientific question V7 should address.I would add a predictive component to V7, but I would formulate it around the transition from brief, recurrent AF to sustained AF, rather than train another model to distinguish AF from sinus rhythm.
We already have extensive AF detection benchmarks. Another detector comparison would not substantially advance our understanding of AF density.
Instead, the predictive experiments should establish whether historical AF organization and ECG representations contain information about what the arrhythmia does next.
1. The closest published prediction problems
I found an especially relevant paper that we should incorporate into the experimental design.
Forecasting Imminent Atrial Fibrillation in Long-Term Electrocardiogram Recordings (2023)
[Journal of Electrocardiology — full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10841237/?utm_source=chatgpt.com)
The authors investigated three different forecasting inputs:
ECG recorded during sinus rhythm before AF.
ECG recorded during shorter AF episodes preceding sustained AF.
ECG containing both preceding rhythm states.
They specifically investigated whether AF episodes lasting five minutes or less provide information about the subsequent occurrence of longer AF episodes. Their reported forecaster achieved 56% sensitivity at 25% false-positive rate with 15 minutes of lead time.
pmc.ncbi.nlm.nih.gov+1
This is probably the closest published prediction problem to the temporal-neighbourhood investigation we just designed.
However, I would make our study more explicit about the contribution of historical AF density, duration classes, recurrence patterns and patient-specific history.
Ngo et al. (2026) — Dynamic prediction using IRIDIA-AF
[Dynamic prediction of paroxysmal atrial fibrillation onset using longitudinal sample entropy in joint models](https://doi.org/10.1186/s12874-026-02773-9?utm_source=chatgpt.com)
This study provides an important statistical precedent using our exact dataset.
Published approach
Result
Original analytical recordings
137
Joint-model time-dependent AUROC
0.644
RBF-SVM AUROC
0.820
Population remaining for conventional ML comparison
72
The authors observed considerable uncertainty in the conventional machine-learning results and convergence problems when increasing model complexity. They excluded subsequent AF episodes within individual recordings from their main investigation.
PMC
That is particularly relevant to our design: the higher SVM AUROC does not automatically mean it is superior because the joint and conventional models were evaluated under different eligibility conditions.
Our benchmark must compare models using the same patients, prediction landmarks, information windows and outcomes.
Published AF termination studies
A 2024 systematic review found that most surface-ECG termination-prediction studies examine signals close to AF termination, rather than predicting episode behaviour substantially earlier.
pmc.ncbi.nlm.nih.gov
That motivates a fixed-landmark prediction design in which the model cannot access ECG immediately preceding a future termination unless that ECG is already available at the prediction time.
2. The prediction problems I would formulate
I would establish three principal predictive tasks and one supporting classification task.
Task
Prediction target
Primary cohort
P01
Will sustained AF begin within the next 15–60 minutes?
SHDB, LTAFDB
P02
Will a newly observed AF episode persist for at least 30 minutes?
IRIDIA, SHDB
P03
What is the remaining duration of an ongoing AF episode?
IRIDIA
P04
Is an ECG excerpt AF, non-AF or technically indeterminate?
Noise Holter
P01 and P02 are the most important because they connect the actual temporal architecture of our cohorts with clinically interpretable prediction targets.
I would additionally investigate future AF burden and density as secondary longitudinal outcomes, but would not make them the primary prediction tasks until we establish adequate patient-level support.
P01 — Predicting the transition to sustained AF
Forecasting sustained AF from preceding rhythm history
Only ECG and rhythm information available before the prediction time may enter the model.
Primary candidate definition:
\[ P(\text{new AF episode lasting }\geq30\text{ min begins within }30\text{ min}\mid\mathcal H_t) \]
Here, \(\mathcal H_t\) contains only information available before prediction time.
A positive outcome requires a new episode that begins within the prediction horizon and subsequently reaches the specified duration.
We can compare against six-minute and 60-minute sustained-AF definitions.
Importantly, prediction landmarks should be selected from the relevant risk population, not exclusively from windows immediately preceding known AF episodes.
P02 — Early prediction of episode persistence
This is particularly well suited to IRIDIA's prolonged PAF episodes.
Predicting episode persistence at an early AF landmark
The primary estimand would be:
\[ P(T_{\mathrm{AF}}\geq30\text{ min}\mid T_{\mathrm{AF}}>60\text{ s},X_{60}) \]
This is explicitly conditional on the episode remaining active at the 60-second landmark.
We could then compare 10-second, 30-second, two-minute and five-minute landmarks.
A later landmark provides more ECG information but also selects a different population. Its apparent performance improvement cannot automatically be attributed to better representations.
P03 — Dynamic prediction of AF termination
Instead of imposing a binary short/long threshold, estimate the conditional probability of AF termination during a future interval.
\[ P(T_{\mathrm{AF}}\leq t+h\mid T_{\mathrm{AF}}>t,\mathcal H_t) \]
This is naturally suited to survival analysis, landmark models and discrete-time hazard models.
I would include remaining duration because it provides a clinically interpretable probabilistic outcome and extends our previous AF-entrenchment experiments.
P04 — Authentic rhythm classification and observability
Noise Holter would provide a separate measurement benchmark:
Current rhythm identification.
Distinguishing AF from flutter and ectopy.
Performance according to expert-labelled signal quality.
Selective prediction and abstention.
Reliability of extracted ECG representations.
It cannot independently validate P01–P03 because it does not contain the necessary continuous future rhythm histories.
3. Which statistical and machine-learning models to include
I would build a hierarchical benchmark, starting with statistically interpretable models and progressing to pretrained waveform representations.
The scientific objective is to determine where the predictive information comes from, not simply identify the architecture with the highest AUROC.
Classical and statistical models
Model
Priority
Purpose
Empirical prevalence baseline
Essential
Establish minimum predictive benchmark
Weibull survival model
Essential
Account for elapsed AF duration
Cox proportional hazards
Essential
Censoring-aware termination analysis
Elastic-net logistic regression
Essential
Strong interpretable classification baseline
GAM
Essential
Nonlinear relationships between burden, density, duration and risk
Explainable Boosting Machine (EBM)
High
Interpretable nonlinear feature effects and interactions
Linear SVM
Include
High-dimensional regularized representation baseline
RBF-SVM
Include
Nonlinear comparator directly relevant to Ngo et al.
Decision tree
Include
Simple clinical decision-rule benchmark
GBDT
Essential
Strong nonlinear tabular benchmark
Random survival forest
Secondary
Nonlinear censoring-aware prediction
Joint longitudinal-survival model
High, if feasible
Dynamic physiological risk estimation
Alternating Hawkes model
High, where supported
History-dependent rhythm-transition benchmark
I am interpreting your reference to EGM as potentially meaning EBM, Explainable Boosting Machine. If you meant intracardiac electrograms, we do not have intracardiac EGM recordings in these four datasets.
I particularly want GAMs and EBMs because they could expose nonlinear relationships that our earlier linear regression analyses missed.
For instance, the relationship between preceding density and subsequent episode persistence might not be adequately described by a single linear coefficient.
Candidate relationships between historical density and predicted AF persistence
Linear
Threshold effect
Nonmonotonic effect
Saturation effect
Hypothetical relationships for model comparison, not observed results.
An important qualification: flexible models can also manufacture apparent nonlinearities in sparse regions of the data. Every GAM or EBM effect plot must therefore include the empirical feature distribution and uncertainty.
4. Which deep-learning architectures to include
I would select four trainable waveform or sequence architectures.
Architecture
Input
Purpose
Existing Raw Mask ResNet
ECG
Established internal baseline
xresnet1d18
ECG
Compact supervised waveform benchmark
GRU-D or compact masked GRU
Sequential physiological features
Historical trajectory modelling
Compact temporal convolutional network
RR or ECG embeddings over time
Temporal representation benchmark
The xresnet1d architecture family has established ECG benchmarking precedent in the PTB-XL literature.
GitHub
I would start with xresnet1d18 rather than xresnet1d101 because our independent patient count is limited. Hundreds of thousands of windows do not constitute hundreds of thousands of independent examples.
For the sequential models, we need the ability to distinguish between:
Recent AF exposure.
Remote AF exposure.
Current physiological state.
Missing observations.
Time since the previous rhythm transition.
This is where a recurrent or temporal convolutional architecture has a meaningful advantage over a classifier using only a single short ECG strip.
I would not initially introduce another large Transformer trained from scratch.
5. Foundation-model selection
You have enough pretrained checkpoints to conduct a substantial representation-learning investigation. However, I would not evaluate all of them indiscriminately.
Tier 1 — Primary foundation-model benchmarks
CSFM Tiny
Cardiac Sensing Foundation Model — 2026Most relevant primary benchmark because its published methodology explicitly addresses heterogeneous cardiac sensing configurations, including single-lead and reduced-lead inputs.
Checkpoint: CSFM_tiny.pth
[Nature Machine Intelligence paper](https://doi.org/10.1038/s42256-026-01180-5?utm_source=chatgpt.com)
CLEF Medium
Clinically guided contrastive ECG representationParticularly useful for investigating whether clinically informed representations transfer to limited-lead AF forecasting.
Checkpoint: clef_medium.ckpt
[Paper and methodology](https://arxiv.org/abs/2512.02180?utm_source=chatgpt.com)
ECGFounder 1-lead
Large-scale supervised ECG pretrainingProvides a different pretraining paradigm from the contrastive and masked-learning models. Its published model supports single-lead ECG.
Checkpoint: 1_lead_ECGFounder.pth
[Published paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12327759/?utm_source=chatgpt.com)
ECG-FM KED
Hybrid self-supervised ECG TransformerUseful as a standard 12-lead-derived representation with random lead masking, provided the actual checkpoint and reduced-lead adaptation are verified.
Checkpoint: best_valid_all_increase_with_augment_epoch_3.pt
[Published paper](https://doi.org/10.1093/jamiaopen/ooaf122?utm_source=chatgpt.com)
CSFM, CLEF and ECGFounder have explicit published support for reduced-lead or single-lead ECG applications. ECG-FM has a lead-masking pretraining strategy, but its applicability to our modified Holter leads still requires evaluation.
doi.org+3
None of these publications establishes that their representations can forecast the persistence of individual AF episodes in our cohorts. That is what we would investigate.
Tier 2 — Internal architectures and controlled comparisons
Model
Decision
Rationale
NEFCore Full
Include conditionally
Tests whether geometry-aware ECG representations transfer to ambulatory AF
NEFCore AE
Include alongside Full
Controls for the additional geometric components
F2 SetOperator
Include conditionally
Tests learned measurement-operator representations
F0 SetOperator
Include alongside F2
Internal controlled comparison
DINO-SR
Include as compact SSL comparator
Self-distillation representation
JEPA
Include as compact SSL comparator
Predictive latent representation
Our earlier SetOperator experiments suggested that RR information dominated some of the morphological branches. That makes it especially important to test whether pretrained representations contribute information after conventional RR features have already been included.
Tier 3 — Defer initially
Model
Reason
CLEF Large
High computational cost; first establish whether Medium transfers
CLEF Small
Useful later for efficiency comparisons
CSFM Base
Scale-up comparison after Tiny
Data2Vec
Secondary pretraining-strategy ablation
HuBERT++
Secondary pretraining-strategy ablation
CPC
Secondary pretraining-strategy ablation
ECGFounder 12-lead
Less directly aligned with available input modality
EchoNext Minimodel
Structural prediction comparator only
EchoNext G1 SetOperator
Structural characterization only, with capacity control
ECG-FM fine-tuned archive
Requires verification of fine-tuning data and labels
WearECG-Reconstruction
Reconstruction objective; no immediate forecasting justification
Semi-Seg ECG
Segmentation objective; not a primary forecasting comparator
EchoNext could be used in SHDB as an exploratory structural representation benchmark, but I would not use model-generated structural disease scores as clinical ground truth.
6. The most important experimental design: information-source ablation
This should be the central benchmark table in the eventual research paper.
Incremental information framework
F0
Clinical and elapsed-time baseline
Age, sex, elapsed AF duration, where available
F1
Recent rhythm history
Previous episode duration, time since termination, recent burden
F2
Complete historical architecture
Episode count, durations, gaps, cumulative exposure
F3
Historical AF density
Add published density and aggregation measures
F4
Conventional ECG physiology
Ventricular rate, CVRR, RMSSD, VCC, sample entropy
F5
Pretrained ECG representation
Frozen foundation-model embeddings
F6
Combined multimodal representation
Historical architecture + density + physiological features + ECG embeddings
The essential comparisons are:
Comparison
Question answered
F2 versus F3
Does density contribute information beyond conventional history?
F3 versus F4
Does measured ECG physiology contribute beyond rhythm organization?
F3 versus F3 + FM
Do pretrained waveform representations add information beyond rhythm history?
F4 versus F6
Do foundation models add information beyond conventional physiological measurements?
F6 versus F6 without density
Does density remain useful after accounting for the complete representation?
Pretrained versus random initialization
Is the advantage specifically attributable to pretraining?
This is much more informative than simply comparing CSFM against ECG-FM against a ResNet.
A foundation model that achieves high AUROC but adds nothing beyond previous episode duration and RR variability would not establish that it learned a distinct representation of AF behaviour.
7. Additional coding-agent directive: V7-P predictive modelling
Append the following instructions to the previous V7 experimental specification.
AF-DENSITY V7-P — Predictive Modelling, Dynamic Survival Analysis and Pretrained ECG Representations
1. Scientific objective
Extend the V7 temporal-phenotype investigation with a controlled prediction framework.
The central objective is to determine whether historical AF temporal concentration and pretrained ECG representations add information about subsequent arrhythmia behaviour beyond conventional episode history and ventricular physiology.
Do not train models simply to maximize AUROC.
Do not replace the original V7 descriptive and clinical investigations with machine-learning benchmarks.
Prediction experiments must have a clearly defined clinical population, prediction time, available information set, outcome horizon and censoring mechanism.
Use the existing four datasets and available model checkpoints.
All historical IRIDIA partition-access restrictions remain applicable. Previous exposure means that the historical test partition cannot be presented as untouched confirmation.
2. Literature-informed formulation
Use these as primary methodological comparators:
Forecasting Imminent Atrial Fibrillation in Long-Term Electrocardiogram Recordings, 2023, PMC10841237.
Ngo et al., 2026, DOI 10.1186/s12874-026-02773-9.
Henriksson et al., 2021, DOI 10.1109/TBME.2020.2995563.
Predicting Spontaneous Termination of Atrial Fibrillation Based on Analysis of Standard Electrocardiograms: A Systematic Review, 2024, PMC11503732.
Van Vreeswijk et al., 2026, DOI 10.1093/europace/euaf300.
Extract the original target definitions, timing assumptions, eligibility criteria, features, validation protocols and performance metrics.
Do not directly compare published AUROC values with our results when the target population or prediction task differs.
Our principal distinction is the combination of repeated AF histories, historical temporal concentration, fixed prediction landmarks and conventional versus pretrained ECG representations.
P01 — Forecasting the onset of sustained AF
P01.1 Population
Primary: SHDB-AF and LTAFDB patients with known AF histories and reference rhythm annotations.
Secondary: IRIDIA-AF.
Construct prediction landmarks from all eligible known non-AF risk periods, not only intervals preceding observed AF onset.
Use a fixed acquisition-anchored landmark grid, initially every five minutes.
Evaluate one-minute and 15-minute grids as sensitivity analyses.
Require adequate preceding rhythm observation.
Do not infer confirmed sinus rhythm from unspecified non-AF.
P01.2 Target
Primary candidate target:
A new AF episode lasting at least 30 minutes begins within 30 minutes after prediction time.
Secondary sustained-duration definitions:
At least 6 minutes.
At least 60 minutes.
Secondary forecasting horizons:
5 minutes.
15 minutes.
60 minutes.
3 hours, where supported.
For every target combination, calculate independent patient counts, positive events, negative landmarks, censoring and overlap.
Select the primary executable target based on eligibility and event support before inspecting model performance.
P01.3 Outcome ascertainment
A positive outcome requires an observed AF onset within the forecast horizon and sufficient subsequent observation to establish the required sustained duration.
A negative outcome requires sufficient follow-up to establish that no qualifying AF episode occurred.
If a candidate AF episode begins near the recording end and its final duration is unknown, handle the target as indeterminate or use an appropriate censoring-aware formulation.
For simple binary classification, a fully observed follow-up of forecasting horizon plus sustained-duration threshold is a conservative eligibility rule.
Do not treat indeterminate cases as negative.
P01.4 Secondary recurrent-episode formulation
Among patients with an observed completed brief AF run, estimate whether a subsequent sustained AF episode begins within the specified horizon.
Use completed brief runs of:
Less than 30 seconds.
30 seconds to less than 2 minutes.
2 to less than 6 minutes.
Set prediction time at observed termination of the index brief run.
Do not use its final duration before termination has occurred.
This is a different conditional population from the general non-AF risk population.
Report both tasks separately.
P02 — Early prediction of AF episode persistence
P02.1 Population
Primary: IRIDIA-AF.
Secondary: SHDB-AF and eligible LTAFDB episodes.
Include episodes with observed onset and known uninterrupted observation through the prediction landmark.
Primary landmark: 60 seconds after AF onset.
Secondary landmarks:
10 seconds.
30 seconds.
2 minutes.
5 minutes.
Use model-compatible ECG lengths.
P02.2 Target
Primary: total AF episode duration reaches 30 minutes.
Secondary:
Reaches 6 minutes.
Reaches 60 minutes.
Reaches 6 hours, if adequately supported.
For each landmark, estimate risk conditional on the episode being active at that landmark.
Exclude target thresholds already reached at the prediction time.
An observed termination before the target duration provides a negative outcome.
Observation of the episode reaching the target provides a positive outcome, even if final termination is censored.
Episodes censored before the threshold without observed termination have indeterminate binary labels.
P02.3 Alternative duration classification
Where support permits, perform ordinal duration classification:
Less than 6 minutes.
6 to less than 30 minutes.
30 to less than 60 minutes.
At least 60 minutes.
Adapt the classes to the prediction landmark so that already impossible classes are excluded.
Do not treat arbitrary duration classes as established clinical disease subtypes.
Use cumulative ordinal probabilities or a coherent survival distribution where feasible.
P03 — Dynamic AF termination and residual-duration prediction
P03.1 Primary formulation
At each eligible elapsed AF landmark, estimate:
P(termination within the next h minutes | episode still ongoing).
Candidate landmarks:
1 minute.
5 minutes.
15 minutes.
30 minutes.
60 minutes.
Candidate horizons:
5 minutes.
15 minutes.
30 minutes.
60 minutes.
Primary candidate: termination within the subsequent 15 minutes at the five-minute landmark.
Use only episodes with observed onset for elapsed-duration prediction.
P03.2 Survival formulation
Estimate the complete conditional survival distribution of remaining AF duration.
Include right-censoring and time-varying physiological observations.
Do not use the observed censored duration as if it were complete episode duration.
Do not assume observed termination was spontaneous unless that can be established from the source data.
P03.3 Required baseline
Construct a training-population-only elapsed-duration conditional survival baseline.
Estimate:
S(t+h)/S(t).
This is essential because our previous experiments demonstrated strong dependence of termination probability on elapsed AF duration.
An ECG model must outperform an appropriate episode-age baseline, not simply a prevalence classifier.
P03.4 Secondary predictions
Where adequate support exists, estimate:
Median remaining AF duration.
Restricted expected remaining duration.
Probability of remaining in AF at each future horizon.
Distribution of future episode duration.
Avoid unrestricted expected duration if the survival tail is not identifiable.
P04 — Authentic rhythm classification and uncertainty
Use Noise Holter for the supporting sensing task.
Evaluate:
AF versus non-AF.
AF versus atrial flutter.
AF versus sinus rhythm with ectopy.
Multiclass rhythm classification where supported by source labels.
Selective classification with abstention.
Retain expert Clean, N1 and N2 labels.
Do not create an artificial ground-truth indeterminate-rhythm class solely from signal quality. Evaluate model abstention separately from the reference rhythm label.
Use existing AF detectors as frozen zero-shot or previously trained comparators.
If adapting a pretrained encoder or classifier to Noise Holter, use patient-disjoint development and evaluation folds.
Do not compare adapted and zero-shot models without explicitly identifying their training regimes.
Noise Holter cannot validate longitudinal P01–P03 outcomes.
P05 — Optional future AF exposure prediction
Only execute if sufficient independent patients have complete future observation.
At a fixed prediction time, estimate:
Future AF burden over 1, 3 and 6 hours.
Probability of any AF.
Probability of continuous all-AF.
Future longest observed AF episode.
Future AF density conditional on defined density.
Do not directly regress an undefined density value.
Use a multistate or hurdle formulation separating:
No AF.
Mixed AF/non-AF.
Entirely AF.
Predict density only for the relevant mixed-rhythm state.
Report conditional density performance and state-classification performance together.
Do not artificially assign no-AF or all-AF observations density zero.
This task is secondary and must not become a new unrestricted model-development campaign.
3. Feature groups and information availability
Construct one immutable, versioned prediction manifest for every task.
Every row must contain:
Dataset.
Patient ID.
Recording ID.
Prediction landmark.
Historical observation boundaries.
Current rhythm.
Elapsed AF duration, if applicable.
Outcome definition.
Outcome event time.
Censoring time.
Feature availability.
ECG availability.
Reference-label provenance.
F0 — Minimal baseline
Current rhythm.
Elapsed episode duration.
Time since most recent observed transition.
Clinical demographics, where consistently available.
Evaluate a physiology-only cross-cohort version and a separate SHDB clinical version.
Do not force missing demographics in other cohorts into an artificial clinical comparison.
F1 — Recent AF history
Immediately preceding complete AF episode duration.
Most recent non-AF gap.
Number of recent observed transitions.
AF exposure during the previous 15, 30 and 60 minutes.
Handle histories containing no previous observed AF separately.
Do not interpret an unknown previous episode duration as zero.
F2 — Conventional historical architecture
Use preceding observation horizons of 30 minutes, 1 hour, 3 hours and 6 hours.
Calculate:
Total AF duration.
AF burden.
Episode count.
Episode frequency.
Longest preceding AF run.
Longest-episode share.
Brief-run count.
Duration-class composition.
Interepisode gap summaries.
Recency of AF exposure.
Previous completed-episode duration statistics.
Use only the observed portion of any episode ongoing at prediction time.
Do not include its eventual duration.
F3 — Density and aggregation
Add:
Independently verified continuous Charitos density.
Published discrete density.
Published aggregation.
Explicit density-availability and history-state indicators.
Primary comparison: continuous density added to F2.
Evaluate aggregation separately and jointly where supported.
Do not assume that continuous and discrete density are interchangeable.
F4 — Conventional ECG physiology
Include existing validated measurements:
Mean ventricular rate.
Median ventricular rate.
RR standard deviation.
RMSSD.
CVRR.
Published VCC.
Sample entropy.
Beat count.
RR distribution percentiles.
Valid beat coverage.
Signal-quality measurements.
Use the published Ngo sample-entropy procedure as a comparator for eligible non-AF windows.
Use separately documented RR processing during AF.
Do not describe AF ventricular irregularity as validated sinus-rhythm autonomic HRV.
For each variable, include recent value, historical mean, variability and trend only where sufficient observations exist.
Do not compute unstable slopes from two points without clearly identifying the limitation.
F5 — Raw ECG and pretrained embeddings
Extract short ECG segments entirely before or ending at the prediction landmark.
Candidate segment lengths:
10 seconds.
30 seconds.
60 seconds.
Use only lengths supported by the encoder.
For pretrained encoders with fixed short input, partition longer ECG history into compatible segments.
Pool embeddings using:
Mean pooling.
Most recent embedding.
Mean and standard deviation.
Compact temporal sequence model, if supported.
Preserve temporal order and missing-segment masks.
Do not interpolate future ECG information.
F6 — Combined representation
Combine F2/F3 historical architecture with F4 physiology and F5 ECG representations.
Use simple regularized late fusion initially.
Do not immediately develop a large multimodal attention architecture.
4. Statistical and machine-learning model registry
Tier A — Mandatory baselines
A0: Empirical training prevalence
Use training-fold outcome prevalence.
For survival tasks, use training-fold empirical survival.
A1: Elapsed-duration baseline
For P02 and P03, use Kaplan–Meier or a parsimonious Weibull model conditional on elapsed episode duration.
A2: Logistic regression
Include:
Unpenalized logistic regression where adequately supported.
L2-regularized logistic regression.
Elastic-net logistic regression.
Fit scaling and imputation within the training folds.
A3: Generalized additive model
Use a binomial GAM for fixed-horizon classification.
Use a survival-compatible GAM or discrete-time hazard GAM for survival tasks.
Model nonlinear effects of:
Historical burden.
Historical density.
Previous episode duration.
Most recent gap.
Elapsed AF duration.
Ventricular rate.
Use penalized low-rank splines.
Restrict effective degrees of freedom according to patient and event support.
Allow density-by-burden interaction only if sufficient common support exists.
Report smooth-function confidence intervals and empirical data support.
A4: Explainable Boosting Machine
Use EBM for nonlinear tabular prediction.
Begin with additive main effects.
Permit a limited number of interactions as a secondary condition.
Use identical feature groups and evaluation populations to logistic regression and GAM.
Report learned shape functions, interaction effects and out-of-fold performance.
Treat feature shapes as predictive associations, not causal effects.
Tier B — Conventional machine learning
B1: Linear SVM
Use for standardized tabular features and frozen high-dimensional embeddings.
B2: RBF-SVM
Tune C and gamma within the inner training folds.
Include this as a specific comparator to Ngo et al.
Fit any probability calibration only using training-fold or inner out-of-fold predictions.
B3: Decision tree
Use a shallow CART classifier.
Tune maximum depth and minimum leaf support.
Report the actual decision rules.
B4: Gradient-boosted decision trees
Primary implementation: XGBoost or LightGBM.
Choose one as the principal GBDT benchmark.
Include CatBoost only where its categorical handling is useful and the comparison is justified.
Restrict model depth, number of boosting rounds and leaf complexity.
Use patient-aware evaluation.
B5: Random forest
Optional secondary comparator.
Do not run multiple redundant tree ensembles unless they answer a specific benchmark question.
Tier C — Survival and dynamic models
Required for P03 and relevant duration analyses:
Cox proportional hazards.
Penalized Cox.
Weibull accelerated failure time.
Discrete-time logistic or complementary-log-log hazard model.
GAM-based discrete-time hazard model.
Random survival forest, if adequately supported.
Survival GBDT, if a suitable objective and validation implementation are available.
Check proportional-hazards assumptions for Cox models.
Report violations and appropriate alternatives.
Include a joint longitudinal-survival model motivated by Ngo et al. only if the number of patients, repeated physiological measurements and event counts justify the model complexity.
For recurrent histories, explicitly address patient-level dependence.
Do not treat a first-event joint model as automatically appropriate for repeated episode termination.
Use alternating Hawkes transition modelling as a history-dependent benchmark for patients meeting the published event-count requirements.
Evaluate predictive likelihood and transition probabilities rather than forcing Hawkes parameters into an inappropriate binary AUROC comparison.
5. Supervised waveform and sequence architectures
D1: Existing Raw Mask ResNet
Reuse the established architecture.
For a new predictive target, replace or retrain the output head as necessary.
Distinguish:
Frozen representation from an existing detector.
Fine-tuned detector encoder.
New architecture trained from random initialization.
An existing detector output is not an episode-persistence probability.
Audit all training-data overlap before using its weights.
D2: xresnet1d18
Implement a compact 1D xResNet-18-style encoder.
Initial architecture:
One-dimensional convolutional stem.
Four residual stages.
Two basic residual blocks per stage.
Candidate widths 32, 64, 128 and 256.
Global average pooling.
Regularized linear prediction head.
Use appropriate temporal strides for the ECG sampling frequency.
Preserve diagnostically relevant ECG bandwidth.
Evaluate:
One-lead input.
Two-lead input with a documented channel-mask mechanism.
Random initialization.
Pretrained initialization only where an architecture-compatible checkpoint exists.
Do not describe the implementation as an exact reproduction of a published xResNet unless the source architecture is actually reproduced.
D3: Compact masked GRU
Input sequential physiological measurements or frozen ECG embeddings.
Include:
Observation mask.
Time since previous observation.
Time since previous rhythm transition.
Elapsed time between feature tokens.
Start with one or two recurrent layers and a small hidden dimension.
Use GRU-D only if its actual decay-based missing-data mechanism is implemented.
D4: Temporal convolutional network
Use a compact causal dilated TCN for sequential RR features or ECG embeddings.
Verify causal receptive fields.
No future time tokens may enter the prediction representation.
Compare recurrent and temporal convolutional models only on matched information histories.
6. Foundation-model checkpoint selection
Locate the user-provided checkpoint paths and validate their actual contents.
Do not infer architecture, input format or parameter count from filename or file size.
Create a model registry with:
Published model.
Publication.
Repository and commit.
Checkpoint path.
Checkpoint hash.
Parameter count.
Pretraining data.
Pretraining objective.
Expected ECG channels.
Expected lead identities.
Expected sampling rate.
Expected signal units.
Expected segment duration.
Normalization.
Attention and channel masks.
Representation extraction point.
Known evaluation-cohort overlap.
FM01: CSFM Tiny — primary
Checkpoint:
/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/Cardiac-Sensing-FM/pretrained/CSFM_tiny.pth
Use the official Cardiac-Sensing-FM implementation.
Extract ECG-only representations.
Verify that the exact checkpoint supports the proposed input configuration.
Run CSFM Base only as a secondary scale comparison after establishing Tiny's technical compatibility.
FM02: CLEF Medium — primary
Checkpoint:
/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/clef/clef_medium.ckpt
Verify the published single-lead interface and model configuration.
Use frozen embeddings initially.
CLEF Small can serve as an efficiency comparison.
Defer CLEF Large.
FM03: ECGFounder 1-lead — primary
Retrieve 1_lead_ECGFounder.pth from:
/data/mithunmanivannan/codex_artifacts/checkpoint_archives/home_models_20260916/project_other_models.tgz
Inspect the archive before extracting the specific checkpoint.
Use the official ECGFounder preprocessing and model architecture.
Extract the representation preceding its diagnostic classification head.
Do not treat diagnostic output probabilities as forecasting probabilities.
FM04: ECG-FM KED — primary conditional comparator
Checkpoint:
/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt
Verify checkpoint identity and source implementation.
Use the supported reduced-lead representation or masking convention.
Do not invent missing leads.
If a fixed 12-lead tensor is required, use documented zero masking only where supported.
A zero-filled channel is not a reconstructed physiological signal.
FM05: DINO-SR and JEPA — compact SSL comparators
Use:
dinosr_e10.ckpt
jepa_e10.ckpt
Within the user-provided ECG pretraining strategies directory.
Verify that these encoders can accept the available Holter lead configurations.
Do not silently alter a 12-lead input layer and claim unchanged pretrained performance.
If adaptation requires retraining an input projection, label that condition separately.
FM06: NEFCore Full and AE — internal comparison
Use:
nefcore_full.pt
nefcore_ae.pt
Evaluate the incremental contribution of geometry-aware representations relative to the base autoencoder.
Confirm actual electrode geometry requirements.
For modified Holter leads, do not assign standard electrode positions without justification.
If geometry cannot be specified faithfully, report incompatibility or explicitly evaluate an approximate geometry condition.
FM07: F2 and F0 SetOperator — internal controlled comparison
Use the supplied pilot_final.pt checkpoints for F2 and F0.
Preserve the architectural differences.
Evaluate the same downstream head, feature set, patient folds and ECG information.
Do not interpret an improvement as an effect of geometric inductive bias if the models differ in other uncontrolled ways.
EchoNext Minimodel and the corresponding G1 geometry SetOperator may be evaluated in SHDB for independently measured LAD or LVEF.
This is a separate structural prediction experiment.
Do not classify AF temporal phenotypes using those scores as independently measured structural labels.
7. Foundation-model adaptation protocol
Stage 1: Compatibility testing
Before any supervised training:
Load every selected checkpoint.
Verify state-dictionary compatibility.
Verify the architecture.
Perform forward passes on representative eligible ECG segments.
Check embedding dimensionality.
Check numerical stability.
Check sensitivity to channel masks.
Check sensitivity to sampling-frequency adaptation.
Check whether embeddings collapse across different inputs.
Use trusted source implementations and safe checkpoint-loading practices.
Do not load arbitrary untrusted executable checkpoint content.
Stage 2: Frozen linear probes
Freeze the full backbone.
Train:
Logistic regression.
Linear SVM.
Regularized linear survival head where applicable.
This is the primary foundation-model comparison.
Use identical patient folds, outcomes and feature availability.
Stage 3: Frozen nonlinear probes
Evaluate:
GBDT on frozen embeddings.
Small regularized MLP.
Optional training-fold-only dimensionality reduction.
Do not allow an excessively flexible probe to obscure differences between pretrained representations.
Stage 4: Historical fusion
Combine frozen ECG embeddings with conventional rhythm-history and RR features.
Begin with concatenation followed by elastic-net logistic regression or GBDT.
Measure the incremental contribution of the embedding.
Stage 5: Parameter-efficient adaptation
Only advance models that demonstrate adequate technical compatibility and meaningful development-fold evidence.
Candidate adaptation:
LoRA where architecturally applicable.
Partial unfreezing of the final encoder block.
Small trainable adapter.
Prespecify selection rules.
Do not choose models for fine-tuning based on final held-out evaluation results.
Stage 6: Full fine-tuning
Not part of the mandatory initial benchmark.
Only consider if the sample size, independent patient support and learning curves justify it.
Given the small physiological prediction populations, full fine-tuning of large foundation models is likely to overfit.
8. Lead, sampling and segment standardization
Run two separate benchmark tracks.
Track A — Matched-information benchmark
All models receive the same physical ECG intervals and one outcome-blind selected lead.
Use a prespecified cohort-specific channel selection rule.
Do not choose the better-performing lead separately for each outcome after evaluation.
Allow each encoder's official resampling and normalization while preserving the same underlying physical signal and duration.
For SHDB, distinguish modified CC5/NASA from standard leads.
Track B — Native-capability benchmark
Permit each architecture to use its supported number of available channels and native input representation.
Report this separately from the matched-information comparison.
Track B evaluates practical maximum capability, not isolated representation quality.
General rules
Preserve physical signal units before model-specific preprocessing.
Use anti-alias filtering where required.
Never fabricate ECG leads.
Never repeat short segments merely to reach a required input length.
Never silently stretch ECG time.
Do not filter across prediction-time boundaries.
Use causal processing for strict real-time claims.
For buffered retrospective processing, explicitly specify acquisition and computation latency.
A model requiring ten seconds of input cannot provide an onset prediction before that input is available.
9. Historical rhythm information: oracle versus deployable tracks
Run two distinct prediction settings.
Reference-history setting
Use original rhythm annotations to calculate preceding burden, density and episode architecture.
This estimates predictive information when rhythm history is measured accurately.
Label it as a reference-history or oracle-history benchmark.
Detected-history setting
Use existing authorized AF detectors and temporal reconstruction methods to generate the preceding rhythm history.
Predictions must be generated without access to future ECG.
Audit D0, D1 and D2 specifically.
A centered median filter or full-record Viterbi/HSMM decode can access future observations.
Such outputs cannot be used in a real-time forecasting claim unless a causal or fixed-lag decoding protocol is explicitly implemented.
Do not transfer reference-history performance directly to a deployment setting.
Compare the impact of detected-history errors on:
Forecast discrimination.
Calibration.
AF persistence classification.
Historical density availability.
Prediction coverage.
10. Required feature-ablation matrix
For every principal task, evaluate:
ID
Information
X00
Minimal clinical or elapsed-duration baseline
X01
Recent episode history
X02
Complete conventional temporal architecture
X03
X02 + density
X04
X02 + aggregation
X05
X02 + density + aggregation
X06
X02 + conventional RR physiology
X07
X03 + conventional RR physiology
X08
Frozen ECG embeddings only
X09
X02 + frozen embeddings
X10
X03 + frozen embeddings
X11
X06 + frozen embeddings
X12
X07 + frozen embeddings
Use a shared analytical population for each direct incremental-information comparison.
Report full-coverage results separately.
The mandatory primary contrasts are:
X02 versus X03.
X06 versus X07.
X06 versus X11.
X07 versus X12.
X11 versus X12.
These determine whether density and learned ECG features contribute complementary information.
Do not run every possible model-feature combination indiscriminately.
Use logistic regression, GAM, EBM and GBDT for the complete tabular ablation sequence.
Use linear probes and GBDT for the principal frozen-embedding comparisons.
Advance sequential and fine-tuned models only after the initial benchmark.
11. Validation protocol
11.1 Patient-disjoint evaluation
No patient may contribute to both training and evaluation within a fold.
Group all recordings, leads, windows, episode pairs and synthetic derivatives from the same patient together.
Do not use random window-level splitting.
Use nested grouped cross-validation where support permits.
Candidate design:
Five outer patient-grouped folds.
Three inner patient-grouped folds.
Reduce the fold count using prespecified event-support rules if necessary.
Do not create folds lacking the outcome classes required by the metric.
All hyperparameter selection, feature selection, dimensionality reduction, calibration and normalization must occur within development data.
11.2 Small-sample validation
For sparse patient populations, consider patient-level bootstrap optimism correction.
Follow the principle of Ngo et al., but resample patients rather than incorrectly treating repeated records as independent.
Report the difference between bootstrap-corrected performance and held-out grouped predictions.
Neither is independent external validation.
11.3 Cohort-specific evaluation
Primary evaluation must be conducted separately for IRIDIA, SHDB and LTAFDB.
Where sufficient common support exists, additionally perform:
Pooled training with cohort-stratified evaluation.
Leave-one-cohort-out transfer.
Training in IRIDIA and evaluation in eligible SHDB prolonged episodes.
Training in SHDB/LTAFDB and evaluation of comparable IRIDIA histories.
Do not interpret failure under substantially different episode-duration distributions as a pure architecture failure.
Report the target-population differences.
11.4 Pretraining contamination audit
For every foundation model, determine whether pretraining or previous fine-tuning included:
IRIDIA-AF.
SHDB-AF.
LTAFDB.
Noise Holter.
Related derivative versions of these data.
The same patients or waveform segments.
Distinguish known overlap, confirmed absence where verifiable and unknown provenance.
A pretrained model exposed to evaluation ECG may still be evaluated as a representation benchmark, but cannot be presented as a fully unseen-cohort generalization result.
Existing internally trained detector checkpoints must also be checked for evaluation-patient overlap.
Train fold-specific versions when required for fair patient-disjoint evaluation.
12. Evaluation metrics
Binary classification
Report:
AUROC.
AUPRC.
Outcome prevalence.
Brier score.
Brier skill score against a training-derived baseline.
Log loss.
Calibration intercept.
Calibration slope.
Sensitivity.
Specificity.
PPV.
NPV.
F1.
Balanced accuracy.
Choose operating thresholds within development data.
Include sensitivity at prespecified false-positive rates.
Do not optimize thresholds on evaluation labels.
Continuous forecasting
For P01 additionally report:
Fraction of sustained episodes preceded by an alert.
Lead time.
False alerts per known non-AF monitoring hour.
Alerts per patient-day.
Time spent in warning state.
Alarm burden.
Repeated alarm merging.
Define a fixed alert refractory period and sensitivity analysis.
An alert matching rule must specify whether multiple warnings preceding one event count as one episode-level success or multiple alert events.
Window-level specificity alone does not establish a clinically useful warning system.
Survival prediction
Report:
Uno's or Harrell's concordance index as appropriate.
Time-dependent AUROC.
Time-dependent Brier score.
Integrated Brier score over supported horizons.
Calibration at specified future horizons.
Observed versus predicted survival.
Estimate inverse-probability censoring weights without evaluation-data leakage.
Do not report ordinary MAE on uncensored episodes as the sole survival-performance measure.
Representation performance
Report:
Total parameter count.
Trainable parameter count.
Embedding dimension.
Inference coverage.
GPU memory.
Feature extraction throughput.
End-to-end inference latency.
Training cost.
Checkpoint size.
Measure runtime using the actual hardware and input configuration.
13. Statistical comparison and interpretation
Generate out-of-fold predictions for every model.
Compare models on identical observations and patient populations.
Use paired patient-level bootstrap confidence intervals for performance differences.
Preserve complete patient histories during resampling.
Report both:
Episode- or landmark-weighted metrics.
Equal-patient-weighted metrics.
Report largest-patient contribution and cohort-specific prevalence.
Do not rely exclusively on pooled AUROC.
Do not apply independent-sample tests to highly overlapping predictions.
For primary feature-increment comparisons, report effect size, uncertainty and calibration changes.
A nonsignificant improvement is not evidence of equivalence.
14. Model interpretation
For statistical models:
Logistic coefficients.
GAM smooths.
EBM feature shapes.
Cox hazard ratios.
AFT time ratios.
For GBDT:
Permutation importance.
Grouped feature ablation.
SHAP where appropriate.
Interpret correlated features jointly.
For waveform models:
Temporal occlusion.
Segment-level representation ablation.
QRS-region versus non-QRS-region occlusion as exploratory analysis.
RR-only versus waveform-only comparison.
Saliency maps do not establish a physiological mechanism.
Investigate whether waveform models primarily encode:
Current rhythm.
Ventricular rate.
RR irregularity.
Lead identity.
Signal quality.
Cohort identity.
Previous episode duration.
Additional morphological information.
Include a diagnostic probe predicting cohort identity from frozen embeddings.
Strong cohort separability is relevant to domain-shift interpretation.
15. Mandatory experimental gates
Before any expensive foundation-model fine-tuning, establish:
Correct source rhythm timelines.
Valid outcome and censoring definitions.
No future-information leakage.
Adequate independent patient support.
Adequate event and non-event support.
Valid model input compatibility.
Appropriate patient-level evaluation.
Pretraining provenance.
Valid conventional baselines.
Meaningful matched-population comparisons.
Perform feasibility simulations or uncertainty-based planning where needed.
Do not declare a fixed patient count universally sufficient.
If a physiological prediction task supports only a very small number of patients, retain it as descriptive or feasibility evidence rather than fitting numerous high-capacity models.
16. Implementation structure
Create:
experiments/af_density_prediction_v7p/
Include:
configs/tasks.yaml
configs/models.yaml
configs/foundation_checkpoints.yaml
configs/statistical_analysis_plan.yaml
src/landmarks.py
src/outcomes.py
src/censoring.py
src/history_features.py
src/ecg_features.py
src/foundation_adapters.py
src/tabular_models.py
src/survival_models.py
src/sequence_models.py
src/validation.py
src/calibration.py
src/alert_evaluation.py
tests/
results/
figures/
reports/
Execution order:
Validate dependencies and source timelines.
Build prediction landmark and outcome manifests.
Produce cohort-specific eligibility tables.
Freeze primary task definitions.
Train empirical and statistical baselines.
Evaluate conventional machine-learning models.
Load and validate foundation-model checkpoints.
Cache eligible frozen embeddings.
Train matched linear and nonlinear probes.
Evaluate incremental historical-density information.
Evaluate ECG representation fusion.
Evaluate compact sequential models where supported.
Conduct censoring-aware survival analysis.
Perform cross-cohort transfer.
Evaluate authentic Noise Holter observability.
Evaluate reference-history versus detected-history predictions.
Generate integrated reports.
Do not regenerate frozen embeddings for every downstream model or fold unless the encoder itself is fitted within that fold.
Cache with checkpoint, input-segment and preprocessing hashes.
17. Required deliverables
Create:
P01_landmark_manifest.parquet
P02_episode_persistence_manifest.parquet
P03_survival_manifest.parquet
P04_noise_classification_manifest.parquet
V7P_task_feasibility.csv
V7P_feature_registry.csv
V7P_foundation_model_registry.csv
V7P_pretraining_overlap_audit.csv
V7P_split_manifest.csv
V7P_model_configurations.csv
V7P_classification_benchmarks.csv
V7P_survival_benchmarks.csv
V7P_alert_benchmarks.csv
V7P_calibration_results.csv
V7P_feature_ablation_results.csv
V7P_foundation_probe_results.csv
V7P_cross_cohort_transfer.csv
V7P_reference_vs_detected_history.csv
V7P_patient_level_results.parquet
V7P_out_of_fold_predictions.parquet
V7P_runtime_benchmarks.csv
V7P_negative_results.csv
V7P_validation_report.md
Generate:
reports/af_density_prediction_v7p_master_report.md
Update the existing AF-density master knowledge base.
For every result include cohort, patient count, episode count, landmark count, outcome prevalence, prediction horizon, model configuration, feature group, evaluation population and artifact path.
18. Final scientific reporting
The primary report must answer these questions separately:
Can historical episode architecture predict progression toward longer AF episodes?
Does historical density add predictive information beyond conventional episode history?
Does early ECG morphology add information beyond RR variability?
Do pretrained ECG representations outperform compact supervised waveform models?
Does pretraining add information when compared against random initialization?
Are predictive relationships consistent across the actual episode-duration distributions of our cohorts?
Does imperfect rhythm-history reconstruction degrade predictive performance?
Which physiological measurements remain observable under authentic noise?
Are any observed gains accompanied by improved calibration and clinically interpretable alert performance?
Do not select a best model solely by pooled AUROC.
Do not present retrospective prediction as proven clinical benefit.
Do not recommend treatment, anticoagulation or cardioversion based on these research models.
Retain all negative, failed and inconclusive results.
The final contribution should explain what information is predictive, under what clinical and temporal conditions, and in which patient populations.
Do not select a manuscript thesis until the full matched-population evidence is available.
8. My proposed primary benchmark suite
To avoid unnecessary computational duplication, this is what I would actually execute first.
Model
P01 Forecast
P02 Persistence
P03 Survival
P04 Rhythm
Empirical baseline
Yes
Yes
Yes
Yes
Elastic-net logistic
Yes
Yes
—
Yes
GAM
Yes
Yes
Yes*
Yes
EBM
Yes
Yes
—
Yes
Linear SVM
Yes
Yes
—
Yes
RBF-SVM
Yes
Yes
—
Yes
Decision tree
Yes
Yes
—
Yes
GBDT
Yes
Yes
Yes*
Yes
Cox / Weibull
Optional*
Optional*
Yes
—
Hawkes
Yes*
Yes*
Yes*
—
Raw Mask ResNet
Yes
Yes
Optional*
Yes
xresnet1d18
Yes
Yes
Optional*
Yes
CSFM Tiny
Yes
Yes
Yes*
Yes
CLEF Medium
Yes
Yes
Yes*
Yes
ECGFounder 1-lead
Yes
Yes
Yes*
Yes
ECG-FM KED
Conditional
Conditional
Conditional
Conditional
DINO-SR
Secondary
Secondary
—
Secondary
JEPA
Secondary
Secondary
—
Secondary
NEFCore Full/AE
Secondary
Secondary
—
Secondary
SetOperator F2/F0
Secondary
Secondary
—
Secondary
*Requires an appropriate time-to-event, hazard or transition-specific implementation. Foundation models require downstream task-specific heads.9. How I would structure the eventual paper
Rather than position it as another ECG foundation-model benchmark, I would frame it as a clinical prediction study with controlled representation comparisons.
Candidate title:
Predicting Sustained Atrial Fibrillation from Ambulatory ECG: Incremental Information from Temporal Episode Organization and Pretrained Cardiac Representations.
The paper would have three central results sections:
1
Temporal AF architecture
Establish how brief and prolonged episodes are related across clinically different Holter populations.
2
Predictive value of historical concentration
Determine whether AF density improves prediction beyond burden, episode duration, recurrence and recent rhythm history.
3
Incremental ECG representation value
Determine whether conventional physiology or pretrained waveform representations improve individualized prediction, including under realistic sensing limitations.
The result I would find most scientifically informative is not necessarily CSFM or CLEF outperforming xResNet.
It would be establishing whether two patients—or two episodes within the same patient—with comparable cumulative AF exposure but different preceding temporal organization have different probabilities of subsequent sustained AF, and whether those differences are reflected in their ECG representations.
That keeps AF density central to the scientific question while allowing statistical learning, classical machine learning, survival analysis and foundation models to contribute meaningfully.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-24T03:22:04-05:00.

The user's current state is as follows:
Other open documents:
- /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/three_dataset_reference_af_density_report.md (LANGUAGE_UNSPECIFIED)
- /data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt (LANGUAGE_UNSPECIFIED)

The user has mentioned some items in the form @[ITEM]. Here is extra information about the items that were mentioned by the user, in the order that they appear:

/goal is a [Slash Command]:
The user has marked this task with /goal, indicating that this task is intended to run for a long time without user input, e.g. overnight. You should be extra thorough and only stop when you are confident the goal has been completely fulfilled. The system will force you to continue execution, prompting you to audit your work until completion. Once complete, include <!-- GOAL_COMPLETE --> in your response. If the user explicitly asked to stop or cancel this goal, include <!-- GOAL_CANCELLED --> in your response to cancel the goal.
/swarm is a [Slash Command]:
<SKILL>The user requested you read and use the "swarm" skill. The path to the skill file is:
/home/mithunmanivannan/.gemini/antigravity/builtin/skills/native_swarm/SKILL.md</SKILL>
</ADDITIONAL_METADATA>