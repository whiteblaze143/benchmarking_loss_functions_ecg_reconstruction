# Prospective Study Plan: Limits and Utility of One-Lead-Conditioned Twelve-Lead ECG Surrogates After AF Ablation or Cardioversion

**Date:** 2026-08-28  
**Purpose:** Supervisor and Samsung pre-meeting decision document  
**Status:** Design proposal; sample size and claims require final biostatistical and regulatory review

## Problem Anchor

- **Bottom-line problem:** Determine which components of a contemporaneous twelve-lead ECG are recoverable from a real wearable one-lead ECG, quantify the irreducible uncertainty, and test whether an uncertainty-qualified surrogate adds clinically useful information for post-procedure rhythm adjudication.
- **Must-solve bottleneck:** A single electrical projection is non-identifying: multiple spatial cardiac vectors and twelve-lead morphologies can produce similar Lead-I-like signals. Retrospective average-error results can conceal regression to the mean, patient-identity loss and clinically dangerous hallucination. They also do not establish performance under wearable hardware, device filtering, temporal mismatch, noise or recurrent rhythms.
- **Non-goals:** Claim exact or unique recovery of eleven unmeasured leads; replace a measured diagnostic twelve-lead ECG; infer AF burden from isolated 30-second recordings; use generated ECGs for autonomous treatment decisions; present a plausible waveform as if it were directly measured.
- **Constraints:** Start with the existing Fitbit feasibility protocol; preserve standard-of-care monitoring; accommodate a possible Samsung collaboration; obtain raw waveforms rather than PDF images; validate a frozen model before any clinical-facing use.
- **Success condition:** Prospectively paired wearable/reference data identify a reproducible recoverable subset of timing/morphology, demonstrate calibrated uncertainty and abstention for non-identifiable cases, and establish whether the surrogate provides incremental rhythm-adjudication value over the native wearable lead alone. A rigorous finding of no incremental value is an informative outcome rather than a failed study.

## Identifiability Position

The generated output is not a recovered measurement. It is an estimate of the conditional distribution

`p(twelve-lead ECG | wearable lead, device, patient context, optional historical ECG)`.

The study should therefore avoid the language “recreate the patient's twelve-lead ECG.” Its central scientific question is where that conditional distribution is narrow enough to support a bounded claim and where it remains too broad.

Likely recoverable information includes shared timing, heart rate, RR irregularity, approximate atrial/ventricular activation timing and some morphology correlated with the observed axis. Information that depends strongly on orthogonal spatial projections—particularly individualized precordial amplitudes, cardiac axis and localized ST-T changes—may be weakly identifiable or non-identifiable from a contemporaneous Lead-I-like signal alone.

## Recommended Scientific Framing

The programme should contain two linked but distinct questions:

1. **Monitoring question:** Does wearable monitoring identify post-procedure atrial tachyarrhythmia earlier or more often than scheduled conventional monitoring?
2. **Surrogate/identifiability question:** When a wearable ECG is recorded, what twelve-lead information is conditionally predictable, how uncertain is it, and does exposing that estimate improve interpretation beyond the native one-lead waveform?

These questions must not share a single primary endpoint. Monitoring intensity determines much of time-to-detection, whereas reconstruction operates only on captured ECG windows. AF itself is commonly identifiable from a native one-lead ECG; the most plausible incremental value of reconstruction is adjudication of ambiguous rhythms, atrial flutter/atrial tachycardia, ectopy, conduction abnormalities, and morphology—not simply another AF label.

### Fundamental information constraint

If the surrogate is a deterministic function of the wearable waveform alone, it cannot create new information about recurrence or rhythm that is absent from that waveform. Under the data-processing inequality, any apparent gain over a sufficiently capable native-lead classifier must arise from model/reader limitations, a useful representation interface, learned population priors, or additional inputs such as a historical twelve-lead ECG—not recovery of new contemporaneous measurements.

Consequently, this programme should not begin as an interventional clinical trial. The appropriate first study is a prospective, silent, observational diagnostic-accuracy and technical-validation study. A randomized clinical-utility trial is justified only if the locked system first passes prespecified technical, safety and incremental-value gates.

## Proposed Staged Design

### Stage 0: Operational bench and data-contract validation

Before enrolling a reconstruction cohort:

- Confirm export of numerical voltage samples, sampling frequency, amplitude units, filter chain, timestamp precision, recording duration, device model, firmware, app version, laterality, contact state, and manufacturer classification.
- Reject PDF-only or screenshot-only acquisition as the principal model input.
- Characterize timebase drift, clipping, missingness, resampling behavior, and device filters using a signal generator or phantom where possible.
- Freeze a common acquisition SOP and a canonical raw-data schema.
- Establish whether model input is the displayed/filtered clinical waveform or a less processed sensor waveform; do not silently mix the two.

**Go/no-go criterion:** Raw waveform access and sufficiently precise temporal linkage to a reference ECG must be demonstrated for each vendor.

### Stage 1: Twenty-participant vanguard acquisition study

Retain the planned 20-patient pilot, but redefine it as a feasibility and data-quality study—not an efficacy or non-inferiority study.

Each participant should contribute paired acquisitions at:

- pre-procedure baseline;
- immediately before cardioversion or ablation when feasible;
- immediately after restoration of sinus rhythm when clinically feasible;
- discharge or early follow-up;
- 3-month and 6-month visits;
- any symptom-triggered or device-alert visit.

At each acquisition episode:

1. Record a continuous clinical reference ECG/telemetry segment long enough to cover all wearable recordings.
2. Record the Fitbit ECG concurrently with the reference.
3. If Samsung is included, record it concurrently with the reference in randomized device order and with minimal delay. Two watches need not be mutually simultaneous if each has its own simultaneous reference segment.
4. Repeat once when feasible to quantify within-session repeatability.
5. Record posture, wrist, finger contact, motion, time since procedure, symptoms, rhythm, heart rate, and signal-quality failures.

**Vanguard endpoints:** proportion of attempted recordings with usable raw data; synchronization error; paired-window yield; device-specific signal-quality distribution; participant adherence; missing metadata; successful linkage to the reference waveform and clinical rhythm.

### Stage 2: Prospective paired diagnostic/reconstruction validation

Use a prospective, multicentre if feasible, rhythm-enriched cohort. Consecutive post-ablation/cardioversion follow-up alone may not supply enough flutter, atrial tachycardia, or ectopy. Add recruitment from cardioversion clinics, ECG clinics, emergency assessment, and EP follow-up while preserving a consecutive longitudinal stratum.

The reconstruction model and all thresholds must be locked before the validation set is opened. Participants—not recordings—must be assigned to development, calibration, and validation groups. No subject may contribute to more than one split.

Recommended analysis strata:

- sinus rhythm;
- atrial fibrillation;
- atrial flutter;
- organized atrial tachycardia;
- frequent PAC/PVC or other ectopy;
- bradycardia/tachycardia and conduction abnormality;
- classifiable versus poor-quality native wearable tracings;
- Fitbit versus Samsung;
- pre- versus post-procedure;
- sex, age, skin tone where appropriately collected, BMI, wrist size, and relevant comorbidity.

### Stage 3: Longitudinal recurrence-effectiveness cohort

Follow patients for at least 6 months; 12 months is preferable for a definitive study. Preserve standard-of-care patch monitoring. The 2024 AF ablation consensus recommends a blanking period when reporting efficacy and, in clinical trials without invasive monitoring, at least periodic Holter-type monitoring, preferably longer 7- or 14-day monitoring.

Define recurrence independently of reconstruction, for example any adjudicated AF, atrial flutter, or atrial tachycardia episode lasting at least 30 seconds after the prespecified blanking period. Report early recurrence separately.

Wearable reconstruction should be evaluated as an aid attached to each captured wearable ECG, not as a substitute for continuous burden measurement.

## Index and Reference Tests

### Reference standard

- Simultaneously recorded clinical twelve-lead ECG whenever reconstruction fidelity is assessed.
- Raw digital waveform, preferably at 500 Hz with calibrated voltage and acquisition metadata.
- Independent rhythm adjudication by at least two blinded electrophysiologists/cardiologists, with a third adjudicator for disagreement.
- Readers should see the real twelve-lead reference only during reference adjudication, not during index-test interpretation.
- Patch or implantable-device evidence can establish longitudinal recurrence but cannot serve as the exact waveform target for a non-simultaneous watch recording.

### Prespecified index conditions

Readers or classifiers should be evaluated under randomized, blinded conditions:

1. native wearable lead only;
2. reconstructed twelve-lead only;
3. native lead plus reconstructed twelve-lead;
4. real twelve-lead reference, used as the performance ceiling rather than an index test.

Include a washout or use different readers across conditions to prevent memory leakage. Record confidence, interpretation time, unclassifiable status, and perceived artifact.

## Outcomes

### Surrogate clinical-utility primary endpoint

Use a clinically anchored endpoint rather than mean squared error alone:

> Paired difference in correctly adjudicated rhythm category between native one-lead interpretation and native-plus-reconstructed interpretation, against the simultaneous real twelve-lead reference, with participant-clustered confidence intervals.

The exact rhythm set should be finalized after the vanguard establishes event prevalence. If the cohort contains too few non-AF rhythms, this should remain a secondary endpoint and the primary endpoint should become a prespecified multicomponent morphology-agreement criterion.

### Technical identifiability co-primary or gate

The model must pass all parts of a hierarchical safety gate:

- per-lead waveform agreement: MAE/RMSE, Pearson correlation, cross-correlation lag, and Bland-Altman bias;
- waveform landmarks: P/QRS/T onset and offset errors, QRS duration, PR, QT/QTc, amplitudes, axes where identifiable, ST-J/ST-segment deviation, and QRS/T areas;
- lower-tail performance rather than averages alone: participant-level median, 5th percentile correlation, and 95th percentile absolute error;
- conditional-coverage performance: whether reference values fall inside model prediction intervals at their nominal rates;
- ambiguity analysis: how much true twelve-lead morphology varies among episodes with closely similar wearable leads;
- comparison with simple conditional-mean and historical-ECG baselines to expose regression-to-the-mean behavior;
- subgroup and device-domain performance;
- calibration of an uncertainty/abstention score;
- hallucination rate, defined prospectively as a clinically material reconstructed feature absent from the reference.

### Longitudinal clinical endpoints

- time to first adjudicated post-blanking atrial tachyarrhythmia;
- proportion with recurrence detected by each monitoring pathway;
- paired detection delay conditional on events detectable by both pathways;
- AF/atrial tachyarrhythmia burden where a continuous reference permits it;
- false alerts, unclassifiable recordings, confirmatory tests, clinical contacts, and time-to-review;
- adherence, wear time, daily ECG completion, and device switching.

Do not describe time to a scheduled 3- or 6-month patch as intrinsic diagnostic delay without acknowledging interval censoring and unequal observation intensity.

## Data to Collect

### Raw device data

- raw or minimally processed ECG sample array;
- calibrated voltage units, sampling rate, clock/timestamp and timezone;
- filter and resampling specifications;
- device serial/model, firmware, app/SDK version;
- wrist/laterality, hand/finger used, posture, recording duration;
- native device label, heart rate, quality/inconclusive code, and reason code;
- electrode/contact and signal-quality telemetry if exposed;
- PPG windows, irregular-rhythm notifications, wear time, activity and sleep metadata if permitted;
- precise provenance for every firmware or algorithm change.

### Clinical waveform and labels

- raw simultaneous twelve-lead ECG, not merely the PDF report;
- continuous rhythm strip spanning device recordings;
- cardiologist rhythm adjudication and confidence;
- fiducials and clinically important morphology, preferably centrally adjudicated on a stratified subset;
- patch raw data or detailed episode ledger, not only a summary report, if obtainable;
- symptoms and patient-trigger time.

### Patient and procedural context

- demographics and relevant social/device-access variables;
- AF subtype and duration, prior cardioversion/ablation, CHA2DS2-VASc variables;
- echocardiographic LA size and LVEF when clinically available;
- antiarrhythmic drugs, beta blockers, anticoagulation and medication changes;
- ablation strategy, energy source, targets, procedural rhythm and complications;
- recurrence-related encounters, cardioversion, repeat ablation and hospitalization.

## Population and Exclusions

Include adults undergoing AF ablation or cardioversion who can provide wearable recordings and consent to waveform linkage. Avoid excluding participants merely because their tracings are noisy or unclassifiable; those are essential real-world outcomes. Instead, prespecify safety exclusions for acquisition and analyze failures explicitly.

The existing “unable to use the smartwatch” exclusion is too vague. Capture the reason—motor limitation, cognitive limitation, language/access barrier, skin/contact issue, incompatible phone, or unwillingness—because exclusion affects generalizability.

## Device Strategy

The final device strategy should follow the partnership and procurement decision rather than assuming Fitbit remains primary. There are three valid designs:

### Scenario A: Fitbit-only

Use the existing Fitbit devices and dashboard for the entire cohort. This is operationally simplest and gives a clean single-device validation claim, but it does not test transportability and may have less vendor support.

### Scenario B: Samsung replaces Fitbit

Use Samsung as the sole study device if Samsung can supply enough devices, research-grade raw waveform access, stable metadata, data rights and operational support. In this design, all vanguard and powered-cohort participants use Samsung; the reconstruction model is calibrated and prospectively locked for Samsung. Existing Fitbit data can remain external historical evidence but should not be pooled into the prospective Samsung primary analysis.

This is preferable to a small Samsung substudy if the intended deployment partner is Samsung: the study should validate the device that would actually be deployed.

### Scenario C: Samsung supplements Fitbit

Increase the total device pool using both vendors. Allocate participants prospectively by randomized device assignment, or use a deliberately paired crossover subset. This supports device-generalization questions but requires more participants and device-stratified inference.

A practical design is:

- most participants randomized 1:1 to Fitbit or Samsung for longitudinal monitoring;
- a crossover subset records both devices against the same continuous clinical reference during clinic visits;
- each wearable recording remains simultaneous with its own reference segment;
- primary monitoring outcomes are stratified by assigned device;
- reconstruction is analyzed separately by vendor before any pooled estimate.

Do not treat Fitbit and Samsung recordings as interchangeable merely because both are Lead-I-like 30-second ECGs. Hardware, electrode geometry, filters, firmware, apps and classification limits differ. A pooled model may be evaluated, but it must be compared with device-specific models or adapters and include a formal device-by-model interaction analysis.

### Decision rule

- Choose **Samsung replacement** if Samsung is the likely deployment partner and meets the raw-data contract before enrollment.
- Choose **dual-vendor supplementation** if vendor robustness is itself an important scientific or commercial goal and the study can afford the larger sample and operational complexity.
- Retain **Fitbit-only** if Samsung cannot provide raw samples, stable versioning or adequate data-use/publication rights.

### Samsung meeting questions

1. Can research partners export the numerical ECG voltage array rather than a PDF?
2. What are the native and exported sample rates, amplitude units, quantization, filters and resampling operations?
3. Are timestamps synchronized to the phone, watch, or server, and what is their precision?
4. Can firmware/app versions and algorithm changes be frozen or logged throughout the study?
5. Are signal-quality/contact indicators and unclassifiable reason codes available?
6. Can PPG alerts and ECG spot checks be linked through stable pseudonymous identifiers?
7. Is there a batch research API/SDK and what are its retention, rate, consent and regional restrictions?
8. Can raw data be stored and processed on Sunnybrook infrastructure, and can derived models/results be published?
9. Will Samsung support device replacement, calibration testing and notification of algorithm updates?
10. What claims are inside versus outside the device's current regulatory indication?

## Model Strategy

Validate two prespecified modes rather than conflating them:

- **Population mode:** wearable ECG alone, with no patient-specific historical twelve-lead.
- **Personalized mode:** wearable ECG plus one pre-index clinical twelve-lead used only as a fixed personalization/calibration input.

The personalized mode is clinically realistic because these patients generally have historical twelve-lead ECGs, but it answers a different question. Historical ECGs must precede the index wearable recording, and their use must be declared. Never allow a same-episode reference target to enter model adaptation.

The population mode estimates a broad conditional distribution and should not be expected to reproduce unique patient-specific spatial morphology. The personalized mode asks a more defensible question: whether a prior measured twelve-lead ECG supplies the patient's spatial template while the current wearable lead supplies contemporaneous rhythm and timing.

The system should emit uncertainty and abstain when identifiability, input quality or domain shift is outside the validation envelope. Generated waveforms should be visibly labeled as model estimates and never silently inserted into the clinical chart as measured leads.

## Statistical Plan

- Use participant-clustered bootstrap confidence intervals or mixed-effects/generalized estimating equations for repeated recordings.
- Use paired tests for native versus reconstruction-assisted interpretation; do not treat repeated ECGs as independent sample-size units.
- Use participant-level multiple imputation only for appropriate covariates, not missing waveform outcomes.
- Analyze unclassifiable recordings under an intention-to-diagnose framework and report a classifiable-only sensitivity analysis.
- Use competing-risk/interval-censored or monitoring-process-aware methods for time-to-detection as appropriate.
- Freeze the model, preprocessing, thresholds, adjudication manual and analysis code before opening the validation labels.
- Prespecify one primary endpoint and control multiplicity for co-primary or hierarchical endpoints.

### Sample-size recommendation

The original 20 participants are appropriate only for feasibility. The stated 96-patient non-inferiority calculation does not justify reconstruction validation and appears to combine a detection-rate contrast with a non-inferiority framework; it should be independently re-derived.

Planning ranges before vanguard estimates:

- **Vanguard:** 20 participants.
- **Rhythm-enriched paired validation:** approximately 100–150 participants, targeting enough independent cases in each clinically important rhythm stratum rather than merely many repeated sinus-rhythm recordings.
- **Longitudinal recurrence study:** plausibly 250–400 participants if recurrence is about 20–30% and the study needs roughly 80–100 recurrent participants for reasonably precise sensitivity and paired diagnostic comparisons.
- **Samsung replacement:** no separate Samsung quota; power the entire cohort around Samsung data after a Samsung-specific vanguard.
- **Dual-vendor supplementation:** allocate enough independent participants per vendor for device-specific estimates. A rough starting point is at least 100–150 participants per vendor in the powered longitudinal cohort, plus a 60–100 participant paired crossover subset if direct cross-device agreement is a stated claim.

These are design ranges, not final power calculations. Final sizing should use vanguard estimates for usable-pair yield, recurrence prevalence, rhythm distribution, within-participant correlation, native-versus-assisted discordance and attrition.

## Bias and Failure Controls

- **Temporal mismatch:** require simultaneous reference acquisition; align by timestamp and waveform cross-correlation while reporting the estimated lag.
- **Spectrum bias:** enrich rare rhythms but preserve and separately report a consecutive clinical cohort.
- **Identity leakage:** participant-level splits and a prospective locked test cohort.
- **Regression to the mean:** report variance, amplitude distributions, extreme morphology preservation and patient-identification/morphology tests, not just average error.
- **Hallucination:** adjudicate clinically material false morphology and require uncertainty-based abstention.
- **Device shift:** version-aware, vendor-stratified evaluation with no unreported pooling.
- **Reader bias:** blinded randomized reader study with a washout or separate reader panels.
- **Verification bias:** apply the same reference adjudication process regardless of watch result.
- **Adherence bias:** log attempted recordings, failures, wear time and reasons for missingness.
- **Post-procedure drift:** evaluate pre/post and medication/rhythm changes explicitly; do not assume a historical baseline remains morphologically valid.

## Decision Gates

1. **Data gate:** raw ECG plus timestamps and device metadata are available.
2. **Feasibility gate:** at least 85–90% of intended paired episodes are technically linkable, with causes of failure understood.
3. **Technical gate:** prespecified lower-tail morphology and hallucination limits are passed on a locked cohort.
4. **Clinical gate:** reconstruction-assisted interpretation improves or is acceptably non-inferior to native-lead interpretation for the prespecified task without materially increasing false positives.
5. **Transport gate:** required only for a dual-vendor or cross-vendor claim. Performance must be acceptable across both vendors without hidden pooling, or separately calibrated device adapters must be validated transparently.
6. **Impact gate:** only after the above should a study test whether exposing reconstructed outputs changes clinical decisions or outcomes.

Failure at any gate stops escalation. Success on mean correlation alone is insufficient.

## Clinical Investigation Readiness

### Study 1: prospective silent validation—not a therapeutic trial

- Generated outputs are hidden from treating clinicians and participants.
- Clinical care and recurrence management follow standard pathways.
- The model, preprocessing, device versions, output definition and thresholds are locked before enrollment or before validation labels are opened.
- The protocol is registered and its statistical analysis plan timestamped.
- Reporting follows STARD-AI for diagnostic accuracy and relevant TRIPOD+AI principles.
- The reference standard, adjudication process, indeterminate-result handling, exclusions and missing-data rules are fixed prospectively.

### Required falsification criteria

Before enrollment, the team must specify values that would invalidate further clinical development, including:

- maximum clinically material hallucination rate;
- minimum prediction-interval coverage;
- maximum T-on/T-off, QRS-duration, QT/QTc and ST-J error where those features are claimed;
- minimum performance in the worst prespecified device and clinical subgroup;
- maximum rate of confidently wrong outputs;
- minimum incremental diagnostic value over both a native-lead clinician and a strong locked native-lead classifier;
- maximum unclassifiable/abstention rate compatible with the intended use.

Thresholds must be set by electrophysiologists and statisticians according to intended use; they cannot be selected after viewing results.

### Study 2: reader-impact study only after Study 1 passes

If the surrogate passes silent validation, conduct a randomized multi-reader, multi-case study comparing native lead with native-plus-surrogate. The unit of inference remains the participant/case, readers are crossed or properly nested, order is randomized, and washout/memory effects are controlled. This establishes interpretation impact—not patient benefit.

### Study 3: clinical workflow trial only after reader impact is established

Only then consider exposing the output in clinical workflow. Prespecify the action triggered by the result, clinician training, override rules, failure handling and downstream harms from false positive and false negative interpretations. Use SPIRIT-AI for the protocol, CONSORT-AI for reporting and DECIDE-AI for early live clinical evaluation.

## Minimal Claim Set

### Dominant claim

Prospectively synchronized wearable and clinical ECG data can establish the empirical identifiability boundary of one-lead-conditioned twelve-lead surrogates: which timing, morphology and diagnostic features are recoverable, with what uncertainty, and under which patient/device conditions.

### Supporting claim

An uncertainty-qualified surrogate—especially when anchored to a historical patient-specific twelve-lead—either improves adjudication of selected ambiguous post-ablation rhythms over the native wearable lead, or demonstrates that no clinically meaningful incremental information is available. Both outcomes directly answer the clinical-utility question.

### Claims explicitly deferred

- exact inversion or equivalence to a measured diagnostic twelve-lead ECG;
- safe ischemia/STEMI exclusion;
- autonomous clinical diagnosis or treatment;
- continuous AF burden estimation from intermittent ECGs;
- universal transport across consumer devices.

## Literature Grounding

- The 2024 international AF ablation consensus distinguishes routine from trial monitoring, recommends a blanking period, and favors longer periodic ambulatory monitoring when continuous invasive monitoring is unavailable: https://pmc.ncbi.nlm.nih.gov/articles/PMC11632303/
- A prospective four-device study used nearly simultaneous twelve-lead and wearable ECGs for Fitbit, Samsung, Apple and AliveCor, demonstrating a practical acquisition model and substantial manufacturer-algorithm inconclusive/accuracy limitations: https://pmc.ncbi.nlm.nih.gov/articles/PMC10879015/
- The BASEL Wearable Study prospectively compared several consumer devices, including Fitbit and Samsung, against cardiologist-interpreted twelve-lead ECG: https://doi.org/10.1016/j.jacep.2022.09.011
- Fitbit's FDA summary describes a 30-second single-channel Lead-I-like ECG intended to supplement AF-versus-sinus classification, not replace traditional diagnosis: https://www.accessdata.fda.gov/cdrh_docs/pdf20/K200948.pdf
- Samsung's FDA summary likewise describes a 30-second Lead-I-like ECG and documents rhythm/heart-rate classification limits: https://www.accessdata.fda.gov/cdrh_docs/pdf20/K201168.pdf
- MCMA/ECGGenEval argues for signal-, feature- and diagnostic-level reconstruction evaluation, but remains based on retrospective clinical datasets rather than prospective consumer-hardware acquisition: https://arxiv.org/abs/2407.11481
- Personalized single-to-twelve-lead synthesis has used historical twelve-lead ECGs, supporting a distinct personalized-mode hypothesis: https://arxiv.org/abs/1811.08035
- A recent feasibility study explicitly warns that reconstruction may regress toward population means and be unsuitable for clinical use without stronger validation: https://pubmed.ncbi.nlm.nih.gov/40281134/
- Recent true-single-lead work reports promising retrospective diagnostic performance but explicitly states that real wearable prospective validation remains necessary: https://pmc.ncbi.nlm.nih.gov/articles/PMC13286904/

## Supervisor Meeting Decisions

1. Is the programme's dominant clinical goal recurrence detection, reconstruction validation, or reconstruction-assisted rhythm differentiation?
2. Is the first paper a technical prospective validation or a longitudinal clinical-effectiveness study?
3. Will historical patient ECGs be allowed for personalization?
4. Which rhythms besides AF must the reconstructed output distinguish?
5. What error/hallucination level would make the model clinically unacceptable?
6. Can we obtain simultaneous raw twelve-lead and raw wearable waveforms at enough visits?
7. Will Samsung replace Fitbit, supplement the available device count, or not enter the prospective cohort? If both vendors are used, is vendor robustness a primary claim or merely an operational convenience?
8. Are 12 months and a larger multicentre cohort feasible after the 20-person vanguard?
9. Who will provide blinded ECG adjudication and statistical oversight?
10. What output, if any, may clinicians see during the study? The recommended validation phase is silent/shadow mode.

## Recommended Immediate Path

1. Amend the existing protocol so the 20-person phase explicitly captures simultaneous raw wearable/reference ECG pairs and all device metadata.
2. Keep the original time-to-recurrence endpoint as the wearable-monitoring objective, but add a separate reconstruction objective and analysis plan.
3. Ask Samsung for the ten data/firmware/regulatory items above before deciding among Fitbit-only, Samsung replacement and dual-vendor supplementation.
4. Run the vanguard and use its usable-pair yield, rhythm mix and repeated-measure correlation to finalize sample size.
5. Lock the reconstruction model and conduct a silent prospective validation before any reconstruction-assisted clinical workflow study.

## 2026 Landscape and Position Relative to 3DRECON

### What 3DRECON-QT already established

3DRECON-QT is not merely a waveform-reconstruction baseline. Its 2026 *Circulation* study used a multitask encoder-decoder to estimate QT/QTc from a nonstandard single lead, included internal and external health-system testing, a serial dofetilide data set, a 1,676-patient real-world class III antiarrhythmic cohort, and limited testing on real insertable cardiac monitor signals. It reported QT MAE of roughly 18–21 ms and AUROC around 0.94 for prolonged QTc. It also linked prolonged QTc to a higher rate of serious ventricular arrhythmias.

The project therefore substantially occupies the claim that reconstructed latent spatial information from a single nonstandard lead can support ambulatory QT surveillance. A new architecture, reconstruction loss or retrospective public-dataset benchmark is not a meaningful clinical contribution beyond it.

Important remaining limitations are explicit in that study:

- retrospective design;
- no prospectively synchronized digital ICM and twelve-lead recordings;
- real ICM/reference pairs could be separated by up to seven days;
- most model development used a simulated V3-V2 ICM-like vector;
- reconstructed waveforms were validated only for QT measurement;
- no consumer Fitbit/Samsung validation;
- no randomized clinical-workflow or patient-outcome evaluation;
- no decisive comparison showing that the reconstruction pathway outperforms a direct single-lead QT model.

### Defensible contribution beyond 3DRECON

If the local clinical population contains enough patients receiving sotalol or dofetilide, the strongest reconstruction-linked intended use is:

> Serial, patient-specific QTc safety surveillance from consumer smartwatch ECGs after rhythm-control intervention, used to trigger confirmatory measured ECG and medication/electrolyte review—not to provide a synthetic diagnostic twelve-lead ECG.

The generated leads remain hidden. Reconstruction is an internal multitask representation, and its necessity must be tested against a direct native-lead QT/QTc model.

The prospective contribution would be:

1. truly simultaneous Fitbit/Samsung and calibrated twelve-lead acquisition;
2. consumer wrist-vector validation rather than a simulated precordial ICM vector;
3. prespecified comparison of direct single-lead QT prediction versus reconstruction-assisted QT prediction;
4. patient-specific baseline calibration and evaluation of absolute QTc and change from baseline;
5. repeated assessment across posture, rhythm, rate, device, firmware and recording quality;
6. uncertainty/abstention and confidently-wrong-event analysis;
7. a silent workflow study measuring confirmatory-ECG yield, alert burden and potential medication-review triggers;
8. subsequent randomized workflow evaluation only if silent validation passes.

### Combined programme, without conflating endpoints

- **Main AF cohort:** native wearable ECG/PPG surveillance for recurrence.
- **Nested QT safety cohort:** participants receiving prespecified QT-prolonging antiarrhythmics, with serial smartwatch ECG and measured twelve-lead references.
- **Research identifiability cohort:** synchronized waveform pairs used to quantify regression-to-the-mean, uncertainty and device transport.

The QT cohort requires a separate primary endpoint and sample-size calculation. It should not be added as an underpowered exploratory analysis merely to preserve a reconstruction claim.

### Go/no-go criterion for this intended use

Before selecting QT safety surveillance, audit the expected number of patients receiving sotalol/dofetilide, the number of initiation or dose-change episodes, and the feasible number of simultaneous reference ECGs. If the projected number of independently prolonged-QTc cases is inadequate, the reconstruction component should remain an identifiability study and the clinical programme should focus on direct native-signal recurrence surveillance.
