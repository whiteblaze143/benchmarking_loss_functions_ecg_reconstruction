# Full Word-for-Word Talk Script: Meeting with Dr. Christopher Cheung & Dr. Alex Mariakakis

**Title**: Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial  
**Authors**: Mithun Manivannan, BSc; Alex Mariakakis, PhD; Christopher Cheung, MD, MPH  
**Study Reference**: *Evaluation of Wearable/FitBit Monitoring to Detect Recurrent Atrial Fibrillation* (Protocol Version 1.1, October 21, 2025)  
**Audience**: Chris (Cardiology / EP PI) and Alex (CS / Machine Learning Co-PI)  
**Total Target Time**: ~18–20 Minutes  
**Slide Count**: 19 Slides (16:9 Widescreen)  
**Deliverables**: `slides/main.pdf` (Beamer), `slides/presentation.pptx` (PowerPoint)  

---

## Slide 1: Title Slide [0:00 – 0:45]

"Good morning everyone. Thank you for making the time to sit down today.

The primary goal of today's meeting is straightforward: we already have an established clinical trial protocol—the Sunnybrook FitBit/Wearable ECG Study.

Our specific objective today is to show how we can cleanly integrate our single-lead to 12-lead AI reconstruction research into this existing trial as a pre-specified secondary translational aim, without adding patient burden, without complicating clinic workflows, and without staking the clinical trial's primary outcome on an unproven algorithmic task."

→ *Transition*: "Let's start by laying out the agenda and core objectives for our discussion today."

---

## Slide 2: Purpose of Today's Meeting [0:45 – 1:45]

"Here is the layout of what we want to accomplish today.

First, in Box 1, we start from our foundation. Our clinical trial protocol—Version 1.1, drafted in October 2025—is an investigator-initiated study at Sunnybrook. It evaluates smartwatch monitoring versus conventional intermittent CardioSTAT patch monitoring for detecting recurrent AF in post-ablation patients. That foundation is rock-solid.

Second, in Box 2, we confront the technical reality. 12-lead reconstruction from a single lead is an underdetermined inverse problem. We know from our benchmarks and the published literature that models suffer from amplitude compression in precordial leads. Staking the trial's primary regulatory endpoint on reconstruction would be scientifically unsound.

Third, in Box 3, we have a unique opportunity. By embedding a 5-minute paired recording protocol during routine clinical visits, we can harvest a prospectively paired wearable-to-12-lead dataset. As demonstrated by recent work in Circulation from Stanford, auxiliary 12-lead reconstruction acts as an electrophysiologic regularizer that encourages the single-lead encoder to learn 3D cardiac dynamics, supporting our primary AF detection while establishing a benchmark for secondary reconstruction.

Today, we want to align on this two-track amendment and freeze our secondary AI endpoints."

→ *Transition*: "Let's briefly review the key parameters of our established clinical protocol."

---

## Slide 3: The Established Clinical Protocol (v1.1) [1:45 – 3:00]

"Let us ground ourselves in the exact text of Protocol Version 1.1.

The study is titled *Evaluation of Wearable/FitBit Monitoring to Detect Recurrent Atrial Fibrillation*. It is a single-center prospective cohort study conducted right here at Sunnybrook Arrhythmia Clinic.

We recruit patients aged 18 and older with persistent AF who are undergoing planned catheter ablation or cardioversion. 

For the intervention, patients wear the smartwatch for 6 months, recording daily 30-second single-lead ECGs and receiving automated photoplethysmography irregular rhythm notifications.

For the standard-of-care comparator, every patient receives a Health Canada-approved 14-day continuous CardioSTAT patch monitor at 3 months and 6 months post-index visit. These are mailed directly to iCentia, which generates clinical-grade de-identified reports for the EP team.

The primary clinical outcome is a comparison of time to AF recurrence detection between the smartwatch and conventional clinical monitoring. 

The protocol is powered for a sample size of 96 patients—assuming a 20% recurrence rate on patch versus 25% on smartwatch, with 80% power, an alpha of 0.05, and a 10% non-inferiority margin. We start with an initial 20-patient vanguard to demonstrate operational feasibility.

This clinical core is sovereign, validated, and ready. Our task is simply how to build upon it."

→ *Transition*: "Now, why can't we simply make 12-lead reconstruction the primary endpoint of this trial?"

---

## Slide 4: Related & Prior Work [3:00 – 4:30]

"Before showing our model and results, let me situate this work against what already exists in the literature — because the limitations of prior studies are exactly what motivated our trial design.

There are three threads.

First, wearable AF detection. Abu-Alrub and colleagues in 2022 showed that smartwatch single-lead monitoring achieves around 94% sensitivity for detecting AF compared to CardioSTAT patch monitoring in post-ablation patients. The study this trial is built on confirmed comparable AF recurrence yield at 3 and 6 months. These are strong results for rhythm surveillance. But neither study integrates 12-lead ECG morphology. They detect whether AF is present; they cannot assess QTc prolongation, conduction delay, or repolarization abnormalities. That is the diagnostic information a 12-lead carries that a single lead does not.

Second, single-lead 12-lead reconstruction. The most relevant recent paper is Ansari and Rogers in Circulation, January 2026 — which is where our architecture takes its multitask training strategy from. They showed that adding a 12-lead auxiliary reconstruction head to a single-lead encoder dramatically improves downstream biomarker correlation, from r of 0.04 to 0.73. That is a compelling finding. But their paired evaluation cohort was 26 patients retrospectively pulled from a CareLink archive, with up to 7 days between the device recording and the clinical 12-lead. Autonomic tone, medications, and fluid balance can all change over a week. And critically, CareLink is a subcutaneous implanted cardiac monitor — it is not a surface smartwatch. Presacan 2025 showed that deep learning models trained on synthetic lead-masking develop amplitude compression on precordial leads, with regression toward population norms. Strodthoff's PTB-XL benchmarks are evaluated on the same dataset and masking distribution the models were trained on — that is a closed-loop evaluation that does not reflect deployment.

Third, the wearable domain gap. Hu et al. in 2026 directly compared hospital Lead I and wrist Lead I signals in 120 patients. They found systematic disagreement exceeding 0.3 mV in precordial-sensitive axes. The physics are straightforward: a Samsung Galaxy Watch uses dry stainless-steel ring electrodes on the wrist, which have higher impedance than wet gel electrodes, a different baseline wander profile, motion artifacts from wrist movement, and proprietary DSP filtering that no publicly available model has been tested against. Every reconstruction model in the literature — including ours — was trained exclusively on hospital-grade 12-lead corpora.

So the gap is clear: no existing study evaluates reconstruction on a simultaneously acquired, real smartwatch signal paired with a clinical 12-lead. That is the gap our Sunnybrook protocol fills. And filling it is itself a dataset contribution, independent of whatever reconstruction performance we ultimately report."

→ *Transition*: "That brings us to the proposed two-track trial structure."

---

## Slide 5: The Clean Integration: A Rigorous Two-Track Protocol [4:15 – 5:15]

"Here is the clean, two-track architecture.

Track 1 is the Primary Clinical Efficacy Aim. The research question is Chris's original question: does longitudinal smartwatch surveillance improve yield and time-to-detection of post-ablation recurrent AF compared to 14-day CardioSTAT patch monitoring? This is powered on 96 patients. It is our inviolable primary endpoint. It guarantees the clinical trial succeeds, and it guarantees high-impact clinical cardiology publications.

"Here is the structure we're proposing.

Track 1 is the primary clinical aim, unchanged from Protocol v1.1. The research question is: does daily smartwatch monitoring detect recurrent AF earlier than 14-day CardioSTAT patch monitoring? This is powered on 96 patients, backed by a clinical-grade reference standard, and guarantees the trial succeeds and produces high-impact clinical publications regardless of what the AI achieves.

Track 2 is our secondary translational AI aim, and the core question is this: can we reconstruct a 12-lead ECG from a smartwatch signal specifically in an AF population?

This framing matters. Generic ECG reconstruction benchmarks are dominated by sinus rhythm data. Patients in this trial are different. They will be recorded in active AF before ablation, immediately post-cardioversion, and then in sinus rhythm at 3 and 6 months of follow-up. Reconstruction in AF is harder — there is no organized P wave, the ventricular rate is irregular, and there is a fibrillatory baseline adding structured noise. Whether the model can recover meaningful 12-lead information in that setting is scientifically open. We do not yet know the answer.

Aim 3 builds on Aim 2: given a baseline 12-lead acquired in clinic, can we track within-patient changes in QTc interval or P-wave remodeling over the 6-month follow-up using the smartwatch signal alone?

Both aims are exploratory. They do not affect the primary trial outcome."

→ *Transition*: "To understand why auxiliary reconstruction helps the single-lead encoder, let me walk through a key paper from Stanford."

---

## Slide 6: The 3DRECON Methodology (Ansari et al., Circulation 2026) [5:15 – 6:45]

"This slide connects our project to recent literature published in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026).

Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing the 12-lead spatial conditioning reduced downstream clinical biomarker prediction from r=0.73 down to 0.04.

We apply this exact paradigm to our trial:
At home, patients record daily single-lead smartwatch ECGs for our primary endpoint: time to AF recurrence.
Lead I alone often misses vector components oriented perpendicular to the frontal plane.
By training an auxiliary 12-lead reconstruction head, the shared encoder is encouraged to preserve 3D atrial vector dynamics without altering single-lead inference at home.

Auxiliary 12-lead reconstruction acts as an inductive regularizer during pretraining."

→ *Transition*: "Alex, let me walk you through the network architecture behind these results."

---

## Slide 7: Figure 1: The ECG-AIM Model Architecture [6:45 – 8:15]

"Alex, this is Figure 1: the architectural blueprint of our champion model, `conv15e_A0_wave_noSSL_gated_add`.

Notice the six core engineering modules:

First, the Patch Tokenizer: it takes the raw single-lead wrist signal sampled at 500 Hz and projects 50 ms non-overlapping temporal windows into high-dimensional tokens.

Second, the Continuous Morlet Wavelet Bank: running in parallel across 32 logarithmic frequency scales from 0.5 to 45 Hz, it extracts localized time-frequency scalograms, isolating fast QRS spikes from slow baseline drift.

Third, the TimeSformer and 1D-CNN backbone: combining cross-patch spatiotemporal self-attention with convolutional receptive fields.

Fourth, our Gated-Add Fusion Module: rather than raw concatenation, a learned sigmoid gate dynamically modulates how much high-frequency wavelet energy to inject into the transformer representations without corrupting baseline morphology.

Fifth, an 8-Layer Transformer Encoder capturing long-range rhythm dependencies.

And sixth, an Axial Transformer Decoder that decodes along temporal and spatial lead dimensions simultaneously into two heads: continuous 12-lead waveforms and explicit P-QRS-T boundary delineations."

→ *Transition*: "Now let's examine Figure 2 to understand how this model is trained on 12-lead data and the multi-task loss formulation."

---

## Slide 8: Figure 2: ECG-AIM Multi-Task Training Pipeline [8:15 – 9:45]

"This figure illustrates how we trained the ECG-AIM champion model before deploying it to single-lead signals.

Notice the training setup across four integrated columns:

First, on the left, we ingest full 12-lead hospital records from PTB-XL and MIMIC-IV. We simulate single-lead wrist input by applying a lead-masking layer ($M_{\text{obs}}$) that zeroes out the 11 unobserved leads, passing only simulated Lead I ($1 \times 5000$).

Second, in the model backbone, the signal feeds in parallel to our patch tokenizer ($50$\,ms windows) and a 32-scale Morlet wavelet bank, combined via gated-add fusion and decoded by a 4-layer axial transformer.

Third, the model outputs both continuous 12-lead waveforms and fiducial delineation logits (P-wave, QRS, T-wave).

Fourth, look at the loss formulation on the right: we supervise waveform reconstruction with a composite loss (MSE, Pearson $r$ correlation, Wyatt ST-T phase loss, and spectral loss) plus Einthoven limb consistency ($L_{\text{limb}}: \text{Lead I} + \text{Lead III} = \text{Lead II}$), alongside multi-task cross-entropy and Dice loss on delineation.

Most importantly, look at the bottom callout banner: in training, the single lead is a synthetic proxy derived from hospital ECG machines on supine patients with wet gel electrodes. It lacks dry-electrode contact noise, wrist impedance, motion artifacts, and wearable DSP filtering.

This fundamental training gap is the exact scientific motivation for why we need our Sunnybrook prospective paired trial data to calibrate and validate the model on real clinical hardware."

→ *Transition*: "Now let's examine the empirical results of this architecture across 175,890 clinical observations."

---

## Slide 9: Empirical Validation of ECG-AIM: Benchmark Performance [9:45 – 11:15]

"This slide presents the empirical benchmark across all 55 models evaluated with patient-clustered MMRM and GEE.

Notice our champion model: `conv15e_A0_wave_noSSL_gated_add`.

Compared to the Stanford 3DRECON-QT model published in Circulation, our model achieves higher precordial chest lead correlation—0.776 vs 0.732—and an anterior Lead V3 correlation of 0.739 vs 0.670.

On conduction delay, it achieves 94.3% concordance with an adjusted odds ratio of 1.84 and 96.5% specificity.

Crucially, on echocardiographic structural heart disease in EchoNext, Ansari's model dropped to 46.5% concordance (and 33.3% on reduced ejection fraction <= 45%) because their scalar QT loss did not constrain chamber geometry.

Our wavelet-gated transformer maintains 71.6% structural concordance and tracks QRS duration with a median error of just 8.0 milliseconds.

This confirms our architecture preserves spatial electrophysiology across structural phenotypes."

→ *Transition*: "Now let's examine multi-cohort zero-shot generalization and arbitrary input lead flexibility."

---

## Slide 10: Zero-Shot Multi-Cohort Generalization & Flexible Lead Inputs [11:15 – 12:30]

"This slide highlights one of the key capabilities of ECG-AIM: arbitrary input lead invariance.

Because we train with dynamic lead token masking, $M_{\text{obs}}$, the exact same model weights accept variable combinations of leads without retraining.

When a patient is at home, they touch the smartwatch bezel to provide single-lead Lead I. That single lead achieves an ECGFounder Macro AUROC of 0.821 and detects AF with an AUROC of 0.952.

When the patient attends clinic, adding Leads II and V2 provides orthogonal axes: AF detection AUROC reaches 0.981, VT detection reaches 0.989, and QRS duration error drops to 8.12 milliseconds.

Across zero-shot external benchmarks: on EchoNext, our model maintains 75.4% AUROC and 81.4% LVEF concordance.

And on our Sunnybrook physical 10-wire XMLs, our model preserves authentic physiological amplitudes and biological noise floors without recalibration."

→ *Transition*: "Let's visually examine the reconstructed waveforms across sensor modalities."

---

## Slide 11: Single-Lead Wearable Gallery: Smartwatch (Lead I) vs. Patch (Lead II) [12:30 – 13:45]

"Here is the direct single-lead comparison between sensor modalities on unseen PTB-XL Record 10283.

On the left, we input single-lead Lead I—the wrist-to-wrist vector acquired by smartwatch devices. The model infers all other 11 leads with a record mean correlation of 0.941. Notice how cleanly it reconstructs the precordial chest leads: V1 is 0.995, V2 is 0.989, and V3 is 0.988. Because Lead I spans the transverse plane, it captures anterior ventricular depolarization with high fidelity.

On the right, we input single-lead Lead II—the vertical vector typically captured by chest patches like CardioSTAT or Zio. Lead II achieves a record mean correlation of 0.919, with recovery of inferior leads like aVF at 0.992 and III at 0.963, but slightly lower anterior resolution.

This demonstrates that ECG-AIM adapts to whichever lead configuration is available."

→ *Transition*: "Next, see how performance scales when moving from single-lead to 3-lead input."

---

## Slide 12: In-Clinic Escalation Gallery: 1-Lead vs. 3-Lead Diagnostic Triad [13:45 – 15:00]

"Now look at how our model scales from at-home ambulatory screening to in-clinic diagnostic triage.

On the left is the at-home smartwatch single-lead input with a mean correlation of 0.941. It is suitable for outpatient rhythm surveillance, but a single lead has an inherent mathematical limitation in resolving orthogonal planes simultaneously.

On the right, when the patient arrives at Sunnybrook for routine follow-up, we place two additional stickers—Lead II on the leg and V2 on the chest.

Instantly, the model locks both the frontal and horizontal planes: Einthoven completion achieves 1.000 across all limb leads, and the precordial chest leads reach correlations between 0.980 and 0.999.

Notice the terminal QRS notch and J-point morphology in V1 and V3—they are reproduced with high accuracy, reducing QRS duration error to 8.12 ms."

→ *Transition*: "Now let's examine zero-shot external validation on Sunnybrook in-clinic recordings."

---

## Slide 13: External Validation: Sunnybrook Physical 10-Wire XMLs [15:00 – 16:15]

"This slide is particularly meaningful because it is evaluated zero-shot on physical ECG data—the Sunnybrook 10-wire XML recordings.

We selected diagnostic record ECG004, which displays robust clinical voltages—over 1.4 mV in Lead II and 2.1 mV in V2.

On the left, we provide only smartwatch Lead I to reconstruct the other 11 leads. Across real physical hospital hardware, with un-derived limb leads and natural biological noise, the model achieves a record mean correlation of 0.865, with limb Lead II at 0.925, aVR at 0.966, and lateral chest leads V5 and V6 reaching 0.934 and 0.929.

On the right, look at the 3-lead physical triad. When we give Leads I, II, and V2, the mean correlation increases to 0.938. All limb leads reach 1.000, anterior chest lead V1 reaches 0.973, and lateral leads reach 0.957.

Crucially, unlike standard baselines that attenuate small-amplitude signals, ECG-AIM preserves authentic physical voltage scaling and sharp QRS deflections."

→ *Transition*: "Next, let's look at external validation on the 100,000-patient Stanford EchoNext cohort."

---

## Slide 14: External Validation: Stanford EchoNext Structural Cohort [16:15 – 17:30]

"This slide shows our external validation on the Stanford EchoNext cohort—100,000 paired echocardiogram-ECG patient tracings.

Here we examine diagnostic record #10, featuring standard normal cardiac voltages up to 1.40 mV in V4 and 1.15 mV in V2.

On the left, 1-lead smartwatch reconstruction achieves a record mean correlation of 0.809, with aVR at 0.967 and chest leads V1 and V5 at 0.942 and 0.915.

On the right, adding the 3-lead triad pushes the mean correlation to 0.964, with near-perfect limb tracking (aVR 1.000, aVF 0.997, III 0.986) and precordial tracking reaching 0.987 on V1 and 0.965 on V6.

Why does this matter clinically?
Because preserving the ST-T wave morphology and true QRS amplitude across these leads is what allows downstream classifiers to predict reduced ejection fraction with 81.4% concordance."

→ *Transition*: "Now let's examine the exact data collection methodology that bridges home surveillance with in-clinic paired acquisition."

---

## Slide 15: Data Collection Methodology: Dual-Stream Protocol [17:30 – 18:45]

"This slide covers the data collection methodology and flags some practical questions we need to work through.

Stream A is the ambulatory protocol from v1.1. Patients wear the smartwatch daily and receive 14-day CardioSTAT patches at 3 and 6 months. This feeds directly into the primary AF recurrence endpoint.

Stream B is the in-clinic paired acquisition. At each of four standard clinical visits—baseline, post-procedure, 3 months, and 6 months—we record a simultaneous 12-lead ECG and a 30-second smartwatch trace. The window is under 5 minutes. It does not change the visit schedule and does not require any additional clinic appointments.

There is one important device-level constraint worth surfacing here. Samsung Galaxy Watch pairs exclusively with Samsung smartphones for data transfer. That has direct implications for participant enrollment: patients without Samsung phones cannot use the watch as specified. We need to decide whether to restrict enrollment to Samsung phone owners, whether the study budget can support providing phones, or whether to switch to a cross-platform wearable like Fitbit, which supports both Android and iOS.

Separately, the CardioSTAT patches worn at 3 and 6 months generate 14 days of continuous single-lead recordings per patient. That is a substantial per-patient time series. One idea worth discussing is whether those recordings could be used for patient-specific model fine-tuning—training a lightweight adapter on each individual's electrophysiology before deploying the reconstruction model."

→ *Transition*: "Slide 16 explains why we need to collect this data prospectively and why no existing dataset fills this role."

---

## Slide 16: Why Paired Data: The Public Dataset Gap [18:45 – 20:00]

"This slide explains why we need to collect paired data prospectively — and why no existing dataset addresses this specific problem.

Let me start with the framing. Our secondary aim is reconstruction in an AF population. That specificity matters. Every major ECG reconstruction benchmark — PTB-XL, MIMIC-IV, Chapman — is sinus-rhythm dominated. PTB-XL has AF cases, but they are a small fraction of the dataset, and even those are recorded from supine hospital patients with wet gel electrodes. When a reconstruction model encounters a patient in active AF — no organized P wave, irregular ventricular rate, fibrillatory baseline across leads — it is seeing a waveform pattern that was underrepresented or absent in its training data. We do not know how well these models perform in that setting, because it has never been tested prospectively.

Our Sunnybrook cohort is the ideal setting to answer this. We will collect paired recordings at four defined clinical milestones: baseline before ablation, when patients may be in AF; post-procedure, when rhythm has just been restored; and at 3 and 6 months, during ongoing rhythm monitoring in sinus. That is three distinct rhythm states across the AF disease trajectory, captured under controlled conditions.

The table on the left compares this with the Stanford retrospective study. Their 26 patients had a clinical 12-lead within 7 days of a CareLink transmission — not simultaneous, not prospective, not a surface wearable. Our protocol acquires simultaneous pairs from a real Galaxy Watch during a routine clinic visit.

The additional point on the right is one that should not be taken for granted: wearable Lead I is not the same signal as clinical Lead I. The physics are different enough that published models cannot be assumed to generalize. Hu et al. measured a disagreement of over 0.3 mV. We need empirical data to characterize that gap, and we cannot get it from any existing public source."

→ *Transition*: "Figure 3 shows how all of this integrates across the full trial pipeline."

---

## Slide 17: Figure 3: Integrated Clinical-AI Trial Pipeline [20:00 – 21:15]

"This is Figure 3: the master architecture diagram showing how the clinical trial and AI engineering integrate.

Stage 1 represents our public foundation pretraining on PTB-XL, MIMIC-IV, and Icentia11k.

Stage 2 is our Sunnybrook Tri-Modal Data Engine: the smartwatch collecting daily at-home rhythms and in-clinic paired Lead I; the 14-day continuous CardioSTAT patch providing ambulatory gold standard at 3 and 6 months; and the hospital 12-lead machine providing diagnostic ground truth during clinic visits.

Stage 3 takes the paired in-clinic data to estimate the device transfer function and fine-tune low-rank adapters (LoRA) for wearable physics.

Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction."

→ *Transition*: "Let's look at how the trial sample size accommodates both aims and how we analyze them with clustered statistical modeling."

---

## Slide 18: Sample Size, Statistical Power & Clustered Modeling [21:15 – 22:15]

"This slide demonstrates total alignment on sample size and statistics.

The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target.

For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes. Per Riley's 2024 BMJ guidelines, this is sufficient to bound QTc measurement error within a 3.5 ms margin of error.

Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering.

Primary sample size is powered on clinical events; secondary AI sample size accounts for repeated measures."

→ *Transition*: "Finally, here are the four decisions we want to finalize today."

---

## Slide 19: Discussion Points & Open Questions [22:15 – 23:30]

"To close, here are the protocol decisions we need to make and the practical questions we need to work through.

On the protocol side, four things:
First, confirm the two-track design—Aim 1 is the primary AF recurrence trial, Aim 2 is the secondary translational AI aim. These need to stay clearly separated in the protocol amendment text.
Second, approve the in-clinic paired recording procedure—simultaneous 12-lead and smartwatch at each of the four standard visits, under 5 minutes added per visit.
Third, finalize which secondary clinical biomarker to prioritize for the paired analysis. QTc interval tracking over 6 months is our current recommendation, because it connects directly to arrhythmia risk and is recoverable from repolarization waveform morphology.
Fourth, authorize the 20-patient vanguard to test the workflow before scaling.

On the practical side, four open questions:
One: Device pairing. Samsung Galaxy Watch requires a Samsung smartphone. Most patients will not own a Samsung phone. We need to decide whether to restrict enrollment, provide phones, or switch to a cross-platform device like Fitbit.
Two: CardioSTAT as fine-tuning data. The 14-day patches at 3 and 6 months generate substantial per-patient single-lead time series. Is it worth pre-specifying patient-specific adapter fine-tuning as part of Aim 2?
Three: Which visit is logistically best for the paired recording? Baseline may be more complex; the 3-month follow-up is likely lower acuity and more consistent.
Four: Does the current REB consent cover secondary AI analysis on in-clinic recordings, or do we need a consent addendum? We should confirm this before data collection begins.

Let's open the floor."

→ *Transition*: "[Open for Discussion & Questions]"
