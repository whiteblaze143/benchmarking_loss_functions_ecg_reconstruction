# Speaker Notes: Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial

**Session**: Research Strategy Alignment with Dr. Christopher Cheung & Dr. Alex Mariakakis  
**Base Protocol**: *Evaluation of Wearable/FitBit Monitoring to Detect Recurrent Atrial Fibrillation* (v1.1, Oct 21, 2025)  
**Total Duration**: ~18–20 Minutes (19 Slides @ ~1 min/slide)  
**Tone**: Methodologically Rigorous, Collaborative, Scientifically Grounded, Clinically Focused  

---

## Slide 1: Title & Strategic Framing
- **What to say**:  
  "Good morning Chris, good morning Alex. The purpose of today's meeting is straightforward: we already have an established clinical trial protocol—the Sunnybrook FitBit/Wearable ECG Study, authored by Chris, Alex, and our team.  
  Our goal today is to establish how we can cleanly integrate our single-lead to 12-lead AI reconstruction research into this existing trial as a pre-specified secondary translational aim, without adding patient burden or endangering the primary clinical trial endpoints."
- **Timing**: 45 seconds.
- **Transition**: "Let's start by laying out the agenda and core objectives for our discussion today."

---

## Slide 2: Purpose of Today's Meeting
- **What to say**:  
  "Here is the layout for today.  
  Box 1 is our foundation: Chris's clinical protocol is locked. It evaluates smartwatch monitoring versus continuous CardioSTAT patch for post-ablation AF recurrence.  
  Box 2 is our methodological reality check: we know 12-lead reconstruction is an underdetermined inverse problem; staking the trial's primary regulatory endpoint on reconstruction would be scientifically unsound.  
  Box 3 is our opportunity: during routine clinical visits, we can capture simultaneous paired watch and 12-lead recordings. This establishes a secondary translational evaluation set to benchmark transfer functions and prior models, while completely protecting the primary clinical trial."
- **Timing**: 60 seconds.
- **Transition**: "Let's briefly review the exact parameters of our established clinical protocol."

---

## Slide 3: The Established Clinical Protocol (v1.1)
- **What to say**:  
  "Let us ground ourselves in the exact text of Protocol Version 1.1.  
  Chris designed this study to evaluate whether daily smartwatch ECG recordings and photoplethysmography irregular rhythm notifications detect recurrent AF earlier than standard-of-care intermittent patch monitoring.  
  Patients receive a 14-day CardioSTAT patch at 3 months and 6 months post-procedure, with clinical-grade de-identified reports from iCentia.  
  The primary endpoint is time to AF recurrence. The sample size is 96 patients, starting with a 20-patient vanguard.  
  This clinical core remains completely untouched."
- **Timing**: 60 seconds.
- **Transition**: "Now, why can't we simply make 12-lead reconstruction the primary endpoint of this trial?"

---

## Slide 4: Why Reconstruction Cannot Be the Primary Endpoint
- **What to say**:  
  "This is a critical distinction.  
  Reviewers and clinical colleagues will rightly push back if anyone claims a smartwatch can replace a diagnostic 12-lead machine today.  
  Electrophysiologically, reconstructing 12 leads from one lead is an underdetermined inverse problem.  
  Presacan in 2025 showed that deep learning models compress R-wave and S-wave amplitudes and pull toward population norms.  
  If a clinical trial staked its primary regulatory endpoint on 12-lead reconstruction accuracy, it would introduce severe clinical and regulatory risk.  
  However, that does not mean the data is useless—it means reconstruction belongs strictly as a secondary translational aim."
- **Timing**: 60 seconds.
- **Transition**: "This brings us directly to our proposed two-track trial structure."

---

## Slide 5: The Clean Integration: A Rigorous Two-Track Protocol
- **What to say**:  
  "This architecture cleanly separates the two efforts.  
  Track 1 is Chris's primary clinical trial: time to recurrent AF detection, backed by the 14-day CardioSTAT patch. This guarantees the study will produce high-impact clinical papers regardless of AI performance.  
  Track 2 is Alex's and our translational AI research: we pre-specify 12-lead reconstruction as a secondary translational aim. We use the in-clinic paired recordings to benchmark transfer functions, domain adaptation, and patient-specific priors.  
  This protects trial integrity while maximizing scientific return."
- **Timing**: 60 seconds.
- **Transition**: "To see why auxiliary reconstruction strengthens single-lead representation learning, let's examine the multitask methodology from Stanford published in Circulation."

---

## Slide 6: The 3DRECON Methodology (Ansari et al., Circulation 2026)
- **What to say**:  
  "Chris and Alex, this slide connects our project to recent literature published in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026).  
  Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing the 12-lead spatial conditioning reduced downstream clinical biomarker prediction from r=0.73 down to 0.04.  
  We apply this paradigm to our trial:  
  At home, patients record daily single-lead smartwatch ECGs for our primary endpoint: time to AF recurrence.  
  Lead I alone often misses vector components oriented perpendicular to the frontal plane.  
  Training an auxiliary 12-lead reconstruction head helps the shared encoder preserve 3D atrial vector dynamics without altering single-lead inference at home."
- **Timing**: 75 seconds.
- **Transition**: "Let's look at Figure 1 to see the model architecture."

---

## Slide 7: Figure 1: The ECG-AIM Model Architecture
- **What to say**:  
  "Alex, this is the architectural engine that we have built: ECG-AIM champion checkpoint `conv15e_A0_wave_noSSL_gated_add`.  
  Notice the core components:  
  1. Patch tokenizer slicing the Lead I input into 50 ms tokens.  
  2. A continuous Morlet wavelet bank across 32 frequency scales from 0.5 to 45 Hz.  
  3. A hybrid TimeSformer and 1D-CNN backbone.  
  4. Our gated-add fusion module that dynamically modulates wavelet energy via a sigmoid gate.  
  5. An 8-layer transformer encoder.  
  6. An axial transformer decoder producing continuous 12-lead waveforms and explicit fiducial delineations."
- **Timing**: 60 seconds.
- **Transition**: "Now let's examine Figure 2 to understand how this model is trained on 12-lead data."

---

## Slide 8: Figure 2: ECG-AIM Multi-Task Training Pipeline
- **What to say**:  
  "Alex and Chris, this figure illustrates how we trained the ECG-AIM model before testing on single-lead signals.  
  First, on the left, we ingest full 12-lead hospital records from PTB-XL and MIMIC-IV. We simulate single-lead input by applying a lead-masking layer ($M_{\text{obs}}$) that passes only simulated Lead I ($1 \times 5000$).  
  Second, the signal feeds into the dual-branch wavelet transformer backbone to predict continuous 12-lead waveforms and fiducial delineations.  
  Third, looking at the loss formulation: we supervise waveform reconstruction with a composite loss (MSE, Pearson r correlation, Wyatt ST-T phase loss, and spectral loss) plus Einthoven limb consistency, alongside multi-task cross-entropy and Dice loss on delineation.  
  Importantly, during training, the single lead is a synthetic proxy derived from hospital ECG machines on supine patients with wet gel electrodes. It lacks dry-electrode contact noise, wrist impedance, and wearable DSP filtering.  
  This gap motivates the need for our Sunnybrook paired trial data to calibrate and validate the model on real clinical hardware."
- **Timing**: 75 seconds.
- **Transition**: "Now let's examine the empirical benchmark across clinical evaluation dimensions."

---

## Slide 9: Empirical Validation of ECG-AIM: Benchmark Performance
- **What to say**:  
  "This slide presents the benchmark across models evaluated with patient-clustered MMRM and GEE.  
  Compared to the Stanford 3DRECON-QT model, our model achieves higher precordial chest lead correlation—0.776 vs 0.732—and an anterior Lead V3 correlation of 0.739 vs 0.670.  
  On conduction delay, it achieves 94.3% concordance with an adjusted odds ratio of 1.84 and 96.5% specificity.  
  On echocardiographic structural heart disease in EchoNext, Ansari's model dropped to 46.5% concordance (and 33.3% on reduced LVEF <= 45%) because their scalar QT loss did not constrain chamber geometry.  
  Our wavelet-gated transformer maintains 71.6% structural concordance and tracks QRS duration with a median error of 8.0 milliseconds."
- **Timing**: 75 seconds.
- **Transition**: "Now let's examine zero-shot generalization across external cohorts and variable lead inputs."

---

## Slide 10: Zero-Shot Multi-Cohort Generalization & Flexible Lead Inputs
- **What to say**:  
  "This slide highlights model evaluation under flexible lead inputs.  
  Because we train with dynamic lead token masking, $M_{\text{obs}}$, the exact same model weights accept variable combinations of leads without retraining.  
  At home, single-lead Lead I achieves an ECGFounder Macro AUROC of 0.821 and detects AF with an AUROC of 0.952.  
  When a patient attends clinic, adding Leads II and V2 resolves orthogonal planes: AF detection AUROC reaches 0.981, VT detection reaches 0.989, and QRS duration error drops to 8.12 ms.  
  On EchoNext, the model maintains 75.4% AUROC and 81.4% LVEF concordance.  
  And on our Sunnybrook physical 10-wire XMLs, the model preserves physiological amplitudes and biological noise floors without recalibration."
- **Timing**: 75 seconds.
- **Transition**: "Let's visually examine the reconstructed waveforms across sensor modalities."

---

## Slide 11: Single-Lead Wearable Gallery: Smartwatch (Lead I) vs. Patch (Lead II)
- **What to say**:  
  "Here is the single-lead comparison between sensor modalities on unseen PTB-XL Record 10283.  
  On the left, we input single-lead Lead I—the wrist-to-wrist vector acquired by smartwatch devices. The model infers the other 11 leads with a record mean correlation of 0.941. Notice the precordial chest lead recovery: V1 is 0.995, V2 is 0.989, and V3 is 0.988. Because Lead I spans the transverse plane, it captures anterior ventricular depolarization accurately.  
  On the right, we input single-lead Lead II—the vertical vector typically captured by adhesive chest patches. Lead II achieves a record mean correlation of 0.919, with recovery of inferior leads like aVF at 0.992 and III at 0.963.  
  This confirms that ECG-AIM adapts to whichever lead is provided."
- **Timing**: 60 seconds.
- **Transition**: "Next, see how performance scales when moving from single-lead to 3-lead input."

---

## Slide 12: In-Clinic Escalation Gallery: 1-Lead vs. 3-Lead Diagnostic Triad
- **What to say**:  
  "This comparison shows scaling from single-lead to 3-lead input.  
  On the left is single-lead Lead I input with a mean correlation of 0.941. While suitable for outpatient rhythm surveillance, a single lead is mathematically limited in resolving orthogonal planes simultaneously.  
  On the right, adding Lead II and V2 provides orthogonal axes.  
  Einthoven completion yields 1.000 across all limb leads, and precordial chest leads reach correlations between 0.980 and 0.999, reducing QRS duration error to 8.12 ms."
- **Timing**: 60 seconds.
- **Transition**: "Now let's examine zero-shot external validation on Sunnybrook in-clinic recordings."

---

## Slide 13: External Validation: Sunnybrook Physical 10-Wire XMLs
- **What to say**:  
  "This slide shows zero-shot evaluation on Sunnybrook hospital XML recordings.  
  Record ECG004 displays clinical voltages—over 1.4 mV in Lead II and 2.1 mV in V2.  
  On the left, using single-lead Lead I across physical hospital hardware with un-derived limb leads and natural noise, the model achieves a record mean correlation of 0.865 (Lead II at 0.925, aVR at 0.966, V5 at 0.934).  
  On the right, with the 3-lead triad, mean correlation increases to 0.938. Limb leads reach 1.000, V1 reaches 0.973, and V6 reaches 0.957.  
  Crucially, unlike models that over-smooth waveforms, ECG-AIM preserves physiological amplitude and sharp QRS deflections."
- **Timing**: 60 seconds.
- **Transition**: "Next, let's look at external validation on the 100,000-patient Stanford EchoNext cohort."

---

## Slide 14: External Validation: Stanford EchoNext Structural Cohort
- **What to say**:  
  "This slide shows external validation on the Stanford EchoNext cohort of paired echocardiogram-ECG patient tracings.  
  Examining Record 10, featuring normal voltages up to 1.40 mV in V4 and 1.15 mV in V2:  
  On the left, 1-lead reconstruction achieves a record mean correlation of 0.809, with aVR at 0.967 and chest leads V1 and V5 at 0.942 and 0.915.  
  On the right, adding the 3-lead triad increases mean correlation to 0.964, with limb tracking between 0.986 and 1.000, and precordial tracking reaching 0.987 on V1 and 0.965 on V6.  
  Preserving ST-T morphology and QRS amplitude allows downstream models to achieve 81.4% concordance on reduced ejection fraction."
- **Timing**: 60 seconds.
- **Transition**: "Now let's examine the data collection workflow bridging home surveillance with in-clinic paired acquisition."

---

## Slide 15: Data Collection Methodology: Dual-Stream Protocol
- **What to say**:  
  "This slide outlines the data collection workflow.  
  It operates as two synchronized streams:  
  Stream A is the ambulatory home surveillance from Protocol v1.1. Patients wear the watch daily for 6 months and wear the 14-day continuous CardioSTAT patch at 3 and 6 months. This feeds directly into our primary clinical efficacy endpoint.  
  Stream B is the in-clinic paired protocol. Patients are already at Sunnybrook for four standard clinical visits: baseline pre-ablation, procedure day, 3 months, and 6 months.  
  During the routine clinical 12-lead ECG, the research coordinator records a 30-second smartwatch tracing simultaneously.  
  It adds less than 5 minutes to routine care, requires zero extra visits, and creates synchronized ground-truth pairs."
- **Timing**: 60 seconds.
- **Transition**: "Slide 16 explains the methodological role of this paired in-clinic dataset."

---

## Slide 16: The Paired In-Clinic Evaluation Set
- **What to say**:  
  "This slide contrasts our prospective paired acquisition with the retrospective analysis in the Stanford paper.  
  Ansari and Rogers searched through Medtronic CareLink transmissions, identifying 26 patients who had a clinical 12-lead within plus-or-minus 7 days. Over a week, autonomic tone, electrolytes, and medications can vary.  
  In our trial, we acquire simultaneous pairs across 96 patients at 4 defined clinical visits.  
  Furthermore, this cohort captures longitudinal transitions: pre-ablation AF, acute post-procedure rhythm, and 3-to-6-month follow-up.  
  This directly addresses the gap between synthetic lead masking and physical wearable recording."
- **Timing**: 60 seconds.
- **Transition**: "Figure 3 illustrates how all these components integrate across the entire trial."

---

## Slide 17: Figure 3: Integrated Clinical-AI Trial Pipeline
- **What to say**:  
  "Figure 3 shows how the clinical trial and AI engineering integrate.  
  Stage 1 represents foundation models pretrained on public datasets.  
  Stage 2 represents data acquisition: smartwatch, 14-day continuous CardioSTAT patch, and hospital 12-lead machine.  
  Stage 3 uses the paired in-clinic data to calibrate the hardware transfer function and adapt the model.  
  Stage 4 shows the separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction."
- **Timing**: 60 seconds.
- **Transition**: "Let's look at sample size and statistical modeling."

---

## Slide 18: Sample Size, Statistical Power & Clustered Modeling
- **What to say**:  
  "This slide outlines sample size and statistical considerations.  
  The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target.  
  For the secondary AI aim, 96 patients across multiple clinic visits provides over 2,500 paired cardiac complexes.  
  Per Riley's 2024 BMJ guidelines, this is sufficient to bound QTc measurement error within a 3.5 ms margin of error.  
  We specify linear mixed-effects models and generalized estimating equations to adjust for intra-patient beat clustering."
- **Timing**: 60 seconds.
- **Transition**: "Finally, here are the decisions we want to finalize today."

---

## Slide 19: Action Items & Alignment for Today's Meeting
- **What to say**:  
  "To close our meeting, here are the 4 decisions we need to agree on:  
  First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction.  
  Second, approve the 5-minute simultaneous recording procedure during in-clinic visits.  
  Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation.  
  Fourth, approve the 20-patient vanguard cohort.  
  Thank you Chris and Alex; let's open the floor for discussion."
- **Timing**: 45 seconds.
- **Transition**: "[Open for Discussion & Questions]"
