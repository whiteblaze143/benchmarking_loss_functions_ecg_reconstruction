notes = """# Speaker Notes: Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial

**Session**: Research Strategy Alignment with Dr. Christopher Cheung & Dr. Alex Mariakakis  
**Base Protocol**: *Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation* (v1.1, Oct 21, 2025)  
**Total Duration**: ~15 Minutes (12 Slides @ ~1–1.5 min/slide)  
**Tone**: Rigorous, Collaborative, Scientifically Honest, Translational  

---

## Slide 1: Title & Strategic Framing
- **What to say**:  
  "Good morning Chris, good morning Alex. The purpose of today's meeting is straightforward: we already have an established, rigorously designed clinical trial protocol—the Sunnybrook FitBit/Wearable ECG Study, authored by Chris, Alex, and our team.  
  Our specific goal today is to establish how we can cleanly integrate our single-lead to 12-lead AI reconstruction research into this existing trial as a pre-specified secondary translational aim, without adding patient burden or endangering the primary clinical trial endpoints."
- **Timing**: 45 seconds.
- **Transition**: "Let's start by laying out the agenda and core objectives for our discussion today."

---

## Slide 2: Purpose of Today's Meeting
- **What to say**:  
  "Here is the layout for today.  
  Box 1 is our foundation: Chris's clinical protocol is locked. It evaluates smartwatch monitoring versus continuous CardioSTAT patch for post-ablation AF recurrence.  
  Box 2 is our methodological reality check: we know 12-lead reconstruction is an ill-posed inverse problem; staking the trial's primary regulatory endpoint on reconstruction would be scientifically unsound.  
  Box 3 is our opportunity: during routine clinical visits, we can capture simultaneous paired watch and 12-lead recordings. This creates the world's premier paired dataset, regularizes our single-lead AF detection models, and establishes a publishable secondary translational AI aim."
- **Timing**: 60 seconds.
- **Transition**: "Let's briefly review the exact parameters of our established clinical protocol."

---

## Slide 3: The Established Clinical Protocol (v1.1)
- **What to say**:  
  "Let us ground ourselves in the exact text of Protocol Version 1.1.  
  Chris designed this study to evaluate whether daily smartwatch ECG recordings and photoplethysmography irregular rhythm notifications detect recurrent AF earlier than standard-of-care intermittent patch monitoring.  
  Patients receive a 14-day CardioSTAT patch at 3 months and 6 months post-procedure, with clinical-grade de-identified reports from iCentia.  
  The primary endpoint is time to AF recurrence. The sample size is 96 patients, starting with a 20-patient vanguard.  
  This clinical core remains completely untouched and sovereign."
- **Timing**: 75 seconds.
- **Transition**: "Now, why can't we simply make 12-lead reconstruction the primary endpoint of this trial?"

---

## Slide 4: Why Reconstruction Cannot Be the Primary Endpoint
- **What to say**:  
  "This is the most critical scientific distinction of the presentation.  
  Reviewers and clinical colleagues will rightly push back if anyone claims a smartwatch can replace a diagnostic 12-lead machine today.  
  Electrophysiologically, reconstructing 12 leads from one lead is an underdetermined inverse problem.  
  Presacan in 2025 proved that deep learning models compress R-wave and S-wave amplitudes and pull toward population norms.  
  If a clinical trial staked its primary regulatory endpoint on 12-lead reconstruction accuracy, it would fail.  
  However, that does not mean the data is useless—it means reconstruction belongs strictly as a secondary translational aim!"
- **Timing**: 75 seconds.
- **Transition**: "This brings us directly to our proposed two-track trial structure."

---

## Slide 5: The Clean Integration: A Rigorous Two-Track Protocol
- **What to say**:  
  "This is the architecture that solves everything cleanly.  
  Track 1 is Chris's primary clinical trial: time to recurrent AF detection, backed by the 14-day CardioSTAT patch. This guarantees the study will produce high-impact clinical papers regardless of AI performance.  
  Track 2 is Alex's and our translational AI research: we pre-specify 12-lead reconstruction as a secondary translational aim. We use the in-clinic paired recordings to benchmark transfer functions, domain adaptation, and patient-specific priors.  
  This completely eliminates trial risk while maximizing scientific return."
- **Timing**: 60 seconds.
- **Transition**: "To see why auxiliary reconstruction makes our single-lead AF detection so much stronger, let's look at the breakthrough methodology from Stanford published just two months ago in Circulation."

---

## Slide 6: The 3DRECON Methodology (Ansari et al., Circulation 2026)
- **What to say**:  
  "Chris and Alex, this slide connects our project to the state-of-the-art literature published in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026).  
  Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing the 12-lead spatial conditioning collapsed downstream clinical biomarker prediction from r=0.73 down to 0.04!  
  We apply this exact paradigm to our trial:  
  At home, patients record daily single-lead smartwatch ECGs for our primary endpoint: time to AF recurrence.  
  Lead I alone often suffers from vector blindness when fibrillatory f-waves are oriented perpendicular to the frontal plane.  
  By training an auxiliary 12-lead reconstruction head, the shared encoder is forced to preserve 3D atrial vector dynamics—giving the smartwatch model superhuman sensitivity for early AF detection!"
- **Timing**: 90 seconds.
- **Transition**: "Let's take a look at Figure 1 to see the model architecture behind these numbers."

---

## Slide 7: Figure 1: The ECG-AIM Model Architecture
- **What to say**:  
  "Alex, this is the architectural engine that we have built: ECG-AIM champion checkpoint `conv15e_A0_wave_noSSL_gated_add`.  
  Notice the 6 core components:  
  1. Patch tokenizer slicing the Lead I input into 50 ms tokens.  
  2. A continuous Morlet wavelet bank across 32 frequency scales from 0.5 to 45 Hz.  
  3. A hybrid TimeSformer and 1D-CNN backbone.  
  4. Our gated-add fusion module that dynamically modulates wavelet energy via a sigmoid gate.  
  5. An 8-layer transformer encoder.  
  6. An axial transformer decoder producing both continuous 12-lead waveforms and explicit fiducial delineations."
- **Timing**: 90 seconds.
- **Transition**: "Now let's examine the exact data collection methodology that bridges home surveillance with in-clinic paired acquisition."

---

## Slide 8: Data Collection Methodology: Dual-Stream Protocol
- **What to say**:  
  "Chris and Alex, this is the exact data collection methodology.  
  It operates as two synchronized streams:  
  Stream A is the ambulatory home surveillance from Protocol v1.1. Patients wear the watch daily for 6 months and wear the 14-day continuous CardioSTAT patch at 3 and 6 months. This feeds directly into our primary clinical efficacy endpoint.  
  Stream B is the in-clinic paired protocol. Patients are already at Sunnybrook for four standard clinical visits: baseline pre-ablation, procedure day, 3 months, and 6 months.  
  During the routine clinical 12-lead ECG, the research coordinator records a 30-second smartwatch tracing at the exact same moment.  
  It adds less than 5 minutes to routine care, requires zero extra visits, and creates perfectly synchronized ground-truth pairs!"
- **Timing**: 75 seconds.
- **Transition**: "Slide 9 explains why this paired in-clinic dataset is the single most important evaluation set in the entire project."

---

## Slide 9: The Paired In-Clinic Evaluation Set: The Gold Standard
- **What to say**:  
  "Alex, this slide explains why our computing science group needs this data so desperately: this evaluation set is the crown jewel of the entire project!  
  Look at the comparison with the Stanford paper in Circulation:  
  Ansari and Rogers had to search retrospectively through old Medtronic CareLink transmissions. They only found 26 patients with a clinical 12-lead within plus-or-minus 7 days! Over 7 days, hydration, potassium, and medication levels change.  
  In our trial, we are prospectively acquiring exact delta-t equals zero pairs across 96 patients at 4 distinct time points—yielding nearly 400 paired recordings!  
  Furthermore, this evaluation set captures true longitudinal transitions: pre-ablation AF, acute sinus restoration with atrial tissue stunning, and 3-to-6-month healed rhythm.  
  This completely breaks the synthetic proxy trap that invalidates other AI papers."
- **Timing**: 90 seconds.
- **Transition**: "Figure 2 illustrates how all these components integrate across the entire trial."

---

## Slide 10: Figure 2: Integrated Clinical-AI Trial Pipeline
- **What to say**:  
  "Alex and Chris, this is Figure 2: the master architecture diagram showing how the clinical trial and AI engineering integrate seamlessly.  
  Stage 1 is our foundation models pretrained on big public data.  
  Stage 2 is our tri-modal data acquisition engine: smartwatch, 14-day continuous CardioSTAT patch, and hospital 12-lead machine.  
  Stage 3 uses the paired in-clinic data to calibrate the hardware transfer function and adapt the model.  
  Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction."
- **Timing**: 90 seconds.
- **Transition**: "Let's look at how the trial sample size accommodates both aims and how we analyze them with clustered statistical rigor."

---

## Slide 11: Sample Size, Statistical Power & Clustered Modeling
- **What to say**:  
  "Chris and Alex, this slide demonstrates total alignment on sample size and statistics.  
  The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target.  
  For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes.  
  Per Riley's 2024 BMJ guidelines, this is more than sufficient to bound QTc measurement error within a tight 3.5 ms margin of error.  
  Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering."
- **Timing**: 75 seconds.
- **Transition**: "Finally, here are the four decisions we want to finalize today."

---

## Slide 12: Action Items & Alignment for Today's Meeting
- **What to say**:  
  "To close our meeting, here are the 4 concrete decisions we need to agree on:  
  First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction.  
  Second, approve the simple 5-minute simultaneous recording SOP during in-clinic visits.  
  Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation.  
  Fourth, green-light the 20-patient vanguard cohort.  
  Thank you Chris and Alex; let's open the floor for discussion."
- **Timing**: 60 seconds.
- **Transition**: "[Open for Discussion & Questions]"
"""

with open("slides/speaker_notes.md", "w") as f:
    f.write(notes)

print("Updated slides/speaker_notes.md successfully to 12 slides")
