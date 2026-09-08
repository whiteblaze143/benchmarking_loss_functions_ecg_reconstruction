script = """# Full Word-for-Word Talk Script: Meeting with Dr. Christopher Cheung & Dr. Alex Mariakakis

**Title**: Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial  
**Authors**: Mithun Manivannan, BSc; Alex Mariakakis, PhD; Christopher Cheung, MD, MPH  
**Study Reference**: *Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation* (Protocol Version 1.1, October 21, 2025)  
**Audience**: Chris (Cardiology / EP PI) and Alex (CS / Machine Learning Co-PI)  
**Total Target Time**: 15:00 Minutes  
**Slide Count**: 12 Slides (16:9 Widescreen)  
**Deliverables**: `slides/main.pdf` (Beamer), `slides/presentation.pptx` (PowerPoint)  

---

## Slide 1: Title Slide [0:00 – 0:45]

"Good morning Chris, good morning Alex. Thank you both for making the time to sit down today.

The primary goal of today's meeting is straightforward: we already have an established, rigorously designed clinical trial protocol—the Sunnybrook FitBit/Wearable ECG Study, authored by Chris, Alex, and our team. 

Our specific objective today is to show how we can cleanly integrate our single-lead to 12-lead AI reconstruction research into this existing trial as a pre-specified secondary translational aim, without adding patient burden, without complicating clinic workflows, and without staking the clinical trial's primary outcome on an unproven algorithmic task."

→ *Transition*: "Let's start by laying out the agenda and core objectives for our discussion today."

---

## Slide 2: Purpose of Today's Meeting [0:45 – 1:45]

"Here is the layout of what we want to accomplish today.

First, in Box 1, we start from our foundation. Our clinical trial protocol—Version 1.1, drafted in October 2025—is an investigator-initiated study at Sunnybrook. It evaluates smartwatch monitoring versus conventional intermittent CardioSTAT patch monitoring for detecting recurrent AF in post-ablation patients. That foundation is rock-solid.

Second, in Box 2, we confront the technical reality. 12-lead reconstruction from a single lead is a mathematically ill-posed inverse problem. We know from our benchmarks and the published literature that models suffer from amplitude compression in precordial leads. Staking the trial's primary regulatory endpoint on reconstruction would be scientifically unsound.

Third, in Box 3, we have a unique opportunity. By embedding a 5-minute paired recording protocol during routine clinical visits, we can harvest the world's premier prospectively paired wearable-to-12-lead dataset. As demonstrated by recent work in Circulation from Stanford, auxiliary 12-lead reconstruction acts as an electrophysiologic regularizer that forces the single-lead model to learn 3D cardiac dynamics, boosting our primary AF detection sensitivity while providing 12-lead reconstruction on the side!

Today, we want to align on this two-track amendment and freeze our secondary AI endpoints."

→ *Transition*: "Let's briefly review the key parameters of our established clinical protocol."

---

## Slide 3: The Established Clinical Protocol (v1.1) [1:45 – 3:00]

"Let us ground ourselves in the exact text of Protocol Version 1.1.

The study is titled *Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation*. It is a single-center prospective cohort study conducted right here at Sunnybrook Arrhythmia Clinic.

We recruit patients aged 18 and older with persistent AF who are undergoing planned catheter ablation or cardioversion. 

For the intervention, patients wear the smartwatch for 6 months, recording daily 30-second single-lead ECGs and receiving automated photoplethysmography irregular rhythm notifications.

For the standard-of-care comparator, every patient receives a Health Canada-approved 14-day continuous CardioSTAT patch monitor at 3 months and 6 months post-index visit. These are mailed directly to iCentia, which generates clinical-grade de-identified reports for Chris and the EP team.

The primary clinical outcome is a comparison of time to AF recurrence detection between the smartwatch and conventional clinical monitoring. 

The protocol is powered for a sample size of 96 patients—assuming a 20% recurrence rate on patch versus 25% on smartwatch, with 80% power, an alpha of 0.05, and a 10% non-inferiority margin. We start with an initial 20-patient vanguard to demonstrate operational feasibility.

This clinical core is sovereign, validated, and ready. Our task is simply how to build upon it."

→ *Transition*: "Now, why can't we simply make 12-lead reconstruction the primary endpoint of this trial?"

---

## Slide 4: Why Reconstruction Cannot Be the Primary Endpoint [3:00 – 4:15]

"This table explains why we must strictly separate the primary clinical trial from the AI reconstruction task.

In clinical trials, primary endpoints must be clinically validated and directly tied to patient outcomes. Time-to-event AF recurrence detection meets this standard: smartwatches already have 94% sensitivity for AF, and identifying recurrence guides immediate clinical decisions—anticoagulation, antiarrhythmic titration, or repeat ablation. The trial risk is low because clinical utility is guaranteed.

In contrast, wearable 12-lead reconstruction is an underdetermined inverse problem. Mathematically, one electrical projection across the wrists cannot deterministically recover 12 spatial vectors through thoracic tissue. 

In our benchmark and in Presacan's 2025 *Communications Medicine* paper, models achieve decent limb correlation, but precordial chest leads—especially V2 through V4—suffer from substantial amplitude compression and variance loss. Furthermore, as Hu et al. published in 2026, hospital Lead I is fundamentally different from a wearable wrist signal due to dry stainless-steel electrode impedance and proprietary DSP filtering.

If we went to the IRB, to Health Canada, or to industry partners claiming that wrist Lead I deterministically replaces an acute diagnostic 12-lead cart, we would be laughed out of the room. Staking the trial's success on acute 12-lead reconstruction would introduce existential trial failure risk.

Instead, we recognize that reconstruction is a secondary translational discovery task."

→ *Transition*: "This leads directly to our proposed two-track protocol architecture."

---

## Slide 5: The Clean Integration: A Rigorous Two-Track Protocol [4:15 – 5:15]

"Here is the clean, two-track architecture.

Track 1 is the Primary Clinical Efficacy Aim. The research question is Chris's original question: does longitudinal smartwatch surveillance improve yield and time-to-detection of post-ablation recurrent AF compared to 14-day CardioSTAT patch monitoring? This is powered on 96 patients. It is our inviolable primary endpoint. It guarantees the clinical trial succeeds, and it guarantees high-impact clinical cardiology publications.

Track 2 is our Secondary Translational AI Aim. Here we ask two distinct computing science questions:
Aim 2: What 12-lead electrical information is truly recoverable from wrist dry electrodes? We evaluate this on simultaneous in-clinic paired recordings.
Aim 3: Does an in-clinic baseline 12-lead prior enable tracking within-patient repolarization shifts—such as QTc prolongation or P-wave remodeling—over the 6-month follow-up?

This is a true win-win: Chris gets an uncompromised clinical trial, while Alex and our AI team get the world's premier paired clinical-wearable dataset."

→ *Transition*: "To see why auxiliary reconstruction makes our single-lead AF detection so much stronger, let's examine the breakthrough methodology from Stanford published just two months ago in Circulation."

---

## Slide 6: The 3DRECON Methodology (Ansari et al., Circulation 2026) [5:15 – 6:45]

"Chris and Alex, this slide connects our project to the state-of-the-art literature published just two months ago in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026).

Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing the 12-lead spatial conditioning collapsed downstream clinical biomarker prediction from r=0.73 down to 0.04!

We apply this exact paradigm to our trial:
At home, patients record daily single-lead smartwatch ECGs for our primary endpoint: time to AF recurrence.
Lead I alone often suffers from vector blindness when fibrillatory f-waves are oriented perpendicular to the frontal plane.
By training an auxiliary 12-lead reconstruction head, the shared encoder is forced to preserve 3D atrial vector dynamics—giving the smartwatch model superhuman sensitivity for early AF detection!

12-lead reconstruction is not just an output—it is an electrophysiologic regularizer that prevents single-lead diagnostic failure."

→ *Transition*: "Alex, let me walk you through the architectural engine behind these results."

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

→ *Transition*: "Now let's examine the exact data collection methodology that bridges home surveillance with in-clinic paired acquisition."

---

## Slide 8: Data Collection Methodology: Dual-Stream Protocol [8:15 – 9:30]

"Chris and Alex, this is the exact data collection methodology.

It operates as two synchronized streams:

Stream A is the ambulatory home surveillance from Protocol v1.1. Patients wear the watch daily for 6 months and wear the 14-day continuous CardioSTAT patch at 3 and 6 months. This feeds directly into our primary clinical efficacy endpoint: time to AF recurrence and cumulative daily AF burden.

Stream B is the in-clinic paired protocol. Patients are already at Sunnybrook for four standard clinical visits: baseline pre-ablation, procedure day, 3 months, and 6 months.

During the routine clinical 12-lead ECG, the research coordinator records a 30-second smartwatch tracing at the exact same moment.

It adds less than 5 minutes to routine care, requires zero extra visits, and creates perfectly synchronized ground-truth pairs!"

→ *Transition*: "Slide 9 explains why this paired in-clinic dataset is the single most important evaluation set in the entire project."

---

## Slide 9: The Paired In-Clinic Evaluation Set: The Gold Standard [9:30 – 11:00]

"Alex, this slide explains why our computing science group needs this data so desperately: this evaluation set is the crown jewel of the entire project!

Look at the comparison with the Stanford paper in Circulation:
Ansari and Rogers had to search retrospectively through old Medtronic CareLink transmissions. They only found 26 patients with a clinical 12-lead within plus-or-minus 7 days! Over 7 days, hydration, potassium, and medication levels change.

In our trial, we are prospectively acquiring exact delta-t equals zero pairs across 96 patients at 4 distinct time points—yielding nearly 400 paired recordings!

Furthermore, this evaluation set captures true longitudinal transitions: pre-ablation AF, acute sinus restoration with atrial tissue stunning, and 3-to-6-month healed rhythm.

99% of published ECG AI papers test only on synthetic derived leads from 12-lead files. That only tests math, not wearable physics (dry electrodes, wrist impedance, DSP filters). Our prospective paired dataset is the first true clinical-hardware benchmark for wearable 12-lead reconstruction."

→ *Transition*: "Figure 2 illustrates how all these components integrate across the entire trial."

---

## Slide 10: Figure 2: Integrated Clinical-AI Trial Pipeline [11:00 – 12:30]

"Alex and Chris, this is Figure 2: the master architecture diagram showing how the clinical trial and AI engineering integrate seamlessly.

Stage 1 represents our public foundation pretraining on PTB-XL, MIMIC-IV, and Icentia11k.

Stage 2 is our Sunnybrook Tri-Modal Data Engine: the smartwatch collecting daily at-home rhythms and in-clinic paired Lead I; the 14-day continuous CardioSTAT patch providing ambulatory gold standard at 3 and 6 months; and the hospital 12-lead machine providing diagnostic ground truth during clinic visits.

Stage 3 takes the paired in-clinic data to estimate the device transfer function and fine-tune low-rank adapters (LoRA) for wearable physics.

Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction."

→ *Transition*: "Let's look at how the trial sample size accommodates both aims and how we analyze them with clustered statistical rigor."

---

## Slide 11: Sample Size, Statistical Power & Clustered Modeling [12:30 – 13:45]

"Chris and Alex, this slide demonstrates total alignment on sample size and statistics.

The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target.

For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes. Per Riley's 2024 BMJ guidelines, this is more than sufficient to bound QTc measurement error within a tight 3.5 ms margin of error.

Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering.

Primary sample size is powered on clinical events; secondary AI sample size exceeds precision requirements."

→ *Transition*: "Finally, here are the four decisions we want to finalize today."

---

## Slide 12: Action Items & Alignment for Today's Meeting [13:45 – 15:00]

"To close our meeting, here are the 4 concrete decisions we need to agree on:

First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction.

Second, approve the simple 5-minute simultaneous recording SOP during in-clinic visits.

Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation.

Fourth, green-light the 20-patient vanguard cohort.

Thank you Chris and Alex; let's open the floor for discussion."

→ *Transition*: "[Open for Discussion & Questions]"
"""

with open("slides/TALK_SCRIPT.md", "w") as f:
    f.write(script)

print("Updated slides/TALK_SCRIPT.md successfully to 12 slides")
