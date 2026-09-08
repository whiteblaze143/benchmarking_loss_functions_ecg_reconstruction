# CMU Forum on Biomedical Engineering 2026 — 5-Minute Talk Script
## "Can We Reconstruct a Diagnostic 12-Lead ECG from Only a Few Measured Leads?"
**Presenter:** Mithun Manivannan, BSc (University of Toronto & Sunnybrook Health Sciences Centre)  
**Co-Authors:** Alex Mariakakis, PhD¹ · Christopher Cheung, MD²  
*¹Department of Computer Science, University of Toronto | ²Division of Cardiology, Sunnybrook Health Sciences Centre*  
**Abstract:** #194 | **Deck Size:** Exactly 8 Slides | **Total Word Count:** 369 words | **Target Runtime:** 3.5 – 4.0 minutes (Strict < 5:00 ceiling)

---

## 8-Slide Presentation Mapping & Timing Overview

| Slide # | Slide Title | Visual Element | Footnote Reference | Spoken Words | Target Time |
|:---:|:---|:---|:---|:---:|:---:|
| **1** | Core Research Question | Institutional Logos (UofT CS + Sunnybrook) & Author Card | Abstract #194 | 34 words | 0:00 – 0:20 |
| **2** | 12 Leads vs. One Wearable Lead | Left: Cardiac Cycle Morphology (`ecg_delineation_components.png`); Right: 12 Leads vs. Smartwatch | Perez 2019; Cheung 2023 | 52 words | 0:20 – 0:50 |
| **3** | The Reconstruction Challenge | Schematic: $\text{1 Observed Lead} \rightarrow \boxed{?} \rightarrow \text{12-Lead}$ + Dual Concept Cards | Frank 1956; Atoui 2010 | 54 words | 0:50 – 1:25 |
| **4** | ECG-AIM Model Architecture | Full-Height Hero Visual (`conv15e_a0_wave_architecture.png`) | van den Oord 2017; Ho 2020 | 29 words | 1:25 – 1:55 |
| **5** | Experimental Setup | 2×2 Clean Benchmark Grid (~175k ECGs, 1–3 Leads, Baselines, Metrics) | Wagner 2020; Gow 2023 | 48 words | 1:55 – 2:30 |
| **6** | Spatial Information Determines Quality | Side-by-Side Panels: 3 Leads ($r > 0.98$) vs. 1 Lead ($r = 0.776$) | Frank 1956; Atoui 2010 | 48 words | 2:30 – 3:10 |
| **7** | Preservation of Clinical Structure | 3-Column Pillars: Morphology ($0.776$), Timing ($8.0\text{ ms}$), EchoNext ($0.775$) | Holste 2024; Surawicz 2009 | 57 words | 3:10 – 3:55 |
| **8** | Conclusions & Next Steps | 3 Core Takeaway Cards + Sunnybrook Protocol in Preparation | Abstract #194 | 51 words | 3:55 – 4:30 |

---

## Slide 1 — Core Research Question [0:00 – 0:20]

**Slide Title:** Can We Reconstruct a Diagnostic 12-Lead ECG from Only a Few Measured Leads?  
**Main Message:** Introducing the research question and clinical collaboration between University of Toronto and Sunnybrook Health Sciences Centre.  
**Visual:** Institutional logos (UofT CS top left, Sunnybrook top right), title hierarchy, author metadata card.  
**Footnote Ref:** *None*

**On-Slide Text:**
```
Can We Reconstruct a Diagnostic 12-Lead ECG from Only a Few Measured Leads?
Discrete Latent Sequence Generation for Limited-Lead ECG Reconstruction

Mithun Manivannan, BSc¹,² • Alex Mariakakis, PhD¹ • Christopher Cheung, MD²
¹Department of Computer Science, University of Toronto | ²Division of Cardiology, Sunnybrook Health Sciences Centre
2026 Carnegie Mellon Forum on Biomedical Engineering • Abstract #194
```

**Speaker Narration (Word count: 34 | Time: 20s):**
> "Hello everyone. I’m Mithun Manivannan from the University of Toronto and Sunnybrook Health Sciences Centre. Today I am presenting our work asking a fundamental question: can we reconstruct a full diagnostic twelve-lead electrocardiogram from only a few measured wearable leads?"

---

## Slide 2 — 12 Leads vs. One Wearable Lead [0:20 – 0:50]

**Slide Title:** 12 Leads vs. One Wearable Lead  
**Main Message:** 12 leads provide comprehensive spatial views of the cardiac cycle; a smartwatch measures only one projection, leaving most spatial information unobserved.  
**Visual:** Left: Authentic ECG waveform delineation diagram (`ecg_delineation_components.png`) showing P, QRS, and ST–T complexes; Right: Dual comparison cards (Clinical 12-Lead vs. Smartwatch ECG).  
**Footnote Ref:** Perez et al., Apple Heart Study, *N Engl J Med* 2019; Cheung et al., *Heart Rhythm* 2023.

**On-Slide Text:**
```
[Left Panel]
Standard ECG Cardiac Cycle & Morphological Intervals
Electrophysiological Fiducials: P wave (atria), QRS (ventricles), and ST–T (repolarization)

[Right Panel]
Clinical 12-Lead ECG: 12 Spatial Views
• Frontal (I–aVF) + Precordial (V1–V6) planes
• Comprehensive 3D cardiac vector field

Smartwatch ECG: 1 Measured Projection
• Single wrist transverse vector (Lead I)
• Completely blind to vertical & chest planes
• Most spatial information is unobserved.
```

**Speaker Narration (Word count: 52 | Time: 30s):**
> "Every heartbeat generates a distinct sequence of electrophysiological waves—P wave, QRS complex, and ST-T segment. Standard clinical care measures these from twelve spatial views across frontal and horizontal planes. Consumer smartwatches, by contrast, measure only a single projection—Lead I across the wrists. While effective for rhythm screening, most spatial information remains unobserved."

---

## Slide 3 — The Reconstruction Challenge [0:50 – 1:25]

**Slide Title:** The Reconstruction Challenge  
**Main Message:** Multiple distinct cardiac electrical states produce similar single-lead traces; the model must infer plausible missing morphology without smoothing it away.  
**Visual:** Top schematic ($\text{1 Observed Lead} \rightarrow \boxed{?} \rightarrow \text{12-Lead}$) + dual concept cards.  
**Footnote Ref:** Frank, *Circulation* 1956; Atoui et al., *IEEE TBME* 2010.

**On-Slide Text:**
```
One Observed Lead    ⟶    [ ? ]    ⟶    12-Lead ECG

Spatial Ambiguity:
Different cardiac electrical states can produce similar single-lead measurements.

Generative Inference:
The model therefore has to infer plausible missing morphology without simply smoothing it away.
```

**Speaker Narration (Word count: 54 | Time: 35s):**
> "This creates a fundamental reconstruction challenge: because multiple distinct cardiac electrical states project onto very similar single-lead signals, the problem is underdetermined. Rather than performing simple pointwise regression that averages away critical spikes, the model must infer plausible missing morphology while preserving sharp physiological features."

---

## Slide 4 — ECG-AIM Model Architecture [1:25 – 1:55]

**Slide Title:** ECG-AIM Model Architecture  
**Main Message:** ECG-AIM combines multiscale wavelets with temporal signals in a compact latent space, jointly reconstructing missing leads.  
**Visual:** **Full-Height Hero Architecture Visual** (`conv15e_a0_wave_architecture.png`).  
**Footnote Ref:** van den Oord et al., VQ-VAE, *NeurIPS* 2017; Ho et al., Axial Attention, *ECCV* 2020.

**On-Slide Text:**
```
[Full Architecture Hero Visual: Continuous Morlet Wavelet SSL + Temporal Patches → Codebook Quantization → Axial Transformer Decoder → 12-Lead Joint Synthesis]
```

**Speaker Narration (Word count: 29 | Time: 30s):**
> "To address this, we developed ECG-AIM. ECG-AIM combines the raw temporal signal with a multiscale wavelet representation, maps these features into a compact latent representation, and jointly reconstructs the missing leads."

---

## Slide 5 — Experimental Setup [1:55 – 2:30]

**Slide Title:** Experimental Setup  
**Main Message:** Evaluated across ~175k ECGs from PTB-XL and MIMIC-IV across 1–3 leads against U-Net and MS-VAE baselines.  
**Visual:** 2×2 clean grid layout with clear icons/badges.  
**Footnote Ref:** Wagner et al., PTB-XL, *Sci Data* 2020; Gow et al., MIMIC-IV-ECG, *PhysioNet* 2023.

**On-Slide Text:**
```
~175k ECGs:
Trained and benchmarked across diverse cohorts from PTB-XL and MIMIC-IV-ECG

1 to 3 Observed Leads:
Evaluated from single wearable Lead I up to 3 complementary leads (I, II, V2)

Competitive Baselines:
Direct head-to-head comparison against continuous 1D U-Net and MultiScale-VAE

Comprehensive Evaluation:
Evaluated on waveform correlation, QRS timing, and downstream clinical preservation
```

**Speaker Narration (Word count: 48 | Time: 35s):**
> "We evaluated our framework across 175,000 hospital ECGs from PTB-XL and MIMIC-IV. We tested configurations ranging from one to three observed leads, compared directly against standard continuous 1D U-Net and MultiScale-VAE baselines, and evaluated both raw waveform fidelity and downstream clinical preservation."

---

## Slide 6 — Spatial Information Determines Reconstruction Quality [2:30 – 3:10]

**Slide Title:** Spatial Information Determines Reconstruction Quality  
**Main Message:** Reconstruction scales sharply with complementary spatial measurements ($r > 0.98$ for 3 leads vs. $r = 0.776$ for 1 lead).  
**Visual:** Side-by-side reconstruction panels (`best_ecg_aim_12lead_reconstruction.png` vs. `best_ecg_aim_lead1_reconstruction.png`) + headline banner.  
**Footnote Ref:** Frank, *Circulation* 1956; Atoui et al., *IEEE TBME* 2010.

**On-Slide Text:**
```
3 Leads Acquired (I, II, V2): r > 0.98    |    1 Lead Acquired (Smartwatch Lead I): r = 0.776
[3-Lead 12-Lead Overlay Panel]            |    [1-Lead 12-Lead Overlay Panel]

Headline: Adding complementary spatial measurements sharply improves reconstruction.
Spatially complementary leads provide substantially more information for reconstruction.
```

**Speaker Narration (Word count: 48 | Time: 40s):**
> "With three complementary leads, reconstruction approaches the measured signal. With a single wearable-like Lead I, substantial morphology remains recoverable, but performance falls because the missing spatial information cannot be directly observed."

---

## Slide 7 — Preservation of Clinically Relevant Structure [3:10 – 3:55]

**Slide Title:** Preservation of Clinically Relevant Structure  
**Main Message:** The reconstruction preserves critical morphology ($r = 0.776$), timing ($8.0\text{ ms}$ QRS error), and downstream foundation model utility ($0.775$ EchoNext AUROC).  
**Visual:** 3 high-contrast stat columns (Morphology, Timing, Downstream Task) + bottom headline banner.  
**Footnote Ref:** Holste et al., EchoNext Structural Prediction, *JACC Adv* 2024; Surawicz et al., *JACC* 2009.

**On-Slide Text:**
```
Morphology                 Timing                     Downstream Task
0.776                      8.0 ms                     0.775
Precordial Lead r          QRS Timing Error           EchoNext Structural AUROC
(vs. 0.571 U-Net,          (Meets < 10.0 ms           (Ground Truth = 0.803)
 0.684 MS-VAE)              Clinical Gate)

Headline: The reconstruction preserved more than waveform similarity.
```

**Speaker Narration (Word count: 57 | Time: 45s):**
> "Crucially, the reconstruction preserved more than waveform similarity. On precordial morphology, ECG-AIM reaches a correlation of 0.776. For clinical timing, it achieves a QRS duration error of 8.0 milliseconds, meeting the clinical gate. And on downstream tasks, EchoNext structural AUROC reaches 0.775, closely tracking the 0.803 ground truth."

---

## Slide 8 — Conclusions and Translational Next Steps [3:55 – 4:30]

**Slide Title:** Conclusions and Translational Next Steps  
**Main Message:** Three core takeaways: spatial bounds, generative preservation, and prospective paired validation at Sunnybrook.  
**Visual:** 3 spacious key takeaway cards + Sunnybrook protocol status & speaker contact info.  
**Footnote Ref:** *None (Clean Protocol Status & Contact Info)*

**On-Slide Text:**
```
1. Reconstruction is limited by the spatial information observed.
Single wearable Lead I recovers useful morphology (r = 0.78); 3 complementary leads unlock near-lossless recovery (r > 0.98).

2. ECG-AIM preserves substantial morphology and timing from limited leads.
Reconstructed signals meet clinical timing gates (8.0 ms QRS error) and retain downstream diagnostic utility (0.775 EchoNext AUROC).

3. Prospective paired smartwatch–12-lead validation is the next test.
Translating from retrospective archives to live clinical deployment requires validating across live hardware factors.

Sunnybrook paired validation protocol in preparation • PI: Dr. Christopher Cheung
Mithun Manivannan • mithun.manivannan@sri.utoronto.ca • Abstract #194
```

**Speaker Narration (Word count: 51 | Time: 35s):**
> "In conclusion: first, reconstruction is limited by the spatial information observed. Second, ECG-AIM preserves substantial morphology and timing from limited leads. Third, prospective paired smartwatch to twelve-lead validation is our next test. Thank you very much for your attention, and I look forward to your questions."

---

### Speaker Delivery Checklist

- [x] **Strict 8-Slide Structure**: Fully aligned across LaTeX, PPTX, and speaker notes.
- [x] **Word Count & Pace**: 369 words at 105 wpm $\rightarrow$ ~3.5 min speaking time, leaving a ~1.5 min cushion before the 5:00 ceiling.
- [x] **No Mini-Lectures**: Merged slides 2 & 3 into a single slide; simplified inverse problem; eliminated separate hardware impedance slide.
- [x] **Clean Claims**: Removed Cartesian dipole basis overclaim; removed MSE categorical failure; harmonized EchoNext AUROC ($0.775$).
- [x] **Clean Visuals**: Zero giant tables; side-by-side 3-lead vs. 1-lead reconstruction on Slide 6; clean 3-pillar metric cards on Slide 7.
