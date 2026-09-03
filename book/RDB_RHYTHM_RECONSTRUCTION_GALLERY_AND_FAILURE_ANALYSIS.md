# External RDB Rhythm Reconstruction Gallery & Deep-Dive Failure Mode Analysis

**Cohort Provenance**: Chinese Cardiologist-Annotated Subset of the Chapman-Shaoxing Database (*Zheng et al., Scientific Data 2020*, Shaoxing People's Hospital, Zhejiang, China / Chapman University).  
**Test Set Scale**: $N = 360$ frozen, patient-disjoint 12-lead ECG records ($10.0\text{ seconds}$ at $500\text{ Hz}$, $5,000\text{ samples/lead}$, preserved in physical millivolts).  
**Evaluated Architecture**: Primary baseline model **`conv15e_A0_raw_s42_l0`** (1D Transformer Autoencoder with 8-layer encoder, 4-block axial space-time decoder, width 768, patch size 25, 0 wavelet/SSL overhead) vs. multi-scale wavelet SSL reference **`conv15e_R7_morlet_mag_ueg_real_s42_l0`**.

---

## 1. Executive Summary & Clinical Findings

Single-lead ECG reconstruction models attempt to recover all 11 missing clinical leads ($\text{II}, \text{III}, a\text{VR}, a\text{VL}, a\text{VF}, V_1\text{--}V_6$) from a single horizontal limb vector (Lead I: Left Arm $-$ Right Arm, $\mathbf{c}_I = [1, 0, 0]^T$). While benchmark summary metrics show strong aggregate performance ($\bar{r} \approx 0.7456$ on PTB-XL, boundary $F_1 = 0.7234$ on RDB), aggregate metrics conceal **severe, rhythm-specific failure modes** that can lead to catastrophic diagnostic errors if deployed without biophysical safety guards.

This study presents a complete **24-case stratified gallery** (Best, Median, and Worst cases across all 8 canonical rhythm classes in RDB) and provides a rigorous, mechanism-driven investigation into why and where the model breaks down.

### Summary Performance Across All 8 Canonical Rhythms ($N = 360$)

The table below summarizes the reconstruction performance of `conv15e_A0_raw` across the 360 held-out test records, grouped by clinical rhythm:

| Canonical Rhythm | Clinical Name | Test Count ($N$) | Mean Missing $r$ | SD | Minimum $r$ (Worst) | Median $r$ | Maximum $r$ (Best) | Clinical Difficulty Rating |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SA** | Sinus Arrhythmia | 60 | **$0.7967$** | $0.1001$ | $0.5015$ | $0.8243$ | **$0.9288$** | Low (Preserved Morphology) |
| **SB** | Sinus Bradycardia | 60 | **$0.7801$** | $0.0945$ | $0.4646$ | $0.7912$ | **$0.9335$** | Low (Long Diastole) |
| **SR** | Normal Sinus Rhythm | 60 | **$0.7755$** | $0.1033$ | $0.2639$ | $0.7838$ | **$0.9242$** | Low–Moderate (Axis Shifts) |
| **AT** | Atrial Tachycardia | 18 | **$0.7201$** | $0.1567$ | $0.3772$ | $0.7787$ | **$0.8920$** | Moderate (Ectopic P-wave) |
| **ST** | Sinus Tachycardia | 21 | **$0.7169$** | $0.1689$ | $0.1639$ | $0.7509$ | **$0.8845$** | Moderate–High (P-on-T Fusion) |
| **AF** | Atrial Flutter | 60 | **$0.6918$** | $0.1406$ | $0.2102$ | $0.7155$ | **$0.9104$** | High (Sawtooth Flutter Wave) |
| **AFIB** | Atrial Fibrillation | 60 | **$0.6904$** | $0.1220$ | $0.3780$ | $0.7155$ | **$0.8563$** | High (Chaotic $f$-waves) |
| **SVT** | Supraventricular Tachycardia | 21 | **$0.6286$** | $0.2196$ | **$-0.0277$** | $0.6878$ | **$0.8521$** | **Severe (Polarity Inversion)** |

---

## 2. Master Stratification Matrix (24 Selected Cases)

For each rhythm class, we selected the **Worst Case** (lowest $\bar{r}$), **Median Case** ($50^{\text{th}}$ percentile), and **Best Case** (highest $\bar{r}$):

| Rhythm | Tier | Record ID | Patient ID | Mean Missing $r$ | RMSE (mV) | Limb $r$ | Precordial $r$ | Lead II $r$ | Lead $a\text{VF}$ $r$ | Lead $V_1$ $r$ | R7 Mean $r$ | $\Delta r (\text{R7} - \text{A0})$ |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AF** | Worst | `AF0228` | `MUSE_20180114_114448_39000` | **$0.2102$** | $0.1431$ | $-0.0042$ | $0.3889$ | $-0.200$ | **$-0.373$** | $0.549$ | $0.1630$ | $-0.0472$ |
| **AF** | Median | `AF0071` | `MUSE_20180712_153004_14000` | **$0.7195$** | $0.1578$ | $0.6012$ | $0.8180$ | $0.769$ | $0.295$ | $0.691$ | $0.7212$ | $+0.0017$ |
| **AF** | Best | `AF0224` | `MUSE_20180114_131049_72000` | **$0.9104$** | $0.0982$ | $0.9381$ | $0.8874$ | $0.935$ | $0.889$ | $0.707$ | $0.9166$ | $+0.0062$ |
| **AFIB** | Worst | `AFIB0293` | `MUSE_20180119_171916_63000` | **$0.3780$** | $0.3489$ | $0.4649$ | $0.3056$ | $0.266$ | **$-0.375$** | $0.151$ | $0.4242$ | $+0.0462$ |
| **AFIB** | Median | `AFIB0178` | `MUSE_20180118_132400_96000` | **$0.7157$** | $0.1592$ | $0.7584$ | $0.6801$ | $0.830$ | $0.751$ | $0.864$ | $0.7300$ | $+0.0143$ |
| **AFIB** | Best | `AFIB0198` | `MUSE_20180113_173023_29000` | **$0.8563$** | $0.1541$ | $0.7818$ | $0.9184$ | $0.872$ | $0.678$ | $0.955$ | $0.8329$ | $-0.0233$ |
| **AT** | Worst | `AT0025` | `MUSE_20180113_130402_38000` | **$0.3772$** | $0.1787$ | $0.3926$ | $0.3645$ | $0.377$ | $0.251$ | $0.509$ | $0.4097$ | $+0.0325$ |
| **AT** | Median | `AT0114` | `MUSE_20180118_175817_58000` | **$0.7911$** | $0.1691$ | $0.7602$ | $0.8168$ | $0.842$ | $0.683$ | $0.943$ | $0.7484$ | $-0.0427$ |
| **AT** | Best | `AT0028` | `MUSE_20180113_131657_13000` | **$0.8920$** | $0.1593$ | $0.8182$ | $0.9535$ | $0.863$ | $0.764$ | $0.955$ | $0.9000$ | $+0.0080$ |
| **SA** | Worst | `SI0012` | `MUSE_20180115_120015_26000` | **$0.5015$** | $0.1512$ | $0.2866$ | $0.6806$ | $0.477$ | **$0.091$** | $0.215$ | $0.5832$ | $+0.0817$ |
| **SA** | Median | `SI0318` | `MUSE_20180114_131208_23000` | **$0.8274$** | $0.1084$ | $0.8752$ | $0.7876$ | $0.935$ | $0.873$ | $0.742$ | $0.8174$ | $-0.0100$ |
| **SA** | Best | `SI0238` | `MUSE_20180115_120332_79000` | **$0.9288$** | $0.0921$ | $0.9248$ | $0.9322$ | $0.923$ | $0.908$ | $0.893$ | $0.9139$ | $-0.0148$ |
| **SB** | Worst | `SB0110` | `MUSE_20180120_120829_81000` | **$0.4646$** | $0.1501$ | $0.2737$ | $0.6238$ | $0.271$ | **$-0.036$** | $0.694$ | $0.4958$ | $+0.0311$ |
| **SB** | Median | `SB0305` | `MUSE_20180116_180934_09000` | **$0.7935$** | $0.1345$ | $0.8651$ | $0.7338$ | $0.862$ | $0.867$ | $0.686$ | $0.7674$ | $-0.0261$ |
| **SB** | Best | `SB0192` | `MUSE_20180119_174514_93000` | **$0.9335$** | $0.0762$ | $0.9634$ | $0.9086$ | $0.970$ | $0.938$ | $0.876$ | $0.9234$ | $-0.0101$ |
| **SR** | Worst | `SR0234` | `MUSE_20180210_132725_66000` | **$0.2639$** | $0.2771$ | $-0.0696$ | $0.5419$ | $0.081$ | **$-0.429$** | $0.929$ | $0.2669$ | $+0.0030$ |
| **SR** | Median | `SR0147` | `MUSE_20180210_123018_80000` | **$0.7845$** | $0.1492$ | $0.6698$ | $0.8801$ | $0.586$ | $0.553$ | $0.943$ | $0.8646$ | $+0.0801$ |
| **SR** | Best | `SR0025` | `MUSE_20180209_131650_53000` | **$0.9242$** | $0.0931$ | $0.9412$ | $0.9100$ | $0.952$ | $0.920$ | $0.875$ | $0.9234$ | $-0.0008$ |
| **ST** | Worst | `ST0030` | `MUSE_20180114_070025_67000` | **$0.1639$** | $0.2578$ | $0.4409$ | $-0.0669$ | $0.347$ | $0.467$ | $0.164$ | $0.4601$ | $+0.2961$ |
| **ST** | Median | `ST0098` | `MUSE_20180112_072239_97000` | **$0.7509$** | $0.1741$ | $0.9324$ | $0.5997$ | $0.958$ | $0.936$ | $0.552$ | $0.7621$ | $+0.0112$ |
| **ST** | Best | `ST0080` | `MUSE_20180113_133051_01000` | **$0.8845$** | $0.1251$ | $0.9281$ | $0.8482$ | $0.934$ | $0.679$ | $0.866$ | $0.8513$ | $-0.0332$ |
| **SVT** | Worst | `VT0122` | `MUSE_20181222_204131_50000` | **$-0.0277$** | $0.3094$ | $0.2205$ | $-0.2346$ | $-0.167$ | **$-0.187$** | **$-0.525$** | $-0.0118$ | $+0.0159$ |
| **SVT** | Median | `VT0095` | `MUSE_20180115_124355_77000` | **$0.6878$** | $0.1983$ | $0.6391$ | $0.7284$ | $0.646$ | $0.258$ | $0.931$ | $0.6944$ | $+0.0066$ |
| **SVT** | Best | `VT0112` | `MUSE_20180116_122703_41000` | **$0.8521$** | $0.1812$ | $0.7782$ | $0.9137$ | $0.794$ | $0.584$ | $0.833$ | $0.8345$ | $-0.0176$ |

---

## 3. Systematic Failure Mode Taxonomy: Why Single-Lead Models Fail

A thorough electrophysiological and biophysical dissection of the worst cases reveals that errors are **not random noise**. They cluster into **5 discrete biophysical failure mechanisms**:

```
                              SINGLE-LEAD (LEAD I) INPUT
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ANATOMIC / SPATIAL FAILURES                    TEMPORAL / RHYTHMIC FAILURES
                 │                                               │
   ┌─────────────┴─────────────┐                   ┌─────────────┴─────────────┐
   ▼                           ▼                   ▼                           ▼
[Failure 1: Vertical        [Failure 4: Precordial [Failure 2: Atrial Chaos    [Failure 3: Tachycardia
Null Space aVF Collapse]    Z-Axis Decoupling]     f-Wave Smoothing]           P-on-T Conflation]
Lead I ⟂ aVF: Polarities    Px(t) blind to Pz(t);  Autoencoder acts as low-    At rates >140 bpm, diastole
invert when heart axis      V1/V2 collapse to      pass filter; fibrillatory   vanishes; P merges into T;
deviates from normal        negative correlation   waves wiped out             phase completely inverts
```

### Failure Mode 1: The Orthogonal Dipole Null Space ($\mathbf{c}_I \perp \mathbf{c}_{a\text{VF}}$)
- **Electrophysiological Law**:
  The heart's frontal electrical activity is modeled as an equivalent time-varying dipole vector $\mathbf{p}(t) = [p_x(t), p_y(t)]^T$.
  $$\Phi_I(t) = \mathbf{c}_I \cdot \mathbf{p}(t) = p_x(t) \qquad (\text{horizontal axis}, 0^\circ)$$
  $$\Phi_{a\text{VF}}(t) = \mathbf{c}_{a\text{VF}} \cdot \mathbf{p}(t) = p_y(t) \qquad (\text{vertical axis}, +90^\circ)$$
  Since $\mathbf{c}_I \perp \mathbf{c}_{a\text{VF}}$, the inner product is identically zero: $\mathbf{c}_I \cdot \mathbf{c}_{a\text{VF}} = 0$.
- **The Empirical Consequence**:
  In records where the patient has a **horizontal or leftward electrical axis** (e.g. `SR0234`, `AF0228`, `AFIB0293`), the vertical projection $p_y(t)$ is small or negatively oriented. Because the model only observes $p_x(t)$, it is mathematically impossible to know whether $p_y(t)$ is positive or negative. The neural network falls back on its learned population prior (which assumes a normal axis of $+60^\circ$), resulting in:
  - Lead $a\text{VF}$ in `SR0234`: $r = \mathbf{-0.429}$ (complete inverted polarity in a healthy sinus rhythm!).
  - Lead $a\text{VF}$ in `AFIB0293`: $r = \mathbf{-0.375}$.
  - Lead $a\text{VF}$ in `AF0228`: $r = \mathbf{-0.373}$.
- **Clinical Implication**: Inferior wall ischemia or myocardial infarction (diagnosed via ST-elevation or Q-waves in Leads II, III, $a\text{VF}$) **cannot be reliably diagnosed from Lead I alone**.

### Failure Mode 2: Unorganized Atrial Chaos & Fibrillatory Wave Dissipation (`AFIB`, `AF`)
- **Electrophysiological Law**:
  In Atrial Fibrillation, multiple reentrant wavelets depolarize the atria randomly at $350\text{--}600\text{ bpm}$, producing small, asynchronous fibrillatory ($f$) waves. In Atrial Flutter, macro-reentrant circuits around the tricuspid annulus generate continuous sawtooth $F$-waves that are maximal in inferior leads ($\text{II}, \text{III}, a\text{VF}$) and lead $V_1$.
- **The Empirical Consequence**:
  Lead I ($0^\circ$) has almost zero voltage projection of right atrial flutter loops. Furthermore, patch-based autoencoders with L1 reconstruction loss penalize high-frequency chaotic variation, acting as an effective **spatial low-pass filter**.
  - In `AFIB0293`, the model generates a clean, flat isoelectric baseline between QRS complexes, completely wiping out the diagnostic $f$-waves.
  - In `AF0228`, the regular $300\text{ bpm}$ flutter sawtooth in Leads II, III, and $a\text{VF}$ is completely missed, replacing it with a flat line.
- **Clinical Implication**: Reconstructed 12-lead ECGs risk **falsely classifying Atrial Fibrillation or Flutter as Normal Sinus Rhythm or junctional rhythm** because the pathognomonic baseline undulations are erased.

### Failure Mode 3: Rate-Dependent Conflation & P-on-T Wave Fusion (`ST`, `SVT`)
- **Electrophysiological Law**:
  As heart rate rises above $130\text{--}150\text{ bpm}$, the diastolic filling period ($T-P$ segment) approaches zero. The P-wave of the next beat depolarizes before the T-wave of the preceding beat has repolarized, creating "P-on-T" wave fusion.
- **The Empirical Consequence**:
  In `ST0030` and `VT0122`, the network cannot disentangle whether an observed deflection in Lead I represents ventricular repolarization (T-wave) or atrial depolarization (P-wave). The axial attention mechanism propagates this phase ambiguity across the leads, leading to:
  - In `VT0122` (SVT, $160\text{ bpm}$): **$r = -0.0277$** (the worst record in the entire database).
  - Precordial leads $V_1$ ($r = -0.525$) and $V_2$ ($r = -0.572$) completely invert their $R/S$ transition.
- **Clinical Implication**: In supraventricular tachycardias, reconstructed signals cannot be trusted to differentiate SVT with aberrancy from ventricular tachycardia, nor can they measure QT intervals reliably.

### Failure Mode 4: Precordial Conduction Decoupling in Paroxysmal SVT
- **Electrophysiological Law**:
  Precordial leads $V_1\text{--}V_6$ measure electrical potentials directly across the anterior-posterior ($z$) and lateral ($x$) axes. Lead I is purely horizontal on the coronal plane ($x$-axis) and contains negligible $z$-axis dipole information.
- **The Empirical Consequence**:
  In `VT0122`, the true physical ECG exhibits a prominent $R$-wave in $V_1$ (typical of right bundle branch block or ectopic origin). However, the model reconstructs a deep, QS-pattern in $V_1$ based on the frontal Lead I prior, completely reversing the precordial transition.

### Failure Mode 5: Architectural Invariance (Wavelets vs. Convolutions)
- **Critical Comparison**:
  Did our multi-scale analytic Morlet wavelet model with BYOL self-supervised learning (`conv15e_R7_morlet_mag_ueg_real`) solve these catastrophic failure cases?
  - On `VT0122` (SVT worst): `A0_raw` $r = \mathbf{-0.0277}$; `R7_wavelet` $r = \mathbf{-0.0118}$.
  - On `AF0228` (AF worst): `A0_raw` $r = \mathbf{0.2102}$; `R7_wavelet` $r = \mathbf{0.1630}$ (wavelets performed *worse*).
  - On `SR0234` (SR worst): `A0_raw` $r = \mathbf{0.2639}$; `R7_wavelet` $r = \mathbf{0.2669}$ (identical).
- **The Verdict**:
  The failure is governed by **Maxwell's equations and volume conduction biophysics**, not by the choice of neural network backbone. No deep learning model—whether convolutional, pure Transformer, or multi-scale Morlet wavelet—can extract information that is not present in the physical lead field.

---

## 4. The Complete 8-Rhythm Gallery

Below is the structured visual inspection of each rhythm class, complete with comparative overlays on authentic clinical ECG grids ($25\text{ mm/s}, 10\text{ mm/mV}$).

---

### 4.1 Atrial Flutter (AF)
- **Clinical Substrate**: Macro-reentrant atrial tachycardia characterized by continuous, rapid, regular atrial undulations (sawtooth $F$-waves) typically at $250\text{--}350\text{ bpm}$, usually with $2:1$ or $4:1$ AV conduction.
- **Observed Cohort Performance**: Mean $r = 0.6918$, range $[0.2102, 0.9104]$.

#### Tri-Way Failure Mode Diagnostic (`AF0228` — Worst Case)
![Tri-Way Diagnostic: Atrial Flutter Worst Case AF0228](figures/gallery/failure_diagnostic_AF_AF0228.png)

#### Clinical 12-Lead Overlays
<!-- slide -->
![AF Worst Case AF0228](figures/gallery/ecg_12lead_AF_worst_AF0228.png)
*Figure 4.1A: Atrial Flutter Worst Case (`AF0228`, Missing $r = 0.2102$). Note the catastrophic polarity inversion in Lead $a\text{VF}$ ($r = -0.373$) and Lead II ($r = -0.200$) while $V_1$ maintains $r = 0.549$.*
<!-- slide -->
![AF Median Case AF0071](figures/gallery/ecg_12lead_AF_median_AF0071.png)
*Figure 4.1B: Atrial Flutter Median Case (`AF0071`, Missing $r = 0.7195$). Precordial leads show solid morphology ($r = 0.818$), but the vertical axis ($a\text{VF}$) remains blunted ($r = 0.295$).*
<!-- slide -->
![AF Best Case AF0224](figures/gallery/ecg_12lead_AF_best_AF0224.png)
*Figure 4.1C: Atrial Flutter Best Case (`AF0224`, Missing $r = 0.9104$). Here the cardiac dipole aligned favorably with Lead I, enabling near-perfect reconstruction across both limb ($r = 0.938$) and precordial ($r = 0.887$) leads.*

---

### 4.2 Atrial Fibrillation (AFIB)
- **Clinical Substrate**: Completely chaotic, disorganized atrial activation with absent P-waves and irregularly irregular ventricular response. Fibrillatory waves are fine ($<0.5\text{ mm}$) to coarse ($>0.5\text{ mm}$).
- **Observed Cohort Performance**: Mean $r = 0.6904$, range $[0.3780, 0.8563]$.

#### Tri-Way Failure Mode Diagnostic (`AFIB0293` — Worst Case)
![Tri-Way Diagnostic: Atrial Fibrillation Worst Case AFIB0293](figures/gallery/failure_diagnostic_AFIB_AFIB0293.png)

#### Clinical 12-Lead Overlays
<!-- slide -->
![AFIB Worst Case AFIB0293](figures/gallery/ecg_12lead_AFIB_worst_AFIB0293.png)
*Figure 4.2A: Atrial Fibrillation Worst Case (`AFIB0293`, Missing $r = 0.3780$). Notice that fibrillatory baseline fluctuations are erased into an isoelectric line on Leads II and III, while Lead $a\text{VF}$ inverts to $r = -0.375$.*
<!-- slide -->
![AFIB Median Case AFIB0178](figures/gallery/ecg_12lead_AFIB_median_AFIB0178.png)
*Figure 4.2B: Atrial Fibrillation Median Case (`AFIB0178`, Missing $r = 0.7157$). Stable QRS tracking across all leads, but fine baseline fibrillation is smoothed away.*
<!-- slide -->
![AFIB Best Case AFIB0198](figures/gallery/ecg_12lead_AFIB_best_AFIB0198.png)
*Figure 4.2C: Atrial Fibrillation Best Case (`AFIB0198`, Missing $r = 0.8563$). High-amplitude precordial forces enable $V_1$ to achieve $r = 0.955$.*

---

### 4.3 Normal Sinus Rhythm (SR)
- **Clinical Substrate**: Normal cardiac impulse originating in the SA node with regular upright P-waves in Lead II and constant PR interval ($120\text{--}200\text{ ms}$).
- **Observed Cohort Performance**: Mean $r = 0.7755$, range $[0.2639, 0.9242]$.

#### Tri-Way Failure Mode Diagnostic (`SR0234` — Worst Case)
![Tri-Way Diagnostic: Normal Sinus Rhythm Worst Case SR0234](figures/gallery/failure_diagnostic_SR_SR0234.png)

#### Clinical 12-Lead Overlays
<!-- slide -->
![SR Worst Case SR0234](figures/gallery/ecg_12lead_SR_worst_SR0234.png)
*Figure 4.3A: Normal Sinus Rhythm Worst Case (`SR0234`, Missing $r = 0.2639$). A textbook demonstration of the orthogonal null space: Lead $V_1$ ($r = 0.929$) and $V_2$ ($r = 0.960$) are reconstructed with surgical perfection, while the vertical limb leads completely invert: Lead $a\text{VF}$ $r = -0.429$, Lead III $r = -0.564$, Lead II $r = 0.081$.*
<!-- slide -->
![SR Median Case SR0147](figures/gallery/ecg_12lead_SR_median_SR0147.png)
*Figure 4.3B: Normal Sinus Rhythm Median Case (`SR0147`, Missing $r = 0.7845$). Balanced reconstruction across the precordium ($r = 0.880$) with moderate limb lead agreement ($r = 0.670$).*
<!-- slide -->
![SR Best Case SR0025](figures/gallery/ecg_12lead_SR_best_SR0025.png)
*Figure 4.3C: Normal Sinus Rhythm Best Case (`SR0025`, Missing $r = 0.9242$). Near-perfect clinical replication across all 12 leads with correct P-wave, QRS, and T-wave polarity.*

---

### 4.4 Sinus Bradycardia (SB) & Sinus Arrhythmia (SA)
- **Clinical Substrate**: SA-node pacing at heart rates $< 60\text{ bpm}$ (Bradycardia) or with respiratory phasic variation in $R-R$ intervals (Arrhythmia).
- **Observed Cohort Performance**: SB Mean $r = 0.7801$ (max $0.9335$); SA Mean $r = 0.7967$ (max $0.9288$).

#### Tri-Way Failure Mode Diagnostics
<!-- slide -->
![Tri-Way Diagnostic: Sinus Bradycardia Worst Case SB0110](figures/gallery/failure_diagnostic_SB_SB0110.png)
*Figure 4.4A: Sinus Bradycardia Worst Case (`SB0110`, Missing $r = 0.4646$). Lead $a\text{VF}$ fails ($r = -0.036$), while $V_1$ succeeds ($r = 0.694$).*
<!-- slide -->
![Tri-Way Diagnostic: Sinus Arrhythmia Worst Case SI0012](figures/gallery/failure_diagnostic_SA_SI0012.png)
*Figure 4.4B: Sinus Arrhythmia Worst Case (`SI0012`, Missing $r = 0.5015$). Lead $a\text{VF}$ stalls at $r = 0.091$, while lateral precordial leads achieve $r = 0.836$.*

#### Selected Clinical Overlays
<!-- slide -->
![SB Best Case SB0192](figures/gallery/ecg_12lead_SB_best_SB0192.png)
*Figure 4.4C: Sinus Bradycardia Best Case (`SB0192`, Missing $r = 0.9335$). The highest-fidelity reconstruction in the entire bradycardia cohort.*
<!-- slide -->
![SA Best Case SI0238](figures/gallery/ecg_12lead_SA_best_SI0238.png)
*Figure 4.4D: Sinus Arrhythmia Best Case (`SI0238`, Missing $r = 0.9288$). Flawless tracking across shifting RR intervals.*

---

### 4.5 Sinus Tachycardia (ST)
- **Clinical Substrate**: Rapid normal sinus pacing ($> 100\text{ bpm}$), resulting in compression of the $T-P$ segment and frequent overlap between ventricular repolarization and subsequent atrial depolarization.
- **Observed Cohort Performance**: Mean $r = 0.7169$, range $[0.1639, 0.8845]$.

#### Tri-Way Failure Mode Diagnostic (`ST0030` — Worst Case)
![Tri-Way Diagnostic: Sinus Tachycardia Worst Case ST0030](figures/gallery/failure_diagnostic_ST_ST0030.png)

#### Clinical 12-Lead Overlays
<!-- slide -->
![ST Worst Case ST0030](figures/gallery/ecg_12lead_ST_worst_ST0030.png)
*Figure 4.5A: Sinus Tachycardia Worst Case (`ST0030`, Missing $r = 0.1639$). At high rate ($135\text{ bpm}$), the precordium decouples completely: $V_1$ drops to $r = 0.164$ and $V_5$ to $r = -0.313$. Interestingly, the multi-scale wavelet model `R7` was able to recover $r = 0.4601$ ($\Delta r = +0.2961$) by leveraging frequency band separation.*
<!-- slide -->
![ST Median Case ST0098](figures/gallery/ecg_12lead_ST_median_ST0098.png)
*Figure 4.5B: Sinus Tachycardia Median Case (`ST0098`, Missing $r = 0.7509$). Limb leads achieve $r = 0.932$, but precordial leads exhibit attenuation ($r = 0.600$).*
<!-- slide -->
![ST Best Case ST0080](figures/gallery/ecg_12lead_ST_best_ST0080.png)
*Figure 4.5C: Sinus Tachycardia Best Case (`ST0080`, Missing $r = 0.8845$). Excellent clinical reproduction across all leads despite elevated heart rate.*

---

### 4.6 Supraventricular Tachycardia (SVT)
- **Clinical Substrate**: Narrow-complex tachycardia arising above the bundle of His (e.g. AVNRT, AVRT) at rates typically $150\text{--}250\text{ bpm}$.
- **Observed Cohort Performance**: Mean $r = 0.6286$, range $[-0.0277, 0.8521]$.

#### Tri-Way Failure Mode Diagnostic (`VT0122` — Worst Case in Entire Benchmark)
![Tri-Way Diagnostic: SVT Worst Case VT0122](figures/gallery/failure_diagnostic_SVT_VT0122.png)

#### Clinical 12-Lead Overlays
<!-- slide -->
![SVT Worst Case VT0122](figures/gallery/ecg_12lead_SVT_worst_VT0122.png)
*Figure 4.6A: SVT Catastrophic Failure Case (`VT0122`, Missing $r = -0.0277$). Global failure across all axes: Lead $a\text{VF}$ $r = -0.187$, Lead II $r = -0.167$, Lead $V_1$ $r = -0.525$, Lead $V_2$ $r = -0.572$. Both `A0_raw` and `R7_wavelet` invert the polarity of the QRS complex because the tachycardia origin breaks standard population axis priors.*
<!-- slide -->
![SVT Median Case VT0095](figures/gallery/ecg_12lead_SVT_median_VT0095.png)
*Figure 4.6B: SVT Median Case (`VT0095`, Missing $r = 0.6878$). Precordial leads recover strongly ($r = 0.728$), with $V_1$ achieving $r = 0.931$.*
<!-- slide -->
![SVT Best Case VT0112](figures/gallery/ecg_12lead_SVT_best_VT0112.png)
*Figure 4.6C: SVT Best Case (`VT0112`, Missing $r = 0.8521$). High-fidelity reconstruction ($r = 0.852$) demonstrating that when ectopic conduction vectors align normally, the model successfully synthesizes 12-lead SVT tracings.*

---

### 4.7 Atrial Tachycardia (AT)
- **Clinical Substrate**: Ectopic atrial pacing from a non-sinus atrial focus at $100\text{--}250\text{ bpm}$, characterized by abnormal P-wave axis and morphology.
- **Observed Cohort Performance**: Mean $r = 0.7201$, range $[0.3772, 0.8920]$.

#### Tri-Way Failure Mode Diagnostic (`AT0025` — Worst Case)
![Tri-Way Diagnostic: Atrial Tachycardia Worst Case AT0025](figures/gallery/failure_diagnostic_AT_AT0025.png)

#### Clinical 12-Lead Overlays
<!-- slide -->
![AT Worst Case AT0025](figures/gallery/ecg_12lead_AT_worst_AT0025.png)
*Figure 4.7A: Atrial Tachycardia Worst Case (`AT0025`, Missing $r = 0.3772$). Low voltage amplitudes across limb leads cause moderate correlation loss ($r = 0.393$).*
<!-- slide -->
![AT Best Case AT0028](figures/gallery/ecg_12lead_AT_best_AT0028.png)
*Figure 4.7B: Atrial Tachycardia Best Case (`AT0028`, Missing $r = 0.8920$). Crisp ectopic P-wave and QRS reconstruction across all precordial leads ($r = 0.954$).*

---

## 5. Clinical Safety Recommendations & Engineering Safeguards

Based on our empirical and biophysical findings, we recommend the following strict safety protocols for any clinical single-lead reconstruction system:

1. **Mandatory Vertical Axis Gating**:
   Because Lead I carries zero mathematical constraint on the orthogonal vertical axis ($\mathbf{c}_I \perp \mathbf{c}_{a\text{VF}}$), the model's reconstructed Leads II, III, and $a\text{VF}$ must be marked with a prominent **"Biophysical Estimate — Axis Unconstrained"** clinical warning banner.
2. **Contraindication for Inferior STEMI & Ischemia Adjudication**:
   Single-lead reconstructed signals **must not be used to rule out or diagnose inferior ST-elevation myocardial infarction (STEMI)**. A true negative or positive in Lead I does not guarantee true morphology in $a\text{VF}$.
3. **Contraindication for Atrial Arrhythmia Classification**:
   Due to the low-pass filtering effect of autoencoders on chaotic fibrillatory wavelets ($f$-waves) and flutter waves, automated arrhythmia classifiers must never rely solely on synthesized missing leads to distinguish AFIB/AF from Sinus Rhythm.
4. **Rate-Dependent Uncertainty Flagging**:
   When detected heart rate exceeds $130\text{ bpm}$, the reconstruction confidence metric must be downgraded due to rate-dependent P-on-T wave fusion risks.
5. **Confidence Scoring via Latent Variance**:
   Deploying an ensemble or Bayesian dropout head to compute epistemic uncertainty on reconstructed leads allows the system to refuse output when the vertical dipole energy is unconstrained.
