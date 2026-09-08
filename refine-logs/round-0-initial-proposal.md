# Research Proposal: Spatial-Dipole Multitask Foundation for Smartwatch Arrhythmia Surveillance (3DRECON-AF)

## Problem Anchor
- **Bottom-line problem**: How to achieve high-sensitivity, early detection of post-ablation atrial fibrillation recurrence and precise continuous AF burden quantification from daily single-lead smartwatch ECGs, while harvesting intermittent in-clinic paired recordings to reconstruct 12-lead electrophysiology and track repolarization dynamics without risking clinical trial failure.
- **Must-solve bottleneck**: Single-lead wearable ECGs (Lead I) suffer from severe spatial blindness (inability to resolve low-amplitude fibrillatory f-waves, atrial P-wave vector shifts, or precordial repolarization), while pure 12-lead reconstruction is mathematically ill-posed and exhibits amplitude compression in precordial leads ($r \approx 0.74$, MAE $\approx 0.13$\,mV on V3). Current models either treat AF detection as a black-box 1D classification (missing spatial vector dynamics) or treat 12-lead reconstruction as an isolated regression task (unusable as an acute regulatory endpoint).
- **Non-goals**: 
  - Claiming that single-lead smartwatch ECGs deterministically replace hospital 12-lead diagnostic carts in acute care.
  - Staking the clinical trial's primary efficacy endpoint on 12-lead reconstruction fidelity.
  - Adding extra clinic visits, blood draws, or invasive procedures beyond standard post-ablation care.
- **Constraints**: 
  - Clinical trial protocol fixed to Sunnybrook Version 1.1 ($N=96$ post-ablation persistent AF patients, 20-patient vanguard).
  - Primary trial endpoint is time to AF recurrence detection compared against 14-day continuous CardioSTAT™ patch monitors at 3 and 6 months.
  - Smartwatch hardware: single-lead Lead I (wrist-to-wrist circuit @ 500 Hz).
  - In-clinic paired recordings: $\le 5$ minutes during routine baseline and follow-up visits.
- **Success condition**: 
  1. The clinical trial succeeds independently on its primary endpoint (superior/non-inferior time-to-recurrence sensitivity $\ge 90\%$ vs. CardioSTAT patch).
  2. Auxiliary 12-lead spatial reconstruction forces the single-lead latent space to learn 3D cardiac dipole dynamics, yielding a statistically significant boost in early AF recurrence detection (higher AUC, lower false-positive rate) compared to single-task 1D baselines.
  3. The secondary paired 12-lead reconstruction recovers clinically actionable repolarization biomarkers (QTc MAE $\le 18$ ms, Conduction Delay AUROC $\ge 0.93$).

---

## Technical Gap
Recent work by Ansari et al. (*Circulation* 2026, 3DRECON-QT) established that learning to reconstruct 12-lead ECGs acts as a transformative inductive bias for single-lead cardiac AI: by forcing an encoder to condition on spatial lead angles $(\theta, \phi)$ to reconstruct the 12-lead ECG, the latent representation is forced to encode the underlying 3D cardiac vector. In Ansari et al., ablating this spatial reconstruction head caused QT interval prediction performance to collapse from Pearson $r = 0.73$ down to $0.04$.

However, 3DRECON-QT and existing spatial reconstruction methods suffer from three critical bottlenecks:
1. **Static Spherical Coordinate Heuristics**: 3DRECON-QT uses fixed scalar angle pairs $(\theta_k, \phi_k)$ based on a 2021 heuristic (Chen et al., IJCAI). In reality, human thoracic volume conduction is inhomogeneous, torso geometry varies widely, and cardiac electrical dipoles loop dynamically in 3D space ($\mathbf{p}(t) \in \mathbb{R}^3$).
2. **Single-Snapshot Scalar Limitation**: 3DRECON-QT was trained on static 10-second snapshots to predict a single scalar interval (QT). It cannot model time-varying rhythm transitions, paroxysmal fibrillatory bursts, or longitudinal disease progression over months.
3. **The Disconnected Clinical Trial Reality**: 3DRECON-QT relied on retrospective datasets (Philips iECG and MUSE-GE) and derived pseudo-ICM signals (V3–V2 differential). It lacked prospectively synchronized paired wearable-to-12L data and had no mechanism to integrate at-home daily surveillance with in-clinic ground-truth calibration.

---

## Method Thesis
- **One-sentence thesis**: By unifying a multi-scale wavelet-attention encoder with a continuous neural dipole observer field, we can multitask the primary clinical objective (longitudinal time-to-AF recurrence and AF burden surveillance) with an auxiliary spatial 12-lead reconstruction head, leveraging in-clinic paired recordings as patient-specific calibration anchors that prevent single-lead spatial blindness.
- **Why this is the smallest adequate intervention**: Instead of inventing complex separate models for AF detection, QTc tracking, and 12-lead reconstruction, a single shared spatiotemporal dipole backbone trains on both tasks simultaneously; the auxiliary 12-lead head regularizes the AF detection latent space at zero added inference cost at home.
- **Why this route is timely in the foundation-model era**: It adopts the modern multi-task representation learning paradigm (pretrain on massive public ECG foundation corpora $\to$ adapt via continuous neural fields $\to$ fine-tune via parameter-efficient low-rank adapters).

---

## Contribution Focus
- **Dominant contribution**: A spatiotemporally grounded multi-task architecture (**3DRECON-AF**) that couples a longitudinal survival/hazard head (Primary Trial Aim) with a neural-dipole 12-lead spatial reconstruction head (Secondary Aim), proving that auxiliary spatial reconstruction directly enhances single-lead AF recurrence sensitivity.
- **Supporting contribution**: A prospective tri-modal trial integration protocol (daily smartwatch at home + 14-day CardioSTAT gold standard + 5-minute paired in-clinic 12L calibration) with patient-specific baseline calibration (Model C).
- **Explicit non-contributions**: We do not claim deterministic zero-shot replacement of hospital 12-lead ECG carts in acute care.

---

## Proposed Method: 3DRECON-AF

### Complexity Budget
- **Frozen / reused foundation backbone**: Pretrained spatiotemporal transformer weights from public corpora (PTB-XL $N=21,837$, MIMIC-IV-ECG $N>800,000$, Icentia11k $N=11,000$).
- **New trainable components**:
  1. *Neural Dipole Observer Field $\mathcal{F}_\psi(\mathbf{c}_k)$*: A lightweight continuous coordinate MLP mapping 3D lead vector directions to projection weights.
  2. *Survival / Hazard Recurrence Head*: A discrete-time hazard survival transformer mapping daily latent trajectories to recurrence probability.
  3. *In-Clinic Low-Rank Adapters (LoRA)*: Patient-specific calibration vectors ($\text{rank}=4$) tuned on in-clinic paired recordings.
- **Tempting additions intentionally excluded**: No massive generative diffusion decoders (too compute-heavy and prone to hallucinated waveforms); no complex multi-sensor PPG fusion (stays pure ECG for regulatory clarity).

### System Overview
```
[At-Home Daily Smartwatch Lead I] (1x5000 @ 500 Hz)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Dual-Branch Spatiotemporal Encoder                       │
│    ├── 32-scale Continuous Morlet Wavelet Bank (0.5-45 Hz)  │
│    └── 50ms Temporal Patch Self-Attention (d=768)          │
│    └── Adaptive Gated-Add Fusion: Z_fused = Z_t + σ(W) ⊙ Z_w│
└──────────────────────────────┬──────────────────────────────┘
                               │ Latent Dipole Dynamics Z ∈ R^{T x D}
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────┐
│ PRIMARY CLINICAL HEAD (Aim 1) │ │ AUXILIARY SPATIAL HEAD (Aim 2)        │
│ At-Home Daily AF Surveillance │ │ In-Clinic 12L Spatial Reconstruction  │
├───────────────────────────────┤ ├───────────────────────────────────────┤
│ • Temporal Survival Decoder   │ │ • Neural Dipole Field F_ψ(c_k)        │
│ • Daily Cumulative AF Burden  │ │   conditioned on lead vector c_k      │
│   b_d ∈ [0, 1]                │ │ • Reconstructs 12 leads:              │
│ • Hazard h(t) for Recurrence  │ │   V_k(t) = Dec(Z, F_ψ(c_k))           │
│ • Supervised by 14-day        │ │ • Supervised by simultaneous in-clinic│
│   CardioSTAT patch (3 & 6 mo) │ │   hospital 12L ECG (L1 + morphology)  │
└───────────────────────────────┘ └───────────────────────────────────────┘
```

### Core Mechanisms
1. **Multi-Scale Wavelet-Attention Encoder**:
   - Single-lead wrist ECG $x(t) \in \mathbb{R}^{1 \times 5000}$ decomposes through a 32-scale Morlet wavelet bank, isolating 4–9 Hz atrial fibrillatory f-waves from rapid ventricular depolarization and low-frequency baseline wander.
   - Patch tokens ($50$ ms) pass through an 8-layer transformer encoder.
   - Gated-Add module dynamically injects high-frequency wavelet cues into the spatiotemporal tokens without baseline distortion:
     $$Z_{\text{fused}} = Z_{\text{time}} + \sigma(W [Z_{\text{time}}, Z_{\text{wave}}]) \odot Z_{\text{wave}}$$

2. **Continuous Neural Dipole Lead Observer Field**:
   - Rather than static spherical coordinate pairs $(\theta_k, \phi_k)$ as in Ansari et al., we parameterize each lead $k$ by its standard anatomical projection vector $\mathbf{c}_k \in \mathbb{R}^3$.
   - A continuous MLP field $\mathcal{F}_\psi(\mathbf{c}_k)$ maps $\mathbf{c}_k$ to spatial modulation weights:
     $$\hat{V}_k(t) = \text{Dec}\left(Z_{\text{fused}}, \mathcal{F}_\psi(\mathbf{c}_k)\right)$$
   - During in-clinic visits, when all 12 leads are simultaneously recorded, this head is trained via a combined loss:
     $$\mathcal{L}_{\text{recon}} = \sum_{k=1}^{12} \left( \| V_k - \hat{V}_k \|_1 + \lambda_{\text{corr}} (1 - \text{Pearson}(V_k, \hat{V}_k)) \right)$$

3. **Survival & AF Burden Head (Primary Trial Aim)**:
   - For daily at-home monitoring, latent embeddings across serial recordings are passed to a temporal pooling transformer.
   - Outputs two clinical quantities:
     a) **Daily AF Burden**: $\hat{b}_d \in [0, 1]$, trained against continuous CardioSTAT patch ground truth.
     b) **Recurrence Hazard**: Discrete-time hazard function $h(t) = P(T = t \mid T \ge t, \mathcal{Z}_{1:t})$, optimized via negative log-likelihood of survival time.

4. **Why Multitask Learning Solves the Clinical Bottleneck**:
   - At home, the watch only records Lead I. A standard 1D CNN often misses low-amplitude f-waves because it lacks spatial perspective.
   - Because the shared encoder $Z_{\text{fused}}$ is forced (during in-clinic training) to reconstruct all 12 leads including precordial leads V1–V6 (where P-waves and f-waves have maximum amplitude), the encoder learns an internal representation of the heart's 3D electrical dipole.
   - When evaluating daily watch ECGs at home, this spatially grounded latent space detects AF recurrence significantly earlier and with far higher specificity than single-task models.

---

## Claim-Driven Validation Sketch

### Claim 1: Auxiliary 12-Lead Spatial Reconstruction Directly Boosts Single-Lead AF Recurrence Detection
- **Minimal experiment**: Train 3DRECON-AF with the auxiliary spatial head versus an ablated single-task model (encoder trained solely on AF classification without the 12L reconstruction head).
- **Baselines**: Single-task ResNet-1D, standard TimeSformer, and native smartwatch irregular rhythm notification (IRR).
- **Evaluation metric**: Time-to-recurrence Hazard Ratio, Area Under ROC Curve (AUROC), and F1 score evaluated against the 14-day continuous CardioSTAT patch ground truth at 8 weeks post-ablation.
- **Expected directional outcome**: Multitask model achieves $\ge 5\%$ absolute higher AUROC (target $\ge 0.92$) and significantly higher sensitivity for paroxysmal AF episodes $\le 30$ seconds.

### Claim 2: In-Clinic Paired Data Calibrates Patient-Specific Transfer Functions
- **Minimal experiment**: Compare Model A (zero-shot public pretraining), Model B (fine-tuned on cohort paired data), and Model C (personalized with patient baseline in-clinic prior via rank-4 LoRA).
- **Metric**: Lead-specific MAE (mV), Pearson correlation $r$, and QTc prediction MAE (ms) on holdout follow-up clinic visits.
- **Expected directional outcome**: Model C reduces QTc MAE from $21.1$ ms to $\le 15.0$ ms and improves precordial lead V3 Pearson $r$ from $0.74$ to $>0.82$.

---

## Compute & Trial Timeline Estimate
- **Pretraining**: 48 GPU-hours on 1x A100 (public PTB-XL, MIMIC-IV, Icentia11k).
- **Trial Adaptation**: $<2$ GPU-hours per patient cohort update.
- **Clinical Timeline**:
  - Months 1–3: 20-patient vanguard feasibility cohort (operational adherence, dashboard data downloads).
  - Months 4–12: Full cohort recruitment ($N=96$), 3-month and 6-month paired in-clinic recordings, and 14-day CardioSTAT patch surveillance.
