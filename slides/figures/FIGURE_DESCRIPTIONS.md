# Formal Figure Descriptions & Technical Specifications

**Study:** Discrete Latent Sequence Generation for 12-lead ECG Reconstruction from Limited Leads  
**Venue:** 2026 Carnegie Mellon Forum on Biomedical Engineering (Abstract #194)  
**Authors:** Mithun Manivannan, Alex Mariakakis, Christopher Cheung  
**Institutions:** University of Toronto & Sunnybrook Health Sciences Centre  

---

## 1. Formal Drawing Descriptions (FIG. 1 – FIG. 8)

### FIG. 1 — Anatomy of the ECG Cardiac Cycle & Morphological Delineation
`slides/figures/ecg_delineation_components.png` (3600 × 1860 px, 300 DPI)
**Description:** FIG. 1 is an electrophysiological schematic illustrating a single cardiac cycle on a standard medical grid (25 mm/s paper speed, 10 mm/mV voltage calibration), identifying critical fiducial intervals and morphological gates.
- **Reference Numerals (100-Series):**
  - **102**: Isoelectric baseline (0.0 mV reference voltage).
  - **104**: P-wave (atrial depolarization, duration 80–100 ms).
  - **106**: PR interval bracket (atrioventricular nodal conduction delay, 120–200 ms).
  - **108**: PR segment (isoelectric AV nodal delay).
  - **110**: QRS complex (ventricular depolarization, normal duration <100 ms).
  - **112**: Q-wave (septal activation, negative deflection).
  - **114**: R-peak (primary ventricular depolarization vector, +1.25 mV).
  - **116**: S-wave (basal ventricular activation, negative deflection).
  - **118**: J-point (junction between QRS termination and ST segment onset).
  - **120**: ST segment (early ventricular repolarization, ischemia/infarction gate).
  - **122**: T-wave (ventricular repolarization, asymmetric morphology).
  - **124**: QT / QTc interval bracket (total ventricular electrical systole, 360–440 ms).

### FIG. 2 — Cardiac Electrical Dipole Projections: 12-Lead vs. Ambulatory Wearables
`slides/figures/wearable_spatial_dipoles.png` (3600 × 1800 px, 300 DPI)
**Description:** FIG. 2 is a comparative structural block diagram illustrating cardiac electrical dipole projections across four monitoring modalities, delineating spatial basis coverage and mathematical observability.
- **Reference Numerals (200-Series):**
  - **202**: Hospital 12-lead standard block (10 physical electrodes capturing complete 3D basis: X, Y, Z).
  - **204**: Smartwatch Lead I block (2 dry metal wrist contacts capturing 1D transverse X-axis projection).
  - **206**: Multi-channel patch block (CardioSTAT / Zio XT capturing 2D planar frontal vectors X and Y).
  - **208**: 3-lead diagnostic Triad block (Leads I, II, and $V_2$ spanning mutually orthogonal 3D cardiac basis).

### FIG. 3 — Systematic 3-Architecture Benchmark Across a $2^4$ Loss Space
`slides/figures/architecture_overview.jpg` (1200 × 896 px, RGB)
**Description:** FIG. 3 is a comprehensive system architecture and benchmarking flowchart illustrating the 3-architecture evaluation framework across the 16-cell factorial loss space.
- **Reference Numerals (300-Series):**
  - **302**: Input acquisition module (sparse 12-lead canvas with 9 unobserved leads masked).
  - **304**: Deterministic U-Net model pathway (1D convolutional multi-channel autoencoder).
  - **306**: MultiScale-VAE model pathway (continuous multi-resolution latent bottleneck).
  - **308**: Discrete Latent ECG-AIM model pathway (patch tokenizer + continuous Morlet wavelet bank).
  - **310**: Factorial loss matrix module ($\mathcal{L}_{\text{MSE}}$, $\mathcal{L}_{\text{corr}}$, $\mathcal{L}_{\text{MMD}}$, $\mathcal{L}_{\Delta}$).
  - **312**: Einthoven Kirchhoff limb conservation constraint ($\text{Lead I} + \text{Lead III} = \text{Lead II}$).
  - **314**: Multi-tiered evaluation engine (signal correlation, fiducial delineation, downstream classifiers).

### FIG. 4 — Dual-Branch Wavelet-Transformer Architecture (ECG-AIM)
`slides/figures/conv15e_a0_wave_architecture.png` (1200 × 896 px, RGB)
**Description:** FIG. 4 is a detailed architectural diagram of the champion ECG-AIM model (`conv15e_A0_wave_noSSL_gated_add`), illustrating dual-branch feature extraction and spatiotemporal axial decoding.
- **Reference Numerals (400-Series):**
  - **402**: Limited lead input tensor ($1 \times 5000$ samples @ 500 Hz).
  - **404**: Patch tokenizer (patch size $P=25$, 50 ms duration, embedding dimension $d=768$).
  - **406**: Temporal Transformer encoder (8 layers, 12 attention heads, learned lead embeddings).
  - **408**: Continuous Morlet wavelet filter bank (32 scales spanning 0.5 to 45.0 Hz).
  - **410**: Spectral feature encoder (TimeSformer + 1D-CNN, dimension $d_w=192$).
  - **412**: Gated-Add fusion module ($Z_{\text{fused}} = Z_{\text{time}} + \sigma(W[Z_t, Z_w]) \odot \text{Proj}(Z_w)$).
  - **414**: Axial Transformer decoder (4 layers, spatiotemporal cross-attention).
  - **416**: Continuous 12-lead waveform reconstruction head ($12 \times 5000$ samples).
  - **418**: Multi-task delineation head (P, QRS, T boundary probability maps).

### FIG. 5 — Multi-Task Training Pipeline & Physical Prior Supervision
`slides/figures/ecg_aim_training_pipeline.png` (1376 × 768 px, RGB)
**Description:** FIG. 5 is a pipeline diagram illustrating the end-to-end training procedure, dynamic lead masking, and physical loss formulation.
- **Reference Numerals (500-Series):**
  - **502**: Retrospective hospital training archive (PTB-XL and MIMIC-IV, $N=175,890$).
  - **504**: Dynamic lead masking operator ($M_{\text{obs}}$ simulating variable lead inputs).
  - **506**: Dual-branch backbone network.
  - **508**: Combinatorial waveform loss function (MSE, Pearson $r$, Wyatt ST-T phase loss).
  - **510**: Kirchhoff limb conservation penalty ($\|\hat{y}_{\text{II}} - (\hat{y}_{\text{I}} + \hat{y}_{\text{III}})\|_2^2$).
  - **512**: Delineation cross-entropy and Dice segmentation loss.
  - **514**: The Hardware Gap callout (synthetic supine gel records vs. ambulatory dry-electrode reality).

### FIG. 6 — Champion 3-Lead Diagnostic Triad Reconstruction
`slides/figures/best_ecg_aim_12lead_reconstruction.png` (4800 × 2550 px, RGBA)
**Description:** FIG. 6 is an authentic high-resolution multi-lead medical waveform plot comparing ground truth (solid black line) against ECG-AIM 3-lead reconstruction (dashed blue line) across all 12 clinical leads on held-out PTB-XL record #10283.
- **Reference Numerals (600-Series):**
  - **602**: Observed Lead I (transverse X-axis anchor, $r=1.000$).
  - **604**: Observed Lead II (vertical Y-axis anchor, $r=1.000$).
  - **606**: Derived frontal leads (III, aVR, aVL, aVF) verified via Kirchhoff conservation ($r=1.000$).
  - **608**: Observed precordial Lead $V_2$ (anterior Z-axis anchor, $r=1.000$).
  - **610**: Reconstructed anterior chest leads ($V_3, V_4, V_5, V_6$) demonstrating near-lossless ST-segment fidelity ($r=0.988 - 0.995$).

### FIG. 7 — Single-Lead Wearable Bounds on Smartwatch Lead I
`slides/figures/best_ecg_aim_lead1_reconstruction.png` (4800 × 2550 px, RGBA)
**Description:** FIG. 7 is a multi-lead comparative plot illustrating 12-lead reconstruction conditioned solely on single-lead smartwatch wrist input (Lead I).
- **Reference Numerals (700-Series):**
  - **702**: Single observed wrist vector (Lead I).
  - **704**: Anterior precordial reconstruction ($V_1$–$V_3$, mean $r=0.776$) capturing horizontal chest vectors.
  - **706**: Inferior limb leads (II, III, aVF) illustrating spatial ambiguity and physiological limits of 1D inverse mapping.

### FIG. 8 — Zero-Shot Hospital Cart Transfer & Analog Noise Floor
`slides/figures/sunnybrook_best_reconstruction_3lead.png` (4800 × 2550 px, RGBA)
**Description:** FIG. 8 is a waveform validation plot showing zero-shot evaluation on raw physical 10-wire XML cart data from Sunnybrook Health Sciences Centre (Record ECG004).
- **Reference Numerals (800-Series):**
  - **802**: Authentic high clinical voltages ($>2.1$ mV in $V_2$).
  - **804**: Preservation of true analog hardware noise floor ($18\,\mu$V).
  - **806**: Unobserved precordial leads ($V_3$–$V_6$) matching diagnostic hospital recording morphology.

---

## 2. Reference Numeral Index

| Numeral | Component Name | Figure | Clinical / Technical Function |
|:-------:|:---------------|:------:|:------------------------------|
| **104** | P-Wave | FIG. 1 | Atrial depolarization (80–100 ms) |
| **106** | PR Interval | FIG. 1 | AV nodal conduction delay (120–200 ms) |
| **110** | QRS Complex | FIG. 1 | Rapid ventricular depolarization (<100 ms) |
| **118** | J-Point | FIG. 1 | ST junction measuring acute ischemia |
| **120** | ST Segment | FIG. 1 | Early repolarization gate (STEMI marker) |
| **124** | QT/QTc Interval | FIG. 1 | Ventricular electrical systole (arrhythmia gate) |
| **202** | 12-Lead Standard | FIG. 2 | Full 3D cardiac vector basis (X, Y, Z) |
| **204** | Smartwatch Lead I | FIG. 2 | 1D transverse wrist projection |
| **206** | Multi-Channel Patch | FIG. 2 | 2D planar frontal vector coverage |
| **208** | 3-Lead Triad | FIG. 2 | Orthogonal 3-axis clinical dipole basis |
| **308** | Discrete ECG-AIM | FIG. 3 | Discrete latent sequence generator |
| **312** | Kirchhoff Limb Law | FIG. 3 | Enforces Lead I + Lead III = Lead II |
| **404** | Patch Tokenizer | FIG. 4 | 50 ms temporal patch encoder ($d=768$) |
| **408** | Morlet Wavelet Bank| FIG. 4 | 32 scales (0.5–45 Hz) analytic scalogram |
| **412** | Gated-Add Fusion | FIG. 4 | Adaptive sigmoid temporal-spectral gate |
| **414** | Axial Decoder | FIG. 4 | Spatiotemporal cross-attention decoder |
| **504** | Dynamic Masking | FIG. 5 | Variable input subset training operator |
| **510** | Limb Loss Penalty | FIG. 5 | Frontal plane vector consistency loss |
| **608** | Lead $V_2$ Input | FIG. 6 | Anterior chest dipole anchor |
| **610** | Precordial Recovery| FIG. 6 | Near-lossless chest reconstruction ($r > 0.98$) |
| **704** | 1-Lead Anterior | FIG. 7 | Transverse vector tracking chest leads |
| **802** | Hospital Cart XML | FIG. 8 | Diagnostic voltages ($>2.1$ mV) preserved |
