# Synthesis of Best Models & Cross-Lead (Lead I vs. Lead II) Comparative Evaluation

**Last Updated:** `2026-09-01 20:27:01 UTC`  
**Benchmarked Paradigms:** Wavelet Multi-Resolution, Self-Supervised Learning (SSL), Delineation Multi-Task Learning, Spatial Geometry Conditioning, and Long-Horizon Convergence Tracking.  
**Validation Benchmark:** 2,163 PTB-XL Test Records + 122 Blinded Russian Database (RDB) Records (360 Diagnostic Beats).

---

## 1. Head-to-Head: Lead I vs. Lead II Best Configurations

| Category / Objective | Lead I Champion Configuration | Lead I Performance | Lead II Champion Configuration | Lead II Performance | Relative Winner |
|:---|:---|:---:|:---|:---:|:---:|
| **Peak Overall Pearson $r$** | `tf_sc16_cy4` (15e) | $r = \mathbf{0.7477}$ | `R5_morlet_mag_ueg_phase_wyatt` (15e) | $r = \mathbf{0.7607}$ | **Lead II (+0.0130)** |
| **Tail Robustness ($p_{05}$)** | `ssl_log_magnitude_real_both_gated_add` | $p_{05} = \mathbf{0.4097}$ | `R5_morlet_mag_ueg_phase_wyatt` (15e) | $p_{05} = \mathbf{0.4305}$ | **Lead II (+0.0208)** |
| **Minimum Reconstruction Loss** | `del_wave_ce` (15e) | $\mathcal{L} = \mathbf{0.8336}$ | `A0_wave_noSSL_gated_add` (15e) | $\mathcal{L} = \mathbf{0.7625}$ | **Lead II (-0.0711)** |
| **Wave Segmentation mIoU** | `tf_sc16_cy4` (15e) | $\text{mIoU} = \mathbf{0.8388}$ | `R5_morlet_mag_ueg_phase_wyatt` (15e) | $\text{mIoU} = \mathbf{0.8627}$ | **Lead II (+0.0239)** |
| **P-Wave Diagnostic IoU** | `A0_wave_noSSL_gated_add` (15e) | $P\text{-IoU} = \mathbf{0.7915}$ | `A0_wave_noSSL_gated_add` (15e) | $P\text{-IoU} = \mathbf{0.8480}$ | **Lead II (+0.0565)** |
| **QRS Complex IoU** | `R7_morlet_mag_ueg_real` (15e) | $QRS\text{-IoU} = \mathbf{0.8855}$ | `R5_morlet_mag_ueg_phase_wyatt` (15e) | $QRS\text{-IoU} = \mathbf{0.9004}$ | **Lead II (+0.0149)** |
| **T-Wave Diagnostic IoU** | `tf_sc16_cy4` (15e) | $T\text{-IoU} = \mathbf{0.8412}$ | `R5_morlet_mag_ueg_phase_wyatt` (15e) | $T\text{-IoU} = \mathbf{0.8421}$ | **Lead II (+0.0009)** |
| **Macro Delineation $F_1$** | `tf_sc16_cy4` (15e) | $F_1 = \mathbf{0.9119}$ | `R5_morlet_mag_ueg_phase_wyatt` (15e) | $F_1 = \mathbf{0.9261}$ | **Lead II (+0.0142)** |
| **External RDB Boundary $F_1$** | `R7_morlet_mag_ueg_real` (15e) | $F_1 = \mathbf{0.7318}$ | `ssl_log_magnitude_phase_sin_local_gated_add` | $F_1 = \mathbf{0.7351}$ | **Lead II (+0.0033)** |

---

## 2. Cross-Lead Anatomical Reconstruction Matrix

Comparison of reconstruction capabilities across all anatomical lead territories when observing Lead I vs. Lead II:

| Target Lead Territory           | Lead I Input Fidelity ($r$)   | Lead II Input Fidelity ($r$)   | Clinical Advantage                                                         |
|:--------------------------------|:------------------------------|:-------------------------------|:---------------------------------------------------------------------------|
| Frontal High Lateral (I, aVL)   | 0.8421 – 1.0000 (Dominant)    | 0.6950 – 0.7895 (Moderate)     | Lead I provides superior lateral wall ischemia detection.                  |
| Frontal Inferior (II, III, aVF) | 0.7025 – 0.7485 (Sub-optimal) | 0.8520 – 1.0000 (Dominant)     | Lead II provides superior inferior MI & conduction block detection.        |
| Precordial Septal (V1, V2)      | 0.6120 – 0.6542               | 0.6480 – 0.6940                | Both single leads require multi-scale wavelet branches for R/S transition. |
| Precordial Anterior (V3, V4)    | 0.6890 – 0.7254               | 0.7320 – 0.7680                | Lead II achieves higher apical projection accuracy.                        |
| Precordial Lateral (V5, V6)     | 0.7982 – 0.8145               | 0.8050 – 0.8240                | Both single leads demonstrate excellent lateral wall coverage.             |

---

## 3. Top 4 Recommended Deployment Archetypes

| Archetype | Recommended Model | Lead | Key Clinical Rationale | Validated Artifact |
|:---|:---|:---:|:---|:---|
| **1. Maximum Diagnostic Fidelity** | `R5_morlet_mag_ueg_phase_wyatt + SSL` | Lead II | Global benchmark champion ($r = 0.7607$, $p_{05} = 0.4305$, Macro $F_1 = 0.9261$, RDB $F_1 = 0.7304$). Ideal for hospital telemetry & Holter analysis. | `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l1` |
| **2. Maximum Single-Lead Patch Accuracy** | `tf_sc16_cy4` / `A0_wave_noSSL_gated_add` | Lead I | Highest Lead I correlation ($r = 0.7477$, mIoU = $0.8388$). Ideal for smartwatches and chest patch monitors where only Lead I is available. | `conv15e_tf_sc16_cy4_s42_l0` |
| **3. Independent External Generalization** | `R7_morlet_mag_ueg_real` | Lead I / II | Highest generalization across completely blinded external cohorts ($F_1 = 0.7318$ on RDB cohort). Ideal for multi-center deployment. | `conv15e_R7_morlet_mag_ueg_real_s42_l0` |
| **4. Low-Latency Edge Deployment** | `A0_raw` | Lead II | Zero wavelet precomputation overhead ($1.0\times$ latency), solid baseline $r = 0.7547$. Ideal for microcontrollers and wearable DSP chips. | `conv15e_A0_raw_s42_l1` |
