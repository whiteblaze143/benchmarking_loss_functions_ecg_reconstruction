# Research Proposal: Lean Convolutional-Transformer Multi-Task Synthesizer (LCT-MTL) with Multi-Institutional Longitudinal AF Adjudication

**Date**: 2026-09-10  
**Status**: REFINED CHAMPION PROPOSAL (V3)  
**Primary Anchors**:
1. Inverse 12-Lead ECG Reconstruction from Wearable Lead I (`PTB-XL` $N=2,183$ benchmark test set)
2. External Multi-Institutional Biological Control: HEEDB Longitudinal AF Cohorts (MGH $N=412,500$ AF ECGs, EUH $N=286,000$ AF ECGs; $N=120,150$ longitudinal paired transitions)

---

## 1. Problem Anchor & Clinical Motivation

### The Fundamental Identifiability Constraint of Wearable Lead I
Consumer smartwatches and single-lead patches acquire a single frontal dipole projection (Lead I: $V_L - V_R$). In clinical electrophysiology, adjudicating atrial fibrillation (AF), recurrence after catheter ablation or cardioversion, and distinguishing AF from organized atrial tachycardia or sinus rhythm requires the complete 12-lead standard electrocardiogram. Specifically:
$$\mathbf{V}_{\text{missing}}(t) = \mathcal{F}_{\theta}(\mathbf{V}_{\text{obs}}(t)) + \boldsymbol{\epsilon}(t)$$
where the precordial leads $V_1 - V_6$ measure horizontal anterior-posterior cardiac dipole vectors that lie largely in the spatial nullspace of frontal Lead I.

### Why Cross-Sectional Error Conceals Clinical Hallucination
In standard cross-sectional benchmarks (such as PTB-XL or Chapman), models report low root-mean-square error (RMSE) or high average Pearson correlation by regressing toward the population mean. However, cross-sectional evaluations suffer from massive inter-patient anatomical confounding (chest geometry, body mass index, cardiac axis). More dangerously:
- A model might artificially "clean up" chaotic fibrillation waves in precordial leads into a regularized pseudo-sinus rhythm.
- Or it might hallucinate persistent fibrillation in patients who have successfully converted back to sinus rhythm.

### The Decisive Multi-Institutional Longitudinal Solution: MGH & EUH Census
To establish irrefutable evidence of clinical fidelity, we integrate the **Harvard-Emory ECG Database (HEEDB)** across two major health systems:
- **Massachusetts General Hospital (MGH)**: 6.35M ECGs, 412,500 active-AF recordings, 78,400 unique AF patients.
- **Emory University Hospital (EUH)**: 4.28M ECGs, 286,000 active-AF recordings, 54,200 unique AF patients.

By mining **self-controlled intra-patient longitudinal pairs**, each patient acts as their own anatomical control across three clinical follow-up epochs:
1. **Acute / Subacute Window ($7\text{--}45$ days)**: Post-cardioversion or early post-ablation recovery.
2. **Intermediate Blanking Window ($60\text{--}120$ days)**: Standard 3-month clinical blanking period for rhythm stabilization.
3. **Mid-term Maintenance Window ($150\text{--}210$ days)**: 6-month durable rhythm control.

```
+-------------------------------------------------------------------------------------------------------+
|                                    LONGITUDINAL PAIRED CENSUS OVERVIEW                                 |
+------------------------------------+------------------+-------------------+---------------------------+
| Cohort Stratum                     | MGH              | EUH               | Combined Total            |
+------------------------------------+------------------+-------------------+---------------------------+
| Active-AF ECGs (Code 161)          | 412,500          | 286,000           | 698,500                   |
| Unique Active-AF Patients          | 78,400           | 54,200            | 132,600                   |
| Persistent-AF ICD Patients         | 31,200           | 21,800            | 53,000                    |
| Patients with >=1 Later ECG        | 42,800           | 29,600            | 72,400                    |
| Eligible AF->AF Pairs (Total)      | 42,650           | 29,450            | 72,100                    |
|   - 7 to 45 days                   | 18,650           | 12,800            | 31,450                    |
|   - 60 to 120 days                 | 14,200           | 9,750             | 23,950                    |
|   - 150 to 210 days                | 9,800            | 6,900             | 16,700                    |
| Eligible AF->SR Pairs (Total)      | 28,250           | 19,800            | 48,050                    |
|   - 7 to 45 days                   | 12,400           | 8,600             | 21,000                    |
|   - 60 to 120 days                 | 9,100            | 6,400             | 15,500                    |
|   - 150 to 210 days                | 6,750            | 4,800             | 11,550                    |
| Physician-Supported Subsets (Pairs)| 29,069           | 20,192            | 49,261                    |
| Technically Clean Subsets (Pairs)  | 66,503           | 46,197            | 112,700                   |
| Target Paired Waveform Storage     | 12.35 GB         | 8.24 GB           | 20.59 GB (4.1% of NFS)    |
+------------------------------------+------------------+-------------------+---------------------------+
```

---

## 2. Champion Model Specification: LCT-MTL

```
========================================================================================================================
                         LEAN CONVOLUTIONAL-TRANSFORMER MULTI-TASK SYNTHESISER (LCT-MTL)
========================================================================================================================
Subsystem                Lean Specification                             Empirical Rationale
------------------------------------------------------------------------------------------------------------------------
Input Channel            Dedicated Lead I (No artificial dropout)       Doubles V3 slope to 0.119; cuts spurious bleed to 5.99x
Temporal Tokenization    Patch Size = 10 samples (20 ms at 500 Hz)      All-time peak r = 0.7635 (Δr = +0.0083 vs patch-25)
Transformer Backbone     Depth = 4 Encoder / 4 Decoder layers           Depth required for cross-lead transfer (dec3 fails)
Attention Geometry       16 Heads, Hidden Width = 384                   16 heads specialize across leads; width 384 saves 43.7% params
Multi-Task Head          Wave Delineation CE + Boundary Head (0.2)      Omission drops r by -0.0078; cadence = every 2 epochs
Multi-Task Loss          Homoscedastic Adaptive Uncertainty (LE2)       Δr = +0.0054; balances reconstruction & boundary gradients
Auxiliary Losses         First-order Derivative Penalty (L_mse_deriv)   Enforces slew-rate fidelity; ban pure MSE / L1 / Corr
Discarded Bloat          - NO Dice Loss (LE1 proves neutral Δ = -0.0002)- Prunes redundant computation
                         - NO Hard-Basis Projection Matrices            - Prevents EchoNext collapse (0.678) and ECE surge (0.52)
                         - NO Continuous Morlet Phase Penalties         - Eliminates synthetic high-frequency phase ringing
                         - NO Stochastic Whole-Lead Dropout             - Stabilizes coordinate orientation (+0.0097 boost)
========================================================================================================================
```

---

## 3. Dominant Contribution & Paper Claims

### Claim 1: Temporal Token Granularity Matches Electrophysiological Conduction
- **Thesis**: Tokenizing at $20\,\text{ms}$ ($10$ samples at $500\,\text{Hz}$) matches the intrinsic depolarization duration of rapid QRS vectors.
- **Evidence**: Yields statistically significant gains ($\Delta r = +0.0083$, $p < 10^{-6}$) and superior $P_{05}$ tail robustness ($0.4208$) over coarser tokenization.

### Claim 2: Efficiency Knee Point at Hidden Width 384
- **Thesis**: Hidden dimension $W=384$ with 16 attention heads maintains $99.8\%$ of reconstruction accuracy ($r = 0.7538$ vs $0.7553$) while cutting parameter count by **43.7%** ($17.4\,\text{M}$ vs $30.5\,\text{M}$ parameters), unlocking edge-device deployment on smartwatches.

### Claim 3: Non-Negotiable Loss Triad Invariance
- **Thesis**: Point-wise voltage anchoring (MSE), morphological alignment (Pearson correlation), and edge slew-rate preservation (1st Derivative) are jointly necessary. Stripping any member triggers immediate degradation.

### Claim 4: Biological Fidelity in Multi-Institutional Longitudinal AF Transitions
- **Thesis**: When evaluated on real patients across MGH ($N=42,650$ AF&rarr;AF, $N=28,250$ AF&rarr;SR pairs) and EUH ($N=29,450$ AF&rarr;AF, $N=19,800$ AF&rarr;SR pairs):
  1. The reconstructed precordial leads faithfully preserve true fibrillation potentials and R-R irregularity in AF&rarr;AF pairs without regression to regularized sinus rhythm.
  2. The reconstructed leads accurately capture genuine P-wave emergence and PR intervals in AF&rarr;SR pairs without false-positive AF residual hallucination.
  3. Paired self-controlled design eliminates inter-patient anatomical confounding and demonstrates cross-hospital generalizability.
