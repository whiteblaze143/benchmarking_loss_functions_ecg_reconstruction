# Research Proposal: Lean-CardioAIM — Parsimonious Multi-Task ECG Reconstruction via Biophysical Inductive Biases

**Date**: 2026-09-08  
**Status**: ACTIVE / REFINED  
**Primary Anchor**: Inverse 12-Lead ECG Reconstruction from Wearable Lead I (`PTB-XL`, $N=2,183$ validation split)

---

## 1. Problem Anchor & Theoretical Formulation

### The Inverse Problem
Wearable smartwatches and patch monitors acquire a single 1D electrical projection (nominally Lead I). In clinical cardiology, diagnostic decisions (STEMI, bundle branch block, long-QT syndrome, ischemia) demand standard 12-lead electrocardiography. The reconstruction of the missing 11 leads constitutes an ill-posed inverse volume-conductor problem:
$$\mathbf{V}_{\text{missing}}(t) = \mathcal{F}_{\theta}(\mathbf{V}_{\text{obs}}(t)) + \boldsymbol{\epsilon}(t)$$
where the precordial leads $V_1 - V_6$ occupy the anatomical anterior-posterior dipole projection that is mathematically in the nullspace of a planar Lead I measurement.

### The Occam's Razor Discovery
Prior work in deep ECG reconstruction deployed massive transformer backbones ($D_{\text{enc}}=8, W=768$, 12 attention heads, >45M parameters) combined with complex multi-loss formulations (MMD distribution alignment, VCG projections, whole-lead dropout). Our empirical Kill-Gate audit on $N=2,183$ records demonstrated:
1. **Capacity Overfitting**: The 8-layer/768-dim encoder memorized noise; pruning to a **4-layer / 512-dim encoder** runs **$\sim 1.5 - 2\times$ faster** while yielding $+0.005$ higher correlation.
2. **Harmful Dropout**: Whole-lead dropout breaks inter-lead spatial coherence; removing it unlocks **$+0.0105$ correlation** and $+0.0179$ tail $p_{05}$ robustness.
3. **Loss Objective Competition**: Combining high-order MMD kernels and VCG penalties creates gradient conflict. The parsimonious triplet loss ($\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{Pearson}} + \mathcal{L}_{\text{Deriv}}$) provides superior gradient signal.
4. **Indispensable Clinical Anchors**: Multi-task delineation heads (P, QRS, T wave segmentation) are non-negotiable; stripping them causes a $-0.0195$ drop in correlation.

---

## 2. The Refined Method: Lean-CardioAIM

`Lean-CardioAIM` embodies the minimalist architecture that maximizes clinical fidelity:

```
                                  ┌──> Multi-Task Delineation Head (P, QRS, T Masks)
                                  │      └──> Auxiliary Cross-Entropy + Dice Loss
Input (Lead I, Z-score) ──> 4-Layer Transformer ─┤
    (W=512, H=8)           Encoder-Decoder      └──> Reconstruction Decoder (D=4)
                                                       └──> 11 Reconstructed Leads
                                                                │
                                                                ▼
                                             Parsimonious Triplet Loss:
                                             L_total = L_MSE + L_Pearson + L_Deriv
```

### Architectural Specifications
- **Encoder Depth**: 4 layers (halved from 8)
- **Decoder Depth**: 4 layers (empirically proven floor; $D=2$ collapses precordial correlation)
- **Hidden Width**: 512 dimensions (8 attention heads)
- **Tokenization**: 25-sample temporal patches (25 ms at 1000 Hz / 50 ms at 500 Hz)
- **Normalization**: Per-sample Z-score standardization ($\mu=0, \sigma=1$)
- **Masking Protocol**: No artificial lead dropout during single-lead training; preserving spatial mapping paths.

---

## 3. The 44-Cell Extended Ablation Matrix

To definitively prove Occam's Razor across all architectural and biophysical dimensions, we execute a comprehensive 44-cell factorial study:
- **Phase 1 (Core Occam)**: Baseline Clean Lean Best (`L1`), No Delineation (`LA1`), Decoder $D=3$ (`LB1`), Width $W=384$ (`LC1`), Encoder $E=3$ (`LD1`), CE-only Delineation (`LE1`), Adaptive Loss (`LE2`).
- **Phase 2 (Loss Triplet Decomposition)**: Pure MSE (`L_mse_only`), Pure Pearson (`L_corr_only`), MSE+Corr (`L_mse_corr`), MSE+Deriv (`L_mse_deriv`), Pure L1 (`L_l1_direct`).
- **Phase 3 (Wavelet Exploration)**: Conv-128, Conv-256, Time-Frequency Transformer, Gated Additive Fusion, Cross-Attention Conditioning.
- **Phase 4 (Temporal Granularity)**: Patch 50, Patch 10, 4 Heads, 16 Heads.
- **Phase 5 (Multi-Task Delineation)**: Cadence 1, Cadence 4, Boundary Loss, Fiducial Loss, Heavy Segmentation, Light Segmentation.
- **Phase 6 (Spatial & Regularization)**: FiLM Conditioning, Panoramic Conditioning, Masking Augmentation, Weight Decay variations.
- **Phase 7 (Adaptive VCG & MMD Frontier)**:
  - `AV1_vcg_adaptive` (`1101000`): MSE + Pearson + Kors 3D VCG loop alignment.
  - `AV2_triplet_vcg_adaptive` (`1111000`): Full biophysical quad (MSE + Pearson + Deriv + VCG).
  - `AM1_mmd_imq_adaptive` (`1100003`): Anatomical block multiscale IMQ kernel MMD.
  - `AM2_mmd_kmeans_adaptive` (`1100004`): Dynamic temporal K-means clustered IMQ kernel MMD.
  - `AM3_mmd_laplace_adaptive` (`1100002`): Anatomical block Laplacian kernel MMD.
  - `AVM1_vcg_mmd_imq_adaptive` (`1101003`): Synergistic VCG + Anatomical IMQ MMD.
  - `AVM2_vcg_mmd_kmeans_adaptive` (`1101004`): Synergistic VCG + Temporal K-Means MMD.
  - `AVM3_vcg_mmd_laplace_adaptive` (`1101002`): Synergistic VCG + Anatomical Laplacian MMD.
  - `AVM4_full_probe_laplace_adaptive` (`1111002`): Full pentad (MSE + Pearson + Deriv + VCG + Anatomical Laplacian).
  - `AVM5_full_probe_imq_adaptive` (`1111003`): Full pentad (MSE + Pearson + Deriv + VCG + Anatomical IMQ).
  - `AVLead1_vcg_lead_adaptive` (`1101010`): 3D VCG loop + Goldberger limb lead consistency.

---

## 4. Key Claims to Validate
- **Claim 1 (Parsimony)**: The 4-layer/512-dim architecture matches or outperforms the 8-layer/768-dim model while reducing FLOPs by 42% and training wallclock by 35%.
- **Claim 2 (Spatial Preservation)**: Removing whole-lead dropout preserves lead projection geometry, yielding significant gains on difficult lateral ($V_5, V_6$) and inferior (Lead II) leads.
- **Claim 3 (Loss Coherence)**: Fixed heuristic weighting for high-order MMD/VCG causes objective competition and gradient interference.
- **Claim 4 (Clinical Grounding)**: Multi-task delineation heads anchor the representation to electrophysiological fiducials without requiring external foundation model distillation.
- **Claim 5 (Adaptive Geometric Reconcilement)**: Homoscedastic uncertainty weighting (`adaptive_composite`) dynamically scales log-variance penalties ($s_i$), allowing high-order 3D dipole loops (VCG) and multi-scale distribution alignment (MMD) to enhance anatomical reconstruction without destabilizing point-wise Pearson correlation.
