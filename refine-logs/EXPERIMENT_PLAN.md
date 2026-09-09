# Experiment Plan: Occam's Razor 33-Cell Lean Ablation Suite

**Date**: 2026-09-08  
**Execution Environment**: Local GPU (NVIDIA A100-PCIE-40GB)  
**Dataset**: PTB-XL ($N=2,183$ validation split, 10-second 12-lead ECGs, 500 Hz)  
**Evaluation Standard**: Paired record-level bootstrap ($B=10,000$, 95% Confidence Intervals, non-inferiority margin $\delta=-0.005$)  
**Ledger Output**: `refine-logs/lean_abl2/summary_ledger.json`

---

## 1. Experiment Grid Overview

The study comprises **44 systematically designed cells**:
- **Phase 1: Lean Core & Occam Simplification (Cells 1–7)**
- **Phase 2: Loss Triplet Decomposition (Cells 8–12)**
- **Phase 3: Wavelet Multi-Spectral Variations on Lean Core (Cells 13–17)**
- **Phase 4: Temporal Patching & Attention Subspaces (Cells 18–21)**
- **Phase 5: Multi-Task Clinical Delineation Cadence & Weighting (Cells 22–27)**
- **Phase 6: Spatial Conditioning & Regularization Robustness (Cells 28–33)**
- **Phase 7: Adaptive VCG & MMD Frontier Suite (Cells 34–44)**

---

## 2. Detailed Cell Definitions

### Phase 1: Core Lean Best & Occam Pruning (Cells 1–7)
| # | Cell ID | Configuration | Question / Hypothesis | Anchor |
|---|:---|:---|:---|:---:|
| 1 | `L1_clean_lean_best` | Enc-4, Width-512, Heads-8, Dec-4, No Dropout, Z-Score, Mask `1110000` | Anchor for lean model: does clean combined model beat A0? | `A0_raw` |
| 2 | `LA1_nodel` | Prune delineation head (`--no-delineation-head`, CE=0, Dice=0) | Is delineation necessary on 512-dim model? | `L1` |
| 3 | `LB1_dec3` | Prune decoder depth to $D=3$ | Can decoder be pruned to 3 layers without collapse? | `L1` |
| 4 | `LC1_width384` | Prune width to $W=384$, heads to $H=6$ | Can capacity be reduced below 512 dimensions? | `L1` |
| 5 | `LD1_enc3` | Prune encoder depth to $E=3$ | Can encoder be pruned below 4 layers? | `L1` |
| 6 | `LE1_nodice` | Delineation with CE only (Dice weight = 0.0) | Is Dice loss essential for delineation guidance? | `L1` |
| 7 | `LE2_adaptive` | Adaptive composite loss weighting (`--reconstruction-loss-type adaptive_composite`) | Does homoscedastic uncertainty weighting improve lean model? | `L1` |

### Phase 2: Loss Triplet Decomposition (Cells 8–12)
| # | Cell ID | Configuration | Question / Hypothesis | Anchor |
|---|:---|:---|:---|:---:|
| 8 | `L_mse_only` | Mask `1000000` (Pure MSE, no Pearson, no derivative) | Quantify correlation drop without Pearson loss | `L1` |
| 9 | `L_corr_only` | Mask `0100000` (Pure Pearson, no MSE, no derivative) | Can Pearson loss train alone without MSE scale anchor? | `L1` |
| 10 | `L_mse_corr` | Mask `1100000` (MSE + Pearson, no first derivative) | Does the first-derivative penalty contribute to QRS sharpness? | `L1` |
| 11 | `L_mse_deriv` | Mask `1010000` (MSE + Derivative, no Pearson) | Can derivative penalty substitute for Pearson correlation? | `L1` |
| 12 | `L_l1_direct` | `--reconstruction-loss-type l1` | Does L1 loss suffer from variance collapse on lean model? | `L1` |

### Phase 3: Wavelet Multi-Spectral Variations on Lean Core (Cells 13–17)
| # | Cell ID | Configuration | Question / Hypothesis | Anchor |
|---|:---|:---|:---|:---:|
| 13 | `W_conv_c128` | Wavelet branch with 1D-CNN (hidden 128) | Does a lightweight 1D-CNN wavelet stream provide T-wave gain? | `L1` |
| 14 | `W_conv_c256` | Wavelet branch with 1D-CNN (hidden 256) | Standard 1D-CNN Morlet representation capacity | `L1` |
| 15 | `W_timesformer_dim256` | Compact Time-Frequency Transformer ($D=2, W=256$) | Transformer vs CNN for time-frequency processing | `L1` |
| 16 | `W_fusion_gated` | Gated residual feature addition fusion | Does gated residual fusion prevent wavelet overfitting? | `L1` |
| 17 | `W_fusion_crossattn` | Cross-attention wavelet conditioning (4 heads) | Cross-attention vs concatenation for multi-spectral fusion | `L1` |

### Phase 4: Temporal Patching & Attention Subspaces (Cells 18–21)
| # | Cell ID | Configuration | Question / Hypothesis | Anchor |
|---|:---|:---|:---|:---:|
| 18 | `T_patch50` | Temporal patch size = 50 samples | Can 2x fewer tokens train faster with comparable accuracy? | `L1` |
| 19 | `T_patch10` | Temporal patch size = 10 samples | Does finer patch resolution improve steep QRS peaks? | `L1` |
| 20 | `T_heads4` | 4 Attention Heads | Does halving attention heads degrade multi-channel subspace mapping? | `L1` |
| 21 | `T_heads16` | 16 Attention Heads | Does higher head count improve cross-lead spatial resolution? | `L1` |

### Phase 5: Multi-Task Clinical Delineation Cadence & Weighting (Cells 22–27)
| # | Cell ID | Configuration | Question / Hypothesis | Anchor |
|---|:---|:---|:---|:---:|
| 22 | `D_cadence1` | Supervise delineation every 1 epoch | Does continuous segmentation guidance boost reconstruction? | `L1` |
| 23 | `D_cadence4` | Supervise delineation every 4 epochs | Minimum delineation supervision required to prevent collapse | `L1` |
| 24 | `D_boundary` | Morphological wave boundary transition penalty (weight 0.2) | Does boundary localization improve onset/offset precision? | `L1` |
| 25 | `D_fiducial` | Peak fiducial point alignment penalty (weight 0.2) | Does fiducial alignment prevent R-peak time jitter? | `L1` |
| 26 | `D_heavy_seg` | Heavy segmentation weight (CE=2.0, Dice=1.0) | Does heavier segmentation penalty overwhelm reconstruction? | `L1` |
| 27 | `D_light_ce` | Soft segmentation weight (CE=0.5, Dice=0.25) | What is the lower bound for effective delineation supervision? | `L1` |

### Phase 6: Spatial Conditioning & Regularization Robustness (Cells 28–33)
| # | Cell ID | Configuration | Question / Hypothesis | Anchor |
|---|:---|:---|:---|:---:|
| 28 | `S_film` | Feature-wise Linear Modulation (FiLM) for lead projection | Does FiLM conditioning outperform additive lead embeddings? | `L1` |
| 29 | `S_panorama` | Panoramic lead geometry spherical embeddings | Do spherical angle coordinates improve precordial transfer? | `L1` |
| 30 | `R_mask15` | 15% random point masking augmentation | Does random point masking improve noise robustness? | `L1` |
| 31 | `R_tempmask15` | 15% contiguous temporal masking augmentation | Does chunk masking encourage temporal context interpolation? | `L1` |
| 32 | `R_wd_low` | Weight decay = $10^{-5}$ | Sensitivity to L2 weight regularization | `L1` |
| 33 | `R_wd_high` | Weight decay = $10^{-3}$ | Impact of stronger L2 penalty on 512-dim model | `L1` |

### Phase 7: Adaptive VCG & MMD Frontier Suite (Cells 34–44)
*All cells evaluate on Clean Lean Core (enc4, dec4, width512, heads8, no_lead_dropout, zscore) with `--reconstruction-loss-type adaptive_composite`.*
| # | Cell ID | Configuration | Mask | Question / Historical Grounding | Anchor |
|---|:---|:---|:---:|:---|:---:|
| 34 | `AV1_vcg_adaptive` | MSE + Pearson + 3D VCG loop | `1101000` | Historical benchmark: $r=0.8026$, AUROC=0.8588. Tests dipole geometry without gradient clash. | `L1` |
| 35 | `AV2_triplet_vcg_adaptive` | MSE + Pearson + Deriv + VCG | `1111000` | Full biophysical quad under dynamic uncertainty weighting. | `L1` |
| 36 | `AM1_mmd_imq_adaptive` | MSE + Pearson + Anatomical IMQ MMD | `1100003` | Historical benchmark: $r=0.8598$, AUROC=0.8594 (#1 correlation winner). Multi-scale kernel distribution matching. | `L1` |
| 37 | `AM2_mmd_kmeans_adaptive` | MSE + Pearson + Temporal K-Means MMD | `1100004` | Historical benchmark: $r=0.8594$, AUROC=0.8593 (#2 correlation winner). Dynamic wave phase clustering. | `L1` |
| 38 | `AM3_mmd_laplace_adaptive` | MSE + Pearson + Anatomical Laplacian MMD | `1100002` | Historical benchmark: $r=0.8340$, AUROC=0.8483. Heavy-tailed anatomical plane matching. | `L1` |
| 39 | `AVM1_vcg_mmd_imq_adaptive` | MSE + Pearson + VCG + IMQ MMD | `1101003` | Historical benchmark: $r=0.7990$, AUROC=0.8582. Joint 3D loop + anatomical distribution matching. | `L1` |
| 40 | `AVM2_vcg_mmd_kmeans_adaptive` | MSE + Pearson + VCG + K-Means MMD | `1101004` | Historical benchmark: $r=0.7968$, AUROC=0.8573. Joint 3D loop + temporal phase matching. | `L1` |
| 41 | `AVM3_vcg_mmd_laplace_adaptive` | MSE + Pearson + VCG + Laplacian MMD | `1101002` | Historical benchmark: $r=0.7804$, AUROC=0.8488. Joint 3D loop + Laplacian regularizer. | `L1` |
| 42 | `AVM4_full_probe_laplace_adaptive` | Full Pentad (MSE+Corr+Deriv+VCG+Laplacian) | `1111002` | Canonical 1111002 probe with adaptive precision to prevent collapse. | `L1` |
| 43 | `AVM5_full_probe_imq_adaptive` | Full Pentad (MSE+Corr+Deriv+VCG+IMQ) | `1111003` | Full 5-objective composite with multi-scale IMQ kernel. | `L1` |
| 44 | `AVLead1_vcg_lead_adaptive` | MSE + Pearson + VCG + Goldberger Lead | `1101010` | Historical benchmark: $r=0.7803$, AUROC=0.8572. Physical limb loop constraint. | `L1` |

---

## 3. Decision Gates & Stopping Criteria
- **Gate 1 (Non-Inferiority)**: A cell passes if $p_{\text{inferior}} < 0.05$ with $\Delta_{\text{mean}} \ge -0.005$ on Missing-11 Pearson correlation.
- **Gate 2 (Tail Robustness)**: The 5th percentile correlation ($p_{05}$) must not drop by more than $\Delta_{p05} = -0.010$.
- **Gate 3 (Precordial Integrity)**: Precordial $V_1 - V_6$ correlation must satisfy non-inferiority independently.
- **Gate 4 (Adaptive Dominance)**: If any Phase 7 model outperforms `LE2_adaptive` ($r > 0.7607$), it establishes that biophysical higher-order geometry *complements* parsimonious architecture once gradient scale interference is solved.
