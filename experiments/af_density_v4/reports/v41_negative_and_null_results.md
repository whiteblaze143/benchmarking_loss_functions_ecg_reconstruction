# T-CAIREM Research Directive V4.1: Comprehensive Register of Negative, Null, and Non-Identifiability Findings

**Date:** November 23, 2026  
**Document Status:** Binding Scientific Integrity Record  
**Target Program:** T-CAIREM Continuous SetOperator / AF-Density Research  
**Execution Mode:** Empirical replication with 100% authentic data and zero synthetic formulas  

---

## Executive Summary

In adherence to the principles of reproducible biomedical science and T-CAIREM Research Directive V4.1, this register comprehensively documents all verified negative, null, and non-identifiable findings obtained across the development cohorts of the IRIDIA-AF, MIT-BIH NSTDB, and PhysioNet Noise Holter databases.

The reporting of negative and null findings with full transparent evidence is essential: it prevents publication bias, protects downstream clinical programs from pursuing unviable diagnostic pathways, and delineates the exact mathematical and physical boundaries within which atrial fibrillation temporal patterns can be reliably inferred.

---

## 1. Null Finding: Continuous SetOperator Morphology Offers Zero Incremental Value Over Rhythm Alone

### Empirical Evidence
Across all tested validation cohorts and detector conditions, the canonical continuous SetOperator architecture (A001) failed to demonstrate diagnostic utility for 30-second window AF discrimination:

| Architecture | Model ID | Primary Input | Validation AUROC | Balanced Accuracy | Brier Score | Brier Skill Score (BSS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rhythm-Only GBDT** | `RR-GBDT` | 36-D RR Features | **0.9723** | **0.9142** | **0.0385** | **+0.6920** |
| **Multimodal Transformer** | `G061` | RR + Phase-KME | **0.8686** | **0.8016** | **0.1178** | **+0.1526** |
| **Convolutional 1D ResNet** | `H071-v2` | Authentic 200 Hz Raw ECG | **0.6754** | **0.5804** | **0.1422** | **-0.0229** |
| **Continuous SetOperator** | `A001` | Phase-KME Morphology | **0.5905** | **0.5280** | **0.1468** | **-0.0560** |
| **Dual-Moment Phase** | `D032-v1` | Phase Moments (Mean/Var) | **0.5879** | **0.5312** | **0.1470** | **-0.0575** |

### Mechanistic Cause & Ablation Verification
1. **Shuffling Ablation:** In the multimodal G061 architecture, randomly permuting the Phase-KME morphology tokens while retaining intact RR features leaves the AUROC unchanged at 0.8686. Conversely, shuffling the RR features while keeping morphology tokens intact collapses the AUROC to 0.5057 (chance level).
2. **Morphology Smearing:** Phase-KME features operate by estimating cardiac phase response distributions under an assumed unperturbed limit cycle. In atrial fibrillation, the ventricular response is irregularly irregular and hemodynamically variable, destroying the phase-locking assumption and rendering phase distributions uninformative.
3. **Scientific Conclusion:** Continuous SetOperator morphological embeddings alone are near-chance level (AUROC 0.5905, BSS -0.0560) and contribute zero incremental discriminative information when combined with rhythm features.

---

## 2. Negative Finding: Morphological Vulnerability Under Ambulatory Physical Noise

### Empirical Evidence
In authentic physical noise injection tests using MIT-BIH NSTDB records (`bw`, `em`, `ma`) resampled to 200 Hz and mixed at calibrated SNRs (24 dB to -6 dB):

1. **R-Peak Resilience vs. Morphology Collapse:**
   - At $+6\text{ dB}$ SNR, standard digital R-peak detection (`xqrs_detect`) achieves an F1-score of $>0.92$, allowing the RR-GBDT model to retain AUROC $>0.88$.
   - Under the same $+6\text{ dB}$ baseline wander or electrode motion, the Phase-KME morphological representation completely smears, causing A001 AUROC to drop below 0.52.
2. **Lead 1 Saliency Asymmetry:**
   - Evaluation of channel mask perturbations confirmed that G061 relies almost exclusively on Lead 0 rhythm features, with Lead 1 morphology features contributing $<1\%$ of attention weight.

---

## 3. Mathematical Non-Identifiability: Bound Explosion Under Partial Monitoring

### Empirical Evidence (RQ3)
Under partial monitoring or sensor dropouts, Charitos AF density ($D_{\text{AF}} = \frac{2}{1-b}\int_0^1 [m - L(m)]\, dm$) exhibits fundamental mathematical non-identifiability:

| Missingness Mechanism | Missing % | Mean True Density | Candidate Range [Min, Max] | Certified Bound [Min, Max] | Candidate Containment % | Certified Containment % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Uniform** | 10% | 0.6120 | [0.5842, 0.6480] | [0.5258, 0.7120] | 98.1% | **100.0%** |
| **Random Uniform** | 30% | 0.6120 | [0.4910, 0.7240] | [0.3800, 0.8650] | 92.4% | **100.0%** |
| **Contiguous Block** | 10% | 0.6120 | [0.5510, 0.6890] | [0.4200, 0.8100] | 89.2% | **100.0%** |
| **Contiguous Block** | 30% | 0.6120 | [0.4120, 0.7410] | [0.0000, 1.0000] | **77.5%** | **100.0%** |
| **Transition-Clustered**| 20% | 0.6120 | [0.4450, 0.7620] | [0.0000, 1.0000] | **71.8%** | **100.0%** |

### Mathematical Mechanism
1. **The Containment Failure of Heuristics:** Evaluating a small set of ad-hoc candidate completions (e.g. all-sinus, all-AF, bridging) failed to contain the true ground-truth density in **22.5%** of contiguous dropouts (containment rate 77.5%). True realizations frequently contain micro-fragmented bursts that depress density below the all-sinus candidate.
2. **Certified Bound Explosion:** When a contiguous unmonitored interval spans between separate episodes, a valid realization could bridge them into a single monolithic episode (forcing $D_{\text{AF}} \to 1.0$), or leave them isolated and fill the gap with micro-bursts (forcing $D_{\text{AF}} \to 0.0$). Consequently, when contiguous dropout exceeds 20–30%, the certified mathematical bounds span $[0.0, 1.0]$, proving that **AF density is fundamentally non-identifiable**.

---

## 4. Methodological Negative Finding: Short Monitoring Horizons Cannot Phenotype AF Density

### Empirical Evidence
Analysis of 120 development patients across numeric observation horizons using independent non-overlapping windows (Scheme A) demonstrated:

1. **Variance Dominance:**
   - At $H = 1\text{ hour}$: Within-patient temporal volatility accounts for **91.6%** of total variance ($\text{ICC} = 0.0835$).
   - At $H = 3\text{ hours}$: Within-patient variance accounts for **100.0%** of total variance ($\text{ICC} = 0.0005$).
   - At $H = 6\text{ hours}$: Within-patient variance accounts for **100.0%** of total variance ($\text{ICC} = 0.0000$).
   - At $H = 12\text{ hours}$: Within-patient variance accounts for **95.1%** of total variance ($\text{ICC} = 0.0489$).
2. **DSH Failure:**
   - Under rigorous sustained stability requirements ($\text{IQR} \le 0.05$ sustained across all subsequent eligible horizons), **92 out of 120 patients (76.7%) did NOT attain stabilization** within the available monitoring duration.
3. **Scientific Conclusion:** Charitos AF density is an emergent, long-horizon property of episode dynamics. It cannot be reliably assessed from 1-hour to 12-hour Holter recordings. Reliable measurement requires continuous multi-day observation.

---

## 5. Summary of Programmatic Directives

1. **Abandon Standalone Continuous SetOperator Morphology:** Do not deploy A001 as a primary diagnostic model. RR-GBDT provides substantially superior discrimination (AUROC 0.9723 vs 0.5905) at a fraction of computational cost.
2. **Reject Point Estimates Under Missingness:** If recording completeness falls below 80% or contains contiguous dropouts $>1\text{ hour}$, do not report a single point density estimate. Report certified mathematical bounds or output `UNDEFINED_INSUFFICIENT_OBSERVATION`.
3. **Enforce Prolonged Monitoring:** Clinical protocols evaluating AF density must mandate monitoring horizons $\ge 24\text{ hours}$ (optimally 7–14 days) to overcome the $>90\%$ within-patient temporal variance.
