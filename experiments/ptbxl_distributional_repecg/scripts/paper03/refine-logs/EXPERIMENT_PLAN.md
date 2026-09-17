# Experiment Plan: Paper 03
# Phase-Path-Signature: Distributional Path Signatures on Cardiac Phase Trajectories

## 1. Locked 15-Variant Claim-Identification Matrix

| Variant Name | Representation Object | Intra-Beat Order | Phase Conditioning | Beat Aggregation | Classifier Head | Primary Mechanistic Claim Tested |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `full` | Depth-3 LogSig + Boundary | \checkmark | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Proposed full distributional path signature model |
| `phase_signature_linear_probe`| Depth-3 LogSig + Boundary | \checkmark | 16 Phase Cells | Nyström KME | Linear ($2048 \to 5$) | Evaluates representation quality without convolutional depth |
| `linear_probe` | Depth-3 LogSig + Boundary | \checkmark | Mean-Pooled Cells | Nyström KME | Linear ($128 \to 5$) | Legacy probe (conflates linearity with phase pooling) |
| `phase_mean_signature` | Beat-Averaged LogSig ($\bar{s}_g$) | \checkmark | 16 Phase Cells | Mean Vector Only | PhaseCNN ($C_{16}$) | **Core Ablation**: Isolates value of KME vs mean signature |
| `whole_beat_signature` | Beat-Level LogSig | \checkmark | None (Full Beat) | Nyström KME | Matched CNN | Isolates value of phase-cell localization |
| `global_signature` | Recording-Level LogSig | \checkmark | None (10s Record) | None (Single Sig) | Matched MLP | **PathFusion Baseline**: Evaluates unaligned rough path |
| `unordered_kme` | Paper 02 Voltage KME | \text{\sffamily X} | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | **Paper 02 Baseline**: Tests path geometry vs static points |
| `order_scrambled` | Internal Order Permuted | \text{\sffamily X} | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Tests reliance on true continuous trajectory ordering |
| `time_reversed` | Time-Reversed ($t \mapsto 1-t$) | Inverted | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Tests sensitivity to trajectory orientation (Lévy area) |
| `monotone_warp_sham` | Non-Linear Monotone Warp | Warped | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | **Negative Control**: Tests Chen reparameterization invariance |
| `depth1` | Level 1 LogSig ($m=1$) | \checkmark | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Isolates net displacement ($8$ dims) |
| `depth2` | Level 1 + 2 LogSig ($m=2$) | \checkmark | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Isolates signed pairwise Lévy area ($36$ dims) |
| `depth3` | Level 1 + 2 + 3 ($m=3$) | \checkmark | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Isolates higher-order Lie brackets ($204$ dims) |
| `signature_plus_mean` | $d^{\text{aug}} = [d^{\text{sig}}, \bar{\gamma}]$ | \checkmark | 16 Phase Cells | Nyström KME | PhaseCNN ($C_{16}$) | Evaluates impact of non-invariant temporal mean |
| `no_circular` | Depth-3 LogSig + Boundary | \checkmark | 16 Phase Cells | Nyström KME | Standard Conv1D | Tests cyclic $C_{16}$ boundary conditions vs zero padding |

---

## 2. Core Scientific Hypotheses

- **Hypothesis 1 ($H_1$, Trajectory vs. Point Cloud)**:
  `full` > `unordered_kme` (Macro AUROC $\Delta > 0.015$). Continuous trajectory geometry and pairwise loop areas provide diagnostic signal beyond static voltage distributions.
- **Hypothesis 2 ($H_2$, Internal Path Ordering)**:
  `full` > `order_scrambled` (Macro AUROC $\Delta > 0.020$). Scrambling internal sample points destroys performance, confirming the network relies on path geometry.
- **Hypothesis 3 ($H_3$, Lévy Area Contribution)**:
  `depth2` > `depth1` (Macro AUROC $\Delta > 0.010$). Signed planar loop area adds decisive diagnostic value over linear lead displacement.
- **Hypothesis 4 ($H_4$, Higher-Order Lie Brackets)**:
  `depth3` $\approx$ `depth2` ($\Delta \le 0.005$). Depth-2 Lévy area captures the vast majority of useful geometric information; depth 3 adds minor marginal gain.
- **Hypothesis 5 ($H_5$, Distributional Aggregation vs. Mean)**:
  `full` > `phase_mean_signature` (Macro AUROC $\Delta > 0.010$). Nonparametric KME beat aggregation outperforms simple mean signature vectors.
- **Hypothesis 6 ($H_6$, Reparameterization Invariance Control)**:
  `full` $\approx$ `monotone_warp_sham` ($|\Delta| < 0.005$). Path signatures are robust to physiological heart-rate warping.

---

## 3. Clinical Stratification across PTB-XL Superclasses
We evaluate macro and class-specific AUROC across all 5 diagnostic superclasses:
1. **NORM** (Normal ECG): Expected high performance across all variants.
2. **MI** (Myocardial Infarction): Strongly benefits from ST-segment baseline morphology ($\gamma(0), \gamma(1)$).
3. **STTC** (ST/T Changes): Benefits from repolarization loop curvature ($m=2$ Lévy area).
4. **CD** (Conduction Disturbance): Benefits decisively from QRS loop orientation and trajectory ordering ($H_1, H_2$).
5. **HYP** (Hypertrophy): Benefits from increased loop area and multilead amplitude dispersion.
