# Claim-Driven Experiment Plan: Paper 01 (Distributional Recurrence Operator)

## 1. Experimental Objectives & Benchmarking Standard
1. **Primary Benchmark**: Evaluate whether Distributional MMD Recurrence achieves predictive noninferiority to supervised raw 1D waveforms ($\Delta \operatorname{AUROC} > -0.01$) on PTB-XL diagnostic superclasses (SOTA baselines: xResNet1d101 ~0.928, ResNet1D-Wang ~0.930).
2. **Sparse-Lead Battery (GraphECG Protocol)**: Test the hypothesis that distributional recurrence provides superior inductive bias in reduced-lead and sparse configurations (6-limb, 3-lead, 2-lead, 1-lead, ICM).
3. **Data Splits**: Strict compliance with official PTB-XL benchmarking protocol:
   - Folds 1–8: Training (17,441 records).
   - Fold 9: Validation / Model Selection (2,193 records).
   - Fold 10: Final Held-Out Test (2,198 records) — never touched during hyperparameter tuning.

---

## 2. Comprehensive 15-Variant Hierarchy

| # | Variant Identifier | Category | Input Representation / Mechanism | Purpose / Falsification Role |
|---|---|---|---|---|
| 1 | **`mmd_recurrence_full`** | Proposed | Empirical beat distribution $P_g$, MMD $U$-statistic RBF kernel, circular 2D CNN | Central proposed contribution |
| 2 | **`euclidean_phase_recurrence`** | Prior Art Baseline | Single mean vector $x_g$, Euclidean distance $R_{gh} = \|x_g - x_h\|^2$ | Direct comparison with existing ECG-RP-CNN literature |
| 3 | **`mean_recurrence`** | Baseline | Cell centroids $\bar{x}_g = \frac{1}{N}\sum X_b$, pairwise centroid distances | Tests whether distribution higher-order moments matter |
| 4 | **`linear_kernel_recurrence`** | Baseline | Linear kernel MMD: $k(x, y) = \langle x, y \rangle$ | First-moment RKHS baseline |
| 5 | **`imq_mmd_recurrence`** | Method Variant | Inverse Multiquadric (IMQ) kernel: $k(x, y) = (c^2 + \|x-y\|^2)^{-1/2}$ | Heavy-tailed distributional recurrence |
| 6 | **`linear_probe`** | Probe | Linear classifier on 120 upper-triangular MMD entries | Representation sufficiency / linear separability |
| 7 | **`mlp_probe`** | Probe | 2-layer MLP on upper-triangular entries without 2D convolution | Non-linear probe without spatial CNN inductive bias |
| 8 | **`raw_phase_cnn`** | Architectural Control | Direct 1D CNN on phase features without recurrence matrix | Isolates benefit of recurrence transformation |
| 9 | **`fixed_phase_permutation`** | Permutation Control | Fixed permutation $\pi$ applied identically to all ECG records | Tests coordinate relabeling robustness |
| 10 | **`recordwise_phase_permutation`** | Falsification Control | Random independent permutation $\pi_i$ drawn per ECG record | Destroys cross-record phase semantics |
| 11 | **`cyclic_shift_no_aug`** | Invariance Test | Circular CNN trained without cyclic shift augmentation | Measures baseline fiducial anchor sensitivity |
| 12 | **`cyclic_shift_aug`** | Invariance Test | Circular CNN trained with random cyclic shift augmentation | Tests achievable cyclic anchor invariance |
| 13 | **`amplitude_removed`** | Amplitude Factorial | Normalized shape-only recurrence $R_{\text{shape}}$ ($\tilde{x} = x/a$) | Measures performance when amplitude is stripped |
| 14 | **`amplitude_appended`** | Amplitude Factorial | Composite model $[R_{\text{shape}}, a]$ (scale disentanglement) | Proves benefit of scale disentanglement over erasure |
| 15 | **`label_permutation`** | Negative Control | Labels randomly shuffled across patients | Pipeline empirical null control (target AUROC = 0.50) |

---

## 3. Prior-Art Replication Baselines
To benchmark against decades of recurrence literature:
1. **Classical Thresholded RP + RQA Descriptors**:
   - Compute binary recurrence plots: $RP_{ij} = \mathbf{1}[\|x_i - x_j\| < \epsilon]$ with recurrence rates $RR \in \{5\%, 10\%, 20\%\}$.
   - Extract standard RQA features: Recurrence Rate, Determinism, Average Diagonal Length, Longest Diagonal, Laminarity, Trapping Time, Entropy.
   - Train an XGBoost classifier on RQA vectors.
2. **Deep Recurrence Plot ResNet-18**:
   - Replicate the architecture of Mathunjwa et al. (2022) using continuous Euclidean RP fed into a standard ResNet-18.
   - Compare parameter efficiency and AUROC against the compact circular `RecurrenceCNN`.

---

## 4. Sparse-Lead & GraphECG Degradation Protocol

To test whether distributional recurrence excels under lead sparsity, execute all models across 7 fixed lead configurations:
1. **12-Lead Standard**: Full diagnostic set (I, II, III, aVR, aVL, aVF, V1–V6).
2. **6-Limb Leads**: Peripheral vector set (I, II, III, aVR, aVL, aVF).
3. **3-Lead Einthoven**: Classic bipolar limb leads (I, II, III).
4. **2-Lead Rhythm**: Bipolar rhythm monitoring (I, II).
5. **1-Lead Single**: Lead I only (wearable smart-watch equivalent).
6. **1-Lead Precordial**: Lead II only.
7. **ICM (Insertable Cardiac Monitor)**: Single subcutaneous vector $V_3 - V_2$.

### Lead Degradation Curve
Evaluate model degradation for random lead subsets $k \in \{1, 2, \dots, 12\}$ using a fixed, pre-computed random subset bank. Compute the advantage metric:
$$\Delta_{\text{MMD-vs-Euclid}}(k) = \operatorname{AUROC}_{\text{MMD}}(k) - \operatorname{AUROC}_{\text{Euclid}}(k)$$
- **Core Hypothesis**: $\Delta_{\text{MMD-vs-Euclid}}(k)$ increases as $k \to 1$, demonstrating that distributional representations preserve critical diagnostic signals under observation sparsity.

---

## 5. Hyperparameter Search & Grid Specification
- **Optimizer**: AdamW with Cosine Annealing scheduler.
- **Learning Rates**: `(1e-4, 3e-4, 1e-3)`.
- **Weight Decays**: `(1e-5, 1e-4, 1e-3)`.
- **Batch Size**: 64 (Production: max 100 epochs, patience 10).
- **Loss Function**: `BCEWithLogitsLoss` with positive class prevalence weights.
- **Selection Metric**: Best Macro-AUROC on PTB-XL Fold 9 validation set.
