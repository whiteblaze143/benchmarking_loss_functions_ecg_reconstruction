# Experiment Plan: Paper 02
# Phase-KME: Phase-Conditioned Kernel Mean Embeddings for Distribution-Valued ECG Representation

## 1. Variant Hierarchy & Claim-Identification Matrix

We define 12 formal variants to exhaustively evaluate each hypothesis:

| Variant | Input Dim | Representation Key | Architecture | Readout | Scientific Question Isolated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `full` | 256 | `kernel` | Circular PhaseCNN | GAP | **Proposed Model** |
| `global_kme` | 256 | `kernel` (mean over phase) | Parameter-matched MLP | Linear | **Prior-Art Baseline**: Scientific Reports 2025 |
| `no_circular` | 256 | `kernel` | Standard PhaseCNN (zero-pad) | GAP | **Ablation**: Discrete $C_{16}$ boundary prior |
| `moments_circular` | 44 | `moments` | Circular PhaseCNN (matched capacity)| GAP | **Ablation**: 1st & 2nd moments vs RKHS |
| `moments_noncircular`| 44 | `moments` | Standard PhaseCNN (zero-pad) | GAP | Factorial interaction of moments & topology |
| `gaussian_surrogate` | 256 | `gaussian` | Circular PhaseCNN | GAP | **Destroyer Control**: Non-Gaussian geometry |
| `linear_kernel` | 8 | `linear` | Circular PhaseCNN | GAP | Classical SAECG (1st moment) baseline |
| `rbf_kme` | 256 | `rbf_kernel` | Circular PhaseCNN | GAP | Kernel sensitivity (IMQ vs Gaussian RBF) |
| `phase_kme_linear_probe` | 4096 | `kernel` (flattened) | Linear Ridge Probe | Linear | Representation linear separability |
| `global_kme_linear_probe`| 256 | `kernel` (mean over phase) | Linear Ridge Probe | Linear | Phase-free KME linear probe |
| `phase_shuffle` | 256 | `kernel` (scrambled bins) | Circular PhaseCNN | GAP | Phase chronological coherence |
| `deepsets_phase` | 256 | Raw beats $\to$ DeepSets | Circular PhaseCNN | GAP | Fixed characteristic RKHS vs learned set encoder |

## 2. Training Protocol & Hyperparameter Search Grid
- **Optimizer**: AdamW with CosineAnnealingLR.
- **Grid (9 cells per variant)**:
  - Learning Rate: $\eta \in \{1 \times 10^{-4}, 3 \times 10^{-4}, 1 \times 10^{-3}\}$
  - Weight Decay: $\lambda \in \{1 \times 10^{-5}, 1 \times 10^{-4}, 1 \times 10^{-3}\}$
- **Batch Size**: 2,048 recordings (shared-tensor single-process streaming).
- **Early Stopping**: Patience 10 epochs on validation Macro AUROC (PTB-XL fold 8).
- **Precision**: `bfloat16` autocast, `torch.backends.cudnn.enabled = False`.

## 3. Evaluation Battery & Equivalence Testing
- **Primary Task Metric**: Macro AUROC across 5 PTB-XL diagnostic classes (NORM, MI, STTC, CD, HYP).
- **Statistical Significance**: Clustered paired bootstrap at patient level (2,000 replicates).
- **Equivalence Margin**: Predefined margin $\delta = 0.005$.
  - Superiority of Phase-KME: $\text{CI}_{95\%,\text{lower}}(\text{AUROC}_{\text{full}} - \text{AUROC}_{\text{control}}) > 0.005$.
  - Non-inferiority: $\text{CI}_{95\%,\text{lower}}(\Delta) > -0.005$.
- **Beat-Count Stratification**: Stratify test set by usable beats ($2-4, 5-7, 8-10, >10$) and report AUROC$(B)$ and split-half stability $S_B$.

## 4. GraphECG Mask-Aware Sparse-Lead Battery
Evaluate model robustness under lead degradation using mask-augmented kernel $\tilde{x} = [m \odot x, \gamma m] \in \mathbb{R}^{16}$:
- Standard 12-lead (represented as 8 independent leads)
- 6 limb leads
- 3 Mason-Likar leads
- 2 bipolar leads (Lead I, Lead II)
- 1 single lead (Lead I, Lead II, $V_2-V_3$)
- Random missing lead ablation ($k=1,\dots,7$ leads masked).
