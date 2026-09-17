# Refinement Report: Evolutionary Trajectory of the repECG Paradigm

## 1. Motivation for Architectural Refinement

Traditional ECG deep learning relies on brute-force empirical optimization over raw 1D voltage traces. Over the course of the repECG development program, we recognized four core evolutionary stages required to transition from fragile deep learning to robust, causal scientific modeling:

```
Stage 1: Raw Voltage CNNs (Legacy Baseline)
   │  [Fragile to noise, non-invariant to heart rate, memorizes hardware filters]
   ▼
Stage 2: Phase-Aligned Distributional RKHS (Papers 01 - 03)
   │  [Heart-rate invariant, physical mV preservation, non-linear MMD recurrence]
   ▼
Stage 3: Dynamical Systems & Invariant Operators (Papers 04 - 08)
   │  [Hankel DMD, Koopman linearity in RKHS, dual-head physical reconstruction]
   ▼
Stage 4: Causal Mechanisms & Out-of-Distribution Invariance (Papers 09 - 15)
   │  [Pearl's Do-Calculus, Invariant Risk Minimization, Independent Causal Mechanisms]
```

---

## 2. Refinement Highlights by Component

### Refinement 1: From Dense Recurrence to Circular Phase Convolutions (Paper 01 vs Paper 02)
- *Initial Attempt*: Form an all-pairs $16 \times 16$ MMD recurrence matrix for every beat. While theoretically expressive, it discards the natural 1D temporal ordering and requires quadratic compute in long sequences.
- *Refined Approach*: Introduce `PhaseCNN` with circular padding residual blocks. This directly operates on the temporal sequence of $16 \times 128$ KME vectors while respecting the cyclical boundary condition of the cardiac cycle (connecting the final repolarization phase back to initial atrial depolarization).

### Refinement 2: Incorporating Geometric Invariance via Rough Paths (Paper 03)
- *Initial Attempt*: Standard dynamic time warping (DTW) to align irregular heartbeats. DTW is non-differentiable, sensitive to outliers, and computationally expensive.
- *Refined Approach*: Lift the sequence into continuous RKHS paths and compute iterated integrals (path signatures). This provides provable invariance to arbitrary monotonic continuous time-warping with zero learned parameters.

### Refinement 3: Linearizing Non-Linear Cardiac Dynamics (Papers 04 & 05)
- *Initial Attempt*: Fit complex non-linear Recurrent Neural Networks (LSTMs / GRUs) to learn cardiac phase dynamics. These networks are prone to gradient vanishing, mode collapse, and lack analytical interpretability.
- *Refined Approach*: Exploit Koopman operator theory. By lifting the state into the infinite-dimensional RKHS, non-linear cardiac dynamics become linearly predictable via finite-rank matrix operators ($Z_{t+1} = K Z_t$), rendering stability analysis and eigenvalue decay directly accessible.

### Refinement 4: Transition from Correlation to Controlled Intervention (Papers 09, 10, 13, 14)
- *Initial Attempt*: Standard data augmentation (random noise, cropping) to improve generalization.
- *Refined Approach*: Rigorous structural causal intervention:
  1. Separation of biological 3D dipole state from measurement lead operator $q$ (Paper 09).
  2. Multi-hospital invariant representation via MMD penalties on known acquisition metadata (Paper 10).
  3. Surgical $do(P_g = P_g^{ref})$ phase substitution to compute necessity and sufficiency scores (Paper 13).
  4. Invariant Risk Minimization (IRM) across 9 diverse clinical hospital datasets (Paper 14).

---

## 3. Rejected Architectural Explorations
1. **Unconstrained Generative Diffusion**: Diffusion models on raw 5000-sample traces generate realistic-looking ECGs that completely hallucinate physiological conduction intervals. Rejected in favor of deterministic RKHS embeddings and closed-form linear operators.
2. **End-to-End Raw Voltage Transformers**: Transformers applied directly to raw high-frequency waveforms require massive parameter counts (>100M) and overfit to high-frequency acquisition noise. Rejected in favor of tokenizing 16 compact phase-cell KMEs.
