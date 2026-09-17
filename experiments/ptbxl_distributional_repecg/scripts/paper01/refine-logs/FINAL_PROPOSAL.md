# Final Proposal: Paper 01 — Distributional Recurrence Operator

## Problem Anchor
Recurrence-plot-based analysis and 2D CNN classification of electrocardiograms (ECGs) already exist in prior art (e.g., Mathunjwa et al. converting ECGs to 2D recurrence plots for ResNet classification; classical Recurrence Quantification Analysis [RQA] in non-linear cardiac dynamics). Conventional recurrence plots construct a trajectory from scalar time series or single-beat phase vectors:
$$x_g \in \mathbb{R}^D \implies R_{gh} = \|x_g - x_h\|_2^2$$
This formulation treats each cardiac phase as an isolated point in Euclidean space, discarding beat-to-beat variability, respiratory modulation, and continuous morphological distributions.

The true problem anchor is: **Prior ECG recurrence methods compare individual trajectory states or scalar samples. How can we construct recurrence between local distributions of electrical states across beats to capture non-linear cardiac dynamics while remaining robust to morphological jitter and lead sparsity?**

---

## Method Thesis: From Euclidean Recurrence to Distributional MMD Recurrence
We discard ordinary Euclidean point recurrence and propose **Distributional Recurrence Operators**.

For a multi-beat recording with $B$ valid cardiac cycles, let each normalized cardiac phase region $g \in \{1, \dots, 16\}$ define an empirical distribution of multi-lead observations across beats:
$$P_g = \mathcal{L}\left\{ X_b(s) : s \in C_g, \; b = 1, \dots, B \right\}$$
where $X_b(s) \in \mathbb{R}^{C}$ is the multi-lead potential vector of beat $b$ at phase offset $s$.

We embed each phase-conditioned distribution $P_g$ into a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$ via its Kernel Mean Embedding (KME):
$$\mu_{P_g} = \mathbb{E}_{X \sim P_g}[\phi(X)] \in \mathcal{H}$$

The pairwise distance between phase regions $g$ and $h$ is defined by the squared Maximum Mean Discrepancy (MMD):
$$D_{gh} = \operatorname{MMD}^2(P_g, P_h) = \|\mu_{P_g} - \mu_{P_h}\|_{\mathcal{H}}^2$$
evaluated using the unbiased $U$-statistic with an RBF kernel $k(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$ or Inverse Multiquadric (IMQ) kernel.

Finally, the **Distributional Recurrence Operator** $R \in \mathbb{R}^{16 \times 16}$ is:
$$\boxed{ R_{gh} = \exp\left( -\frac{\operatorname{MMD}^2(P_g, P_h)}{\tau} \right) }$$
where $\tau > 0$ is a temperature parameter chosen via the median heuristic across all phase pairs.

---

## Qualification of Invariance Claims & Duration Disentanglement
1. **No Claims of "Exact" Invariance**: Phase normalization provides invariance only to uniform affine time reparameterization. In real ECGs, non-uniform systolic/diastolic warping, R-peak jitter, and ectopic beats violate exact invariance. We explicitly measure sensitivity to these factors.
2. **Explicit Factorization of Morphology and Duration**: Because critical clinical biomarkers reside in intervals ($PR$, $QRS$, $QT$, $RR$), phase-normalized recurrence inherently strips duration. Rather than asserting duration is "implicitly present", we formulate explicit factorization:
   - Primary representation (Duration-free): $R_{\text{distributional}} \in \mathbb{R}^{16 \times 16}$.
   - Duration feature vector: $d = [d_{RR}, d_{PR}, d_{QRS}, d_{QT}] \in \mathbb{R}^4$.
   - Secondary composite model: $Z_{\text{record}} = [R_{\text{distributional}}, d]$.
   - Falsification ablation: Compare **Recurrence only** vs **Duration only** vs **Recurrence + Duration**.

---

## Amplitude Treatment: Disentangling Scale from Shape
We reject intra-record variance normalization as a default, as it erases vital clinical voltage markers (low-voltage QRS in amyloidosis/effusion, Sokolow-Lyon hypertrophy criteria, bundle branch attenuation). Instead, amplitude is treated via:
1. **Factorial Amplitude Matrix**: Evaluate representations under:
   - $X_{\text{physical}}$ (calibrated $\mu\text{V}$, preserving absolute voltage criteria).
   - $X_{\text{train standardized}}$ (dataset-level centering and scaling).
   - $X_{\text{record standardized}}$ (unit variance per recording).
2. **Explicit Scale Disentanglement**:
   $$x(t) = a \cdot \tilde{x}(t), \quad \text{where } a = \sqrt{\frac{1}{T}\sum_t \|x(t)\|^2}, \quad \tilde{x}(t) = \frac{x(t)}{a}$$
   Construct distributional recurrence $R_{\text{shape}}$ on scale-free $\tilde{x}$, and append scalar energy $a$ separately as $[R_{\text{shape}}, a]$.

---

## Cyclic Shift & Model Equivariance
If cyclic boundary invariance is desired, the standard 2D CNN must be replaced or augmented:
- Standard 2D convolution breaks cyclic equivariance.
- We implement **2D Circular Convolutional Residual Blocks**:
  $$x = \operatorname{Conv2d}(\operatorname{pad}(x, (1, 1, 1, 1), \text{mode}='circular'))$$
- We measure cyclic equivariance directly via cyclic shift discrepancy:
  $$\Delta_{\text{cyclic}} = \frac{1}{16} \sum_{k=0}^{15} |p(R) - p(P_k R P_k^\top)|$$
  and Centered Kernel Alignment: $\operatorname{CKA}(z(R), z(P_k R P_k^\top))$.

---

## Benchmarking Targets & Decision Verdict
- **Prior PTB-XL Benchmarks**: Established supervised 1D baselines (xResNet1d101: 0.928 macro-AUROC; ResNet1D-Wang: 0.930 macro-AUROC). An arbitrary target of $0.75$ is rejected as unpublishable.
- **Defensible Target**:
  $$\boxed{ \operatorname{AUROC}_{\text{repECG}} \ge \operatorname{AUROC}_{\text{strong representation baseline}} }$$
  and noninferiority to raw 1D waveform models ($\Delta \operatorname{AUROC} > -0.01$) under full 12 leads, while demonstrating statistically significant superiority in **sparse-lead regimes** (1-lead, 2-lead, 3-lead, ICM).
- **Official Splits**: PTB-XL Folds 1–8 for training, Fold 9 for validation/model selection, Fold 10 strictly held out for final test.

**Decision Verdict**: `MAJOR REVISION — NOT YET PRODUCTION READY` (requires updating representation pipeline from Euclidean to MMD recurrence before production launch).
