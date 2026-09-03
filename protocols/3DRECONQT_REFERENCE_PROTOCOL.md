# 3DRECON-QT Architectural Reference & Spatial-Semantics Benchmark Protocol

**Project Name**: `3DRECONQT_REFERENCE`  
**Status**: Revised Protocol with Sampled-Query Primary Reference  
**Scientific Purpose**: Determine whether the physical-theta coordinate effect that was practically undetectable in the D-series ($\Delta = +0.0016$) emerges when the spatial-conditioning mechanism is embedded in an architecture substantially closer to 3DRECON-QT (Ansari et al., 2026; Chen et al., 2021).

---

## 1. Strict Non-Destructive Separation

- **D-Series** ($D_0$–$D_8$): Designated **"3D-theta mechanism ablation in ECG-AIM"**. Evaluates spatial conditioning inside a Transformer encoder-decoder architecture with patch tokenization and target-query cross-attention.
- **RQ-Series** ($RQ_1Q$, $RQ_2Q$, etc.): Designated **"3DRECON-QT architectural reference"**. Evaluates spatial conditioning inside the convolutional SE-ResNeXt + Z1/Z2 temporal-attention + convolutional upsampling architecture published in Ansari et al. (2026).
- **Rule**: Never replace, alter, rename, or overwrite $D_0$–$D_8$ checkpoints, scripts, or evaluation tables.

---

## 2. The Sampled-Query Formulation (Primary Reference)

As revealed by inspection of the public WhatAShot/Electrocardio-Panorama codebase (`random.sample(rest_index, 1)`) and Ansari et al. (2026) Figure 2:
- **Training Formulation**: During training, each patient example receives a deterministically scheduled **target viewpoint**:
  - For **Lead I deployment source**: $l_i \sim \mathcal{U}\{\text{II}, \text{III}, \text{aVR}, \text{aVL}, \text{aVF}, \text{V}_1, \dots, \text{V}_6\}$ ($11$ missing leads; observed Lead I is strictly excluded: $\text{count}(l == 0) = 0$).
  - For **subcutaneous ICM differential vector** ($V_3 - V_2$): $l_i \sim \mathcal{U}\{\text{I}, \dots, \text{V}_6\}$ (all 12 standard leads are legitimate targets).
  The shared latent $Z_i$ is multiplicatively conditioned on that single target query $e_{l_i}$, decoded through the single shared convolutional decoder, and evaluated against that single target waveform:
  $$\mathcal{L}_i = \|\hat{X}_{i, l_i} - X_{i, l_i}\|_1$$
- **Inference Formulation**: At validation and test time, the model queries all 12 viewpoints simultaneously from the shared latent $Z_i$, reconstructing the full 12-lead ECG tensor for comprehensive multi-lead evaluation.

### Two Explicit Training Modes Supported:
1. `--target-training sampled_query` (**PRIMARY 3DRECONQT_REFERENCE**): Single sampled target query per training example; matches public solver and Figure 2.
2. `--target-training simultaneous_12` (**SECONDARY VARIANT `RQ_A12`**): Simultaneous 12-lead joint decoding and L1 penalty across all leads every iteration.

---

## 3. Architecture & Provenance Status Registry

Every single component is cataloged in `refine-logs/3dreconqt_reference/architecture_provenance.yaml`:

| Component | Status | Source / Value |
|:---|:---:|:---|
| **Input Signal** | `PAPER_CONFIRMED` | 10-second duration, 500 Hz sampling rate |
| **Bandpass Filtering** | `PAPER_CONFIRMED` | 0.5–40 Hz, fifth-order zero-phase Butterworth |
| **Waveform Reconstruction Loss** | `PAPER_CONFIRMED` | L1 waveform loss on reconstructed ECG |
| **Training Target Formulation** | `UNCONFIRMED` | Public code uses sampled single-query; paper text does not confirm simultaneous full-12 training |
| **Development Source Vector** | `PAPER_CONFIRMED` | $x_{\text{ICM}} = V_3 - V_2$ (subcutaneous ICM orientation) |
| **Deployment Source Vector** | `PAPER_CONFIRMED` | $x_I = \text{Lead I}$ (wearable single-lead deployment) |
| **Encoder Backbone** | `PAPER_CONFIRMED` | 1D Squeeze-and-Excitation ResNeXt (`SE-ResNeXt-1D`) |
| **SE-ResNeXt Depth / Cardinality** | `ASSUMPTION` | Canonical 32x4d, `layers=(3, 4, 6, 3)`, `groups=32`, `width_per_group=4`, `se_reduction=16` |
| **Feature Extraction & Split** | `FIGURE_CONFIRMED` | 1D Convolution into $Z_1$ and $Z_2$ streams (Figure 2) |
| **Z2 Temporal Attention** | `FIGURE_CONFIRMED` | Multi-head self-attention on $Z_2$ tokens (Figure 2) |
| **Z2 LayerNorm** | `FIGURE_CONFIRMED` | LayerNorm directly following temporal attention (Figure 2) |
| **Z1/Z2 Concatenation** | `FIGURE_CONFIRMED` | Concatenation of $Z_1$ and transformed $Z_2$ into shared latent $Z$ |
| **Angular Encoding Representation** | `PAPER_CONFIRMED` | $[\theta, \phi, \theta+\phi, \theta-\phi, \sin, \cos]$ 12-D encoding (Chen et al., 2021) |
| **Exact ThetaEncoder Implementation** | `PUBLIC_CODE_EXACT` | Released WhatAShot/Electrocardio-Panorama code (SHA256 `ef03ca7e...`) |
| **Lead Angle Coordinate Table** | `PUBLIC_CODE_EXACT` | 12 angular coordinate pairs from public `ptbv2.py`, reordered to standard convention |
| **Query Projection MLP** | `FIGURE_CONFIRMED` | `Linear(12, 128) -> Mish() -> Linear(128, 256)` (Figure 2) |
| **Query-Latent Conditioning** | `PUBLIC_CODE_EXACT` | Channelwise multiplicative gating: $Z_l = Z \odot e_l$ |
| **Reconstruction Decoder** | `PUBLIC_CODE_ADAPTED` | Shared 1D Upsample + DoubleConv blocks, adapted for 5,000 samples |
| **Decoder Output Activation** | `PUBLIC_CODE_ADAPTED` | Unbounded linear (no sigmoid/tanh; preserves signed continuous ECG) |
| **Auxiliary QT Branch** | `PAPER_CONFIRMED` | Transformer encoder over shared latent $Z$ without target conditioning |
| **Optimizer** | `PAPER_CONFIRMED` | SGD with initial LR $= 10^{-3}$, weight decay $= 10^{-5}$, momentum $= 0.9$ |
| **Learning Rate Schedule** | `PAPER_CONFIRMED` | `CosineAnnealingWarmRestarts` ($T_0=10$, $T_{\text{mult}}=1$, $\eta_{\min}=10^{-6}$) |
| **Batch Size** | `PAPER_CONFIRMED` | Effective batch size 128 (physical batch 32 with 4x gradient accumulation) |
| **Observed VRAM Footprint** | `VERIFIED_EMPIRIC` | **$4.83\text{ GB}$ peak allocated, $4.99\text{ GB}$ peak reserved** at physical batch 32 on A100 |

---

## 4. Revised Experiment Matrix & Phased Execution Strategy

Do not run five expensive 100-epoch cells at once. The scientifically prioritized sequence is:

### Phase 1: Primary Decisive Falsification Screen (Lead I, Seed 42)
Run strictly:
1. **`RQ1Q_theta_l1`**: Sampled query, true physical angles, Lead I source.
2. **`RQ2Q_permuted_theta_l1`**: Sampled query, fixed permuted derangement (no fixed points), Lead I source.

#### Primary Contrast:
$$\Delta_{\text{geom}, I} = M(RQ1Q_I) - M(RQ2Q_I)$$
- Since $RQ1Q$ and $RQ2Q$ have identical architectures, parameter counts (26.5M), and amount of target identity information, any performance difference isolates **physical viewing-angle semantics**.

### Phase 2: Target Code Structure Screen (Conditional)
If $|\Delta_{\text{geom}, I}| < 0.003$ in validation missing-11 Pearson:
3. **`RQ3Q_learned12_l1`**: 12-D learned continuous query codes.
4. **`RQ4Q_random12_l1`**: 12-D fixed standardized random codes.

### Phase 3: Precordial Source-Vector Interaction (Subcutaneous ICM Vector $V_3 - V_2$)
Repeat $RQ1Q$ and $RQ2Q$ under the paper's original subcutaneous ICM vector:
5. **`RQ1Q_theta_l1_v3_v2`**
6. **`RQ2Q_permuted_theta_l1_v3_v2`**

Compute the two-factor interaction:
$$\Delta_{\text{geom}, \text{ICM}} = M(RQ1Q_{\text{ICM}}) - M(RQ2Q_{\text{ICM}})$$
$$\Delta_{\text{source} \times \text{geom}} = \Delta_{\text{geom}, \text{ICM}} - \Delta_{\text{geom}, I}$$
Tests whether physical viewing angles provide value specifically for nonstandard subcutaneous orientations.

### Phase 4: Simultaneous Full-Target Supervision Diagnostic
7. **`RQ1A12_theta_l1`** vs. **`RQ2A12_permuted_theta_l1`**
Tests whether simultaneous 12-target training vs. sampled single-query training alters the physical coordinate effect.

### Role of RQ0 (Negative Identifiability Control):
- Constant spatial conditioning ($q_I = q_{II} = \dots = q_{V6}$) removes target identity entirely; the model can only produce a target-averaged waveform. $RQ0$ serves strictly as a **sanity check on target identifiability**, not a measure of physical geometry.

---

## 5. Evaluation Protocol

All models evaluated on PTB-XL validation fold 9 ($N = 2,183$ patients; fold 10 is untouched test set):
1. **Paper-Style Metric**: All-12 mean Pearson $\bar{r}_{12} = \frac{1}{12}\sum_{l=1}^{12} r_l$.
2. **Strict Deployment Metrics**:
   - Missing-11 mean Pearson $\bar{r}_{\text{missing}}$
   - Tail robustness $r_{p05}$ (5th percentile patient correlation)
   - Precordial $V_1$–$V_6$ mean Pearson $\bar{r}_{\text{chest}}$
   - Precordial transition progression $\Delta V_k = V_{k+1} - V_k$
   - Full 12-lead reconstruction L1 error (mV) and MSE error
   - Per-lead Pearson correlations for all 12 individual leads
3. **Physical Inconsistency Audit**:
   - Einthoven residual: $E_{III} = |II - I - III|$
   - Goldberger residuals: $E_{aVR}, E_{aVL}, E_{aVF}$
4. **Statistical Inference**:
   - 10,000-sample paired patient bootstrap for $\Delta_i = r_i(RQ1Q) - r_i(RQ2Q)$ on PTB-XL validation cohort.

---

## 6. Scientific Precision Standards

1. **Fusion Mechanism**: "No practically detectable fusion effect was observed at seed 42 under the tested optimization budget ($|\Delta r| \le 0.0003$)."
2. **Limb-Lead Algebra**: "Hard algebraic enforcement improves limb/output-level consistency and aggregate reconstruction metrics without retraining. Capacity reallocation requires a trained independent-basis model ($D_9$) to show improvement on the predicted independent outputs: Lead II and/or V1–V6."
