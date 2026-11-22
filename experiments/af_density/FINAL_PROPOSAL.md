# Research Proposal: Continuous q-conditioned Temporal Phase-KME with Ordered-Beat Dynamics, Distributional Morphology, and Residual ResNet Fusion for Atrial Fibrillation Density Estimation

## 1. Problem Anchor
- **Bottom-line problem**: Faithful reconstruction of atrial fibrillation (AF) temporal episode dynamics and continuous temporal density estimation from long-term, multi-channel / multi-lead ambulatory Holter ECG recordings under arbitrary lead configurations, missing channels, and acquisition dropouts.
- **Must-solve bottleneck**:
  1. *Representation Collapse in Categorical Set Operators*: The existing categorical SetOperator adapter (and its channel-identity-ablated control) experiences complete activation collapse in the Transformer set encoder. Across 498,057 validation evaluations, the active-token latent standard deviation collapses to $\sim 3.45 \times 10^{-6}$ (categorical) and $\sim 2.24 \times 10^{-6}$ (identity-ablated), resulting in a degenerate, invariant predicted AF probability of $\approx 0.2284$ regardless of cardiac rhythm or input waveform.
  2. *Loss of Sequential Temporal Rhythm Dynamics*: Canonical SetOperators treat heartbeats within an observation window as an unordered set, discarding inter-beat interval ($RR$) transitions, irregularity dynamics, and rhythm phase reset, which are the primary hallmarks of AF.
  3. *Uncalibrated Temporal Emissions*: Window-level classifiers produce uncalibrated probabilities that corrupt downstream temporal decoders (HSMM / median filter), degrading continuous density integration and episode boundary delineation.
- **Non-goals**:
  - NO speculative downstream clinical endpoint claims (e.g., CHA2DS2-VASc stroke prediction, systemic thromboembolism prevention, or all-cause mortality), which cannot be validated without multi-year longitudinal clinical trial registries.
  - NO unvalidated 16-channel fusions or arbitrary synthetic lead projections; all geometric operators must correspond to physically observed clinical lead coordinates.
  - NO heuristic uncalibrated HMM/HSMM post-processing without explicit duration likelihood modeling and temperature/Platt calibration.
  - NO data snooping, threshold retuning, or model selection using held-out test data.
  - NO fabricating ground truth or imputing synthetic heartbeats during channel dropouts.
- **Constraints**:
  - Compute budget: 1x NVIDIA A100-SXM4-80GB GPU.
  - Datasets: IRIDIA-AF benchmark (504,777 train, 166,047 val, 131,667 test 30-second windows at 200 Hz across 152 patients); external validation on SHDB (98 records, 93 subjects, 279,686 windows) and LTAF (84 records, 232,233 windows).
  - Exact Parameter Target: 1,027,009 trainable parameters in the diagnostic forward path (matching EchoNext benchmark specifications).
- **Success condition**:
  - Elimination of representation collapse: validation prediction standard deviation $> 0.25$ (vs $8.48 \times 10^{-8}$ in collapsed categorical model).
  - Statistically significant reduction in Charitos continuous density MAE and binned density MAE over matched raw ResNet and observed-RR baselines under patient-clustered bootstrap.
  - True lead-dropout and channel-permutation invariance (divergence $< 0.15\%$ in burden and density under random lead swap/dropout).

---

## 2. Technical Gap
Current automated AF detection methods predominantly operate on fixed-lead raw voltage waveforms (via 1D CNNs/ResNets) or extracted R-peak intervals ($RR$ interval statistics). Both approaches suffer from structural limitations:

1. **Rigid Lead Geometries**: Standard 1D ResNets assume fixed channel ordering (e.g., Channel 0 = Lead I, Channel 1 = Lead II). When electrodes peel off, noise forces a channel shutdown, or wearable patches provide non-standard lead vectors, CNNs fail or require separate models trained for every possible lead combination ($2^M - 1$ models).
2. **Failure of Unconditioned Set Encoders**: While permutation-invariant SetOperators theoretically accept arbitrary subsets of leads, treating channels as discrete categorical IDs ($0, 1$) fails to supply spatial metric geometry. In the previous implementation, the cross-attention and self-attention layers in the set encoder mapped identical categorical tokens into a uniform attractor, causing representation collapse.
3. **Loss of Beat Dynamics in Set Encoders**: Standard set pooling averages beat tokens across time, obliterating the temporal order of beats. However, AF is characterized by marked *temporal irregularity* (Lorentz plot dispersion, sample entropy, Markovian transition dynamics). A set of beat morphology tokens without temporal order lacks the rhythm information required to differentiate AF from sinus arrhythmia or frequent premature atrial contractions (PACs).
4. **Disconnection Between Window Classification and Continuous Density**: Episode reconstruction requires temporal decoders (Viterbi HSMM). When window scores are uncalibrated or boundary-blurred, episode boundaries are shifted, causing massive errors in Charitos continuous density, which integrates the shortest-window envelope analytically.

---

## 3. Method Thesis
**Core Thesis**: By conditioning kernel mean embeddings (Phase-KME) on continuous lead projection vectors $q \in \mathbb{R}^d$ in the canonical 8-lead clinical basis, augmenting the beat representation with sequential inter-beat dynamics (ordered-beat sequence encoding) and distributional morphology statistics (mean + variance across phase response atoms), and fusing this invariant latent representation residually with a raw-waveform ResNet pathway, we break representation collapse, preserve lead-invariance, and achieve superior episode boundary delineation and continuous AF density estimation.

- *Smallest Adequate Intervention*: Rather than building a massive multi-modal foundation model or unconstrained 16-channel graph network, we introduce continuous geometric projection vectors $q$ into the operator MLP, replace unordered set pooling with an order-preserving sequential beat encoder, and provide a direct residual bypass for raw waveform features.
- *Modern Foundation-Model Era Relevance*: Employs continuous operator conditioning (analogous to continuous neural operators in physical systems) and kernel mean embeddings in reproducing kernel Hilbert spaces (RKHS), bridging non-parametric statistics with deep representation learning.

---

## 4. Contribution Focus
- **Dominant Contribution**:
  - The formulation and validation of the **Continuous $q$-Conditioned Temporal Phase-KME SetOperator**, which maps variable-channel ECG observations into a metric space defined by canonical lead geometry, breaking the representation collapse that crippled categorical set encoders.
- **Optional Supporting Contribution**:
  - The integration of **Ordered-Beat Dynamics (D037)** and **Distributional Morphology Pooling (D032)** with a **Residual ResNet Fusion Pathway (I082)**, proving that temporal order and raw voltage residuals resolve boundary-jitter in continuous AF density calculation.
- **Explicit Non-Contributions**:
  - We do not claim novel clinical biomarkers for stroke risk or mortality.
  - We do not claim universal synthetic lead reconstruction (12-lead synthesis from 1 lead).
  - We do not claim uncalibrated heuristic HSMM optimality.

---

## 5. Proposed Method

### 5.1 Complexity Budget
- **Frozen / Reused Components**:
  - PCHIP phase normalization (256 phase samples per beat, 16 cells).
  - Unsupervised Phase-KME Nyström kernel map (128 landmarks fitted on 737,280 training atoms across 90 training patients with Inverse Multiquadric kernel $c^2=1$).
  - Frozen duration model: Prospectively fitted training-only censored lognormal HSMM duration PMFs (599 runs across 90 training patients).
- **New Trainable Components (Strictly Capped at 1,027,009 Trainable Parameters)**:
  - Continuous Operator MLP: $\mathbb{R}^8 \to \mathbb{R}^{64} \to \mathbb{R}^{64}$.
  - ModulatedPhaseCNN: 1D convolutional response encoder operating on 16x128 feature responses.
  - Ordered-Beat Sequence Encoder: Bidirectional recurrent / causal 1D token dynamics encoder over beat tokens within the 30s window.
  - Residual Fusion Module: Linear fusion connecting the SetOperator latent ($d=256$) with the 8-block raw ResNet latent ($d=256$).
- **Tempting Additions Intentionally Excluded**:
  - NO auxiliary reconstruction decoders in the diagnostic path (excluded to keep parameter count strictly at 1,027,009).
  - NO complex heuristic post-processing or dynamic programming bridges without prior calibration.

### 5.2 System Overview & Architecture Diagram

```
Raw Multi-Channel ECG (B, C, 6000) @ 200 Hz
   │
   ├───> [Stream 1: Raw Voltage Mask-ResNet] ────────────────────────────────────────┐
   │        ├── 8-Block 1D ResNet                                                   │
   │        └── Latent z_raw ∈ ℝ^256                                                │
   │                                                                                │
   └───> [Stream 2: Continuous Phase-KME SetOperator]                              │
            ├── R-peak detection & PCHIP Phase Normalization                        │
            ├── Nyström IMQ Kernel Mean Embedding (16 cells × 128 features)         │
            ├── Lead Coordinate Vector q_c ∈ ℝ^8 (Canonical Clinical Basis)         │
            │      └── Continuous Operator MLP: ℝ^8 → ℝ^64 → ℝ^64                   │
            ├── ModulatedPhaseCNN (encodes phase atoms conditioned on q_c)          │
            ├── Ordered-Beat Sequence Encoder (captures RR & morphology dynamics)   │
            ├── Distributional Morphology Pooling (Mean + Variance across atoms)    │
            ├── 2-Layer Transformer Set Encoder (d_model=192, 4 heads)              │
            └── Latent z_set ∈ ℝ^256                                                │
                                                                                    │
   ┌────────────────────────────────────────────────────────────────────────────────┘
   ▼
[Residual Fusion: z_fused = W_f [z_set || z_raw] + z_set] ∈ ℝ^256
   │
   ▼
Linear AF Classification Head: ℝ^256 → ℝ^1 (Logit s_t)
   │
   ▼
Temperature & Platt Probability Calibration: p_t = σ(s_t / T_cal)
   │
   ▼
[Temporal Hierarchy: D0 → D1 → D2]
   ├── D0: Calibrated Window Thresholding (p_t ≥ τ)
   ├── D1: Median Filtering (odd-width, non-AF gap merge ≤ 60s within blocks)
   └── D2: Numerical Viterbi HSMM (censored lognormal duration priors)
   │
   ▼
Continuous & Binned AF Density Estimation + Episode Delineation
```

### 5.3 Mathematical Formulation of Core Components

#### 5.3.1 Continuous Operator Geometry ($q \in \mathbb{R}^8$)
Each ECG channel $c \in \{1, \dots, C\}$ is associated with a physical lead projection vector $q_c \in \mathbb{R}^8$ in the canonical 8-lead clinical basis $[I, II, V_1, V_2, V_3, V_4, V_5, V_6]^T$:
- Nominal Lead I: $q_I = [1, 0, 0, 0, 0, 0, 0, 0]^T$
- Nominal Lead II: $q_{II} = [0, 1, 0, 0, 0, 0, 0, 0]^T$
- Non-standard / wearable vectors: projected via orthogonal transformation into $\mathbb{R}^8$.

The continuous operator encoder $g_\phi: \mathbb{R}^8 \to \mathbb{R}^{d_q}$ is an MLP with GeLU activations:
$$e_c = g_\phi(q_c) = W_2 \text{GeLU}(W_1 q_c + b_1) + b_2, \quad d_q = 64$$
Because $q_c$ is continuous and geometrically grounded, $e_c$ uniquely identifies the lead orientation without discrete ID collapse, and allows seamless interpolation across arbitrary lead combinations.

#### 5.3.2 Modulated Phase-KME with Distributional Morphology
For each detected beat $k$ and channel $c$, the phase-normalized voltage profile is mapped via Nyström kernel landmarks $Z \in \mathbb{R}^{128 \times D}$ into an empirical kernel mean embedding:
$$\mu_{c, k} = \frac{1}{|P_{c, k}|} \sum_{x \in P_{c, k}} k(x, Z) \in \mathbb{R}^{16 \times 128}$$
The ModulatedPhaseCNN modulates the feature maps with the operator embedding $e_c$:
$$h_{c, k} = \text{PhaseCNN}(\mu_{c, k}) \odot \sigma(W_m e_c)$$
To prevent morphology variance collapse, we compute distributional statistics across the beat set:
$$z_{c}^{\text{dist}} = [\text{Mean}_{k}(h_{c, k}) \,\|\, \text{Var}_{k}(h_{c, k})] \in \mathbb{R}^{2 \cdot d_h}$$

#### 5.3.3 Ordered-Beat Sequence Dynamics (D037)
AF is fundamentally a rhythm disorder defined by irregular inter-beat intervals $\Delta t_k = t_{k} - t_{k-1}$. Treating beats as an unordered set discards this signal. We encode the beat sequence using an order-preserving sequential module:
$$u_{c, k} = [h_{c, k} \,\|\, \text{PE}(k) \,\|\, \log(\Delta t_k) \,\|\, \Delta t_k / \bar{\Delta t}]$$
$$v_{c, k} = \text{BiGRU}(u_{c, k})$$
The resulting sequence captures both local morphology and transition dynamics.

#### 5.3.4 Residual ResNet Fusion (I082)
To combine the invariant representation of the SetOperator with the raw high-frequency fidelity of the 1D ResNet, we perform residual fusion:
$$z_{\text{set}} = \text{SetTransformer}(\{v_{c, k}\}_{c \in \mathcal{C}_{\text{active}}}) \in \mathbb{R}^{256}$$
$$z_{\text{raw}} = \text{ResNet1D}(X_{\text{raw}}, M_{\text{mask}}) \in \mathbb{R}^{256}$$
$$z_{\text{fused}} = \text{LayerNorm}(z_{\text{set}} + W_f [z_{\text{set}} \,\|\, z_{\text{raw}}])$$
When channels are missing or raw waveforms are corrupted, $z_{\text{set}}$ maintains robustness, while $z_{\text{raw}}$ provides exact boundary sharpness when channels are present.

### 5.4 Training Plan & Loss Function
- **Objective**: Patient-weighted binary cross-entropy with temperature calibration:
  $$\mathcal{L} = - \sum_{i=1}^B w_i \left[ y_i \log \sigma(s_i) + (1 - y_i) \log (1 - \sigma(s_i)) \right]$$
  where $w_i = \frac{1}{N_{p(i)} \cdot T_{p(i)}}$ enforces equal total weight per training patient ($sum = 1$ per patient).
- **Multi-Acquisition Schedule Cycling**:
  Every training batch cycles through three acquisition masks: Channel 0 only ($M=[1, 0]$), Channel 1 only ($M=[0, 1]$), and Both channels ($M=[1, 1]$), ensuring equal exposure and robust single-lead fallback.
- **Optimization**: AdamW, learning rate $1 \times 10^{-4}$, cosine decay, weight decay $1 \times 10^{-2}$, batch size 256, BF16 mixed-precision training with strict FP32 checkpoint selection.
- **Deterministic Checkpoint Selection**: Checkpoint selected strictly by minimum patient-weighted validation BCE loss across all three acquisition schedules.

---

## 6. Claim-Driven Validation Sketch

### Claim 1: Non-Degeneracy & Representation Stability
*Claim*: Continuous $q$-conditioning and ordered-beat dynamics completely eliminate representation collapse, producing diverse active-token latents ($SD > 0.10$) and meaningful validation score variance ($SD > 0.25$) compared to collapsed categorical SetOperators ($SD < 1 \times 10^{-7}$).
- *Minimal Experiment*: Development evaluation on 166,047 validation windows under all three acquisition modes comparing A001 (Continuous SetOperator) vs A002 (Categorical) vs A003 (Identity-Ablated).
- *Decisive Metric*: Active token activation standard deviation and prediction score standard deviation.
- *Expected Evidence*: A001 maintains score $SD \ge 0.30$, whereas A002 collapses to $SD < 1 \times 10^{-6}$.

### Claim 2: Superior Episode Delineation & Continuous Density Accuracy
*Claim*: The proposed architecture significantly reduces Charitos continuous density MAE, binned density MAE, and episode boundary error compared to matched raw ResNet and observed-RR baselines under D2 HSMM temporal decoding.
- *Minimal Experiment*: Full validation cohort evaluation (60 daily recordings, 30 patients, 11 informative multi-interval patients) under patient-clustered bootstrap ($B=2,000$).
- *Decisive Metric*: Continuous density MAE, Binned density MAE, Episode Sensitivity, PPV, and Mean Boundary Error (seconds).
- *Expected Evidence*: Continuous density MAE reduction $\ge 25\%$ relative to raw ResNet, with boundary onset/offset error $< 8.0$ seconds.

### Claim 3: Invariance to Lead Dropout and Channel Permutation
*Claim*: Permuting channels or dropping a lead produces a shift of $< 0.15\%$ points in estimated AF burden and density, proving true operator invariance.
- *Minimal Experiment*: Permutation audit on all 166,047 validation windows swapping Channel 0 and Channel 1 with corresponding $q$ vectors.
- *Decisive Metric*: Maximum absolute logit difference $|\hat{s}(C_0, C_1) - \hat{s}(C_1, C_0)|$.
- *Expected Evidence*: Logit difference $< 1 \times 10^{-5}$ (machine precision parity).

---

## 7. Experiment Handoff Inputs
- **Must-Prove Claims**: Elimination of representation collapse; superior density MAE; lead permutation invariance.
- **Must-Run Ablations**:
  1. A001 (Continuous D1 SetOperator) vs A002 (Categorical SetOperator).
  2. D032 (Distributional morphology) vs standard mean pooling.
  3. D037 (Ordered beat dynamics) vs unordered set.
  4. I082 (Residual ResNet fusion) vs SetOperator-only vs ResNet-only.
  5. F053 (No Transformer baseline).
- **Critical Datasets**: IRIDIA-AF (internal benchmark), SHDB (external 98 records), LTAF (external 84 records).
- **Highest-Risk Assumptions**: Stationarity of the censored lognormal duration distributions across different clinical populations.

---

## 8. Compute & Timeline Estimate
- **Model Training**: ~3.5 hours per full training run (3 epochs cycling acquisition masks, 504,777 windows) on 1x A100-SXM4-80GB.
- **Validation Replay & Temporal Decoding**: ~45 minutes for full D0/D1/D2 pipeline across 60 validation files.
- **External Evaluation**: ~1.2 hours for SHDB (279,686 windows) and ~1.0 hour for LTAF (232,233 windows).
- **Total Compute Budget**: ~24 GPU-hours.
