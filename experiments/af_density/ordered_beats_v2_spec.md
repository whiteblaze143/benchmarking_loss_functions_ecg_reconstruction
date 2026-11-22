# Architecture Specification: True Ordered-Beat Sequence Dynamics (D037-v2)

**Author:** Model & Registry Architect (`@architect`)  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Program  
**Status:** Approved Technical Specification (Stage 3 Core Milestone)  
**Date:** November 22, 2026  
**Implementation Target:** `experiments/af_density/ordered_beats_v2.py` / `D037-v2` in `continuous_setoperator_models.py`  

---

## 1. Scientific Background & The Sequence Modeling Problem

In the Stage 1 pilot experiments, model `D037-v1` was described as an "order-preserving beat sequence encoder". However, code inspection revealed that `OrderPreservingBeatSetOperatorD037` applied 1D convolutions along the **16 phase cells of an already time-aggregated response token**:
$$\mathbf{H} = \text{Conv1d}_{1 \to 3}(\mathbf{R}_{B \times M, 128, 16})$$

Because the 16 phase cells represent spatial locations along the cardiac cycle (P-QRS-T), this convolution modeled **intra-cycle phase continuity**, not the **temporal sequence of successive heartbeats over time**. It did not ingest individual beats, beat order, timestamps, or RR intervals.

**Stage 3 establishes `D037-v2`**: an authentic sequence-modeling architecture that ingests the chronological sequence of $K$ distinct heartbeats occurring over a 30-second window:
$$t_1 < t_2 < \dots < t_K$$
with exact inter-beat intervals $\Delta t_k = t_k - t_{k-1}$ and beat-specific morphology representations.

---

## 2. Mathematical Formulation of Ordered Beat Inputs

Let a 30-second ECG window contain $K$ detected heartbeats indexed chronologically by $k \in \{1, \dots, K\}$.

```
ECG Voltage Trace (30s)
  │
  ├── Beat 1 (t_1) ────► [Morphology Vector e_1] ──► [Δt_1, log Δt_1]
  ├── Beat 2 (t_2) ────► [Morphology Vector e_2] ──► [Δt_2, log Δt_2]
  ├── Beat 3 (t_3) ────► [Morphology Vector e_3] ──► [Δt_3, log Δt_3]
  │   ...
  └── Beat K (t_K) ────► [Morphology Vector e_K] ──► [Δt_K, log Δt_K]
            │
            ▼
   Beat Token Matrix B ∈ R^(B × M × K_max × D_beat)
   Beat Mask M_beat ∈ {0, 1}^(B × M × K_max)
            │
            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Sequence Backbone (BiGRU / TCN / RelTime-Transformer)   │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
            Pooled Beat Sequence Token h_seq ∈ R^128
                              │
                              ▼
        [Cat with Continuous Operator Q] ──► SetOperator
```

### 2.1. Beat Token Vector Construction
For heartbeat $k$ in channel $m$, the beat feature vector $\mathbf{b}_{m, k} \in \mathbb{R}^{D_{\text{beat}}}$ is defined as:
$$\mathbf{b}_{m, k} = \left[ \mathbf{e}_{\text{morph}}(m, k); \; \Delta t_k; \; \log(\Delta t_k + \epsilon); \; \Delta^2 t_k; \; q_{m, k} \right]$$
where:
* $\mathbf{e}_{\text{morph}}(m, k) \in \mathbb{R}^{128}$ is the 128-dimensional Phase-KME beat morphology embedding (or pooled phase representation) for beat $k$.
* $\Delta t_k = t_k - t_{k-1}$ is the current RR interval in seconds ($0.2 \le \Delta t_k \le 3.0$).
* $\log(\Delta t_k + \epsilon)$ provides log-linear sensitivity to heart rate acceleration/deceleration.
* $\Delta^2 t_k = \Delta t_k - \Delta t_{k-1}$ represents beat-to-beat interval acceleration (second derivative).
* $q_{m, k} \in [0, 1]$ is a signal quality / delineation confidence score for beat $k$.

Total beat vector dimension: $D_{\text{beat}} = 128 + 1 + 1 + 1 + 1 = 132$ (projected to 128 via linear projection).

### 2.2. Variable Sequence Length Handling
Because physiological heart rates vary (e.g. 30 bpm to 180 bpm), the number of beats $K$ in a 30-second window ranges from $K_{\min} = 15$ to $K_{\max} = 90$.
* All batches are padded to $K_{\max} = 96$ beats.
* A boolean beat mask $\mathbf{M}_{\text{beat}} \in \{0, 1\}^{B \times M \times K_{\max}}$ is passed:
  $$M_{\text{beat}}(b, m, k) = \begin{cases} 0 & \text{if } k \le K_{b, m} \text{ (valid beat)} \\ 1 & \text{if } k > K_{b, m} \text{ (padding)} \end{cases}$$

---

## 3. Sequence Architecture Portfolio (Stage 3 Comparisons)

Stage 3 implements and benchmarks four distinct temporal sequence encoders for the beat tokens:

### 3.1. Ordered-Beat Bidirectional GRU (`D037-v2-BiGRU`)
Processes the beat sequence forward and backward in time:
$$\mathbf{h}_k^{(f)} = \text{GRU}_{\text{fwd}}(\mathbf{b}_k, \mathbf{h}_{k-1}^{(f)}), \quad \mathbf{h}_k^{(b)} = \text{GRU}_{\text{bwd}}(\mathbf{b}_k, \mathbf{h}_{k+1}^{(b)})$$
$$\mathbf{h}_k = [\mathbf{h}_k^{(f)}; \; \mathbf{h}_k^{(b)}] \in \mathbb{R}^{256}$$
* **Architecture:** 2-layer BiGRU, hidden dimension 128 per direction, total 256.
* **Masked Pooling:** Masked mean pooling across valid beats $k \in \{1, \dots, K\}$ followed by linear projection to 128-D response token.
* **Parameter Count:** ~395,000 parameters.

### 3.2. Temporal Convolutional Network (`D037-v2-TCN`)
Stack of dilated 1D causal convolutions over the beat index dimension:
* **Kernel Size:** $K=3$.
* **Dilations:** $d \in \{1, 2, 4, 8, 16\}$, providing an effective receptive field of $\sum 2 \times 3 \times d > 64$ beats, covering the entire 30-second window.
* **Residual Connections:** Weight normalization, GELU, and residual skip connections per block.
* **Parameter Count:** ~280,000 parameters.

### 3.3. Beat-Level Relative-Time Transformer (`D037-v2-RelTrans`)
Multi-head self-attention where attention weights are modulated by continuous temporal distances between heartbeats:
$$A_{i, j} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} + \psi(t_i - t_j)$$
where $\psi(t_i - t_j) = -\gamma |t_i - t_j|$ is an exponential decay prior based on physical elapsed time in seconds, rather than discrete beat indices.
* **Architecture:** 2 layers, $d_{\text{model}}=128$, 4 attention heads, feedforward dimension 256.
* **Parameter Count:** ~320,000 parameters.

### 3.4. Compact Selective State-Space Model (`D037-v2-S4D`)
Continuous diagonal state-space model parameterizing beat-to-beat recurrence with input-dependent discretization. Provides linear $O(K)$ computational complexity.

---

## 4. Mandatory Negative Control Suite (P0.3 Audit Controls)

To scientifically determine whether AF classification benefits from **temporal beat order**, **interval dynamics**, or merely **morphology aggregation**, `D037-v2` mandates four controlled ablations:

| Control ID | Experimental Manipulation | Invariant Preserved | Hypotheses Isolated |
| :--- | :--- | :--- | :--- |
| **Control 1 (Temporal Shuffling)** | Randomly permute beat order: $k \sim \pi(k)$ independently per window | Morphology distribution and RR histogram perfectly preserved | If performance drops, **temporal beat order** is causally necessary. If performance is unchanged, the model operates as an unordered "bag of beats". |
| **Control 2 (Timestamp Ablation)** | Set $\Delta t_k = 0$ and $t_k = 0$ for all beats | Only sequential morphology transitions $\mathbf{e}_k \to \mathbf{e}_{k+1}$ preserved | Tests whether **rhythm dynamics** (RR irregularity) drive performance vs. sequential morphology change. |
| **Control 3 (Morphology Ablation)** | Replace $\mathbf{e}_k$ with constant mean vector $\bar{\mathbf{e}}$ | True RR interval sequence $\Delta t_1, \dots, \Delta t_K$ preserved | Isolates the predictive power of **pure ventricular rhythm chaos** without morphology. |
| **Control 4 (Renewal RR Replacement)** | Replace $\Delta t_k$ with independent draws from the patient's global RR distribution | Mean and variance of RR preserved; Markov transition structure destroyed | Tests whether **beat-to-beat transition structure** matters vs. stationary interval dispersion. |

---

## 5. Sequence Data Contract & Provenance

```python
# Specification for D037-v2 forward pass signature
def forward(
    self,
    beat_tokens: torch.Tensor,       # (B, M, K_max, 128) per-beat morphology
    rr_intervals: torch.Tensor,      # (B, M, K_max) delta_t in seconds
    beat_timestamps: torch.Tensor,   # (B, M, K_max) physical timestamp in seconds
    operators: torch.Tensor,         # (B, M, 8) continuous channel operators
    beat_mask: torch.Tensor,         # (B, M, K_max) bool mask (True = padding)
    channel_mask: torch.Tensor,      # (B, M) bool mask (True = unobserved lead)
) -> Dict[str, torch.Tensor]:
    ...
```

### Data Extraction Pipeline Requirements
1. **R-Peak Detector:** Pan-Tompkins + Hamilton-Tompkins consensus detector applied to raw voltage at 200 Hz.
2. **Interval Validation:** Filter physiological outliers ($\Delta t_k < 0.20\text{s}$ or $\Delta t_k > 3.0\text{s}$ marked as ectopic/artifact with quality flag $q_k = 0$).
3. **Window Alignment:** Beat timestamps are anchored to the physical start of each 30.0-second window ($0.0 \le t_k \le 30.0$).
4. **Boundary Censoring:** Intervals crossing window boundaries are flagged with boundary status $\text{censored} = 1$.

---

## 6. Clinical Electrophysiological Significance

Atrial fibrillation exhibits two hallmark electrophysiological phenomena:
1. **Irregularly Irregular Ventricular Response:** Decremental conduction through the atrioventricular (AV) node causes chaotic, non-renewal RR interval sequences with heavy-tailed dispersion and lack of rhythmic periodicity.
2. **Absence of Discrete P-Waves & Fibrillatory Baseline Drift:** Between ventricular depolarizations, atrial tissue generates high-frequency, disorganized micro-reentrant fibrillatory wavelets (350–600 bpm), manifesting as baseline morphological irregularity.

`D037-v2` unites these two phenomena into a single unified temporal sequence model, allowing the neural network to learn the exact cross-covariance between beat morphology degradation and interval chaos.
