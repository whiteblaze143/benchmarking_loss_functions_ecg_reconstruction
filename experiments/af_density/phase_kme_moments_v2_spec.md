# Architecture Specification: True Across-Beat Phase-KME Distributional Moments (D032-v2)

**Author:** Model & Registry Architect (`@architect`)  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Program  
**Status:** Approved Technical Specification (Stage 2 Core Milestone)  
**Date:** November 22, 2026  
**Implementation Target:** `experiments/af_density/phase_kme_moments_v2.py` / `D032-v2` in `continuous_setoperator_models.py`  

---

## 1. Scientific Background & The Distributional Imperative

In the Stage 1 pilot experiments, model `D032-v1` achieved the top performance among all models (Local AUROC 0.8523, Balanced Accuracy 0.7751, Continuous Timeline Density MAE 0.2575). However, architectural audit revealed that `D032-v1` computed its standard deviation **across the 16 phase cells** of the already temporally pooled Phase-KME representation:
$$\sigma_{\text{phase}, d} = \sqrt{\frac{1}{16}\sum_{p=1}^{16} \left( x_{p, d} - \bar{x}_d \right)^2}$$

Because phase cells $p \in \{1, \dots, 16\}$ index standardized temporal locations within a canonical cardiac cycle (e.g. $p=1..3$ P-wave, $p=6..9$ QRS, $p=12..15$ T-wave), this standard deviation captured **intra-beat spatial morphology dispersion across the cardiac cycle**, rather than **inter-beat variability across heartbeats**.

While intra-beat phase dispersion was shown to be highly predictive of AF (likely because atrial fibrillation alters baseline phase variance between QRS complexes), it did **not** capture true cardiac cycle-to-cycle variability, morphology alternans, or fibrillatory wave dynamics.

**Stage 2 establishes `D032-v2`**: an architecture that computes the true across-beat empirical distribution over $K$ individual detected cardiac cycles in each 30-second window.

---

## 2. Mathematical Formulation of True Across-Beat Moments

Let a 30-second window contain $K$ detected heartbeats indexed by $k \in \{1, \dots, K\}$, each delineated into 16 standardized cardiac phase cells $p \in \{1, \dots, 16\}$.

For channel $m \in \{1, \dots, M\}$, let:
$$\Phi_{k, p, :} \in \mathbb{R}^{D}$$
represent the $D$-dimensional Nyström kernel mean embedding feature vector for heartbeat $k$ at phase cell $p$ (standard $D=128$).

```
[Raw 30s ECG Window]
       │
       ▼
 [R-Peak Detection & Delineation] ──► K individual cardiac cycles (k = 1, ..., K)
       │
       ▼
 [Phase-KME Embedding per Beat]  ──► Φ[k, p, d] ∈ R^(K × 16 × 128)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 [Across-Beat Mean]             [Across-Beat Variance]
 μ[p, d] = (1/K) Σ_k Φ[k, p, d]   σ^2[p, d] = (1/K) Σ_k (Φ[k, p, d] - μ[p, d])^2
       │                                 │
       └────────────────┬────────────────┘
                        ▼
            [Concatenation & Projection]
     R[p, :] = W_proj [ μ[p, :] ;  √(σ^2[p, :] + ε) ] + b_proj  ∈ R^(16 × 128)
                        │
                        ▼
             [SetOperator Response Encoder]
```

### 2.1. First Moment: Empirical Mean Representation
The empirical mean vector captures the stationary, time-averaged morphological profile across all heartbeats:
$$\mu_{p, d} = \frac{1}{K} \sum_{k=1}^K \Phi_{k, p, d}, \quad \forall p \in \{1, \dots, 16\}, \; d \in \{1, \dots, D\}$$
In matrix form: $\boldsymbol{\mu}_p \in \mathbb{R}^D$.

### 2.2. Second Moment: Across-Beat Variance Representation
The empirical variance vector captures the cycle-to-cycle morphological dispersion at each specific phase of the cardiac cycle:
$$\sigma_{p, d}^2 = \frac{1}{K} \sum_{k=1}^K \left( \Phi_{k, p, d} - \mu_{p, d} \right)^2, \quad \forall p \in \{1, \dots, 16\}, \; d \in \{1, \dots, D\}$$
In matrix form: $\boldsymbol{\sigma}_p^2 \in \mathbb{R}^D$.

### 2.3. Dual-Moment Response Fusion
The distributional response token $\mathbf{R}_p$ for each phase cell is formed by projecting the concatenated mean and standard deviation:
$$\mathbf{R}_p = \text{GELU}\left( \mathbf{W}_{\text{proj}} \left[ \boldsymbol{\mu}_p; \; \sqrt{\boldsymbol{\sigma}_p^2 + \epsilon} \right] + \mathbf{b}_{\text{proj}} \right) \in \mathbb{R}^{D}$$
where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{D \times 2D}$, $\mathbf{b}_{\text{proj}} \in \mathbb{R}^{D}$, and $\epsilon = 1 \times 10^{-6}$ ensures numerical gradient stability.

---

## 3. Small-Beat-Count Policy and Boundary Conditions

In ambulatory monitoring, severe bradycardia, extreme pause events, or transient signal loss can result in low heartbeat counts $K$ within a 30-second window. `D032-v2` enforces a strict, deterministic boundary policy:

| Heartbeat Count $K$ | Clinical Context | Statistical Validity | Variance Formulation $\boldsymbol{\sigma}_p^2$ | Confidence Flag $c_K$ |
| :--- | :--- | :--- | :--- | :--- |
| **$K = 0$** | Asystole, lead disconnect, or complete detection failure | **Invalid / Indeterminate** | Undefined $\to$ raises `IndeterminateWindowError` | $0.0$ |
| **$K = 1$** | Extreme bradycardia / ventricular pause (1 beat / 30s) | Single sample: variance mathematically undefined | Forced to $\mathbf{0}$ ($\boldsymbol{\sigma}_p^2 = \mathbf{0}$); only $\boldsymbol{\mu}_p$ evaluated | $0.2$ (Low confidence) |
| **$2 \le K < 5$** | Severe bradycardia (4–10 bpm) | Low degrees of freedom | Sample variance with small-sample Bessel correction: $\frac{1}{K-1}\sum(\Phi - \mu)^2$ | $K / 5.0$ |
| **$K \ge 5$** | Normal physiological heart rates ($\ge 10$ bpm) | High degrees of freedom | Standard empirical variance ($N$-denominator for maximum likelihood stability) | $1.0$ (Full confidence) |

```python
# Deterministic small-beat-count implementation
if K == 0:
    raise ValueError("Zero beats detected: window must remain indeterminate")
elif K == 1:
    mu = phi.squeeze(0)  # (16, D)
    sigma = torch.zeros_like(mu)
    conf = 0.2
else:
    mu = phi.mean(dim=0)
    sigma = phi.std(dim=0, unbiased=False)
    conf = min(1.0, K / 5.0)

fused_response = self.proj(torch.cat([mu, sigma], dim=-1)) * conf
```

---

## 4. Distributional Variations and Matched Controls (Stage 2 Matrix)

To rigorously verify that second-moment variance adds genuine clinical information beyond mean morphology, Stage 2 mandates an eight-variant ablation suite:

| Ablation ID | Model Name | Response Formulation $\mathbf{R}_p$ | Hypothesis Tested |
| :--- | :--- | :--- | :--- |
| **C021** | Mean Only | $\mathbf{W} \boldsymbol{\mu}_p$ | Classical stationary Phase-KME baseline |
| **C022** | Variance Only | $\mathbf{W} \boldsymbol{\sigma}_p$ | Tests whether variance alone is sufficient to diagnose AF |
| **D032-v2** | Mean + Variance (Sample) | $\mathbf{W} [\boldsymbol{\mu}_p; \boldsymbol{\sigma}_p]$ | Primary distributional hypothesis |
| **C023** | Mean + Robust MAD | $\mathbf{W} [\text{median}(\Phi_p); \text{MAD}(\Phi_p)]$ | Robustness against single ectopic beats (PVCs) |
| **C024** | Mean + Quantiles | $\mathbf{W} [Q_{0.25}; Q_{0.50}; Q_{0.75}]$ | Full non-parametric distributional shape |
| **C025** | Low-Rank Covariance | $\mathbf{W} [\boldsymbol{\mu}_p; \text{diag}(\mathbf{U}\mathbf{S}\mathbf{U}^\top)]$ | Cross-phase covariance structure |
| **C026** | Quality-Weighted Moments | $\mathbf{W} [\sum w_k \Phi_k; \sum w_k (\Phi_k - \mu)^2]$ | Mitigates contamination by noisy beats |
| **D032-v1** | Phase-Cell Dispersion | Intra-beat dispersion over 16 phase cells | Benchmark comparison against Stage 1 top performer |

### Robust MAD Formulation (C023)
$$\text{MAD}_p = \text{median}_{k} \left| \Phi_{k, p, :} - \text{median}_{j}(\Phi_{j, p, :}) \right|$$
The Median Absolute Deviation provides a breakdown point of 50%, ensuring that occasional premature ventricular contractions (PVCs) or motion artifacts do not falsely inflate the AF variance metric.

---

## 5. Feature Cache Data Contract & Leakage Prevention

### 5.1. Per-Beat Feature Cache Specification
Generating `D032-v2` requires a versioned, immutable HDF5/NPZ per-beat feature cache:
* **Storage Path:** `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/beat_features_v2/`
* **Tensor Layout:**
  * `phi_beats`: Shape `(N_windows, 2_channels, K_max, 16, 128)`, `float32`.
  * `beat_counts`: Shape `(N_windows, 2_channels)`, `int16`.
  * `beat_timestamps_ms`: Shape `(N_windows, 2_channels, K_max)`, `int32`.
  * `beat_valid_mask`: Shape `(N_windows, 2_channels, K_max)`, `bool`.
  * `window_ids`: Shape `(N_windows,)`, `int64`.

### 5.2. Strict Leakage Isolation
1. **Landmark Fitting:** The 128 Nyström landmark points must be fitted **strictly on the 90 training patients**.
2. **Whitening Transforms:** Whitening matrices $\mathbf{W}_{\text{white}}$ must be computed strictly on training patient beats.
3. **Zero Test Access:** No per-beat features from the 32 held-out test patients may be generated or loaded during Stage 2 development.

---

## 6. Clinical and Electrophysiological Interpretation

In human cardiac electrophysiology, true across-beat variance $\boldsymbol{\sigma}_p^2$ reflects distinct pathological mechanisms across specific phase cells:

1. **Phase Cells 1–4 (Atrial Depolarization / Baseline):** In normal sinus rhythm, atrial depolarization produces consistent, low-variance P-waves ($\boldsymbol{\sigma}_{1..4}^2 \approx 0$). In atrial fibrillation, coordinated P-waves are replaced by continuous, chaotic fibrillatory waves (f-waves) with variable amplitude and frequency, resulting in **dramatically elevated across-beat baseline variance**.
2. **Phase Cells 6–9 (Ventricular Depolarization / QRS):** In AF with narrow QRS, ventricular depolarization morphology remains relatively constant across beats ($\boldsymbol{\sigma}_{6..9}^2 \approx \text{baseline}$). However, the presence of aberrant ventricular conduction (Ashman phenomenon) will produce high QRS variance.
3. **Phase Cells 12–15 (Ventricular Repolarization / T-Wave):** Reflects cycle-length-dependent repolarization changes and T-wave alternans.

By preserving phase-specific across-beat moments, `D032-v2` allows the SetOperator to clinically localize morphological chaos specifically to atrial vs. ventricular phases.
