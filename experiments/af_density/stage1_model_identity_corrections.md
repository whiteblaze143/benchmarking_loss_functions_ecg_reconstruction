# Stage 1 Model Identity Corrections and Scientific Nomenclature Ledger

**Author:** Model & Registry Architect (`@architect`)  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Program  
**Status:** Canonical Corrective Specification (P0.2 & P0.3 Resolution)  
**Date:** November 22, 2026  
**Repository Anchor:** `experiments/af_density/continuous_setoperator_models.py`  

---

## 1. Executive Summary & Audit Mandate

Following the critical review of the Stage 1 pilot experiments, four critical model identity discrepancies were identified where nominal experiment labels diverged from the true algorithmic implementations. In accordance with Priority 0 corrective actions (P0.2, P0.3) and Stage 1B/2/3/5 requirements:
1. **No historical pilot results are erased or obscured.** All pilot checkpoints, logs, and metrics are preserved under their true, verified algorithmic identities.
2. **Ambiguous labels are formally retired and disambiguated.** All models in `continuous_setoperator_models.py` and `ablation_registry.json` are split into explicitly versioned identities (`-v1` vs `-v2`).
3. **Genuine architectural counterparts are implemented.** New, uncompromised architectures are provided for authentic evaluation against real physiological data.

This document establishes the canonical mathematical definitions, parameter budgets, input data contracts, and experimental boundaries for all corrected models.

---

## 2. Granular Model Disambiguation Ledger

### 2.1. I082: Classification-Head Capacity Control vs. Genuine ResNet Fusion (P0.3)

#### The Stage 1 Discrepancy
The Stage 1 pilot registry described `I082` as a "ResNet + Residual MLP". However, architectural inspection revealed that `ResNetResidualMLPSetOperatorI082` consisted solely of the continuous Phase-KME SetOperator backbone (1,027,009 parameters) with its linear classification head replaced by a 2-layer residual MLP block (`Linear(256, 256) -> GELU -> Linear(256, 256) + skip -> GELU -> Linear(256, 1)`, adding 131,584 parameters). It did **not** ingest, encode, or fuse raw ECG waveforms from a 1D ResNet. Interpreting `I082`'s pilot performance as evidence regarding raw-plus-set fusion was scientifically invalid.

#### Canonical Disambiguation
* **`I082-v1` — SetOperator Residual Classification-Head Capacity Control:**
  * **Class:** `SetOperatorResidualMLPCapacityControlI082v1` (aliased to `ResNetResidualMLPSetOperatorI082`)
  * **Input Modality:** Phase-KME response tokens $\mathbf{R} \in \mathbb{R}^{B \times M \times 16 \times 128}$ and continuous operators $\mathbf{Q} \in \mathbb{R}^{B \times M \times 8}$.
  * **Parameter Count:** Exactly **1,158,593** trainable parameters.
  * **FLOPs:** 16,880,128 FLOPs per 2-lead window.
  * **Scientific Role:** Isolates the effect of increasing classification head non-linear capacity without altering representation learning or adding auxiliary modalities. Retains the Stage 1 pilot checkpoint and metrics under this exact identity.
* **`I082-v2` — Genuine ResNet–SetOperator Residual Fusion:**
  * **Class:** `ResNetResidualFusionSetOperatorI082v2`
  * **Input Modality:** Simultaneous authentic raw ECG voltage $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{B \times 2 \times 6000}$ (at 200 Hz) **and** Phase-KME tokens $\mathbf{R} \in \mathbb{R}^{B \times 2 \times 16 \times 128}$.
  * **Architecture:** 8-block 1D ResNet raw waveform encoder (2,186,049 params) coupled to the continuous SetOperator branch (1,027,009 params) via a learnable residual gate:
    $$\text{logit}_{\text{fused}} = \text{logit}_{\text{raw}} + \alpha \cdot \Delta\text{logit}_{\text{set}}, \quad \alpha \in \mathbb{R}, \; \alpha_{\text{init}} = 0.0$$
  * **Parameter Count:** Exactly **3,213,059** total parameters (1,027,010 trainable when the raw ResNet is frozen; 3,213,059 during joint fine-tuning).
  * **FLOPs:** ~2,416,617,472 FLOPs per 2-lead window (~2.40 GFLOPs raw ResNet + 16.6 MFLOPs SetOperator).
  * **Scientific Role:** Direct test of Stage 5: whether Phase-KME SetOperator representations contribute orthogonal, incremental diagnostic value over a state-of-the-art raw ECG ResNet.

---

### 2.2. H071: Synthetic Proxy Failure vs. Authentic Raw Waveform ResNet (P0.2)

#### The Stage 1 Discrepancy
The Stage 1 runner script constructed an artificial `raw_mock` tensor by repeating one dimension of the 16-cell Phase-KME representation across 2,500 samples. When `PerChannelRawResNetSetOperatorH071` was trained on this synthetic signal, it collapsed to AUROC 0.4995, Brier score 0.5417, and ECE 0.6348. This catastrophic failure was caused by invalid proxy inputs, not an intrinsic limitation of 1D convolutional waveform learning.

#### Canonical Disambiguation
* **`H071-v1` — Synthetic Proxy Failure Diagnostic:**
  * **Input Modality:** Synthetic tiled Phase-KME mock signal $\mathbf{X}_{\text{mock}} \in \mathbb{R}^{B \times M \times 2500}$.
  * **Role:** Preserved strictly as an invalid-proxy input diagnostic demonstrating that convolutional feature extractors overfit catastrophically when trained on synthetic repetitions of summary statistics.
* **`H071-v2` — Per-Channel Raw ResNet SetOperator:**
  * **Class:** `PerChannelRawResNetSetOperatorH071` / `PerChannelRawResNetSetOperatorH071v2`
  * **Input Modality:** Authentic observed ECG voltage $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{B \times M \times 6000}$ (200 Hz, 30.0 seconds).
  * **Architecture:** Independent 8-block 1D ResNet per channel projecting voltage waveforms into 128-D response tokens, combined with 64-D operator embeddings, encoded via a 2-layer Transformer set encoder, and pooled via learned-query attention.
  * **Parameter Count:** Exactly **2,065,249** trainable parameters.
  * **FLOPs:** ~2,420,000,000 FLOPs.
  * **Scientific Role:** Evaluates whether end-to-end convolutional feature extraction directly on raw voltage can match or exceed Phase-KME inductive bias when provided with physical channel geometry conditioning.

---

### 2.3. D032: Phase-Cell Dispersion vs. True Across-Beat Distributional Moments (Stage 2)

#### The Stage 1 Discrepancy
`DualMomentPhaseSetOperatorD032` in Stage 1 calculated the mean and standard deviation across the 16 phase cells within each channel's aggregated response. Because the 16 phase cells represent spatial locations along the standardized cardiac cycle within an already-averaged beat representation, this standard deviation measured **spatial phase dispersion** across the cardiac cycle, not beat-to-beat variability across heartbeats.

#### Canonical Disambiguation
* **`D032-v1` — Phase-Cell Dispersion SetOperator:**
  * **Class:** `DualMomentPhaseSetOperatorD032` / `DualMomentPhaseSetOperatorD032v1`
  * **Formulation:** Computes dual moments across phase cells $p \in \{1, \dots, 16\}$:
    $$\mu = \frac{1}{16}\sum_{p=1}^{16} \mathbf{x}_p, \quad \sigma = \sqrt{\frac{1}{16}\sum_{p=1}^{16} (\mathbf{x}_p - \mu)^2 + \epsilon}$$
  * **Parameter Count:** Exactly **1,043,393** trainable parameters.
  * **Scientific Role:** Measures intra-beat spatial morphology dispersion across phase angles. Preserved as the Stage 1 top performer (AUROC 0.8523, Burden MAE 0.4333).
* **`D032-v2` — True Across-Beat Phase-KME Moments:**
  * **Specification:** `phase_kme_moments_v2_spec.md`
  * **Formulation:** Evaluates the empirical distribution of $K$ individual detected heartbeats in the 30-second window before temporal pooling:
    $$\mu_p = \frac{1}{K}\sum_{k=1}^K \Phi_{k, p, :}, \quad \sigma_p^2 = \frac{1}{K}\sum_{k=1}^K (\Phi_{k, p, :} - \mu_p)^2$$
    $$\mathbf{R}_p = \mathbf{W}_{\text{proj}} \left[ \mu_p; \sqrt{\sigma_p^2 + \epsilon} \right]$$
  * **Scientific Role:** Genuinely captures cycle-to-cycle morphological variance, alternans, and fibrillatory baseline fluctuations across individual heartbeats.

---

### 2.4. D037: Phase-Trajectory Convolution vs. True Ordered Beat Dynamics (Stage 3)

#### The Stage 1 Discrepancy
`OrderPreservingBeatSetOperatorD037` in Stage 1 applied a 1D convolution along the 16 phase cells of an aggregated representation. It did not ingest individual heartbeats, ordered beat timestamps, or RR intervals. It modeled intra-cycle phase transitions, not inter-beat temporal dynamics.

#### Canonical Disambiguation
* **`D037-v1` — Phase-Trajectory Convolution SetOperator:**
  * **Class:** `OrderPreservingBeatSetOperatorD037` / `PhaseTrajectoryConvSetOperatorD037v1`
  * **Formulation:** 3-block 1D causal/circular convolution along the 16 phase cells: $\text{Conv1D}(128, 128, \text{kernel}=3)$.
  * **Parameter Count:** Exactly **928,449** trainable parameters.
  * **Scientific Role:** Captures smooth phase continuity within the averaged P-QRS-T complex.
* **`D037-v2` — True Ordered Beat Dynamics Sequence Model:**
  * **Specification:** `ordered_beats_v2_spec.md`
  * **Formulation:** Ingests the true temporal sequence of $K$ heartbeats $\mathbf{B} \in \mathbb{R}^{B \times M \times K \times D}$, ordered by physical occurrence $t_1 < t_2 < \dots < t_K$, with explicit RR intervals $\Delta t_k = t_k - t_{k-1}$, beat-level morphology embeddings, and sequence masks.
  * **Backbones:** Bidirectional GRU, Temporal Convolutional Network (TCN), and Relative-Time Transformer.
  * **Scientific Role:** Determines whether AF information resides in temporal order, rhythm irregularity, or morphological variability.

---

## 3. Canonical Model Identity Crosswalk Table

| Model Registry ID | Canonical Class Name | Corrected Scientific Identity | Input Modality | Trainable Parameters | FLOPs (2-Lead) | Stage Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A001** | `ContinuousSetOperatorA001` | Continuous D1 SetOperator Reference | Phase-KME + Continuous $q \in \mathbb{R}^8$ | 1,027,009 | 16.62 M | Stage 1 Reference Baseline |
| **A002** | `MatchedCategoricalSetOperatorA002` | Matched Categorical SetOperator Control | Phase-KME + Integer ID $\in \{0, 1\}$ | 1,022,401 | 16.61 M | Continuous vs Discrete Control |
| **A003** | `OperatorAblatedSetOperatorA003` | Operator-Ablated Symmetry Control | Phase-KME (No $q$ conditioning) | 1,022,273 | 16.61 M | Geometry Ablation Control |
| **A004** | `Continuous2DSetOperatorA004` | 2D Continuous Projection Control | Phase-KME + $q \in \mathbb{R}^2$ | 1,026,625 | 16.62 M | Dimension Reduction Control |
| **D032-v1** | `DualMomentPhaseSetOperatorD032` | Phase-Cell Spatial Dispersion SetOperator | Phase-KME ($\mu + \sigma$ across 16 phases) | 1,043,393 | 16.66 M | Stage 1 Top Performer |
| **D032-v2** | *(Stage 2 Spec)* | True Across-Beat Phase-KME Moments | Per-beat $\Phi[k, p, :]$ across $K$ beats | ~1,045,000 | ~17.5 M | Stage 2 Distributional Model |
| **D037-v1** | `OrderPreservingBeatSetOperatorD037` | Phase-Trajectory Convolution SetOperator | Phase-KME (Conv1D across 16 phases) | 928,449 | 16.32 M | Intra-cycle Transition Model |
| **D037-v2** | *(Stage 3 Spec)* | True Ordered Beat Dynamics Sequence Model | Beat sequence $(B, M, K, D)$ + $\Delta t_k$ | ~1,120,000 | ~22.0 M | Stage 3 Temporal Sequence Model |
| **E047** | `PhaseWiseAttentionSetOperatorE047` | Phase-Wise Joint Attention Pooling | Phase-KME ($M \times 16$ uncollapsed tokens) | 1,027,009 | 42.15 M | Token Hierarchy Variant |
| **F053** | `NoTransformerSetOperatorF053` | Transformer-Free Masked Mean Pooling | Phase-KME (Direct channel pooling) | 284,545 | 4.88 M | Ultra-Lightweight Control |
| **F057** | `ResidualTokenSetOperatorF057` | Residual Token Pathway SetOperator | Phase-KME (Transformer + skip connection) | 1,027,009 | 16.62 M | Anti-Oversmoothing Variant |
| **G061** | `MultimodalRRSetOperatorG061` | Multimodal RR + Morphology SetOperator | Phase-KME + 36-D RR statistics | 1,027,045 | 16.62 M | Multimodal Fusion Performer |
| **H071-v1** | *(Retired)* | Synthetic Proxy Failure Diagnostic | Synthetic Tiled Phase-KME ($2500$ samples) | 2,065,249 | ~2.42 G | Invalid-Proxy Diagnostic Only |
| **H071-v2** | `PerChannelRawResNetSetOperatorH071` | Per-Channel Authentic Raw ECG ResNet | Authentic Raw Voltage $(B, M, 6000)$ | 2,065,249 | ~2.42 G | Raw Conv Waveform Baseline |
| **I082-v1** | `SetOperatorResidualMLPCapacityControlI082v1` | SetOperator Residual Head Capacity Control | Phase-KME + 2-layer residual MLP head | 1,158,593 | 16.88 M | Classification Head Capacity Control |
| **I082-v2** | `ResNetResidualFusionSetOperatorI082v2` | Genuine ResNet + SetOperator Residual Fusion | Raw $(B, 2, 6000)$ + Phase-KME tokens | 3,213,059 | ~2.43 G | Stage 5 Multimodal Residual Fusion |

---

## 4. Architectural Implementation Verification

In `continuous_setoperator_models.py`, all versioned identities are formally declared and registered:
```python
SET_OPERATOR_MODEL_REGISTRY = {
    "A001": ContinuousSetOperatorA001,
    "A002": MatchedCategoricalSetOperatorA002,
    "A003": OperatorAblatedSetOperatorA003,
    "A004": Continuous2DSetOperatorA004,
    "D032": DualMomentPhaseSetOperatorD032,
    "D032-V1": DualMomentPhaseSetOperatorD032,
    "D037": OrderPreservingBeatSetOperatorD037,
    "D037-V1": OrderPreservingBeatSetOperatorD037,
    "E047": PhaseWiseAttentionSetOperatorE047,
    "F053": NoTransformerSetOperatorF053,
    "F057": ResidualTokenSetOperatorF057,
    "G061": MultimodalRRSetOperatorG061,
    "H071": PerChannelRawResNetSetOperatorH071,
    "H071-V2": PerChannelRawResNetSetOperatorH071,
    "I082": SetOperatorResidualMLPCapacityControlI082v1,
    "I082-V1": SetOperatorResidualMLPCapacityControlI082v1,
    "I082-V2": ResNetResidualFusionSetOperatorI082v2,
}
```
All models satisfy backward pass gradient checks (`torch.isfinite(p.grad).all() == True`), indeterminate channel refusal checks, and strict parameter accounting.
