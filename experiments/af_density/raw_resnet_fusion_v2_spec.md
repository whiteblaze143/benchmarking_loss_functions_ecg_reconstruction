# Architecture Specification: Genuine ResNet–SetOperator Residual Fusion (I082-v2)

**Author:** Model & Registry Architect (`@architect`)  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Program  
**Status:** Approved Technical Specification (Stage 1B & Stage 5)  
**Date:** November 22, 2026  
**Implementation:** `ResNetResidualFusionSetOperatorI082v2` in `continuous_setoperator_models.py`  

---

## 1. Motivation and Theoretical Objective

In the Stage 1 pilot experiments, the model labeled `I082` evaluated a 2-layer residual MLP classification head attached to the Phase-KME SetOperator latent, rather than a true fusion of raw ECG waveforms and SetOperator representations. Consequently, the core scientific question of **Stage 5** remained completely untested:

> *Do continuous Phase-KME SetOperator representations provide complementary, incremental diagnostic information for AF episode detection beyond a high-capacity, end-to-end 1D raw ECG ResNet?*

`I082-v2` (`ResNetResidualFusionSetOperatorI082v2`) provides the rigorous, authentic architectural implementation required to test this hypothesis. It couples a frozen or fine-tuned 1D raw ECG ResNet with a continuous Phase-KME SetOperator branch via a zero-initialized residual scalar connection, ensuring strict performance monotonicity and graceful fallback under missing data.

---

## 2. Mathematical Architecture Formulation

```
                                  [Authentic Raw ECG]
                                X_raw: (B, 2, 6000) @ 200 Hz
                                          │
                                          ▼
                                ┌───────────────────┐
                                │  8-Block 1D ResNet│ (2,186,049 params)
                                │  RawMaskResNet    │
                                └─────────┬─────────┘
                                          │
                                          ▼ logit_raw ∈ R^B
                                          │
[Phase-KME Responses]                     │
 R: (B, 2, 16, 128)                       │
[Continuous Operators]                    │
 Q: (B, 2, 8)                             │
         │                                │
         ▼                                │
┌──────────────────┐                      │
│ Continuous       │                      │
│ SetOperator      │ (1,027,009 params)   │
│ Branch           │                      │
└────────┬─────────┘                      │
         │                                │
         ▼ Δlogit_set ∈ R^B               │
         │                                │
         └───────► [ × α ] ◄──────────────┘  (α_init = 0.0)
                     │
                     ▼ (+)
              logit_fused ∈ R^B
```

### 2.1. Raw Waveform ResNet Branch ($\text{logit}_{\text{raw}}$)
The raw ECG branch consumes the authentic two-channel voltage waveform $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{B \times 2 \times 6000}$ (30-second window at 200 Hz) along with the boolean channel availability mask $\mathbf{A} \in \{0, 1\}^{B \times 2}$ (where $A_{b, m} = 1$ if channel $m$ is observed, $0$ if missing).
* **Stem:** $\text{Conv1d}(4 \to 32, \text{kernel}=15, \text{stride}=2, \text{padding}=7)$, followed by $\text{BatchNorm1d}$, $\text{ReLU}$, and $\text{MaxPool1d}(3, \text{stride}=2, \text{padding}=1)$.
  * Input channels are constructed as:
    $$\mathbf{X}_{\text{stem}} = \left[ \mathbf{X}_{\text{raw}} \odot \mathbf{A}; \; \mathbf{A} \otimes \mathbf{1}_{T} \right] \in \mathbb{R}^{B \times 4 \times 6000}$$
* **Residual Backbone:** Eight 1D residual blocks arranged across 4 stages with strides $[1, 1, 2, 1, 2, 1, 2, 1]$ and channel capacities $[32, 32, 64, 64, 128, 128, 256, 256]$.
* **Global Pooling & Head:**
  $$\mathbf{z}_{\text{raw}} = \frac{1}{T'} \sum_{t=1}^{T'} \mathbf{H}_8[:, :, t] \in \mathbb{R}^{B \times 256}$$
  $$\text{logit}_{\text{raw}} = \mathbf{w}_{\text{raw}}^\top \mathbf{z}_{\text{raw}} + b_{\text{raw}} \in \mathbb{R}^B$$
* **Parameter Count:** Exactly **2,186,049** parameters.

### 2.2. Continuous Phase-KME SetOperator Branch ($\Delta\text{logit}_{\text{set}}$)
The auxiliary morphology branch consumes the 16-cell Phase-KME representation $\mathbf{R} \in \mathbb{R}^{B \times 2 \times 16 \times 128}$ and the 8-dimensional continuous channel operator vectors $\mathbf{Q} \in \mathbb{R}^{B \times 2 \times 8}$.
* **Operator Encoder:** 2-layer MLP $\text{Linear}(8 \to 64) \to \text{GELU} \to \text{Linear}(64 \to 64)$ (4,736 params).
* **Response Encoder:** Modulated circular 2-block PhaseCNN projecting $(16, 128) \to 128$ (230,144 params).
* **Set Encoder:** 2-layer Transformer encoder with $d_{\text{model}}=192$, 4 attention heads, feedforward dimension 384, and GELU activation (738,816 params).
* **Attention Pooling:** Learned query $\mathbf{q}_{\text{pool}} \in \mathbb{R}^{1 \times 1 \times 192}$ pooling over active channels (148,416 params).
* **Latent Projection & Residual Delta Head:**
  $$\mathbf{z}_{\text{set}} = \text{GELU}(\mathbf{W}_{\text{lat}} \mathbf{h}_{\text{pool}} + \mathbf{b}_{\text{lat}}) \in \mathbb{R}^{B \times 256}$$
  $$\Delta\text{logit}_{\text{set}} = \mathbf{w}_{\text{delta}}^\top \mathbf{z}_{\text{set}} + b_{\text{delta}} \in \mathbb{R}^B$$
* **Parameter Count:** Exactly **1,027,009** parameters (identical to A001).

### 2.3. Zero-Initialized Residual Coupling
The final model prediction is formed by residual addition:
$$\text{logit}_{\text{fused}} = \text{logit}_{\text{raw}} + \alpha \cdot \Delta\text{logit}_{\text{set}}$$
where $\alpha \in \mathbb{R}$ is a learnable scalar parameter initialized to zero:
$$\alpha_{\text{init}} = 0.0$$

#### Theoretical Advantages of Zero-Initialization:
1. **Strict Monotonicity:** At iteration zero ($t=0$), $\alpha = 0$, guaranteeing that $\text{logit}_{\text{fused}} \equiv \text{logit}_{\text{raw}}$. The fused model begins exactly at the performance of the raw ResNet baseline.
2. **Gradient-Driven Relevance:** The gradient with respect to $\alpha$ is:
   $$\frac{\partial \mathcal{L}}{\partial \alpha} = \sum_{b=1}^B \left( \sigma(\text{logit}_{\text{fused}, b}) - y_b \right) \Delta\text{logit}_{\text{set}, b}$$
   If the SetOperator prediction $\Delta\text{logit}_{\text{set}}$ is orthogonal or uncorrelated with the residual error of the raw ResNet, $\frac{\partial \mathcal{L}}{\partial \alpha} \approx 0$, and $\alpha$ remains near zero, preventing destructive interference.
3. **Capacity Protection:** If the SetOperator branch learns noise or overfits, regularization and early stopping can easily keep $\alpha$ small.

---

## 3. Missing Modality & Graceful Degradation Contract

In real-world ambulatory monitoring, Phase-KME extraction may fail due to severe baseline wander, missing R-peak delineations, or excessive muscle artifact, even while raw voltage continues to stream. Conversely, single-lead disconnects may occur.

`I082-v2` enforces the following mathematical fallback rules:

| Condition | Raw Status | Phase-KME Status | Model Behavior | Output Logit |
| :--- | :--- | :--- | :--- | :--- |
| **Normal 2-Lead** | 2 Valid Channels | 2 Valid Channels | Full dual-branch inference | $\text{logit}_{\text{raw}} + \alpha \Delta\text{logit}_{\text{set}}$ |
| **Single-Lead Dropout** | 1 Active, 1 Masked | 1 Active, 1 Masked | Masked inference on both branches | $\text{logit}_{\text{raw}} + \alpha \Delta\text{logit}_{\text{set}}$ |
| **Phase-KME Failure** | Valid Raw ECG | Failed Extraction ($\mathbf{M}_{\text{phase}} = \mathbf{0}$) | Mask $\Delta\text{logit}_{\text{set}} \to 0$ | $\text{logit}_{\text{raw}}$ (Raw ResNet Fallback) |
| **All Channels Lost** | 0 Valid Channels | 0 Valid Channels | Indeterminate acquisition | **Raises ValueError / Returns Indeterminate** |

```python
# Graceful fallback implementation in continuous_setoperator_models.py
if not has_valid_phase.all():
    delta_logit = torch.where(has_valid_phase, delta_logit, torch.zeros_like(delta_logit))
logit_fused = logit_raw + self.alpha * delta_logit
```

---

## 4. Parameter and Computational Resource Ledger

| Component | Sub-Module | Dimension / Configuration | Parameter Count |
| :--- | :--- | :--- | :--- |
| **Raw ResNet Stem** | Conv1d + BN + MaxPool | $4 \to 32, K=15, S=2$ | 1,984 |
| **Raw ResNet Layer 1-2** | 2 Residual Blocks | $32 \to 32, K=7, S=1$ | 28,928 |
| **Raw ResNet Layer 3-4** | 2 Residual Blocks | $32 \to 64 \to 64, K=7, S=2$ | 103,040 |
| **Raw ResNet Layer 5-6** | 2 Residual Blocks | $64 \to 128 \to 128, K=7, S=2$ | 410,880 |
| **Raw ResNet Layer 7-8** | 2 Residual Blocks | $128 \to 256 \to 256, K=7, S=2$ | 1,640,960 |
| **Raw ResNet Head** | Linear | $256 \to 1$ | 257 |
| **Subtotal Raw ResNet** | `self.raw_branch` | Full 1D ResNet Backbone | **2,186,049** |
| **Operator MLP** | 2-Layer MLP + GELU | $8 \to 64 \to 64$ | 4,736 |
| **Response Encoder** | PhaseCNN | $128 \to 128$ (2 circular blocks) | 230,144 |
| **Set Encoder** | 2-Layer Transformer | $d_{\text{model}}=192, H=4, d_{\text{ff}}=384$ | 738,816 |
| **Pool Query & MHA** | Learned Query Attention | $192 \to 192$ | 148,416 |
| **Latent Projection** | Linear + GELU | $192 \to 256$ | 49,408 |
| **Residual Delta Head** | Linear | $256 \to 1$ | 257 |
| **Subtotal SetOperator** | Continuous SetOperator | Phase-KME Diagnostic Path | **1,027,009** |
| **Residual Gate** | Scalar Parameter | $\alpha \in \mathbb{R}$ | **1** |
| **TOTAL (I082-v2)** | `ResNetResidualFusionSetOperatorI082v2` | Dual-Branch Fusion | **3,213,059** |

---

## 5. Two-Stage Training Protocol

To distinguish true representation synergy from joint optimization instability, `I082-v2` is trained in two distinct phases:

### Phase A: Frozen Raw ResNet Residual Training
1. **Initialize Raw ResNet:** Load the converged, frozen checkpoint of `RawMaskResNet` (AUROC ~0.84, trained on training partition).
2. **Freeze Raw Parameters:**
   ```python
   model.freeze_raw_branch()
   # Trainable parameters: exactly 1,027,010 (SetOperator + alpha)
   ```
3. **Train Residual Branch:** Optimize the SetOperator branch to predict the residual error of the raw ResNet:
   $$\mathcal{L} = \frac{1}{B}\sum_{b=1}^B \text{BCEWithLogits}(\text{logit}_{\text{raw}, b} + \alpha \Delta\text{logit}_{\text{set}, b}, y_b)$$
   * Optimizer: AdamW, $\text{lr} = 1 \times 10^{-4}$, weight decay 0.01, cosine decay over 50 epochs.
   * Monitored Metric: Development partition patient-grouped cross-validation loss.

### Phase B: Controlled End-to-End Fine-Tuning
1. **Unfreeze Raw Parameters:**
   ```python
   model.unfreeze_raw_branch()
   # Trainable parameters: all 3,213,059
   ```
2. **Differential Learning Rates:**
   $$\text{lr}_{\text{raw}} = 1 \times 10^{-5}, \quad \text{lr}_{\text{set}} = 5 \times 10^{-5}, \quad \text{lr}_{\alpha} = 1 \times 10^{-4}$$
3. **Patience:** 10 epochs early stopping based on development AUROC.

---

## 6. Experimental Verification & Capacity Control Baseline

To prevent confounding parameter scale with multimodal fusion benefit:
1. **Baseline 1:** Frozen `RawMaskResNet` alone (2,186,049 params).
2. **Baseline 2:** `ContinuousSetOperatorA001` alone (1,027,009 params).
3. **Control 3 (`I082-v1`):** `ContinuousSetOperatorA001` + 2-layer residual MLP head (1,158,593 params) — parameter capacity control.
4. **Control 4:** Frozen `RawMaskResNet` + 2-layer residual MLP head (131,584 params) — raw capacity control without Phase-KME.
5. **Experimental Model (`I082-v2`):** Dual-Branch ResNet + SetOperator Fusion (3,213,059 params).

If `I082-v2` significantly outperforms Control 3 and Control 4 on patient-grouped cross-validation ($p < 0.01$), the scientific claim that *Phase-KME and continuous operator geometry contribute orthogonal physiological signal to 1D raw ECG* is formally supported.
