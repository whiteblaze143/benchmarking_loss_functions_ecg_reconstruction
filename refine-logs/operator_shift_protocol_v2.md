# Operator Shift Protocol v2: Continuous Geometric Equivariance & Rotation Invariance Audit
## T-CAIREM Continuous SetOperator AF-Density Research Program

**Document Version**: 2.0 (Stage 1B Corrective Protocol)  
**Date**: 2026-11-22  
**Author**: Statistical Governance & Refinement Planner (@planner)  
**Target Repository**: `experiments/af_density/`  
**Storage Destination**: `operator_shift_protocol_v2.md`, `refine-logs/operator_shift_protocol_v2.md`, and `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/operator_shift_protocol_v2.md`  

---

## 1. Executive Summary & Physical Rationale

Standard deep learning models for multi-lead electrocardiography (e.g., 1D ResNets) treat input leads as fixed, indexed channels (Channel 0 = Lead I, Channel 1 = Lead II). When the physical orientation of the recording electrodes shifts—such as when a patient changes posture, when ambulatory Holter patches are placed at arbitrary angles on the torso, or when leads are mathematically rotated—fixed-channel models suffer from severe representation degradation or output unpredictable predictions.

The **Continuous $q$-conditioned SetOperator** is grounded in continuous operator theory. In the frontal plane, the human cardiac electrical dipole vector $\mathbf{p}(t) \in \mathbb{R}^2$ generates voltages across arbitrary lead axes according to Einthoven's and Goldberger's laws. For any frontal angle $\theta \in [0, 2\pi)$, a physical lead vector is:
$$\mathbf{l}_\theta = [\cos\theta, \, \sin\theta]^T$$
The resulting scalar ECG voltage signal $y_\theta(t)$ is the inner product:
$$y_\theta(t) = \langle \mathbf{p}(t), \, \mathbf{l}_\theta \rangle = \cos(\theta) x_{\text{I}}(t) + \sin(\theta) x_{\text{II}}(t)$$
where $x_{\text{I}}(t)$ and $x_{\text{II}}(t)$ are the canonical orthogonal Lead I and Lead II projections (in the Einthoven plane).

In the Continuous SetOperator architecture, each channel's voltage profile $y_\theta(t)$ is accompanied by its corresponding continuous projection coordinate vector:
$$q_\theta = [\cos\theta, \, \sin\theta, \, 0, \, 0, \, 0, \, 0, \, 0, \, 0]^T \in \mathbb{R}^8$$
in the canonical 8-lead clinical basis $\mathcal{B} = [I, II, V_1, V_2, V_3, V_4, V_5, V_6]^T$.

This protocol establishes the rigorous mathematical audit for **joint physical-operator rotation equivariance**, proving that the model's prediction is invariant under joint spatial rotation of the physical voltage signals and the geometric query vectors $q$.

---

## 2. Mathematical Formulation of the Joint Rotation Audit

```
             ┌─────────────────────────────────────────────────────────┐
             │            Canonical ECG Signals: x_I(t), x_II(t)       │
             └────────────────────────────┬────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     Physical Signal Rotation                       Geometric Query Rotation
     y_θ(t) = cos(θ)x_I(t) + sin(θ)x_II(t)          q_θ = [cos(θ), sin(θ), 0...]^T
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Continuous SetOperator Model M     │
                      │       M({(y_θ, q_θ), (y_θ', q_θ')})   │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      Predicted Logit: s(θ) ≈ s_canonical (Invariant)
```

### 2.1 The Three Transformation Conditions
For any rotation angle $\theta \in [0, 2\pi)$, we evaluate three distinct conditions to isolate the geometric operator mechanism:

1. **Condition 1: Joint Physical & Operator Rotation (Matched Equivariance)**:
   - Physical Signals:
     $$y_1(t) = \cos(\theta) x_{\text{I}}(t) + \sin(\theta) x_{\text{II}}(t)$$
     $$y_2(t) = -\sin(\theta) x_{\text{I}}(t) + \cos(\theta) x_{\text{II}}(t)$$
   - Geometric Queries:
     $$q_1 = [\cos\theta, \, \sin\theta, \, 0, \, 0, \, 0, \, 0, \, 0, \, 0]^T$$
     $$q_2 = [-\sin\theta, \, \cos\theta, \, 0, \, 0, \, 0, \, 0, \, 0, \, 0]^T$$
   - **Hypothesis**: The SetOperator recognizes that $y_1(t)$ corresponds to $q_1$ and $y_2(t)$ corresponds to $q_2$. The pooled latent representation $z_{\text{set}}(\theta)$ and final logit $\hat{s}_{\text{joint}}(\theta)$ must be strictly invariant to $\theta$:
     $$|\hat{s}_{\text{joint}}(\theta) - \hat{s}_{\text{canonical}}| \le \epsilon_{\text{joint}}$$

2. **Condition 2: Signal-Only Rotation (Uninformed / Misaligned Physics)**:
   - Physical Signals: Rotated by $\theta$ as in Condition 1 ($y_1(t), y_2(t)$).
   - Geometric Queries: Kept unrotated at canonical coordinates ($q_1 = q_{\text{I}}, q_2 = q_{\text{II}}$).
   - **Hypothesis**: The model is fed rotated signals without the corresponding geometric metadata. Prediction logs drift significantly, establishing that the operator embeddings $q$ are actively guiding feature extraction rather than being ignored.

3. **Condition 3: Query-Only Rotation (Phantom Operator Perturbation)**:
   - Physical Signals: Kept at canonical unrotated voltages ($x_{\text{I}}(t), x_{\text{II}}(t)$).
   - Geometric Queries: Rotated by $\theta$ ($q_1 = q_\theta, q_2 = q_{\theta + \pi/2}$).
   - **Hypothesis**: Verifies that the model is sensitive to operator geometry when voltages do not match the declared lead vector.

---

## 3. Discretized Angular Audit Grid

To thoroughly test the $S^1$ rotation group, evaluations are conducted over a 16-point uniform angular grid:
$$\Theta = \left\{ 0^\circ, \, 15^\circ, \, 30^\circ, \, 45^\circ, \, 60^\circ, \, 75^\circ, \, 90^\circ, \, 105^\circ, \, 120^\circ, \, 135^\circ, \, 150^\circ, \, 165^\circ, \, 180^\circ, \, 225^\circ, \, 270^\circ, \, 315^\circ \right\}$$

### 3.1 Anatomically Significant Landmark Angles:
- $\theta = 0^\circ$: Canonical Einthoven Lead I ($\mathbf{l}_{\text{I}}$).
- $\theta = 60^\circ$: Canonical Einthoven Lead II ($\mathbf{l}_{\text{II}}$).
- $\theta = 90^\circ$: Canonical aVF (vertical inferior lead).
- $\theta = 120^\circ$: Canonical Einthoven Lead III ($\mathbf{l}_{\text{III}} = \mathbf{l}_{\text{II}} - \mathbf{l}_{\text{I}}$).
- $\theta = 180^\circ$: Polarity inversion (Lead $-I$). Tests model invariance to electrode cable reversal ($q \to -q$).
- $\theta = 270^\circ$: Inverted aVF.

---

## 4. Evaluation Metrics & Quantifiable Success Bars

For each evaluated model (A001, A002, A003, A004, D032, F053, G061, H071-v2, I082-v2) on the validation cohort (166,047 windows / 498,057 evaluations):

### 4.1 Metric 1: Joint Rotation Invariance Divergence ($\Delta_{\text{joint}}$)
$$\Delta_{\text{joint}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left| \sigma(\hat{s}_{\text{joint}}^{(i)}(\theta)) - \sigma(\hat{s}_{\text{canonical}}^{(i)}) \right|$$
- **Maximum Angular Divergence**: $\Delta_{\max} = \max_{\theta \in \Theta} \Delta_{\text{joint}}(\theta)$.
- **Mean Angular Divergence**: $\bar{\Delta} = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \Delta_{\text{joint}}(\theta)$.
- **Target Bar for Continuous Models (A001, I082-v2)**:
  - $\bar{\Delta} \le 0.025$ (mean probability shift $\le 2.5\%$).
  - $\Delta_{\max} \le 0.050$ (maximum probability shift $\le 5.0\%$).

### 4.2 Metric 2: Operator Sensitivity Ratio (OSR)
Measures the protective effect of matching operator coordinates versus uninformative physical rotation:
$$\text{OSR}(\theta) = \frac{\Delta_{\text{signal-only}}(\theta)}{\Delta_{\text{joint}}(\theta) + \epsilon}$$
- **Target Bar**: $\text{OSR} \ge 3.0$ across $\theta \in [30^\circ, 150^\circ]$, proving that supplying matched $q_\theta$ reduces prediction error by at least $3\times$ compared to unguided rotated signals.

### 4.3 Metric 3: Polarity Inversion Equivariance ($\theta = 180^\circ$)
At $\theta = 180^\circ$, physical voltages are negated ($y = -x$) and queries are negated ($q = -q$).
$$\Delta_{\text{polarity}} = \frac{1}{N} \sum_{i=1}^N \left| \sigma(\hat{s}( -x, -q )) - \sigma(\hat{s}( x, q )) \right|$$
- **Target Bar**: $\Delta_{\text{polarity}} \le 0.015$ (within $1.5\%$ probability margin).

### 4.4 Metric 4: Lead Permutation Symmetry ($\theta = 90^\circ$ swap)
Swapping Channel 0 and Channel 1 with corresponding $q$ coordinates:
$$\Delta_{\text{perm}} = \max_{i} \left| \hat{s}(C_1, C_0) - \hat{s}(C_0, C_1) \right|$$
- **Target Bar**: $\Delta_{\text{perm}} \le 1.0 \times 10^{-5}$ (machine-precision symmetry).

---

## 5. Execution Workflow & Implementation Specification

The audit will be executed by running `experiments/af_density/audit_operator_shift.py`:

```python
# Implementation Pseudo-Specification
import torch
import numpy as np

def generate_rotated_batch(x_I, x_II, theta_rad):
    """
    x_I, x_II: (B, 1, T) raw or phase-feature representations.
    Returns:
      y_1 = cos(theta)*x_I + sin(theta)*x_II
      y_2 = -sin(theta)*x_I + cos(theta)*x_II
      q_1 = [cos(theta), sin(theta), 0, ...]
      q_2 = [-sin(theta), cos(theta), 0, ...]
    """
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    y_1 = cos_t * x_I + sin_t * x_II
    y_2 = -sin_t * x_I + cos_t * x_II
    y_rotated = torch.cat([y_1, y_2], dim=1) # (B, 2, T)
    
    q_1 = torch.tensor([cos_t, sin_t, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    q_2 = torch.tensor([-sin_t, cos_t, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    q_rotated = torch.stack([q_1, q_2], dim=0).unsqueeze(0).expand(x_I.size(0), -1, -1)
    
    return y_rotated, q_rotated
```

### 5.1 Step-by-Step Execution Sequence
1. Load validation cohort feature cache and model checkpoints (A001, A002, A003, H071-v2, I082-v2).
2. For each angle $\theta \in \Theta$:
   - Synthesize physical signal rotation on 200 Hz raw ECG (for H071, I082-v2) and phase-normalized responses (for A001-A004, D032).
   - Evaluate Condition 1 (Joint), Condition 2 (Signal-Only), and Condition 3 (Query-Only).
   - Compute prediction probabilities, AUROC, Brier score, and logit divergence.
3. Compute summary statistics: $\bar{\Delta}$, $\Delta_{\max}$, and OSR.
4. Export results to `operator_shift_results.csv` and `operator_shift_audit_receipt.json`.

---

## 6. Model Comparative Expectations Matrix

| Model ID | Model Description | Expected Joint Divergence $\bar{\Delta}$ | Expected Signal-Only Divergence | Expected OSR | Outcome / Diagnosis |
|---|---|---|---|---|---|
| **A001** | Continuous SetOperator | **$\le 0.020$ (Invariant)** | $> 0.120$ | $\ge 6.0$ | **Pass**: Continuous $q$ coordinates successfully steer attention and preserve invariance. |
| **A002** | Categorical SetOperator | Undefined (Collapsed) | Undefined | $\sim 1.0$ | **Fail**: Discrete IDs cannot represent rotated angles; suffers from score collapse. |
| **A003** | Operator-Ablated Control | $> 0.150$ (Degraded) | $> 0.150$ | $1.0$ | **Fail**: Lacks operator guidance entirely; prediction shifts arbitrarily with signal rotation. |
| **A004** | 2D Continuous $q$ | $\le 0.025$ (Invariant) | $> 0.110$ | $\ge 4.4$ | **Pass**: 2D projection handles planar rotation, but lacks 8-lead clinical embedding basis. |
| **H071-v2** | Raw 1D ResNet Baseline | $> 0.180$ (Severe Drift) | $> 0.180$ | $1.0$ | **Fail**: Fixed 1D convolution filters fail when Lead I and Lead II axes rotate. |
| **I082-v2** | ResNet-Residual Fusion | **$\le 0.030$ (Invariant)** | $> 0.100$ | $\ge 3.3$ | **Pass**: SetOperator branch stabilizes the raw ResNet branch, maintaining spatial invariance. |

---

## 7. Deliverables & Audit Trail Manifest
- **Execution Script**: `experiments/af_density/audit_operator_shift.py`
- **Output Artifacts**:
  - `operator_shift_results.csv` (16 angles $\times$ 10 models $\times$ 3 conditions)
  - `operator_shift_audit_receipt.json` (SHA-256 bound execution receipt)
  - `figures/operator_shift_polar_invariance.png` (polar plot of probability error vs angle)
- **Reporting Gate**: Stage 5 of the master GPU campaign.
