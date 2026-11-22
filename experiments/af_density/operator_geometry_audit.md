# Continuous SetOperator Geometry & Coordinate Basis Audit

**Document:** `operator_geometry_audit.md`  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Program  
**Auditor:** Data Provenance & Integrity Auditor (`integrity`)  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

This audit establishes the formal mathematical basis, nominal coordinate assignments, and anatomical uncertainty boundaries for the **Continuous SetOperator** applied to ambulatory two-lead Holter monitoring (IRIDIA-AF cohort).

In the previous categorical SetOperator formulation (`set_models_v1`), channel identities were provided solely as categorical tokens ($0$ or $1$) or ablated to identity ($q$-ablated). As documented in `representation_collapse_diagnostics.json`, the categorical tokens lacked continuous spatial structure, causing the active-token representations of the Transformer set encoder to collapse to near-zero standard deviation ($	ext{SD} pprox 3.45 	imes 10^{-6}$ for categorical, $2.24 	imes 10^{-6}$ for $q$-ablated), generating near-constant output probability ($\hat{p} pprox 0.2284$).

The **Continuous SetOperator** resolves this failure mechanism by projecting channels into the standard **8-lead clinical ECG coordinate basis** $\mathbb{R}^8$, breaking token symmetry with continuous geometric inductive bias.

---

## 2. Mathematical Definition of the 8-Lead Clinical Coordinate Basis

### 2.1 Subspace Dimension of Clinical ECG
The standard 12-lead electrocardiogram records 12 voltage signals:
$$\mathbf{X}(t) = [V_I(t), V_{II}(t), V_{III}(t), V_{aVR}(t), V_{aVL}(t), V_{aVF}(t), V_{V1}(t), \dots, V_{V6}(t)]^T \in \mathbb{R}^{12}$$

By electrical network theory and Einthoven's triangle law, the limb leads are linearly dependent:
$$V_{III}(t) = V_{II}(t) - V_I(t)$$

Furthermore, Goldberger's augmented limb leads are defined relative to Wilson's Central Terminal ($WCT = rac{RA + LA + LL}{3}$), yielding exact linear relationships:
$$V_{aVR}(t) = -rac{V_I(t) + V_{II}(t)}{2}$$
$$V_{aVL}(t) = V_I(t) - rac{V_{II}(t)}{2}$$
$$V_{aVF}(t) = V_{II}(t) - rac{V_I(t)}{2}$$

Consequently, the 6 limb leads have algebraic rank **2**, and the full 12-lead ECG has algebraic rank **$2 + 6 = 8$**. The linear space of clinical ECG signals is spanned by the canonical 8-lead clinical basis:
$$\mathcal{B} = \{I, II, V1, V2, V3, V4, V5, V6\}$$

Any lead $L$ in the 12-lead system can be expressed as an operator vector $\mathbf{q}_L \in \mathbb{R}^8$:
$$V_L(t) = \mathbf{q}_L^T \mathbf{B}(t)$$
where $\mathbf{B}(t) = [V_I(t), V_{II}(t), V_{V1}(t), V_{V2}(t), V_{V3}(t), V_{V4}(t), V_{V5}(t), V_{V6}(t)]^T$.

---

## 3. Nominal Holter Lead Coordinates

In the IRIDIA-AF dataset, recordings consist of two ambulatory ECG channels, designated `channel 0` and `channel 1`.

Under the nominal Einthoven limb coordinate hypothesis, `channel 0` corresponds nominally to Lead I, and `channel 1` corresponds nominally to Lead II. In the 8-lead basis $\mathcal{B}$, their nominal operator coordinates $\mathbf{q}_{	ext{nominal}} \in \mathbb{R}^8$ are defined as:

### 3.1 Lead I Canonical Basis Vector
$$\mathbf{q}_{	ext{Lead I}} = egin{bmatrix} 1 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \end{bmatrix}$$

### 3.2 Lead II Canonical Basis Vector
$$\mathbf{q}_{	ext{Lead II}} = egin{bmatrix} 0 \ 1 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \end{bmatrix}$$

In the PyTorch `ContinuousAFSetOperator`, these nominal vectors are registered as persistent buffers:
```python
nominal = torch.zeros(2, 8, dtype=torch.float32)
nominal[0, 0] = 1.0  # Channel 0: Nominal Lead I
nominal[1, 1] = 1.0  # Channel 1: Nominal Lead II
```

---

## 4. Holter Electrode Placement Uncertainty

### 4.1 Anatomical & Engineering Reality
While nominal Lead I and Lead II coordinates provide an initial inductive anchor, **actual ambulatory Holter electrode placement is not guaranteed to coincide with canonical Einthoven Lead I and Lead II**:

1. **Modified Bipolar Torso Configurations:**
   - Canonical Einthoven leads require electrodes placed on the distal limbs (right arm RA, left arm LA, left leg LL).
   - In 24-hour ambulatory Holter monitoring, limb electrodes are clinically unacceptable due to severe muscle tremor (electromyographic artifact), cable pulling, and patient movement.
   - Holter monitors utilize torso-mounted modified lead configurations (such as modified chest leads MCL1, CM5, or modified limb placements on the infraclavicular fossae and lower ribs).
   - Therefore, the true physical vector recorded on Channel 0 and Channel 1 is an unknown linear projection:
     $$\mathbf{q}_{	ext{actual}} = lpha \mathbf{q}_I + eta \mathbf{q}_{II} + \sum_{k=1}^6 \gamma_k \mathbf{q}_{Vk} + oldsymbol{\epsilon}$$

2. **Electrode Displacement & Angle Tilt:**
   - Patch application by nursing staff or patient re-application introduces angle variations ($\pm 15^\circ$ to $\pm 35^\circ$ rotation).
   - Thoracic impedance differences, adiposity, breast tissue, and pectus excavatum distort the electric dipole projection.

3. **Postural Dynamics & Cardiac Axis Drift:**
   - Ambulatory patients continuously transition between supine, prone, sitting, walking, and standing states.
   - Positional changes cause anatomical shifting of the cardiac apex and electrical axis, dynamically modulating the effective projection vector $\mathbf{q}(t)$.

4. **Sign & Polarity Inversions:**
   - Inadvertent lead wire swapping (reversing positive and negative poles) induces a polarity inversion:
     $$\mathbf{q} 	o -\mathbf{q}$$
   - The projective outer product representation $\mathbf{Q} = \mathbf{q} \mathbf{q}^T \in \mathbb{RP}^7$ used in `OperatorSetModel(operator_mode="projective")` provides strict mathematical invariance under polarity negation:
     $$(-\mathbf{q})(-\mathbf{q})^T = \mathbf{q} \mathbf{q}^T$$

---

## 5. Architectural Implications & Audit Recommendations

1. **Continuous Operator Encoding Breaks Token Collapse:**
   - Passing continuous vectors $\mathbf{q}_0 = [1, 0, \dots, 0]$ and $\mathbf{q}_1 = [0, 1, \dots, 0]$ into the 2-layer MLP operator encoder produces distinct spatial embeddings $\mathbf{e}_0 
e \mathbf{e}_1$ ($L_2$ distance $> 0.5$).
   - This provides the Transformer set encoder with explicit asymmetry, preventing the self-attention collapse observed in categorical models.

2. **Robustness to Coordinate Noise:**
   - Because true Holter vectors deviate from canonical $e_1, e_2$, models must not overfit to crisp binary vectors.
   - Small continuous perturbations $\mathbf{q} + oldsymbol{\delta}$ with $\|oldsymbol{\delta}\| \le 0.15$ must yield stable latent representations without exploding gradients.

3. **Strict Handling of Indeterminate Channels:**
   - When all channels are unobserved or invalid ($	ext{active} = [0, 0]$), the model must raise an explicit indeterminate exception or output an indeterminate flag.
   - Under no circumstances should the model fabricate synthetic beats or hallucinate an AF probability when no physiological signal is present.

---

## 6. Audit Conclusion

The nominal Lead I coordinate $\mathbf{q}_1 = [1, 0, 0, 0, 0, 0, 0, 0]$ and nominal Lead II coordinate $\mathbf{q}_2 = [0, 1, 0, 0, 0, 0, 0, 0]$ in the 8-lead clinical basis $\mathbb{R}^8$ are mathematically verified and implemented in `set_waveform_model.py`. The documented anatomical placement uncertainty mandates continuous representation conditioning and robustness validation.
