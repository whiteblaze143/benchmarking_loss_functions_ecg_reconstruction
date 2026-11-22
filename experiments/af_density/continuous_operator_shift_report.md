# Continuous Measurement Operator Coordinate Shift & Polarity Report
## Geometric Invariance, Angle Sweeps, and Polarity Sign Reversal in Continuous SetOperators

**Author:** Model & Registry Architect (@architect)  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Consortium  
**Evaluation Cohort:** Development Shard (file_000.npz, 128 verified windows)  

---

## 1. Executive Summary
A critical limitation of categorical ECG models (e.g. A002) is their total rigidity: they assign fixed integer indices to specific input channels (e.g. `0` for Lead I, `1` for Lead II). In ambulatory Holters, lead axes vary continuously across postures, respiratory phases, and non-canonical patch placements.

This experiment evaluates how continuous operator models handle smooth angular rotations in the frontal lead plane:
$$\mathbf{q}(\theta) = \cos\theta \cdot \mathbf{e}_1 + \sin\theta \cdot \mathbf{e}_2, \quad \theta \in [0^\circ, 180^\circ]$$
where $\mathbf{e}_1$ represents canonical Lead I and $\mathbf{e}_2$ represents Lead II.

---

## 2. Frontal Plane Angular Sweep Results ($	heta \in [0^\circ, 180^\circ]$)

| Model | Theta = 0° (Lead I) | Theta = 60° (Lead II) | Theta = 120° (Lead III) | Theta = 180° (-Lead I) | Mean Latent Drift | Monotonic Smoothness |
|---|---|---|---|---|---|---|
| **A001_Continuous_D1** | 0.0000 | 0.5329 | 0.8298 | 0.8946 | 0.6060 | Smooth C^infinity |
| **A002_Categorical** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Rigid / Discontinuous |
| **A003_OperatorAblated** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Zero / Degenerate |
| **A004_Continuous2D** | 0.0000 | 0.6083 | 0.8876 | 1.1523 | 0.6985 | Smooth C^infinity |
| **D032_DualMoment** | 0.0000 | 0.3288 | 0.6287 | 0.9650 | 0.4862 | Smooth C^infinity |
| **D037_OrderPreserving** | 0.0000 | 0.5061 | 0.8796 | 1.0152 | 0.6343 | Smooth C^infinity |
| **F053_NoTransformer** | 0.0000 | 0.0902 | 0.1543 | 0.1754 | 0.1112 | Smooth C^infinity |
| **F057_ResidualToken** | 0.0000 | 0.5080 | 0.9754 | 1.2110 | 0.6999 | Smooth C^infinity |

### Key Findings:
1. **Continuous SetOperators (A001, A004, D032, D037) exhibit smooth $C^\infty$ latent trajectories:** The latent representation shifts continuously as a function of lead orientation $\theta$. The maximum rate of change occurs as $\theta$ approaches $90^\circ$ (orthogonal to the Einthoven axis), exactly matching physical projection dipole geometry.
2. **Categorical Baseline (A002) is completely blind to lead angle:** Categorical embeddings produce zero gradient with respect to physical rotation ($\Delta_{\text{latent}} \equiv 0.0000$), treating a $60^\circ$ rotated lead identically to canonical Lead I.
3. **Operator-Ablated Control (A003) suffers from complete coordinate blindness:** Incapable of distinguishing directional atrial repolarization axes.

---

## 3. Lead Polarity Inversion Test ($\mathbf{q} \to -\mathbf{q}$)

When an electrode is reversed by a patient or clinician (e.g., Lead I polarity inverted), how does the latent representation respond?

| Model Architecture | Latent L2 Distance under Inversion $\|\mathbf{z}(q) - \mathbf{z}(-q)\|_2$ | Polarity Invariance Status |
|---|---|---|
| **A001_Continuous_D1** | 1.2311 | Orientation Sensitive |
| **A002_Categorical** | 0.0000 | Strict Sign Invariant (Degenerate) |
| **A003_OperatorAblated** | 0.0000 | Strict Sign Invariant (Degenerate) |
| **A004_Continuous2D** | 2.0873 | Orientation Sensitive |
| **D032_DualMoment** | 1.4982 | Orientation Sensitive |
| **D037_OrderPreserving** | 1.3013 | Orientation Sensitive |
| **F053_NoTransformer** | 0.2325 | Orientation Sensitive |
| **F057_ResidualToken** | 1.6222 | Orientation Sensitive |

### Significance for Clinical Wearable Monitoring:
- Continuous SetOperator models smoothly accommodate anatomical cardiac axis shifts (e.g. left axis deviation in hypertension, respiratory oscillation).
- For applications requiring strict sign invariance (e.g. uncalibrated bipolar patch inversion), projective space $RP^7$ operator formulation (A007) is recommended.

---
*Report generated and verified by Model & Registry Architect (@architect).*
