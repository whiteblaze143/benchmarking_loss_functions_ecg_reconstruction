# Scientific Protocol: 3DRECON-QT 3D Theta Spatial Reconstruction Benchmark (ECG-AIM-3Dθ)

**Author**: Cardiology Machine Learning & Bioengineering Core  
**Date**: September 2, 2026  
**Status**: ACTIVE EXPERIMENTAL PROTOCOL  
**Tracking Issue**: Strict 1-Lead $\to$ 12-Lead ECG Reconstruction & Spatial Conditioning Isolation  

---

## 1. Core Scientific Hypothesis

Does 3DRECON-QT-style target spatial coordinate conditioning $(\theta_l, \phi_l)$ provide reconstruction information beyond arbitrary lead identity in strict one-lead $\to$ 12-lead ECG reconstruction?

$$\boxed{\text{Does 3DRECON-QT-style target spatial conditioning help our strict 1}\to\text{12 task?}}$$

Prior work in this repository explored spatial gating (author Panorama branch) and learned target query attention (`TargetQuerySpatialDecoder`), but did not isolate the exact mechanism from Ansari et al. (*Circulation* 2026) and Chen et al. (*IJCAI* 2021):
$$\mathbf{H}_{\text{source}} = E(x_{\text{source}}) \in \mathbb{R}^{P \times D}$$
$$\mathbf{u}_l = \Theta(\theta_l, \phi_l) \in \mathbb{R}^{12}, \quad \mathbf{g}_l = \text{MLP}_\theta(\mathbf{u}_l) \in \mathbb{R}^D$$
$$\mathbf{Z}_l(t) = \mathbf{H}_{\text{source}}(t) \odot \mathbf{g}_l$$
$$\hat{\mathbf{X}}_l = D_{\text{shared}}(\mathbf{Z}_l)$$

Where $D_{\text{shared}}$ is **one shared temporal transformer decoder** for all target leads. The target identity arrives strictly through the spatial code.

---

## 2. Lead Angles and Anatomical Coordinate Table

Spherical lead coordinates $(\theta, \phi)$ in radians following Chen et al. (2021) and mapped into canonical ECG-AIM order:

| Index | Lead Name | Polar Angle $\theta$ (rad) | Azimuthal Angle $\phi$ (rad) | Anatomical Vector Orientation |
|:---:|:---:|:---:|:---:|:---|
| 0 | **I** | $\pi / 2 \approx 1.5708$ | $\pi / 2 \approx 1.5708$ | Frontal horizontal axis ($0^\circ$) |
| 1 | **II** | $5\pi / 6 \approx 2.6180$ | $\pi / 2 \approx 1.5708$ | Frontal inferior axis ($+60^\circ$) |
| 2 | **III** | $5\pi / 6 \approx 2.6180$ | $-\pi / 2 \approx -1.5708$ | Frontal inferior-right axis ($+120^\circ$) |
| 3 | **aVR** | $\pi / 3 \approx 1.0472$ | $-\pi / 2 \approx -1.5708$ | Frontal superior-right axis ($-150^\circ$) |
| 4 | **aVL** | $\pi / 3 \approx 1.0472$ | $\pi / 2 \approx 1.5708$ | Frontal high lateral axis ($-30^\circ$) |
| 5 | **aVF** | $\pi \approx 3.1416$ | $\pi / 2 \approx 1.5708$ | Frontal vertical inferior axis ($+90^\circ$) |
| 6 | **V1** | $\pi / 2 \approx 1.5708$ | $-\pi / 18 \approx -0.1745$ | 4th ICS, right sternal border |
| 7 | **V2** | $\pi / 2 \approx 1.5708$ | $\pi / 18 \approx 0.1745$ | 4th ICS, left sternal border |
| 8 | **V3** | $19\pi / 36 \approx 1.6581$ | $\pi / 12 \approx 0.2618$ | Midway between V2 and V4 |
| 9 | **V4** | $11\pi / 20 \approx 1.7279$ | $\pi / 6 \approx 0.5236$ | 5th ICS, left midclavicular line |
| 10 | **V5** | $16\pi / 30 \approx 1.6755$ | $\pi / 3 \approx 1.0472$ | 5th ICS, left anterior axillary line |
| 11 | **V6** | $16\pi / 30 \approx 1.6755$ | $\pi / 2 \approx 1.5708$ | 5th ICS, left midaxillary line |

### Encoded 12-D Theta Features (Provenance Baseline)
Angular projection $u_l = \text{stack}[\theta, \theta+\phi, \theta-\phi, \sin(\cdot), \cos(\cdot)]$:
- Lead I: `[1.5708, 1.5708, 3.1416, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]`
- Lead aVF: `[3.1416, 1.5708, 4.7124, 1.5708, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0]`

---

## 3. Geometric Derangement Permutation Control

To test whether the network exploits **genuine spatial orientation** rather than merely using $\mathbf{u}_l$ as 12 arbitrary distinct identifiers, cell **D5** uses a deterministic derangement:
$$\text{perm} = (i + 5) \pmod{12}$$

| Target Lead | True Angles $(\theta, \phi)$ | Permuted Assigned Angles | Assigned Lead Geometry |
|:---:|:---:|:---:|:---:|
| **I** | I ($\pi/2, \pi/2$) | aVF ($\pi, \pi/2$) | Vertical Inferior |
| **II** | II ($5\pi/6, \pi/2$) | V1 ($\pi/2, -\pi/18$) | Septal Precordial |
| **III** | III ($5\pi/6, -\pi/2$) | V2 ($\pi/2, \pi/18$) | Septal Precordial |
| **aVR** | aVR ($\pi/3, -\pi/2$) | V3 ($19\pi/36, \pi/12$) | Anterior Precordial |
| **aVL** | aVL ($\pi/3, \pi/2$) | V4 ($11\pi/20, \pi/6$) | Anterior Precordial |
| **aVF** | aVF ($\pi, \pi/2$) | V5 ($16\pi/30, \pi/3$) | Lateral Precordial |
| **V1** | V1 ($\pi/2, -\pi/18$) | V6 ($16\pi/30, \pi/2$) | Lateral Precordial |
| **V2** | V2 ($\pi/2, \pi/18$) | I ($\pi/2, \pi/2$) | Frontal Horizontal |
| **V3** | V3 ($19\pi/36, \pi/12$) | II ($5\pi/6, \pi/2$) | Frontal Inferior |
| **V4** | V4 ($11\pi/20, \pi/6$) | III ($5\pi/6, -\pi/2$) | Frontal Inferior-Right |
| **V5** | V5 ($16\pi/30, \pi/3$) | aVR ($\pi/3, -\pi/2$) | Frontal Superior-Right |
| **V6** | V6 ($16\pi/30, \pi/2$) | aVL ($\pi/3, \pi/2$) | Frontal High Lateral |

**Derangement Property**: Zero fixed points ($\text{perm}[i] \neq i, \forall i$). Target waveform labels remain in canonical order. If $D_3 \approx D_5$, theta coordinates function merely as categorical identifiers.

---

## 4. Normalization and Zero-Leakage Policy

### Audit of `--zscore-norm`
- **Finding**: Record-wide normalization across all 12 leads before masking computes $\mu(X_{12}), \sigma(X_{12})$.
- **Verdict**: **`TARGET_NORMALIZATION_LEAKAGE_NONDEPLOYABLE`**.
- In strict 1-lead clinical deployment (e.g., smartwatches, single-patch monitors), the 11 hidden leads are unavailable at test time.
- **Protocol Mandate**: All Stage A models consume **raw physical millivolts** of the single observed lead. No statistics from unobserved leads enter preprocessing.

---

## 5. Stage A Experimental Matrix

| Cell | Run Name | Spatial Code | Fusion Mode | Reconstruction Loss | Purpose & Hypothesis |
|:---|:---|:---:|:---:|:---:|:---|
| **D0** | `D0_current_id_currentloss` | Learned Lead ID | Additive ($H + e_l$) | Current (MSE) | Frozen baseline anchor |
| **D1** | `D1_theta_mul_currentloss` | Exact $(\theta, \phi)$ | Multiplicative ($H \odot g_l$) | Current (MSE) | Theta under established objective |
| **D2** | `D2_current_id_l1` | Learned Lead ID | Additive ($H + e_l$) | Genuine L1 | Isolate 3DRECON L1 loss effect |
| **D3** | `D3_theta_mul_l1` | Exact $(\theta, \phi)$ | Multiplicative ($H \odot g_l$) | Genuine L1 | **Primary 3DRECON-QT cell** |
| **D4** | `D4_learned12_mul_l1` | Learned $12 \times 12$ matrix | Multiplicative ($H \odot g_l$) | Genuine L1 | Capacity / lead-ID control |
| **D5** | `D5_permuted_theta_mul_l1` | Permuted $(\theta, \phi)$ | Multiplicative ($H \odot g_l$) | Genuine L1 | Physical-geometry control |

### Hypothesis Contrasts
1. **$D_1 - D_0$**: Does theta help under the established MSE objective?
2. **$D_2 - D_0$**: Does switching from MSE to L1 improve baseline performance?
3. **$D_3 - D_2$**: Does theta conditioning improve reconstruction once the loss matches 3DRECON?
4. **$D_3 - D_4$**: Are physical spherical coordinates superior to arbitrary learned continuous representations?
5. **$D_3 - D_5$**: Does the actual physical mapping between coordinates and anatomical leads matter?

---

## 6. Precordial Progression Metric ($\Delta V_k$)

In addition to aggregate Pearson $r$ and precordial mean $r_{V1:V6}$, we compute the differential transition between adjacent chest leads:
$$\Delta V_k(t) = V_{k+1}(t) - V_k(t), \quad k \in \{1, 2, 3, 4, 5\}$$
$$r_{\text{trans}} = r(\widehat{\Delta V_k}, \Delta V_k)$$

This prevents models that generate generic, overly smoothed precordial waveforms from appearing artificially competitive.
