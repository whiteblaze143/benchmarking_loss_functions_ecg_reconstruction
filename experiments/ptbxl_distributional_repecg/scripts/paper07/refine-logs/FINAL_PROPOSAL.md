# Final Proposal: Paper 07 — Continuous Measurement-Operator ECG

## 1. Problem Anchor & Clinical Motivation
In conventional clinical practice, standard 12-lead electrocardiography treats multichannel voltage recordings as a static, ordered matrix $X \in \mathbb{R}^{T \times 12}$ (or 8 independent voltage channels: I, II, V1–V6). However, in realistic monitoring environments:
1. **Missing or detached leads**: Telemetry in intensive care units, emergency ambulances, and wearable patches regularly observe only a sparse subset of leads ($m \in [1, 6]$).
2. **Electrode displacement**: Misplacement of precordial leads V1–V6 by as little as 1.5 cm significantly alters the observed waveform morphologies.
3. **Non-standard configurations**: Pediatric leads (V3R, V4R), posterior leads (V7–V9), Mason-Likar exercise placements, and Frank XYZ vectorcardiographic leads do not conform to the standard 12-lead coordinate system.
4. **Physical Linearity of the Electrical Field**: Under quasi-static Maxwell equations in biological volume conductors, an electrode voltage is a linear functional of the cardiac current source density $\mathbf{J}(\mathbf{r}, t)$ projected along a lead vector $\mathbf{c}_i \in \mathbb{R}^3$:
   $$v_i(t) = \mathbf{c}_i^\top \mathbf{p}(t)$$
   In 8-lead physical voltage space, an arbitrary lead projection is an operator $q \in \mathbb{S}^7 \subset \mathbb{R}^8$:
   $$x_q(t) = q^\top x(t)$$

Paper 07 shifts the paradigm: instead of treating leads as fixed, discrete categorical channels, we represent the ECG as an **unordered set of continuous operator-response pairs**:
$$\mathcal{D} = \big\{(q_1, r_{q_1}), (q_2, r_{q_2}), \dots, (q_m, r_{q_m})\big\}, \quad q_i \in \mathbb{S}^7, \; r_{q_i} \in \mathbb{R}^{16 \times 128}$$

---

## 2. Core Inductive Biases and Formal Mathematics

### 2.1 Continuous Operator Representation vs Categorical Lead Tokens
- **Continuous Operator Embedding $E(q)$**:
  $$E(q) = W_2 \, \text{GELU}(W_1 q), \quad W_1 \in \mathbb{R}^{64 \times 8}, \; W_2 \in \mathbb{R}^{64 \times 64}$$
  Because $E$ is a continuous mapping on $\mathbb{S}^7$, it can interpolate smoothly across novel, unseen lead projections:
  $$q_{\text{interp}} = \frac{(1-\alpha) I + \alpha V_2}{\|(1-\alpha) I + \alpha V_2\|}$$
  and generalize out-of-distribution to derived limb leads (III, aVR, aVL, aVF) and dense random projections.
- **Categorical Lead Embedding Control $E_{\text{cat}}(\text{id})$**:
  Maps discrete lead indices $\{0, 1, \dots, 7\}$ to learned vectors via an embedding table. Any unseen lead orientation maps to `UNKNOWN_ID` ($-1$), producing a degenerate constant vector that is completely blind to spatial geometry.

### 2.2 Response Atom Phase-Distribution Embedding
For a unit measurement operator $q \in \mathbb{S}^7$, the projected waveform and its temporal derivative define the physical phase-space response atom:
$$a_q(t) = \left[ \frac{q^\top x(t)}{V_{\text{scale}}}, \; \frac{d}{dt}\left(\frac{q^\top x(t)}{V_{\text{scale}}}\right) \right] \in \mathbb{R}^2$$
These atoms are grouped across 16 circular cardiac phases and embedded into an RKHS via a frozen 128-landmark Nyström map:
$$r_q \in \mathbb{R}^{16 \times 128}$$

### 2.3 Set Transformer Context Aggregator
Given an arbitrary observed context set $\{(q_1, r_{q_1}), \dots, (q_m, r_{q_m})\}$ of size $m \in [1, 6]$:
1. Feature extraction: $h_i = \text{PhaseCNN}(r_{q_i}) \in \mathbb{R}^{128}$.
2. Operator-response fusion: $u_i = [E(q_i); h_i] \in \mathbb{R}^{192}$.
3. Permutation-equivariant set attention:
   $$\{v_1, \dots, v_m\} = \text{TransformerEncoder}(\{u_1, \dots, u_m\})$$
4. Permutation-invariant attention pooling with a learned query vector:
   $$z = \text{MultiHeadAttention}(\text{query}, \{v_i\}, \{v_i\}) \in \mathbb{R}^{256}$$
5. Multi-label diagnosis: $\hat{y} = \text{Head}(z) \in \mathbb{R}^5$.

### 2.4 Physical Orientation Law & Polarity Invariance
Linearity of volume conduction implies:
$$x_{-q}(t) = -q^\top x(t) = -x_q(t) \implies a_{-q}(t) = -a_q(t)$$
Reversing the polarity of an electrode ($q \to -q$) inverts the observed waveform, but the underlying patient pathology is physically invariant. We enforce this through explicit inverted training pairs and a symmetric Bernoulli KL loss:
$$\mathcal{L}_{\text{orientation}} = \mathcal{D}_{\text{KL}}^{\text{sym}}\Big(\sigma(\hat{y}(q)), \sigma(\hat{y}(-q))\Big)$$

### 2.5 Reconstructive Inversion (Auxiliary Task)
To regularize the patient representation $z$, an auxiliary decoder predicts the unseen response $r_{q^*}$ along an arbitrary query operator $q^* \in \mathbb{S}^7$:
$$\hat{r}_{q^*} = \text{Decoder}(z, E(q^*)) \in \mathbb{R}^{16 \times 128}$$
$$\mathcal{L}_{\text{recon}} = \|\hat{r}_{q^*} - r_{q^*}\|^2$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{bce}}(\hat{y}, y) + 0.1 \mathcal{L}_{\text{recon}} + 0.05 \mathcal{L}_{\text{orientation}}$$

---

## 3. Prior Art & Defensible Novelty Boundary

1. **Dower (1988) & Kors (1990) Transforms**:
   Linear regression matrices mapping Frank XYZ $\leftrightarrow$ 12-lead. Paper 07 is not a static linear regression matrix; it operates dynamically on arbitrary variable-sized subsets $m \in [1, 6]$ in a continuous operator space.
2. **Conditional Neural Processes (Garnelo et al., 2018)**:
   CNPs define distributions over functions given context pairs $(x_c, y_c)$. Paper 07 grounds this in physical cardiac measurement operators $q \in \mathbb{S}^7$, where inputs are lead orientation vectors and responses are phase-stratified distribution embeddings (Phase-KME).
3. **Flexible-Lead ECG Models**:
   Recent deep learning models mask discrete lead tokens. Paper 07 demonstrates why **continuous geometric operators $q \in \mathbb{S}^7$ strictly outperform categorical lead embeddings**, especially on out-of-distribution lead geometries.
