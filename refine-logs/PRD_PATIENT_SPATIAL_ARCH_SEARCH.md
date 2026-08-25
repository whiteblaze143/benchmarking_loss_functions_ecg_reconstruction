# PRD — Patient-Conditioned Spatial Architecture Search for Strict 1→12 ECG Reconstruction

**Date:** August 25, 2026  
**Status:** Successor to active Wavelet-SSL program  
**Dataset:** PTB-XL (21,799 records / 18,869 patients)  
**Primary observed lead:** Lead I ($l_0$)  
**Robustness observed lead:** Lead II ($l_1$), finalists only  
**Inference ECG input:** Exactly one lead ($x_{\text{source}} \in \mathbb{R}^{1 \times 5000}$)  
**Primary objective:** Unchanged `1110000` (MSE + Pearson Correlation + First-Derivative $L_1$)  
**Deployment heads:** Unchanged waveform reconstruction + multi-class delineation  

---

## 1. Hard Experimental Rule

For every deployable model:
$$x_{\text{source}} \in \mathbb{R}^{1 \times 5000}$$
is the only measured ECG input at test/deployment time.

### Allowed Additional Inputs:
$$\boxed{\text{age, sex, height, weight}}$$
because these are independently available, non-ECG patient characteristics, not quantities reconstructed or inferred from hidden leads.

### Allowed Deterministic Transformations of the Observed Lead:
- Time derivatives ($\dot{x}, \ddot{x}$),
- Multi-scale temporal sampling resolutions (500 Hz, 250 Hz, 100 Hz),
- Beat segmentation and median beat extraction from the single observed lead,
- R-peak and cardiac-phase coordinates.

### Prohibited Inputs (Strict Boundary):
- Diagnosis labels / SCP statements / infarction locations,
- Electrical axis from the 12-lead clinical ECG,
- PR / QRS / QTc intervals measured from the full 12-lead,
- Any additional ECG lead at deployment time.

---

## 2. Baseline & Study Preservation

* **Reconstruction Objective**: `1110000`
* **Optimizer**: AdamW ($\text{LR} = 10^{-4}, \text{weight decay} = 10^{-4}$)
* **Batch Size**: 32 (effective)
* **Epochs**: 3 for initial broad screen
* **Seed**: 42
* **Patient Folds**: Disjoint train / validation / test splits

---

## 3. Core Architectural Concept

```
                         Lead I (1 x 5000)
                                 |
                     source morphology encoder
                                 |
                          H_source [200, D]
                                 |
             +-------------------+-------------------+
             |                                       |
             |                                patient metadata
             |                          [age, sex, height, weight]
             |                                       |
             |                                metadata encoder
             |                                       |
             +-------------------+-------------------+
                                 |
                         spatial decoder
                                 |
                        12 x 200 latent grid
                                 |
                         waveform decoder
                                 |
                      12 reconstructed leads
```

The search focuses on **three unresolved mappings**:
1. $\text{Lead I} \rightarrow \text{cardiac state}$
2. $(\text{cardiac state}, \text{patient}) \rightarrow \text{spatial lead representation}$
3. $\text{spatial lead representation} \rightarrow V_1, \ldots, V_6$

---

## 4. Metadata Pipeline & Schema

Canonical metadata vector $m \in \mathbb{R}^8$:
$$m = \begin{bmatrix} \text{age}_z, & \text{sex}, & \text{height}_z, & \text{weight}_z, & m_{\text{age}}, & m_{\text{sex}}, & m_{\text{height}}, & m_{\text{weight}} \end{bmatrix}^T$$

Where continuous fields are standardized strictly using **training patients only**:
- $\text{sex} \in \{0: \text{Male}, 1: \text{Female}\}$, missing imputed with training mode.
- $\text{height}_z, \text{weight}_z$: z-scored against training set medians and IQRs.
- $m_{\text{var}} \in \{0, 1\}$: missingness indicators.

---

## 5. Complete 48-Cell Search Matrix (Lead I Screen)

### Block A — Anchors & Metadata Conditioning
* `A00_base_frozen`: Exact frozen winning architecture from current sweep.
* `A01_param_matched`: Scaled-width baseline matching added parameter count.
* `A02_meta_age_sex_concat`: Late concatenation of age + sex embeddings.
* `A03_meta_all4_concat`: Late concatenation of all 4 metadata variables.
* `A04_meta_patient_token`: Metadata encoded into a dedicated patient token $e_p \in \mathbb{R}^{768}$.
* `A05_meta_lead_embedding`: Patient metadata modulates target lead embeddings $q_l = g(e_l, e_p)$.
* `A06_meta_film_grid`: Metadata FiLM affine scaling on the full $12 \times 200$ latent grid.
* `A07_meta_film_leadwise`: Lead-specific FiLM modulation $h'_{l,t} = \gamma_l(e_p) \odot h_{l,t} + \beta_l(e_p)$.
* `A08_meta_target_query`: Patient token integrated into target query generation.
* `A09_meta_hyperdecoder`: Hypernetwork generating decoder modulation weights from $(e_p, e_l)$.
* `A10_meta_age_sex_film`: Age and sex only version of A07 FiLM.
* `A11_meta_all4_bmi_film`: All 4 metadata variables + derived BMI feature.
* `A12_meta_shuffled_control`: A07 architecture with patient metadata permuted across patients (negative control).

### Block B — Target-Query Spatial Decoders
* `B13_target_query_1x`: 1 cross-attention block ($Q_{\text{target}}$ queries $H_{\text{source}}$).
* `B14_target_query_2x`: 2 cross-attention blocks.
* `B15_target_query_selfattn`: Cross-attention followed by inter-lead target-target self-attention.
* `B16_target_query_gated`: Gated source injection rather than unconstrained residual addition.

### Block C — Hierarchical Lead Structure
* `C17_plane_tokens`: Separate Frontal ($I, II, III, aVR, aVL, aVF$) and Precordial ($V_1\text{--}V_6$) parent tokens.
* `C18_plane_experts`: Dedicated Frontal and Precordial decoder expert branches.
* `C19_plane_shared_then_split`: Shared trunk decoder followed by plane-specific branches.
* `C20_precordial_refiner`: Dedicated $V_1\text{--}V_6$ chest-lead refinement network.

### Block D — Spatial Graph Decoders
* `D21_graph_fixed`: Fixed anatomical ECG adjacency graph ($V_1 \leftrightarrow V_2 \leftrightarrow \dots \leftrightarrow V_6$).
* `D22_graph_geometry`: Edge weights derived from 3D electrode coordinates.
* `D23_graph_learnable`: Learnable adjacency matrix initialized from lead geometry.
* `D24_graph_patient`: Geometry graph conditioned on patient metadata.

### Block E — Target Specialization & Refiners
* `E25_shared_head`: Standard shared waveform synthesis head.
* `E26_per_lead_heads`: 12 separate small decoder heads.
* `E27_shared_plus_refiner`: Shared trunk decoder + per-lead residual blocks $\hat{x}_l = D_{\text{shared}}(h_l) + R_l(h_l)$.
* `E28_grouped_heads`: Grouped heads by territory (limb, septal, anterior, lateral).

### Block F — Low-Dimensional Cardiac Basis
* `F29_basis_k3`: 3 global latent temporal basis components ($K=3$).
* `F30_basis_k6`: 6 global latent temporal basis components ($K=6$).
* `F31_basis_k8`: 8 global latent temporal basis components ($K=8$).
* `F32_basis6_residual`: 6 global basis components + lead-specific residual synthesis.

### Block G — Universal Single-Lead Pretraining
* `G33_random_source8`: Trained with 1 random independent source lead per batch ($I, II, V_1\dots V_6$).
* `G34_balanced_source8`: Balanced source identity scheduling across all 8 independent leads.
* `G35_source8_then_I`: Universal 8-source pretraining $\rightarrow$ fine-tuned on strict Lead I.
* `G36_source8_geometry`: Universal model with explicit source-target relative geometry.

### Block H — Masked-Channel Curricula
* `H37_one_random_lead`: Pretraining reveals exactly 1 random lead per record.
* `H38_random_1to3`: Pretraining reveals 1–3 random leads.
* `H39_random_1to8`: Pretraining reveals 1–8 independent leads.
* `H40_curriculum_8_4_2_1`: Progressive lead masking curriculum ($8 \rightarrow 4 \rightarrow 2 \rightarrow 1$).
* `H41_fullmask_then_I`: Broad lead-mask pretraining $\rightarrow$ strict Lead-I fine-tune.

### Block I — Same-Lead Deterministic Representations
* `I42_raw_d1`: Raw Lead I + first derivative ($\dot{x}$).
* `I43_raw_d1_d2`: Raw Lead I + first and second derivatives ($\dot{x}, \ddot{x}$).
* `I44_multires_raw`: Multi-resolution filter bank (500 Hz, 250 Hz, 100 Hz).
* `I45_full_plus_medianbeat`: 10-s raw recording + single-lead median beat encoder.
* `I46_full_plus_beatstack`: Global recording + beat-level stack encoder.
* `I47_rphase_encoding`: Raw ECG + source-derived cardiac-phase positional encoding.

---

## 6. Execution Lifecycle

```
WAIT_WAVELET_COMPLETION (1110000 & 1111002)
  ├── BUILD_METADATA_PARQUET
  ├── PREFLIGHT_8_CELLS
  ├── FAST_48_LEADI_SCREEN (3 Epochs)
  ├── PROMOTE_TOP_12_MODELS (5 Epochs)
  ├── LEAD_II_ROBUSTNESS (Finalists)
  ├── COMBINATION_STAGE (6 Cells)
  └── 10_EPOCH_CONFIRMATION
```

---

## 7. Primary Evaluation Targets

1. **Missing-Lead Pearson Correlation**: Target $r_{\text{mean}} \ge 0.740$ (Baseline: $0.706$).
2. **5th-Percentile Worst-Case Correlation**: Target $r_{05} \ge 0.430$ (Baseline: $0.365$).
3. **Precordial Leads ($V_1\text{--}V_3$) Pearson**: Target $r_{V1-V3} \ge 0.720$.
4. **Delineation Consistency**: Maintain Macro $F_1 \ge 0.840$ with reduced boundary hallucination.
