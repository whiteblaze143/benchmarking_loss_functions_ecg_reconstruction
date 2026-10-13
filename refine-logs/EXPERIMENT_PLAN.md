# Experiment Plan: Calibration-Efficient Patient-Specific Physiological Foundation Models

**Central Thesis**: Universal ECG representations contain persistent subject-specific structure; the key question is whether that structure can be converted into clinically useful state-change information with very little calibration.  
**Comparator**: Cardiac Sensing Foundation Model (CSFM Base, $d=768$, Gu et al., *Nature Machine Intelligence* 2026) and ECG-LFM.  
**Date**: 2026-09-30  

---

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Block | Status |
| :--- | :--- | :--- | :--- | :--- |
| **C0: Patient identity in representations** | Establishes whether patient-specific structure is unique to SetOperator or generic to cardiac FMs | Head-to-head PTB-XL audit: AUROC, Top-$K$ retrieval, and variance $R^2$ under Native and Matched PCA ($d=128, 256$) | **Block 0-FM (M0-FM)** | **FIRST RUN TO LAUNCH** |
| **C1: Calibration efficiency & saturation** | Clinical feasibility constraint: which representation gains utility fastest from sparse history? | $G(K) = \text{Perf}(K) - \text{Perf}(0)$ for $K \in \{0, 1, 2, 3, 5\}$ under identical mean-baseline machinery | **Block 1-C (M1-C)** | Scheduled |
| **C2: Set aggregation value** | Disentangles representation quality from permutation-invariant history pooling | $2 \times 2$ factorial evaluation: SetOperator vs CSFM $\times$ Empirical Mean vs Learned Set Reference | **Block 1-S (M1-S)** | Scheduled |
| **C3: Sparse-lead resilience** | Wearable translation: does geometry survive when measurement burden drops? | Patient verification AUROC under $12L, II, I+II, I+II+V1, I+II+V5$ | **Block 2 (Sparse-Lead)** | Scheduled |
| **C4: Dynamic disease preservation** | Proves model retains acute clinical sensitivity while holding stable patient coordinate | Triad audit on longitudinal transitions: state accessibility, patient preservation, and relative change gain | **Block 3 (Disease-Preserve)** | Scheduled |
| **C5: Clean external generalization** | Evaluates out-of-domain transfer without pretraining contamination | Clean cohorts with no pretraining overlap: Kingston ICU AF, SJLIFE, LUDB, EchoNext | **Block 4 (Clean-External)** | Scheduled |
| **C6: Multi-modal reference alignment** | Enables cross-sensor calibration between wearable PPG and clinical ECG | Cross-modal evaluation of CSFM ($z_{\text{CSFM}}^{\text{ECG}}, z_{\text{CSFM}}^{\text{PPG}}, z_{\text{CSFM}}^{\text{ECG+PPG}}$) vs SetOperator | **Block 5 (MIMIC-PPG)** | Scheduled |

---

## Detailed Experiment Blocks

### Block 0-FM: Foundation-Model Patient-Geometry Benchmark (Milestone M0-FM / M0-C)
- **Objective**: Directly test whether SetOperator preserves patient-specific structure more strongly than a state-of-the-art generic cardiac foundation model.
- **Backbone Models**:
  1. Frozen SetOperator E6 ($z_{\text{SO}} \in \mathbb{R}^{256}$)
  2. Frozen CSFM Base with identity classification head ($z_{\text{CSFM}} \in \mathbb{R}^{768}$)
- **Cohort & Harmonization**:
  - Exact same PTB-XL repeated-patient cohort: **1,693 patients, 4,087 ECGs**.
  - CSFM Preprocessing: 250 Hz resampling, NeuroKit2 cleaning (`nk.ecg_clean`), per-channel z-score normalization, 2,500 samples (10 s).
  - Deterministic evaluation: For recordings $>10$ s, evaluate deterministic non-overlapping 10 s windows and mean-pool window representations.
- **Dimensionality Control**:
  - Evaluate **Native Representations** ($d=256$ vs $d=768$).
  - Evaluate **Matched Dimensionality**: Fit PCA strictly on training patients only and transform representations to $d=128$ and $d=256$. Never fit PCA on test patients.
- **Reporting Protocol**:
  - Same-patient Euclidean verification AUROC
  - Same-patient Cosine verification AUROC
  - Top-1, Top-5, Top-10 closed-set retrieval accuracy & Median Patient Rank
  - Descriptive variance decomposition: Patient $R^2$ vs Diagnostic-Superclass $R^2$
  - Cross-disease verification AUROC (same-patient diagnostic transitions vs impostors)
  - Time-stratified verification AUROC: $<7\text{ d}$, $7-30\text{ d}$, $30-180\text{ d}$, $>180\text{ d}$
  - *No composite score; report each metric directly.*

### Block 1-C: Calibration-Efficiency Comparison (Milestone M1-C)
- **Objective**: Determine which representation becomes clinically useful fastest when sparse personal history becomes available.
- **Conditions**:
  - Cold Start ($K=0$): $X_0 = z_t$
  - One Prior ($K=1$): $X_1 = [z_t, z_t - z_{t-1}]$
  - $K$ Priors ($K \in \{2, 3, 5\}$): $b_p^{(K)} = \frac{1}{K}\sum_{i=1}^K z_{p, t-i}$, $X_K = [z_t, z_t - b_p^{(K)}]$
- **Standardized Machinery**: Identical mean baseline, shrinkage rule, chronological orderings, downstream linear classifiers, hyperparameter grids, patient splits, and bootstrap samples.
- **Primary Endpoint**: Calibration scaling curves:
  $$G_{\text{SO}}(K) = \text{Perf}_{\text{SO}}(K) - \text{Perf}_{\text{SO}}(0) \quad \text{vs} \quad G_{\text{CSFM}}(K) = \text{Perf}_{\text{CSFM}}(K) - \text{Perf}_{\text{CSFM}}(0)$$

### Block 1-S: Set Aggregation Experiment (Milestone M1-S)
- **Objective**: Disentangle whether performance advantages stem from the ECG representation or from permutation-invariant set pooling.
- **Factorial Design ($2 \times 2$)**:
  1. SetOperator + Empirical Mean Reference ($b_{\text{mean}}$)
  2. CSFM + Empirical Mean Reference ($b_{\text{mean}}$)
  3. SetOperator + Learned Set Reference ($b_{\text{set}} = \rho(\sum \phi([z_i, \Delta t_i, q_i]))$)
  4. CSFM + Learned Set Reference ($b_{\text{set}}$)

### Block 2: Sparse-Lead Representation Audit
- **Objective**: Quantify patient information retention as sensing burden decreases.
- **Configurations**:
  - $12L$ (Full standard diagnostic)
  - Lead $II$ (Standard wearable single-lead)
  - Lead $I + II$ (Dual-lead)
  - Lead $I + II + V1$ (3-lead rhythm + precordial)
  - Lead $I + II + V5$ (3-lead limb + lateral precordial)
- **Mechanism**: SetOperator measurement operators vs CSFM native channel index API.

### Block 3: Disease-Preservation Comparison
- **Objective**: Verify that patient identity preservation does not sacrifice acute disease decodability.
- **Protocol**: Freeze both encoders; fit identical linear probes for `NORM`, `MI`, `STTC`, `CD`, `HYP`.
- **Evaluation**: Track longitudinal transitions ($NORM \to MI$, $NORM \to STTC$, $CD^- \to CD^+$).
- **Simultaneous Triad**:
  1. State accessibility ($z_t \to \text{disease}$)
  2. Patient preservation (same-patient verification survives transition)
  3. Patient-relative change ($z_t - b_p$ improves transition discrimination)

### Block 4: Clean External Generalization & Pretraining Ledger
- **Datasets**:
  - Kingston ICU AF: Clean external rhythm & AF transition benchmark (`CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE`).
  - SJLIFE: Clean external multi-year pediatric cancer survivor cohort (`CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE`).
  - LUDB: Clean external detailed wave delineation (`CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE`).
  - EchoNext: Clean external structural heart disease / LVEF progression (`CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE`).
  - Chapman / Shaoxing: Clean external multi-label diagnostic cohort (`CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE`).
  - MIMIC-III-Ext-PPG & MIMIC-IV: Explicitly flagged as `CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE`; evaluated for internal mechanics only.

### Block 5: Multi-Modal Wearable Physiology (MIMIC-PPG)
- **Objective**: Characterize cross-modal patient coordinates using CSFM's native multi-modal capacity.
- **Conditions**: $z_{\text{CSFM}}^{\text{ECG}}$, $z_{\text{CSFM}}^{\text{PPG}}$, and $z_{\text{CSFM}}^{\text{ECG+PPG}}$.
- **Comparison**: Compare against SetOperator dual-branch configuration $[z_E, z_P]$.

---

## Run Order & Immediate Next Steps
1. **Launch Block 0-FM (M0-FM)**:
   - Extract frozen CSFM Base features ($d=768$) for all 4,087 PTB-XL repeated-patient records.
   - Run the exact M0 audit side-by-side with SetOperator ($d=256$).
   - Fit PCA ($d=128, 256$) strictly on training patients and re-evaluate both encoders.
   - Launch in a persistent detached `tmux` session (`prism_m0_fm`).
2. **Launch Block 1-C (M1-C)**:
   - Compute calibration curves $K \in \{0, 1, 2, 3, 5\}$ for both SetOperator and CSFM.
3. **Launch Block 1-S (M1-S)**:
   - Run the $2 \times 2$ set aggregator factorial.
