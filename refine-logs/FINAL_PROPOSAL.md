# Research Proposal: Calibration-Efficient Patient-Specific Physiological Foundation Models

## Revised Project Thesis
> **Universal ECG representations contain persistent subject-specific structure; the key question is whether that structure can be converted into clinically useful state-change information with very little calibration.**

This reframing shifts the scientific positioning from a speculative claim that SetOperator has discovered an exclusive "patient physiological coordinate manifold" to a rigorous, defensible empirical question:
1. Does contemporary large-scale cardiac self-supervised pretraining (e.g. CSFM, ECG-LFM) naturally encode persistent subject identity into latent representations, or is this geometry distinctive to measurement-aware operators?
2. When sparse personal historical observations become available ($K \in \{0, 1, 2, 3, 5\}$), which representation and reference construction framework converts personal history into clinical state-change sensitivity most efficiently?

---

## Problem Anchor & Clinical Constraints
- **Bottom-Line Problem**: Standard population ECG models map individual recordings to population diagnostic probabilities ($x_t \rightarrow \hat{y}_t$). However, clinical decision-making inherently asks: *"Has this patient meaningfully changed from their personal baseline?"* (e.g., distinguishing chronic pre-existing bundle branch block from new-onset acute ischemia, or baseline rate fluctuations from acute arrhythmia conversion).
- **The Calibration Bottleneck**: Prior personalization methods either demand dense longitudinal histories (RNNs / longitudinal Transformers) or expensive per-patient gradient fine-tuning. In real-world clinical care, calibration data is a severe bottleneck; models must function immediately at cold-start ($K=0$) and opportunistically improve as sparse historical records accumulate ($K \in \{1, 2, 3, 5\}$).
- **Central Research Question**: What is the minimum amount of patient-specific information required to make physiological AI meaningfully more accurate than population models, and how do general cardiac foundation representations compare against measurement-aware operators along this calibration–utility frontier?
- **Operational Representation Model**:
  $$z_t = f_\theta(x_t) \approx s_p + \delta_t + \epsilon$$
  where $s_p$ represents an individual patient reference component and $\delta_t$ captures dynamic physiological deviation. We do **not** claim this decomposition is mathematically exact, orthogonal, or that $\delta_t$ is a "pure state perturbation"; rather, we evaluate it as an operational hypothesis across static vs dynamic diagnostic tasks.
- **Explicit Non-Goals**:
  - We do *not* train neural networks per patient, nor run per-patient fine-tuning loops or hypernetwork adapters.
  - We do *not* claim "personalized AI" in an unconstrained sense.
  - We do *not* assert that SetOperator is the sole architecture capable of preserving patient identity without empirical benchmarking against leading foundation models.

---

## Technical Gap & Key Insights
1. **Representational Gap**: While recent foundation models (e.g., CSFM, Gu et al., *Nature Machine Intelligence* 2026; ECG-LFM, Nature Communications 2026) are trained on biosignals from over 1.7M to 10M individuals, their latent geometries have predominantly been evaluated on static, population-level classification tasks. It remains unknown whether these universal embeddings can act as stable patient-specific reference frames.
2. **Methodological Insight**: Clinical history is naturally an irregular, unordered set of observations $H_p = \{x_1, \dots, x_K\}$, not an evenly sampled dense time series. Personalizing the *reference coordinate* rather than personalizing *model parameters* allows instant inference without retraining.
3. **The Calibration–Utility Frontier**: Rather than declaring a single model "better," the core scientific contribution is mapping performance gains $G(K) = \text{Performance}(K) - \text{Performance}(0)$ as history size $K$ grows, evaluating which representations achieve the highest state-change discrimination with the least calibration burden.

---

## Foundation-Model Comparison Track (CSFM & ECG-LFM)
### Primary Comparator: Cardiac Sensing Foundation Model (CSFM)
- **Pretraining**: Self-supervised masked autoencoding on 1.7 million individuals across MIMIC-III-WDB, MIMIC-IV-ECG, and CODE-FULL (Gu et al., *Nature Machine Intelligence* 2026).
- **Official Weights & API**: Evaluated strictly using the official pretrained weights (`CSFM_base.pth`, $d=768$, and `CSFM_tiny.pth`) with classification head replaced by `nn.Identity()`.
- **Primary Comparison**: Frozen encoder vs frozen encoder under identical downstream machinery:
  $$f_{\text{SO}}(x) \to z_{\text{SO}} \quad \text{vs} \quad f_{\text{CSFM}}(x) \to z_{\text{CSFM}}$$
- **ECG-LFM Track**: Where available, benchmark alongside ECG-LFM (Nature Communications 2026, 10M ECGs, multi-segment contrastive objective explicitly targeting patient-level consistency).

### Critical Dimensionality Control
- SetOperator produces $z_{\text{SO}} \in \mathbb{R}^{256}$. CSFM Base produces $z_{\text{CSFM}} \in \mathbb{R}^{768}$.
- High-dimensional representations can behave differently under Euclidean distances and variance partitioning.
- **Protocol**: Every patient-geometry audit is evaluated in two parallel forms:
  1. **Native Representations**: $z_{\text{SO}} \in \mathbb{R}^{256}$ vs $z_{\text{CSFM}} \in \mathbb{R}^{768}$.
  2. **Matched Dimensionality**: Fit Principal Component Analysis (PCA) strictly on training patients only, projecting both representations to common dimensions:
     $$d = 128 \quad \text{and} \quad d = 256$$
     *Rule*: PCA projection is **never** fit on test patients.

---

## Experimental Milestones & Validation Roadmap

### Milestone M0-FM: Foundation-Model Patient-Geometry Benchmark
- **Question**: Does SetOperator preserve patient-specific structure more strongly than a state-of-the-art generic cardiac foundation model, or is identity preservation a shared property of large-scale cardiac representations?
- **Cohort**: Exact same PTB-XL repeated-patient cohort:
  - 1,693 repeated patients, 4,087 ECGs.
  - Identical patient pairs, identical time strata (<7d, 7–30d, 30–180d, >180d).
  - Identical cross-disease pairs and identical 10× matched impostor sampling.
- **Reported Metrics** (Direct reporting, no single aggregated score):
  - Same-patient Euclidean verification AUROC (Native, $d=256$, $d=128$)
  - Same-patient Cosine verification AUROC (Native, $d=256$, $d=128$)
  - Closed-set patient retrieval: Top-1, Top-5, Top-10 accuracy and Median Rank
  - Descriptive variance partitioning: Patient $R^2$ vs Diagnostic-Superclass $R^2$
  - Cross-disease verification AUROC (same patient with diagnostic transition vs impostor)
  - Time-stratified verification AUROC: $<7\text{ d}$, $7-30\text{ d}$, $30-180\text{ d}$, $>180\text{ d}$

### Milestone M1-C: Calibration-Efficiency Comparison
- **Question**: Which representation becomes clinically useful fastest when sparse personal history becomes available?
- **Protocol**: For each frozen encoder independently ($z_t = f(x_t)$), construct identical calibration schemes:
  - Cold Start ($K=0$): $X_0 = z_t$
  - One Prior ($K=1$): $X_1 = [z_t, z_t - z_{t-1}]$
  - $K$ Priors ($K \in \{2, 3, 5\}$): $b_p^{(K)} = \frac{1}{K}\sum_{i=1}^K z_{p, t-i}$, $X_K = [z_t, z_t - b_p^{(K)}]$
- **Standardized Machinery**: Both encoders receive identical mean baselines, shrinkage rules, chronological orderings, downstream linear classifiers, hyperparameter grids, patient splits, and bootstrap samples.
- **Endpoint**: Calibration scaling curves:
  $$G_{\text{SO}}(K) = \text{Perf}_{\text{SO}}(K) - \text{Perf}_{\text{SO}}(0) \quad \text{vs} \quad G_{\text{CSFM}}(K) = \text{Perf}_{\text{CSFM}}(K) - \text{Perf}_{\text{CSFM}}(0)$$

### Milestone M1-S: Set Aggregation vs Mean Reference ($2 \times 2$ Factorial)
- **Question**: Does a learned permutation-invariant set aggregator add value over an empirical mean, and does this benefit depend on the encoder?
- **History Set**: $H_p^{(K)} = \{z_{p, t_1}, \dots, z_{p, t_K}\}$.
- **Aggregators**:
  - Empirical mean: $b_{\text{mean}} = \frac{1}{K}\sum_i z_i$
  - Low-capacity set reference: $b_{\text{set}} = \rho\left(\sum_i \phi([z_i, \Delta t_i, q_i])\right)$
- **Factorial Grid ($2 \times 2$)**:
  1. SetOperator + Empirical Mean Reference
  2. CSFM + Empirical Mean Reference
  3. SetOperator + Learned Set Reference
  4. CSFM + Learned Set Reference
- **Significance**: Isolates whether representation quality vs set-based historical pooling drives performance.

### Input Harmonization & Sparse-Lead Audit
- **CSFM Preprocessing**: Resample to 250 Hz, clean ECG via NeuroKit2 (`nk.ecg_clean`), per-channel z-score standardization, 2,500 samples (10 s).
- **Deterministic Evaluation Modification**: Avoid stochastic window cropping. For recordings $>10$ s, evaluate deterministic non-overlapping 10 s windows, extract embeddings, and mean-pool window representations into a single recording representation while logging window indices.
- **Sparse-Lead Configurations**:
  - Test lead configurations: $12L$, Lead $II$, $I+II$, $I+II+V1$, $I+II+V5$.
  - Evaluate how patient verification and calibration efficiency degrade as sensing burden drops.

### Disease-Preservation Comparison
- Freeze both encoders; fit identical linear probes for primary diagnostic categories (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
- Evaluate longitudinal pairs with disease transitions ($NORM \to MI$, $NORM \to STTC$, $CD^- \to CD^+$).
- Simultaneously measure:
  1. **State Accessibility**: Can $z_t$ decode disease?
  2. **Patient Preservation**: Does the same-patient relationship survive the transition?
  3. **Patient-Relative Change**: Does $z_t - b_p$ improve transition discrimination over raw $z_t$?
- Core requirement: Simultaneous demonstration of *stable patient context*, *dynamic disease sensitivity*, and *sparse-calibration utility*.

---

## External Dataset Evaluation Strategy & Pretraining Overlap Ledger
To guarantee scientific integrity, all external evaluation cohorts must explicitly track pretraining overlap in their manifests:

| Dataset | Modality / Task | SetOperator Overlap | CSFM Overlap | Pretraining Domain Overlap Flag | Role in Benchmarking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PTB-XL** | 12-lead multi-visit repeated cohort | Pretraining Source | None (Clean) | `CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE` | Primary M0/M1 benchmark cohort |
| **Kingston ICU AF** | Continuous rhythm / AF conversion | None (Clean) | None (Clean) | `CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE` | **Clean external** transition benchmark |
| **SJLIFE** | Multi-year childhood survivor follow-up | None (Clean) | None (Clean) | `CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE` | **Clean external** long-term aging/stability |
| **LUDB** | Comprehensive ECG wave delineation | None (Clean) | None (Clean) | `CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE` | **Clean external** morphological probing |
| **EchoNext ECGs** | Structural heart disease (LVEF $\le 45\%$) | None (Clean) | None (Clean) | `CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE` | **Clean external** structural progression |
| **Chapman / Shaoxing** | Multi-label arrhythmia diagnosis | None (Clean) | None (Clean) | `CSFM_PRETRAIN_DOMAIN_OVERLAP = FALSE` | **Clean external** arrhythmia generalization |
| **MIMIC-III-Ext-PPG** | ICU rhythm & multi-modal ECG/PPG | None (Clean) | **Pretrained (WDB)** | `CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE` | Multi-modal mechanics & PPG reference |
| **MIMIC-IV-ECG** | Massive clinical 12-lead cohort | None (Clean) | **Pretrained (MIMIC-IV)** | `CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE` | In-domain reference scaling |

*Strict Policy*: Datasets with pretraining overlap (`TRUE`) are used for longitudinal mechanics and multi-modal alignment, but are **never** described as clean external generalization against CSFM.
