# Review Summary: Calibration-Efficient Patient-Specific Physiological Foundation Models

**Problem**: Population ECG models detect population abnormalities, but medicine requires detecting change relative to an individual patient's personal baseline. Conventional personalization fails due to calibration data bottlenecks.  
**Revised Method**: Frame the project around the **calibration–utility frontier**: characterizing which representations convert personal history into clinical state-change sensitivity with minimal calibration ($K \in \{0, 1, 2, 3, 5\}$).  
**Foundation Model Comparator Track**: Integrate **Cardiac Sensing Foundation Model (CSFM)** as a first-class frozen encoder comparator ($f_{\text{SO}}(x) \to z_{\text{SO}}$ vs $f_{\text{CSFM}}(x) \to z_{\text{CSFM}}$), with dimensionality-matched controls (PCA $d=128, 256$ fit on train patients only).  
**Date**: 2026-09-30  
**Verdict**: READY  

---

## Critical Feedback & Refinement History

### Round 0 & 1: Problem Anchor & Set-Based Reference
- **Initial Framing**: "Personalized ECG deep neural network."
- **Critique**: Per-patient retraining or fine-tuning is clinically unfeasible and computationally prohibitive.
- **Refinement**: Personalize the *reference frame* ($r_p$) over frozen universal representations rather than personalizing neural network weights.

### Round 2: Calibration Burden as Central Constraint
- **Core Constraint**: Clinicians require models that work immediately ($K=0$) and opportunistically improve as sparse history accumulates.
- **Hypothesis**: Operational decomposition $z_t \approx s_p + \delta_t + \epsilon$.
- **Validation**: Completed empirical runs R000–R010 across PTB-XL, MIMIC-III, Kingston, SJLIFE, and EchoNext.

### Round 3 (Current): Scientific Nuance & Foundation-Model Benchmarking
- **Conceptual Correction on Identity**:
  - ECG is a known biometric modality; high patient verification AUROC (0.9412) does not by itself prove a "physiological coordinate manifold."
  - *Correction*: Adopt conservative terminology: *"the latent contains strong, longitudinally persistent subject-specific structure."*
- **Variance Partitioning Interpretation**:
  - Patient $R^2$ (87.7%) vs Disease $R^2$ (16.4%) is a descriptive variance decomposition across repeated records, not a measure of representational priority.
- **Temporal Stability**:
  - AUROCs across time strata (~0.92–0.94) establish that *"no monotonic degradation was detected across tested time bins,"* not "zero temporal decay."
- **Substrate Decomposition Nuances (Aim 3)**:
  - Do *not* claim $z_t = s_p + \delta_t + \epsilon$ is "mathematically exact" or that $\delta_t$ is "pure state perturbation."
  - Conduction defects and hypertrophy are *"relatively stable substrate-related attributes"* (which can fluctuate clinically), not "fixed anatomy."
  - State the empirical result: static attributes remain linearly decodable from $s_p$ and weakly decodable from residual $\delta_t$.
- **Statistical Wording**:
  - Replace "97.9% confidence" with *"the same-patient reference outperformed the matched wrong-patient reference in 97.9% of donor permutations."*
  - Replace "captures 100% of gain" with *"the largest observed gain occurred by $K=3-5$, with no additional improvement from using the full available history."*
- **Audit Fidelity Nuances**:
  - Aim 2 MIMIC: Clearly state that this was an event-level transition feasibility audit on 752 conversion events (376 SR $\to$ AF, 376 AF $\to$ SR), not full waveform feature validation.
  - Aim 5: Refer to task evaluations as *frozen-feature transfer*, reserving "zero-shot" strictly for downstream heads evaluated without training.
- **Foundation Model Benchmarking Requirement**:
  - Include CSFM (Gu et al., *Nature Machine Intelligence* 2026, 1.7M patients) and ECG-LFM (Nature Communications 2026, 10M ECGs) as first-class frozen comparators.
  - Dimensionality control is mandatory: evaluate both native embeddings and matched dimensionality (PCA $d=128, 256$ fit strictly on training splits).
  - Explicit pretraining overlap accounting: flag MIMIC-III and MIMIC-IV as `CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE` and evaluate clean generalization on Kingston, SJLIFE, LUDB, and EchoNext.

---

## Final Status & Dominant Contribution
- **Anchor**: Characterizing the calibration–utility trade-off across foundation representations.
- **Dominant Contribution**: Demonstrating whether patient-specific structure is unique to measurement-aware operators or emergent in large-scale cardiac foundation models, and evaluating which representation most efficiently converts sparse history into state-change detection.
- **Immediate Execution Priority**: Milestone **M0-FM** (Head-to-head PTB-XL frozen CSFM audit on 1,693 patients / 4,087 ECGs with PCA dimensionality matching).
