# Refinement Report: Calibration-Efficient Patient-Specific Physiological Foundation Models

**Problem**: Clinicians interpret ECGs against a patient's personal history, but current AI evaluates against population distributions. Personalization fails in practice due to calibration data bottlenecks.  
**Initial Approach**: Train a "personalized model" per patient.  
**Refined Approach**: Characterize the calibration–utility frontier across universal representations, benchmarking SetOperator against Cardiac Sensing Foundation Model (CSFM) under identical calibration machinery.  
**Date**: 2026-09-30  
**Final Verdict**: READY  

---

## Problem Anchor & Boundary Conditions
- **Bottom-Line Problem**: Standard AI evaluates whether an ECG is abnormal in the general population. Clinical decision-making evaluates whether an ECG represents a meaningful deviation from an individual patient's baseline.
- **Must-Solve Bottleneck**: Calibration efficiency. Real-world clinical workflows provide sparse, irregularly sampled historical ECGs ($K \in \{0, 1, 2, 3, 5\}$). Models must function at zero calibration and scale gracefully as sparse history accumulates.
- **Non-Goals**: No per-patient parameter fine-tuning, no continuous backpropagation, no ungrounded claims of "discovering a unique physiological coordinate system."
- **Success Criteria**: Mapping the empirical calibration gain $G(K) = \text{Perf}(K) - \text{Perf}(0)$ across foundation representations and testing whether patient-relative geometry is unique to SetOperator or shared with large-scale self-supervised models (CSFM).

---

## Method Evolution & Review History

| Round | Reviewer / Supervisor Feedback | Refinement Actions Taken | Resolution Status |
| :--- | :--- | :--- | :--- |
| **0** | "Personalized model per patient is unscalable and unviable clinically." | Shifted from per-patient parameter retraining to personalizing the *reference frame* ($r_p$) over frozen foundation latents. | Resolved |
| **1** | "Dense sequential history (RNN/Transformer) fails because clinical data is sparse and unordered." | Introduced a permutation-invariant SetOperator formulation ($r_p = \text{SetOperator}(H_p)$) treating history as an unordered set of measurements. | Resolved |
| **2** | "Formulate the project around the calibration bottleneck." | Designed calibration scaling experiments ($K \in \{0, 1, 3, 5, 10\}$) testing the operational decomposition $z_t \approx s_p + \delta_t + \epsilon$. | Resolved |
| **3** | **Critical Supervisor Review**: (1) Do not overclaim a "unique physiological coordinate manifold" when ECG biometric verification is a known phenomenon; (2) Tonally correct variance ratio (descriptive decomposition, not representational priority); (3) Tonally correct Aim 3 (substrate decodability, not mathematically exact decomposition or pure perturbation); (4) Fix statistical phrasing (97.9% of donor permutations, not 97.9% confidence); (5) Distinguish event-level metadata audit from waveform encoding on MIMIC; (6) **Mandate CSFM and foundation model comparisons** (M0-FM, M1-C, M1-S) with rigorous training-only PCA dimensionality matching. | (1) Revised thesis to focus on calibration–utility frontier; (2) Corrected all reporting claims to empirical precision; (3) Added CSFM as a first-class frozen encoder comparator; (4) Formalized PCA dimensionality matching ($d=128, 256$ fit on train patients only); (5) Established strict pretraining overlap ledger (`CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE` for MIMIC). | **FULLY RESOLVED** |

---

## Method Architecture Snapshot
1. **Backbones**:
   - SetOperator E6: Continuous measurement operator with phase-CNN responses ($z_{\text{SO}} \in \mathbb{R}^{256}$).
   - CSFM Base: Multi-modal Transformer foundation model pretrained on 1.7M patients across MIMIC-III, MIMIC-IV, and CODE-FULL ($z_{\text{CSFM}} \in \mathbb{R}^{768}$).
   - ECG-LFM: 10M-patient contrastive foundation model.
2. **Harmonized Dimensionality**:
   - Native: $d=256$ vs $d=768$.
   - Dimension-matched: PCA $d=128$ and $d=256$ fit strictly on training splits.
3. **Reference Construction Schemes**:
   - Mean baseline: $b_{\text{mean}}^{(K)} = \frac{1}{K}\sum_{i=1}^K z_{p, t-i}$
   - Learned SetOperator: $b_{\text{set}}^{(K)} = \rho\left(\sum_i \phi([z_i, \Delta t_i, q_i])\right)$
4. **State-Change Probing**:
   - Cold start ($K=0$): $X_0 = z_t$
   - Personalized reference: $X_K = [z_t, z_t - b_p^{(K)}]$

---

## Pretraining Domain Overlap Accounting
- **MIMIC-III and MIMIC-IV**: Marked `CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE` because CSFM was pretrained on MIMIC-III-WDB and MIMIC-IV-ECG. Used strictly for longitudinal mechanics, not clean external generalization.
- **Clean External Cohorts**: Kingston ICU AF, SJLIFE, LUDB, EchoNext, and Chapman/Shaoxing. Overlap is verified as `FALSE` for both encoders.

---

## Immediate Next Steps
- Execute Milestone **M0-FM**: Run the completed M0 patient-clustering audit using frozen CSFM Base on the exact same 1,693-patient / 4,087-ECG PTB-XL cohort, reporting native and dimension-matched verification AUROC, retrieval ranks, and variance partitioning.
