# Experiment Plan: T-CAIREM Continuous SetOperator AF-Density Research Program

**Problem**: Faithful AF episode reconstruction and continuous temporal density estimation from long-term multi-channel / multi-lead ambulatory Holter ECG recordings.  
**Method Thesis**: Continuous $q$-conditioned Temporal Phase-KME with ordered-beat dynamics, distributional morphology, and residual ResNet fusion breaks representation collapse, preserves lead-dropout invariance, and delivers superior episode boundary delineation and density accuracy.  
**Date**: 2026-11-22  

---

## 1. Claim Map

| Claim ID | Claim Statement | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|---|
| **C1** | Continuous $q$-conditioning and ordered-beat dynamics eliminate representation collapse. | Categorical set encoders produced degenerate constant scores ($SD < 10^{-7}$). | Latent token activation $SD > 0.10$ and validation prediction $SD > 0.25$ with near-zero training loss on balanced overfit benchmark. | Block 1, Block 2 |
| **C2** | Continuous SetOperator with residual ResNet fusion achieves superior episode reconstruction and density estimation. | Episode boundary errors distort Charitos continuous density, leading to diagnostic error. | Statistically significant reduction in Continuous Density MAE ($\ge 20\%$) and Boundary Error ($< 8$s) vs raw ResNet and observed-RR baselines under patient-clustered bootstrap. | Block 1, Block 3 |
| **C3** | The architecture is strictly invariant to lead dropout, permutation, and unobserved channels. | Ambulatory Holter monitoring frequently encounters electrode detachment and variable lead placement. | Swapping Lead I and Lead II yields logit difference $< 10^{-5}$; single-lead dropout shifts burden by $< 0.15\%$ points without performance collapse. | Block 4 |
| **C4** | Simplicity & Deletion: Each component (continuous $q$, ordered beats, distributional pooling, residual fusion) is strictly necessary. | Defends against reviewer skepticism regarding architectural overbuilding. | Deleting continuous $q$ causes collapse; deleting ordered beats degrades sensitivity by $> 15\%$; deleting residual fusion increases boundary error by $> 5$s. | Block 3 |
| **C5** | Model generalizes to external cohorts (SHDB, LTAF) without test-guided threshold retuning. | Proves robustness against demographic shift and different sampling rates (128 Hz / 200 Hz). | Maintained high specificity ($> 85\%$) and burden correlation ($r > 0.70$) on SHDB (98 records) and LTAF (84 records) using frozen thresholds. | Block 5 |

---

## 2. Paper Storyline
- **Main Paper Must Prove**:
  1. *Table 1 (Primary Benchmark)*: Patient-weighted comparison of Continuous SetOperator (A001/I082) vs Categorical SetOperator (A002), Matched Raw Mask-ResNet, and Observed-RR Baseline across Burden MAE, Continuous Density MAE, Binned Density MAE, Episode Sensitivity, PPV, and Boundary Error with 95% Patient-Clustered Bootstrap CIs.
  2. *Figure 2 (Representation Non-Collapse)*: Layer-wise token activation variance and score distribution across the 498,057 validation evaluations showing collapse in A002 and healthy dispersion in A001.
  3. *Table 2 (Component Ablation Matrix)*: Stepwise removal of continuous $q$ (A003), ordered beats (D037 ablation), distributional pooling (D032 ablation), and residual ResNet fusion (I082 ablation).
  4. *Table 3 (External Validation)*: Zero-retuning performance on SHDB (98 records, 93 subjects) and LTAF (84 records).
- **Appendix Can Support**:
  1. Full Manski partial-identification bounds $[Burden_{\min}, Burden_{\max}]$ across wear completeness thresholds (10% to 90%).
  2. Synthetic channel shutdown analysis (gradual lead detachment scenarios).
  3. Continuous vs binned Charitos density mathematical derivation and empirical discrepancy analysis.
- **Experiments Intentionally Cut**:
  - NO 16-channel synthetic diffusion reconstructions.
  - NO stroke risk / CHA2DS2-VASc predictive modeling.
  - NO heuristic uncalibrated HMM post-processing.

---

## 3. Experiment Blocks

### Block 1: Main Anchor Result — Episode Reconstruction & Density Benchmarking
- **Claim Tested**: C1, C2
- **Purpose**: Establish primary superiority on the locked IRIDIA-AF benchmark under frozen protocol.
- **Dataset / Split**: IRIDIA-AF (504,777 train / 166,047 val / 131,667 test 30s windows; 152 patients: 90 train / 30 val / 32 test). Primary evaluation on 11 informative multi-interval validation patients (15 segments) and 13 informative test patients.
- **Compared Systems**:
  1. *Proposed*: Continuous $q$-conditioned SetOperator with Ordered Beats, Distributional Morphology, and Residual ResNet Fusion (I082).
  2. *Baseline 1*: Matched Raw Mask-ResNet (8-block 1D CNN).
  3. *Baseline 2*: Matched Observed-RR HGB (32 rhythm features).
  4. *Baseline 3*: Categorical SetOperator (A002).
- **Metrics**: Charitos Continuous Density MAE, Charitos Binned Density MAE, Burden MAE, Episode Sensitivity, Episode PPV, Mean Boundary Onset/Offset Error (seconds), ECE, Brier Score.
- **Setup Details**: 3-epoch availability schedule cycling; batch size 256; AdamW lr=1e-4; BF16 mixed-precision; FP32 validation checkpoint selection; D2 HSMM temporal decoding with prospectively fitted censored lognormal priors.
- **Success Criterion**: Continuous Density MAE $\le 0.08$ (relative reduction $\ge 20\%$ over Raw ResNet); Boundary Error $\le 8.0$ seconds; Prediction score $SD \ge 0.30$.
- **Priority**: **MUST-RUN**

### Block 2: Representation Collapse Resolution & Novelty Isolation
- **Claim Tested**: C1, C4
- **Purpose**: Empirically prove that continuous $q$-conditioning breaks the token symmetry collapse that caused categorical set encoders to produce invariant scores ($\approx 0.2284$).
- **Dataset / Split**: Validation subset (256 both-channel-valid windows across 4 patients for detailed layer-wise probing; full 166,047 validation windows for population activation distribution). Overfit sanity split: 128 AF-positive and 128 AF-negative windows.
- **Compared Systems**:
  - A001: EchoNext continuous D1 SetOperator ($q \in \mathbb{R}^8 \to \text{MLP}_{64} \to \text{PhaseCNN} \to \text{Transformer}$).
  - A002: Matched categorical operator control ($ID \in \{0, 1\} \to \text{Embedding}_{64}$).
  - A003: Channel-agnostic operator-ablated control ($q = \mathbf{0}$).
  - A004: 2D continuous $q$ projection.
  - F053: No Transformer (Masked Mean Pooling baseline).
- **Metrics**: Token-level activation standard deviation at: (i) PhaseCNN output, (ii) Transformer Layer 1, (iii) Transformer Layer 2 active tokens, (iv) Attention pooling output, (v) Final prediction logits.
- **Success Criterion**: Active token activation $SD \ge 0.10$ for A001 (vs $< 5 \times 10^{-6}$ for A002 and A003); Overfit sanity reaches training BCE loss $< 0.05$ and AUROC $> 0.98$.
- **Priority**: **MUST-RUN**

### Block 3: Simplicity & Architectural Deletion Matrix
- **Claim Tested**: C4
- **Purpose**: Prove that each added mechanism is justified and that the architecture cannot be pruned without performance degradation.
- **Dataset / Split**: IRIDIA-AF Validation cohort (166,047 windows, 30 patients).
- **Compared Systems**:
  1. Full Model (I082: Continuous $q$ + Ordered Beats + Distributional Morphology + Residual ResNet).
  2. Ablation 1 (Remove Residual ResNet Fusion: SetOperator alone).
  3. Ablation 2 (Remove Ordered Beats D037: unordered beat token set).
  4. Ablation 3 (Remove Distributional Morphology D032: standard mean-only pooling).
  5. Ablation 4 (Remove Continuous $q$: A003).
  6. Ablation 5 (Replace Transformer with Masked Mean Pooling: F053).
- **Metrics**: Continuous Density MAE, Episode Boundary Error, AUROC, Patient-Weighted Brier Score.
- **Success Criterion**: Removing any single component causes a statistically significant degradation in at least one decisive metric (e.g., removing ResNet increases boundary error by $> 4$s; removing ordered beats increases density MAE by $> 0.03$).
- **Priority**: **MUST-RUN**

### Block 4: Multi-Lead Invariance, Permutation, and Shutdown Robustness
- **Claim Tested**: C3
- **Purpose**: Validate mathematical operator invariance and performance retention under lead dropout and channel detachment.
- **Dataset / Split**: IRIDIA-AF Validation cohort evaluated under: (i) Channel 0 only, (ii) Channel 1 only, (iii) Both channels, (iv) Permuted channels (swap $C_0$ and $C_1$), (v) 6 synthetic shutdown schedules (abrupt lead loss at 1h, 2h, 4h, 8h, 12h, 16h).
- **Metrics**: Logit symmetry error $|\hat{s}(C_0, C_1) - \hat{s}(C_1, C_0)|$, Burden divergence between single-lead and two-lead acquisition, Manski bounds coverage $[Burden_{\min}, Burden_{\max}]$.
- **Success Criterion**: Logit permutation difference $< 1 \times 10^{-5}$; single-lead vs two-lead burden difference $< 0.15\%$ points; 100% coverage of reference burden within Manski bounds.
- **Priority**: **MUST-RUN**

### Block 5: External Validation on Independent Clinical Cohorts
- **Claim Tested**: C5
- **Purpose**: Verify generalizability on independent hospital databases without any test-guided tuning or threshold adjustment.
- **Datasets**:
  1. *SHDB (St. Petersburg Institute of Cardiological Technics 12-lead Arrhythmia Database)*: 98 annotated records, 93 subjects, 279,686 native-clock 30s windows at 200 Hz; 794 AF intervals; 49 multi-interval subjects.
  2. *LTAF (Long-Term Atrial Fibrillation Database)*: 84 records, 232,233 native-clock windows at 128 Hz; 7,358 AF intervals; 66 multi-interval records.
- **Setup Details**: Frozen model weights, frozen classification threshold $\tau$ (from IRIDIA validation), frozen D2 HSMM duration parameters. Lead projections mapped via canonical basis $q \in \mathbb{R}^8$.
- **Metrics**: Patient Specificity (on zero-AF records), False Burden (%), Episode Sensitivity, Episode PPV, Burden Pearson $r$ and Spearman $\rho$.
- **Success Criterion**: Patient Specificity $\ge 85.0\%$ on negative controls; False Burden $\le 1.5\%$; Burden Pearson $r \ge 0.70$ on multi-interval patients.
- **Priority**: **MUST-RUN**

---

## 4. Run Order and Milestones

| Milestone | Stage Name | Target Runs | Decision Gate / Success Bar | Est. Cost | Risk & Mitigation |
|---|---|---|---|---|---|
| **M0** | Preflight & Integrity Sanity | R001-R003 | Test stage 0 integrity passes; tensor shapes `(B, 2, 16, 128)` verified; zero NaNs in gradients. | 0.5 GPU-h | Numerical precision overflow in BF16 $\to$ use FP32 for kernel whitening and validation. |
| **M1** | Representation Non-Collapse | R004-R007 | Overfit test converges (BCE $< 0.05$, AUROC $> 0.98$); A001 active token $SD > 0.10$ vs A002 $< 10^{-6}$. | 1.0 GPU-h | If A001 collapses, increase operator MLP capacity or adjust GeLU gain. |
| **M2** | Master Model Training | R008-R012 | Complete 3-epoch multi-schedule training for A001, A002, D032, D037, I082. Select checkpoints via strict min validation BCE. | 15.0 GPU-h | GPU memory limit $\to$ verified batch size 256 fits in 31.2 GiB. |
| **M3** | Temporal Decoding & Benchmarking | R013-R017 | Run D0, D1, D2 HSMM across 60 validation files. Compute patient-clustered bootstrap intervals ($B=2,000$). | 3.5 GPU-h | Censoring edge cases in HSMM $\to$ use verified stationary residual-life law. |
| **M4** | External Evaluation & Governance | R018-R020 | Execute zero-retuning inference on SHDB (98 records) and LTAF (84 records). Produce final comparative tables. | 4.0 GPU-h | Sampling rate mismatch (LTAF 128 Hz) $\to$ PCHIP phase normalization resamples to 256 points per beat, making it sample-rate agnostic. |

---

## 5. Compute and Data Budget
- **Total Compute**: 24 GPU-hours on 1x NVIDIA A100-SXM4-80GB.
- **Disk Storage**:
  - Phase-KME Feature Cache: ~45 GB (802,491 windows $\times$ 2 channels $\times$ 16 cells $\times$ 128 float32 responses).
  - Model Checkpoints: ~1.2 GB (5 models $\times$ 2 checkpoints $\times$ 120 MB).
  - Timeline Predictions: ~3.5 GB.
- **Data Integrity**: SHA-256 verification of all raw waveform shards and input contracts; strictly zero held-out test labels loaded into memory during forward inference.

---

## 6. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| **Transformer Latent Underflow in BF16** | Low | High | Use FP32 for attention softmax and layer-norm operations; validate gradient norms every 100 steps. |
| **Over-smoothing in Long AF Episodes by HSMM** | Medium | Medium | Include D1 median filter as a direct benchmark; verify that duration PMF support extends to 30 days. |
| **Electrode Misalignment on Wearables** | High | Medium | Continuous $q$ embedding allows continuous interpolation across non-standard vector projections. |
| **Low Statistical Power on Held-Out Test Set** | Medium | High | Patient-clustered bootstrap ($B=2,000$) prevents overconfident p-values; primary external validation on SHDB provides 49 additional multi-interval subjects. |

---

## 7. Final Checklist
- [x] Main paper tables and figures are covered (Table 1, Table 2, Table 3, Figure 2).
- [x] Representation collapse resolution is rigorously isolated in Block 2.
- [x] Architectural simplicity is defended via deletion matrix in Block 3.
- [x] Parameter target is strictly met: 1,027,009 trainable parameters.
- [x] External validation cohorts (SHDB, LTAF) are evaluated without test-guided retuning.
- [x] Nice-to-have experiments are cleanly separated from must-run milestones.
