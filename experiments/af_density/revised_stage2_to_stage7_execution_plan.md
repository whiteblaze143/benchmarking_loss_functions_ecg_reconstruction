# Revised Stage 2 to Stage 7 Execution Plan
## T-CAIREM Continuous SetOperator AF-Density Research Program

**Document Version**: 2.0 (Post-Stage 1 Critical Review Master Roadmap)  
**Date**: 2026-11-22  
**Author**: Statistical Governance & Refinement Planner (@planner)  
**Target Repository**: `experiments/af_density/`  
**Execution Environment**: NVIDIA A100-SXM4-80GB GPU, persistent tmux session `hrs2027_gpu_12h`  
**Storage Destination**: `revised_stage2_to_stage7_execution_plan.md`, `refine-logs/revised_stage2_to_stage7_execution_plan.md`, and `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/revised_stage2_to_stage7_execution_plan.md`  

---

## 1. Master Pipeline Architecture & Dependency DAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 1B: Corrective Replication on Actual Data               │
│  (10 models on patient-balanced cohort; verified test isolation; tmux session)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Passing Stage 1B Gate)
┌──────────────────────────────────────▼──────────────────────────────────────┐
│       STAGE 2: Full-Cohort Window Extraction & Calibration Validation       │
│    (Complete 166,047 val windows + 98 SHDB records; temperature calibration)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│        STAGE 3: Specificity & Negative Control Audit (False Alarms)         │
│     (34 SHDB zero-AF controls + IRIDIA non-AF files; target Spec ≥ 85%)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│      STAGE 4: Temporal Decoding Hierarchy & Duration HSMM Ablation          │
│       (D0 Thresholding vs D1 Median vs D2 HSMM on informative cohort)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│       STAGE 5: Multi-Lead Invariance, Rotation, and Dropout Audit           │
│     (Joint operator-shift protocol, lead dropout, 6 shutdown schedules)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│       STAGE 6: Wearable Completeness & Manski Burden Identification         │
│     (10%–90% wear time, 2x30s spot-check simulation, [Burden_min, max])     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│      STAGE 7: Submission Manuscript & Artifact Synthesis (T-CAIREM)         │
│    (Publication-ready tables, Patient-Clustered Bootstrap CIs, 6-page PDF)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage-by-Stage Detailed Protocols

### Stage 2: Full-Cohort Window Extraction & Calibration Validation
- **Objective**: Generate complete timeline window predictions across all 166,047 validation windows (30 patients, 60 daily files) and the 98 external annotated records in SHDB (279,686 windows), followed by temperature scaling and Platt calibration.
- **Inputs**:
  - Sealed checkpoints from Stage 1B (`checkpoints/best_*.pt`).
  - Feature caches: `val_phase_features.pt` and `shdb_phase_cache/`.
  - Raw ECG tensors: `/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5/`.
- **Key Tasks**:
  1. Forward pass across all 10 models under three acquisition schedules (Channel 0, Channel 1, Both).
  2. Fit univariate temperature parameter $T_{\text{cal}} > 0$ on development validation windows via negative log-likelihood minimization:
     $$p_t = \sigma(s_t / T_{\text{cal}})$$
  3. Validate post-calibration Expected Calibration Error (ECE, 10 bins) $\le 0.18$ and Brier Score $\le 0.15$.
  4. Seal prediction arrays in `annotated_window_preds_stage2.pt` with SHA-256 integrity receipt.
- **Stopping Gate**:
  - If any model produces prediction score standard deviation $< 0.10$, halt immediately for representation collapse.
  - Test access lock: strictly zero held-out test windows evaluated.

---

### Stage 3: Specificity Audit on Negative Controls
- **Objective**: Quantify false alarm rates, false burden generation, and patient-level specificity on subjects with confirmed zero atrial fibrillation.
- **Cohorts**:
  - 34 SHDB negative control subjects with zero reference AF intervals.
  - IRIDIA validation recordings exhibiting zero documented AF.
- **Key Tasks**:
  1. Run timeline decoding on negative controls using frozen thresholds $\tau$ and frozen HSMM decoders.
  2. Compute Patient Specificity: proportion of subjects with 0 false episodes detected.
  3. Compute False Burden (%): cumulative false AF duration divided by total monitoring time.
  4. Compute False Episode Count and mean duration of false episodes.
- **Quantitative Target Bars**:
  - Patient Specificity $\ge 85.0\%$.
  - Mean False Burden $\le 1.35\%$ across negative control cohorts.
  - Mean false episode duration $\le 90$ seconds (short transient artifacts filtered by D1/D2).

---

### Stage 4: Temporal Decoding Hierarchy & Duration HSMM Ablation
- **Objective**: Benchmark the temporal decoding hierarchy (D0 raw thresholding, D1 median filtering with gap merge, D2 Viterbi HSMM with censored lognormal priors) across all models.
- **Population**: Primary Informative Multi-Interval Cohort (11 IRIDIA validation patients / 15 segments; 49 SHDB subjects; 66 LTAF records).
- **Key Tasks**:
  1. **D0 Decoding**: Raw thresholding at fixed policy threshold $\tau \in [0.01, 0.99]$.
  2. **D1 Decoding**: Odd-width temporal median filter with 60s non-AF gap merge within observed blocks.
  3. **D2 HSMM Decoding**: Numerical Viterbi decoding with prospectively fitted censored lognormal duration PMFs:
     - Initial run: stationary residual-life distribution.
     - Final run: survival distribution.
     - Likelihood: prior-corrected emission probabilities $P(Y_t \mid S_t) \propto P(S_t \mid Y_t) / P(S_t)$.
  4. Evaluate decisive metrics:
     - Charitos Continuous Density MAE (analytic envelope integration).
     - Charitos Binned Density MAE (echaritos/AF-density R v3 verified).
     - Continuous AF Burden MAE.
     - Isolated 1-to-1 Episode Boundary Error (onset/offset error in seconds).
- **Quantitative Target Bars**:
  - D2 HSMM reduces Continuous Density MAE by $\ge 20\%$ relative to D0.
  - Mean boundary onset/offset error $\le 8.0$ seconds for I082-v2.
  - Episode Sensitivity $\ge 88.0\%$ with PPV $\ge 80.0\%$.

---

### Stage 5: Multi-Lead Invariance, Permutation, and Shutdown Audit
- **Objective**: Validate continuous geometric operator theory under lead rotation, channel permutation, and gradual lead detachment.
- **Key Tasks**:
  1. **Joint Rotation Audit**: Execute `operator_shift_protocol_v2.md` over the 16-angle grid $\Theta \in [0, 2\pi)$:
     - Condition 1: Joint signal rotation $y_\theta(t) = \cos\theta x_{\text{I}} + \sin\theta x_{\text{II}}$ and query $q_\theta = [\cos\theta, \sin\theta, \dots]$.
     - Condition 2: Signal-only rotation (uninformed).
     - Condition 3: Query-only rotation (perturbed).
     - Compute Angular Divergence $\bar{\Delta}$ and Operator Sensitivity Ratio (OSR).
  2. **Permutation Symmetry**: Verify logit discrepancy $|\hat{s}(C_1, C_0) - \hat{s}(C_0, C_1)| \le 1.0 \times 10^{-5}$.
  3. **Lead Dropout Robustness**: Compare single-lead (Channel 0 alone vs Channel 1 alone) vs dual-lead acquisition. Burden spread must be $< 0.15\%$ points.
  4. **Synthetic Shutdown Schedules**: Evaluate 6 channel shutdown schedules (lead loss at 1h, 2h, 4h, 8h, 12h, 16h), verifying graceful fallback without non-finite outputs.

---

### Stage 6: Wearable Completeness & Manski Burden Identification
- **Objective**: Establish clinical validity under partial wear time and missing data, comparing continuous monitoring against intermittent spot-checking.
- **Key Tasks**:
  1. **Manski Worst-Case Bounds**: Compute $[Burden_{\min}, Burden_{\max}]$ across wear completeness thresholds from 10% to 90% (in 10% increments):
     $$Burden_{\min} = \frac{T_{\text{AF, obs}}}{T_{\text{total}}}, \quad Burden_{\max} = \frac{T_{\text{AF, obs}} + T_{\text{unobs}}}{T_{\text{total}}}$$
     Verify 100% empirical coverage of true reference burden.
  2. **Wear Completeness Threshold**: Identify minimum monitoring completeness required to bound burden uncertainty within $\pm 2.5\%$ (nominal threshold: 60% wear time).
  3. **Intermittent Spot-Check Audit**: Simulate standard wearable spot-checking (two 30-second resting checks per day). Measure episode miss rate for episodes $\ge 6$ minutes and indeterminate rate.
- **Expected Outcome**: Intermittent spot-checks miss $> 95\%$ of paroxysmal episodes with $> 98\%$ indeterminate density; continuous SetOperator achieves $< 5\%$ miss rate.

---

### Stage 7: Manuscript & Final Report Synthesis
- **Objective**: Compile the publication-ready, fully governed research paper and audit ledger for submission to T-CAIREM / HRS 2027.
- **Key Deliverables**:
  1. **Table 1 (Master Benchmark)**: All 10 models evaluated on IRIDIA, SHDB, and LTAF across Burden MAE, Continuous Density MAE, Binned Density MAE, Episode Se/PPV, and Boundary Error with 95% Patient-Clustered Bootstrap CIs ($B=2,000$).
  2. **Table 2 (Component Ablation Matrix)**: Stepwise removal of continuous $q$ (A003), ordered beats (D037), distributional pooling (D032), and residual ResNet fusion (I082).
  3. **Figure 1 (Architecture & Rotation Geometry)**: Continuous operator projection and joint equivariance polar plot.
  4. **Figure 2 (Representation Non-Collapse)**: Layer-wise token activation variance comparison (A001 vs A002).
  5. **Figure 3 (Continuous Density & Manski Bounds)**: Shortest-window envelope integration under varying wear completeness.
  6. **LaTeX Manuscript Compilation**: Automated build of `manuscript/main.pdf` via `pdflatex` / `latexmk`.

---

## 3. GPU Resource Allocation & Execution Timeline

| Stage | Estimated Duration | GPU Memory (Peak) | Process Manager / Command |
|---|---|---|---|
| **Stage 1B** | 1.8 hours | 31.2 GiB | `tmux send-keys -t stage1b_corrective_gpu "python run_stage1b_corrective_queue.py" Enter` |
| **Stage 2** | 0.8 hours | 24.5 GiB | Chained in `hrs2027_gpu_12h`: `python extract_and_calibrate_stage2.py` |
| **Stage 3** | 0.4 hours | 18.2 GiB | Chained: `python audit_negative_controls_stage3.py` |
| **Stage 4** | 1.2 hours | 26.0 GiB | Chained: `python run_temporal_hsmm_stage4.py` |
| **Stage 5** | 0.7 hours | 22.0 GiB | Chained: `python audit_operator_shift.py` |
| **Stage 6** | 0.5 hours | 16.5 GiB | Chained: `python evaluate_wearable_completeness_stage6.py` |
| **Stage 7** | 0.3 hours | CPU only | Chained: `bash build_manuscript_stage7.sh` |
| **Total** | **5.7 hours** | 31.2 GiB | Fully autonomous, chained execution in tmux session |

---

## 4. Quality Assurance, Incident Handling, and Stopping Rules

- **Stopping Rule 1 (Score Collapse)**: Any model exhibiting prediction score $SD < 1.0 \times 10^{-4}$ on valid inputs is automatically flagged as collapsed and excluded from downstream temporal decoding.
- **Stopping Rule 2 (Non-Finite Outputs)**: Any `NaN` or `Inf` loss or gradient during inference triggers immediate process termination and logs a core dump.
- **Stopping Rule 3 (Data Leakage)**: If any process opens `test_windows.npz` or reads held-out patient IDs without explicit authorization, the execution lock is broken and the run is invalidated.
- **Audit Verification**: Every stage emits a cryptographically signed JSON receipt (SHA-256 of inputs, code, model weights, and outputs) into `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/`.
