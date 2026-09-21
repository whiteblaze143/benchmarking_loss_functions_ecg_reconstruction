# Experiment Tracker: Acquisition-Configuration Shift Benchmark

**Status Date**: September 21, 2026  
**Master Queue**: `experiments/ptbxl_distributional_repecg/scripts/run_lean_queue.sh` (Active in `gpu_queue` tmux session)

---

## 1. Stage-by-Stage Execution Tracker

| Stage | Target / Description | Status | Key Checkpoint / Result |
| :--- | :--- | :--- | :--- |
| **Stage 0** | Historical Fold 8 Evaluations (P01, P02, P03, P05, P06, P07, P14) | **COMPLETED** | P07 `continuous_primary` (**0.9230**), P02 `moments` (**0.9183**), P03 `unordered_kme` (**0.8899**) |
| **Stage 1** | Paper 02 `no_circular` Baseline Variant | **COMPLETED** | Macro AUROC: **0.8762** (all 6 P02 variants fully trained) |
| **Stage 2** | Paper 08 Full Capped Factorial (Routing $\times$ Representation) | **IN PROGRESS** | 55/56 cells done, completing `continuous__uniform_attention` |
| **Stage 3** | Claims Papers (P10, P11, P13, P15) | **QUEUED** | Scripts validated; runs sequentially after Stage 2 |
| **Stage 4** | Post-Queue Fold 8 Standardized Evaluations | **QUEUED** | Automated evaluation generating standardized fold8 JSONs |
| **Stage 5** | **GraphECG PTB-XL Full Benchmark Training** | **WIRED & QUEUED** | Script: `scripts/graphecg/run_graphecg.sh` (Dry-run verified on GPU) |
| **Stage 6** | **Tier 4 Acquisition-Configuration Shift Battery** | **WIRED & QUEUED** | Script: `scripts/evaluation/evaluate_tier4_shift.py` (Clinical + 255 Combinatorial Subsets) |

---

## 2. Model Evaluation Checklist for Tier 4 Shift Suite

- [x] **GraphECG Integration**:
  - [x] Dependencies installed (`torch_geometric`, `xxhash`).
  - [x] Author code wrapped in `src/repecg/graphecg/`.
  - [x] Forward/backward GPU test passed without cuDNN bugs (`ptrDesc->finalize()` averted).
  - [x] Full PTB-XL training pipeline implemented (`train_graphecg_ptbxl.py`).
  - [ ] Full training converged on PTB-XL (Stage 5 in queue).
  - [ ] Frozen Fold 8 checkpoint evaluated on Tier 4 (Stage 6 in queue).

- [ ] **SetOperator (Paper 07 `continuous_primary_best.pt`, AUROC 0.9230)**:
  - [x] Converged checkpoint available at `outputs/paper07_operator_reconstruction/continuous_primary_best.pt`.
  - [ ] Tier 4A clinical subsets evaluated.
  - [ ] Tier 4B 255 combinatorial subsets evaluated.
  - [ ] Tier 4C 100 continuous dual operators evaluated.

- [ ] **Moments Model (Paper 02 `moments_best.pt`, AUROC 0.9183)**:
  - [x] Converged checkpoint available at `outputs/paper02_kernel_mean/development_training/moments_best.pt`.
  - [ ] Tier 4A clinical subsets evaluated via marginal submatrix extraction.
  - [ ] Tier 4B 255 combinatorial subsets evaluated.

- [ ] **FixedTensor Baseline ($Z_0$ Imputation)**:
  - [ ] Evaluated under standard zero-imputation baseline across all Tier 4 subsets.

---

## 3. Tier 4 Measured Metrics (Interpretation B: Zero Probes, Pre-Trained Diagnostic Head)

| Model | Full $Q_8$ (8 Leads) | $S_6$ (Precordial) | $S_3$ (ICU Telemetry) | $S_2$ (Bipolar I, II) | $S_1$ (Smartwatch I) | $S_1$ (Lead II) | Retention $R_2$ | Retention $R_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`FixedTensor_P02` ($Z_0$)** | 0.8716 | 0.8017 | 0.6987 | 0.6596 | 0.6118 | 0.6700 | 75.7% | 70.2% |
| **`GraphECG` (Ansari et al.)** | 0.9127 | 0.8783 | 0.8737 | 0.8628 | 0.7675 | 0.8248 | 94.5% | 84.1% |
| **`P07_ROBUSTNESS_TRAINED`** | 0.9309 | 0.9167 | 0.9028 | 0.8849 | 0.8425 | 0.8412 | **95.0%** | **90.5%** |
| **`P07_ROBUSTNESS_TRAINED_AUX`** | **0.9322** | **0.9178** | **0.9058** | **0.8831** | **0.8420** | **0.8418** | **94.7%** | **90.3%** |
| **`P07_FULLLEAD_ONLY`** | *Queued* | *Queued* | *Queued* | *Queued* | *Queued* | *Queued* | *Ablation arm* | *Ablation arm* |
