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

| Model | Full $Q_8$ (8 Leads) | $S_6$ (Precordial) | $S_6$ (Limb) | $S_3$ (ICU V1) | $S_3$ (ICU V5) | $S_2$ (Bipolar I, II) | $S_1$ (Smartwatch I) | $S_1$ (Lead II) | $S_{\rm ICM}$ ($V_3-V_2$) | Retention $R_2$ | Retention $R_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`FixedTensor_P02` ($Z_0$)** | 0.8716 | 0.8017 | 0.6596 | 0.6987 | 0.6848 | 0.6596 | 0.6118 | 0.6700 | 0.6654 | 75.7% | 70.2% |
| **`GraphECG` (Ansari et al.)** | 0.9127 | 0.8783 | 0.8800 | 0.8737 | 0.8719 | 0.8628 | 0.7675 | 0.8248 | **0.7197** | 94.5% | 84.1% |
| **`P07_FULLLEAD_ONLY`** | **0.8786** | 0.8578 | 0.7710 | 0.8484 | 0.8463 | 0.8105 | 0.7550 | 0.7812 | 0.5152 | **92.2%** | **85.9%** |
| **`P07_ROBUSTNESS_TRAINED`** | **0.9309** | 0.9167 | 0.8105 | 0.9028 | 0.9056 | **0.8849** | **0.8425** | **0.8412** | 0.4423 | **95.0%** | **90.5%** |
| **`P07_ROBUSTNESS_TRAINED_AUX`** | **0.9322** | **0.9178** | 0.7633 | **0.9058** | **0.9050** | 0.8831 | 0.8420 | **0.8418** | 0.4379 | 94.7% | 90.3% |

---

## 4. Task-Native Decoupling Tracker (External Clinical Tasks - Foundation Encoders)

**Protocol (Interpretation B)**: Foundation pre-trained encoders ($\theta$) frozen; task-native head $\phi_{\text{task}}$ trained strictly on full leads ($m=8/12$); evaluated across configuration shift battery ($Q_8 \to S_6 \to S_3 \to S_2 \to S_1 \to S_{\rm ICM}$) with **zero target configuration probes or gradient updates**.

### Key Highlights Across 14 Models:
- **LUDB (8 Diagnostic Categories)**: `graphecg` achieves **0.8161** full-lead AUROC, retaining **94.8%** on $S_2$ and **92.4%** on $S_1$. `paper14` achieves **0.6201** full-lead AUROC and retains **96.3%** on $S_1$.
- **Zhejiang (RVOT vs LVOT Arrhythmia Origin)**: Reveals severe degradation for fixed-lead encoders under lead loss:
  - `fixed_tensor`: drops from **0.9637** to **0.4487 (46.6% retention)** on single smartwatch lead $S_1$.
  - `moments`: drops from **0.9444** to **0.5000 (52.9% retention)**.
  - `paper14`: drops from **0.8483** to **0.4530 (53.4% retention)**.
  - `paper03`: drops from **0.7842** to **0.4380 (55.9% retention)**.
  - In contrast, `graphecg` retains **98.9% (0.7457 AUROC)** and `set_operator_robust` retains **93.0% (0.5662 AUROC)** on $S_1$.
- **Kingston-ICU (AFIB/AFLT Telemetry Rhythm)**: `graphecg` achieves **0.7973** AUROC, retaining **94.5%** on $S_2$ and **91.7%** on $S_1$.
- **EchoNext (12 SHD phenotypes, 5,442 records)**: Download progressing via persistent wget stream on NFS (~67%).

Full results matrix available at [`outputs/task_native_evaluation/task_native_decoupling_matrix.md`](outputs/task_native_evaluation/task_native_decoupling_matrix.md) and `.json`.


