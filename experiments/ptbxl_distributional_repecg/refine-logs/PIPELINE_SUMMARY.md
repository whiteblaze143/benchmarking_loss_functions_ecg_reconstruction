# Pipeline Summary

**Problem**: Can an ECG representation preserve diagnostic information under a change in measurement operator?  
**Final Method Thesis**: Formulating ECG observations as continuous dual operator-response pairs $(q, s_q(t))$ with projective gauge symmetry $\mathbb{RP}^7$ provides measurement-invariant cardiac representations that retain diagnostic accuracy under combinatorial lead omission and novel measurement operators without post-hoc probing or retraining.  
**Final Verdict**: READY  
**Date**: September 21, 2026  

---

## 1. Final Deliverables
- **Proposal**: `refine-logs/FINAL_PROPOSAL.md`
- **Review Summary**: `refine-logs/REVIEW_SUMMARY.md`
- **Refinement Report**: `refine-logs/REFINEMENT_REPORT.md`
- **Experiment Plan**: `refine-logs/EXPERIMENT_PLAN.md`
- **Experiment Tracker**: `refine-logs/EXPERIMENT_TRACKER.md`
- **Evaluation Suite (Interpretation B)**: `scripts/evaluation/evaluate_ptbxl_configuration_shift.py`
- **Empirical Configuration-Shift Matrix**: `outputs/ptbxl_configuration_shift/configuration_shift_matrix.md`
- **Ablation Training Script**: `scripts/paper07/train_paper07_fulllead_only.py`

---

## 2. Contribution Snapshot
- **Dominant Contribution**: Continuous dual operator-response set encoder ($\mathbb{RP}^7$) retaining **95.0%** of diagnostic performance on 2 bipolar leads and **90.5%** on 1 smartwatch lead under strict Interpretation B evaluation (zero probes, frozen pre-trained diagnostic head).
- **Secondary Protocol**: Task-native training for external tasks (EchoNext SHD) followed by frozen zero-shot configuration-shift evaluation, strictly rejecting heuristic cross-task proxy mappings.
- **Explicitly Rejected Complexity**: Discrete torso mesh solvers, heuristic cross-dataset proxy label mappings, and post-hoc linear probe retraining.

---

## 3. Methodological Contract: Interpretation B
$$\boxed{\textbf{No parameter may be trained after the model has seen the evaluation configuration.}}$$

- **Primary Protocol (PTB-XL Tier 4)**:
  $$\text{Train on full leads} \rightarrow \text{Freeze encoder } \theta \text{ and diagnostic head } \phi \rightarrow \text{Evaluate under } O_S$$
- **Measured Diagnostic Retention on PTB-XL Fold 8**:
  - `P07_ROBUSTNESS_TRAINED`: Full $Q_8$ = **0.9309** | 2-Lead $S_2$ = **0.8849** ($R_2 = \mathbf{95.0\%}$) | 1-Lead $S_1$ = **0.8425** ($R_1 = \mathbf{90.5\%}$)
  - `P07_FULLLEAD_ONLY` (Pure Inductive Bias, Zero Augmentation): Full $Q_8$ = **0.8786** | 2-Lead $S_2$ = **0.8105** ($R_2 = \mathbf{92.2\%}$) | 1-Lead $S_1$ = **0.7550** ($R_1 = \mathbf{85.9\%}$)
  - `GraphECG` (Ansari et al.): Full $Q_8$ = 0.9127 | 2-Lead $S_2$ = 0.8628 ($R_2 = 94.5\%$) | 1-Lead $S_1$ = 0.7675 ($R_1 = 84.1\%$) | Novel $S_{\rm ICM}$ = **0.7197** ($R_{\rm ICM} = \mathbf{78.8\%}$)
  - `FixedTensor_P02`: Full $Q_8$ = 0.8716 | 2-Lead $S_2$ = 0.6596 ($R_2 = 75.7\%$) | 1-Lead $S_1$ = 0.6118 ($R_1 = 70.2\%$)
- **Combinatorial Single-Lead & Pair Sweeps**:
  - SetOperator outperforms FixedTensor across all 15 configurations by **+0.0960 to +0.2773 AUROC**.
- **Task-Native Decoupling across Clinical Datasets (Foundation Encoders)**:
  - Validated across LUDB (8 diagnostic categories), Zhejiang (RVOT vs LVOT origin), ISP (sex classification), and Kingston-ICU (rhythm detection).
  - GraphECG achieves **0.8528 AUROC** on LUDB (retaining 82.0% on smartwatch), **0.8803 AUROC** on Zhejiang (retaining 73.3% on smartwatch), and **0.7525 AUROC** on Kingston-ICU (retaining 92.7% on smartwatch).
  - SetOperator achieves **0.7051 AUROC** on Zhejiang (retaining 83.3% on smartwatch) and eliminates the catastrophic collapse seen in FixedTensor (which plummets from 0.9615 to **0.3996**, a 58.4% drop).
  - Demonstrates cross-domain structural invariance of continuous and geometric representations under task-native training with zero post-hoc probes.

---

## 4. Active Background Queue Status
1. `task_native_queue`: **COMPLETED**. Generated master matrix at `outputs/task_native_evaluation/task_native_decoupling_matrix.md`.
2. `p07_fulllead`: **COMPLETED** (Best Val AUROC `0.9162`, evaluated on Tier 4).
3. `graphecg_train`: **COMPLETED** (Fold 8 Test Macro AUROC `0.9264`, evaluated on Tier 4).
4. `ptbxl_shift_eval`: **COMPLETED** (Generated `outputs/configuration_shift_evaluations/ptbxl/configuration_shift_matrix.md`).
5. `ptbxl_sweeps`: **COMPLETED** (Generated `outputs/configuration_shift_evaluations/ptbxl/combinatorial_sweeps.md`).
6. `gpu_queue`: Running Paper 15 (`capacity_matched_shared`, final claims paper).
7. `download_echonext`: Wget mirror streaming to NFS `/data/mithunmanivannan/echonext/` at **~63%** (~2.5 / 4.1 GB).


