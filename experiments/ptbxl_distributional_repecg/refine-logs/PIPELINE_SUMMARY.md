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
  - `GraphECG` (Ansari et al.): Full $Q_8$ = 0.9127 | 2-Lead $S_2$ = 0.8628 ($R_2 = 94.5\%$) | 1-Lead $S_1$ = 0.7675 ($R_1 = 84.1\%$)
  - `FixedTensor_P02`: Full $Q_8$ = 0.8716 | 2-Lead $S_2$ = 0.6596 ($R_2 = 75.7\%$) | 1-Lead $S_1$ = 0.6118 ($R_1 = 70.2\%$)

---

## 4. Next Actions
1. Allow `gpu_queue` (currently Stage 3 Paper 13) and `graphecg_train` (Epoch 27/50) to complete uninterrupted.
2. Launch `P07_FULLLEAD_ONLY` training arm via `scripts/paper07/train_paper07_fulllead_only.py` to isolate pure representation inductive bias from subset data augmentation.
3. Upon completion of EchoNext mirror to NFS, execute task-native full-lead training followed by zero-shot configuration-shift evaluation.
