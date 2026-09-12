# R3 Linear Accessibility Audit Plan

**Problem**: Select a leakage-safe frozen GRAIL reference latent before any 4,095-subset analysis.
**Method thesis**: Frozen representations must be compared with identical patient-aware probes whose tuning never touches Fold 8.
**Date**: 2026-09-11

## Planning gate

- Dominant contribution: representation accessibility and geometry, not further encoder training.
- Frozen models: UB128, B0, B1, 001, 101, 111.
- Rejected complexity: encoder retraining and premature subset-lattice evaluation.
- Primary claim: 101 provides complementary transferable information beyond B1.
- Anti-claim: apparent gains arise only from leakage, capacity, rotations, or patient duplication.

## Must-run execution

1. Export folds 1–8 with IDs, folds, labels, demographics, and slot tensors.
2. Select logistic-probe regularization on folds 1–6 versus Fold 7; refit on 1–7; evaluate Fold 8 once.
3. Run per-concept accessibility, 20-repeat shared-patient low-shot probing, and train-fitted PCA dimensionality.
4. Follow with residual-slot, cross-representation geometry, residual-subspace, patient-excluded retrieval, and paired patient bootstrap audits.

## Compute

- A100: frozen embedding extraction, batch 2,048; witnessed 21–25 GiB and 100% compute utilization.
- CPU: deterministic scikit-learn LBFGS probes after embedding export.
- Stop gate: any checkpoint mismatch, record-order mismatch, Fold-8 fitting, or unconverged probe is invalid.

