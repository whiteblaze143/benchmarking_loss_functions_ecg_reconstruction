# Experiment Plan: Predictable Residual Subspace Completion

**Problem**: Separate low-rank representation capacity from latent predictability in Lead-I to seven-independent-lead ECG reconstruction.
**Method thesis**: A tiny orthonormal residual output subspace learned jointly with the predictor may align better with information observable from Lead I than direct regression or frozen PCA.
**Date**: 2026-09-10

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1: A compact residual manifold improves over direct regression. | C1 or C2 minus B1 >=0.005 mean patient independent r, paired 95% CI low >0, no harm failure. | B1, B2 |
| C2: Predictability-aware orientation matters beyond PCA variance. | C2 materially exceeds C1 at identical rank/training, with subspace-angle evidence and no harm. | B2, B3 |
| Anti-claim: gains come from oracle access, unequal optimization, or test selection. | No latent supervision; identical contract; rank fixed before training; external tests sealed. | B0-B3 |

## Paper storyline

- Main paper: independently freeze rank, then distinguish B1/C1/C2 on fold 9.
- Appendix: full rank metrics, dependent-limb QC sensitivity, latent predictability, and principal angles.
- Cut: neural VCG, loss/architecture/rank sweeps after training, probabilistic completion, ECGFounder, fold 10, EchoNext, coordinate-physiology claims.

## Experiment blocks

### B0 — Closed-form rank gate

- Dataset: folds 1-8 fit; fold 9 oracle; folds 10/external sealed.
- Systems: uncentered residual PCA ranks 1-6 only.
- Primary: training residual energy, mean/p05 patient independent r, independent MSE.
- Secondary: per-lead R2, P/QRS/T IoU, SemiSeg mIoU, QRS MAE, Sokolow-Lyon MAE, average and per-precordial variance retention.
- QC: retain ECGs 5930/15867 for independent endpoints. Flag source max-absolute limb residual >0.01 mV; report dependent-limb metrics full and excluding flagged ECGs.
- Selection: smallest k with energy >=0.90 and G_(k+1)<0.005; otherwise smallest k reaching 0.90.
- Output: immutable residual_rank_curve and K_STAR_manifest artifacts.
- Priority: MUST-RUN before neural code.

### B1 — Equality and sanity

- Systems: B1 direct seven output, C1 fixed-PCA K_STAR, C2 learned QR-orthonormal K_STAR.
- Contract: same Lead I, T_patch10 trunk, shared trunk tensor initialization/hash, records/order, augmentation, preprocessing, optimizer/scheduler, balanced objective, clipping, seed 42, 15 epochs.
- Head initialization: same frozen initializer/seed; shape necessarily differs. C2 basis initializes from PCA.
- Tests: observed-I passthrough, shapes, frozen C1 gradients absent, C2 Q-transpose-Q equals identity and gradients present, derived limbs, one-batch overfit, checkpoint metric.
- Priority: MUST-RUN.

### B2 — Seed-42 matched comparison

- Selection: best epoch by patient-weighted r_ind,val only; all secondary metrics from it.
- Comparisons: C1-B1, C2-B1, and mechanistic C2-C1.
- Statistics: 10,000 paired patient bootstraps, seed 20260910. Recompute p05 inside each resample.
- Success: delta mean r >=0.005 and CI low >0 against B1.
- Harm margins: delta mIoU >-0.005; P-IoU >-0.01; T-IoU >-0.01; QRS MAE <+1 ms; LVH MAE <+0.05 mV; variance-retention log error no more than +0.05.
- Stop if neither C1 nor C2 advances.
- Priority: MUST-RUN.

### B3 — Mechanistic diagnostics and conditional replication

- C1: coordinate correlation/R2 against oracle PCA coefficients without latent training; attributable reconstruction by component.
- C2: principal angles to PCA subspace; no coordinate matching.
- Seeds 43/44 only for seed-42 advancing systems plus B1.
- Priority: diagnostics MUST-RUN; extra seeds CONDITIONAL.

## Run order

| Milestone | Runs | Gate | Approximate cost |
|---|---|---|---|
| M0 | PCA oracle ranks 1-6 | Freeze K_STAR | <0.5 GPU-hour |
| M1 | Unit/smoke/equality audit | All invariants and hashes pass | <0.2 GPU-hour |
| M2 | B1/C1/C2 seed42, 15 epochs | Bootstrap plus harm screen | 3 established runs |
| M3 | Evaluation and latent diagnostics | Interpret hierarchy | <1 GPU-hour |
| M4 | Seeds43/44 if authorized | Replicate advancement | conditional |
| M5 | External confirmation | Only after final freeze | deferred |

## Frozen implementation decisions

- c and PCA basis use folds 1-8 only; C1 basis is a buffer.
- C2 stores one unconstrained 7-by-K_STAR parameter and applies reduced QR; no orthogonality loss.
- Loss is seven-lead reconstructed-signal loss only. No latent/PCA/VCG/uncertainty loss.
- C2 axes have no physiological interpretation.
- FIXED_VCG_SPECIFICITY = NOT_SUPPORTED; old artifacts remain unchanged.
- No generative primitive is claimed; probabilistic completion is conditional future work.
- The selected trunk's inherited record-wide z-score uses full-record mean/scale. A pre-launch fold-9 audit found negligible impact on the primary Pearson endpoint (global 0.822248, Lead-I-only 0.822290, raw 0.822289), so it is retained identically across B1/C1/C2 for this controlled comparison. It is not deployment-clean for amplitude metrics; any later confirmation must include observable-only normalization sensitivity and may not conceal this limitation.

## Final checklist

- [x] Anchor and anti-claim frozen.
- [ ] Rank curve complete and K_STAR immutable.
- [ ] B1/C1/C2 equality tests pass.
- [ ] Seed-42 selection uses only r_ind,val.
- [ ] Paired mean/tail bootstrap and harm screen complete.
- [ ] Conditional seed decision recorded.
- [x] Fold 10 and pristine EchoNext sealed.
