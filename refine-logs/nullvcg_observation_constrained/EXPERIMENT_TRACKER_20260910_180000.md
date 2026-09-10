# Geometry-specific latent completion experiment tracker

Created: 2026-09-10 18:00:00 UTC

## Terminal state for this protocol

- Geometry-specificity oracle: **FAIL (data-integrity stop)**.
- Neural B1/M1/C1 training: **not launched**.
- PTB-XL fold 10: **sealed**.
- EchoNext waveform evaluation: **sealed**; metadata-only audit froze 4,097 patient-disjoint ECG keys for possible future confirmation.

## Completed evidence

- Plan: `EXPERIMENT_PLAN_20260910_173054.md`.
- Full run: 17,418 fold-1–8 ECGs (87,090,000 time samples) fit the uncentered residual PCA; 2,183 fold-9 ECGs from 1,942 patients were evaluated.
- Geometry tests: 5/5 passed.
- Limb audit: failed. Two original PTB-XL records contain gross isolated source-lead violations: ECG 5930 (aVL/aVF ≈2.36 mV residual) and ECG 15867 (III 0.767 mV residual). See `LIMB_ALGEBRA_OUTLIER_AUDIT_20260910T175900Z.md`.
- G0 fixed VCG: mean independent r 0.867166; p05 0.690804; MSE 0.0466963.
- P0 training PCA: mean independent r 0.942731; p05 0.832630; MSE 0.0230896.
- T0 frozen patch-10: mean independent r 0.822248; p05 0.551181; MSE 0.0495120.
- Paired 10,000-bootstrap G0−P0 delta: −0.075565, 95% CI [−0.078360, −0.072952].
- P0−T0 delta: +0.120483, 95% CI [+0.115370, +0.125776].
- G0−T0 delta: +0.044918, 95% CI [+0.040230, +0.049753].
- Among 100 frozen random rank-2 bases, G0 was at the 98th percentile, P0 at the 100th; best random mean r was 0.898427.
- VCG failed the preregistered clinical-compensation rule (one win; large losses on Sokolow–Lyon error and variance-retention fidelity).

## Interpretation

Even apart from the integrity stop, the preregistered scientific classification would be `LOW_RANK_ONLY`, not `VCG_SUPPORTED`: P0 exceeds G0 by 0.075565, far beyond the 0.005 boundary, with no clinical compensation. The promising effect is training-derived low-rank residual structure. Fixed VCG geometry is better than almost every random subspace but is decisively inferior to the learned linear PCA subspace.

## Authorized next protocol

Create a fresh preregistration that freezes a robust treatment for the two source-corrupt validation records before recomputation. The smallest justified follow-up is B1 direct seven-output versus C1 frozen-PCA two-output, seed 42, with M1 retained only as a negative control. Do not claim VCG-specific benefit and do not open fold 10 or EchoNext until that comparison and seeds are frozen.

