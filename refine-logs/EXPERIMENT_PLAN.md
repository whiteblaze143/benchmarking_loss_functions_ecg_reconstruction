# Braid Clinical Extension Experiment Plan

**Problem:** determine whether PTB-XL-trained Braid representations match SetOperator and GraphECG under the same frozen-encoder clinical configuration-shift protocol.

**Method thesis:** Braid is evaluated as a PTB-XL-trained encoder; each native cohort receives only a full/native-training-split task head, which is frozen before reduced-lead evaluation.

## Claim Map

| Claim | Minimum evidence | Block |
|---|---|---|
| Braid changes flexible-lead clinical representations relative to SetOperator and GraphECG | Same patient/split/configuration predictions with patient-level uncertainty | B1 |
| Any temporal benefit is present on native annotated boundaries rather than inferred labels | Fixed native delineation splits, direct sequence head, per-class and boundary metrics | B2 |

## Block B1: Cross-Dataset Task-Native Decoupling

- **Systems:** GraphECG, three SetOperator checkpoints, five fully trained PTB-XL Braid checkpoints, and the existing fixed-tensor control.
- **Cohorts:** LUDB, Zhejiang, ISP, Kingston-ICU, then RDB, HEEDB, and EchoNext.
- **Protocol:** extract a frozen encoder representation under the shared Q8, S6, S3, S2, S1, and ICM configurations; fit a task head only on the cohort's full/native training split; select it only on that cohort's validation split; freeze both components for shift testing.
- **Metrics:** task-appropriate AUROC/AUPRC; absolute full-to-shift delta; retention; class-wise denominators; patient-equal aggregation whenever patient IDs exist.
- **EchoNext gate:** train/validation waveforms must be restored and verified against the official split before any head fitting. The local 5,442-record official test archive is evaluation-only.
- **HEEDB-derived Emory-MUSE gate:** use the audited 12SL diagnosis table only after per-row WFDB materialization is reconciled, empty-code exclusions are frozen, the label vocabulary/support rule is fit on training patients only, and patient-disjoint hash splits are emitted. This is distinct from a claim about the full multi-site HEEDB corpus.
- **RDB gate:** use the frozen cache train/validation/test directories and payload rhythm labels, never filename-derived labels.

## Block B2: Native Delineation

- **Cohorts:** ISP, LUDB, RDB, Zhejiang.
- **Protocol:** direct sequence head over each frozen encoder's time-resolved representation. Do not score a generic NeuroKit delineator on reconstructed waveforms as an encoder comparison.
- **Labels:** native P/QRS/T masks and six onset/offset fiducials; preserve each dataset's sampling rate and documented coordinate map.
- **Metrics:** class Dice/IoU, per-fiducial event F1 at prespecified millisecond tolerances, absolute boundary error in milliseconds, and valid/ineligible sample counts.
- **Splits:** ISP official train/test; LUDB frozen patient folds; RDB frozen cache splits; Zhejiang frozen record-hash folds. No test-derived normalization, thresholds, or early stopping.

## Block B3: Statistical Inference and Ablations

- **Bootstrap:** 2,000 paired resamples of test patients; use records only where patient IDs are unavailable and label that exception.
- **Pairwise discrimination:** DeLong tests only for matched predictions on the same binary endpoint/test set; alpha 0.05 with a prespecified multiple-testing correction family.
- **Shift comparison:** paired bootstrap on per-patient metric contributions and paired full-versus-reduced deltas; report 95% CIs and explicit denominators.
- **Braid ablations:** field versus each topology variant under identical PTB-XL data, seed, optimizer, selection split, and parameter-budget disclosure. Do not rescue a negative field-richness result with post-hoc architecture changes.

## Run Order

| Milestone | Gate | Status | Next action |
|---|---|---|---|
| M0 | Five Braid PTB-XL checkpoints | Running | Three-way GPU-saturated wave, then remaining two variants |
| M1 | Existing four-cohort task-native comparison | Pending M0 | Run the shared registry only after every Braid checkpoint exists |
| M2 | RDB rhythm/delineation adapter | Ready data, adapter pending | Implement/test frozen-cache native-label adapter |
| M3 | ISP/LUDB/Zhejiang direct delineation heads | Labels audited, heads pending | Implement/test one common sequence-head contract without coordinate coercion |
| M4 | HEEDB-derived Emory-MUSE admission | Complete | Frozen v1 admission has 941,679 records / 343,424 patients, zero split crossings, all 180 training-derived codes, and verified waveform pairs for every diagnosis-eligible row |
| M5 | EchoNext task-native head | Blocked by missing train/validation waveforms | Restore and hash official training/validation inputs; retain test sealed |
| M6 | Statistical report | Pending M1-M5 | Generate paired bootstrap/DeLong artifacts from frozen prediction tables |

## Non-negotiable exclusions

- No fitting on EchoNext's official test waveforms.
- No fabricated PTB-XL labels for native cohorts.
- No pooled-record inference when a patient-level analysis is possible.
- No topology threshold, checkpoint, or hyperparameter selection from a reduced configuration or external test outcome.
