# Experiment Plan: Beat-Structured ECG-AIM

## Claims and Decisive Tests

| Claim | Comparison | Primary metric | Pass condition |
|---|---|---|---|
| C1 morphology utility | proposed embedding vs dimension-matched current pooling | held-out QRS-onset-to-T-offset MAE | CI for paired MAE improvement excludes zero |
| C1 rhythm preservation | proposed embedding vs current pooling | AF/AFIB AUROC | non-inferior within 0.02 absolute AUROC |
| C2 component separation | full embedding vs delete structure/dynamics/rhythm | endpoint-specific AUROC/MAE | morphology degrades most without structure; rhythm degrades most without rhythm |

## Stage 0: Required Data Gate

Materialize beat boundaries for train/validation/test without using test labels for method selection. Audit:

- every record has at least one valid beat;
- boundaries are ordered, in range, and left-aligned after padding;
- number of beats and RR distributions by split;
- failure rate by diagnostic superclass;
- detector configuration and source hash.

Stop if more than 1% of records fail or failure is diagnosis-dependent by more than 2 percentage points. The current implementation intentionally has no fallback.

## Stage 1: Representation Pilot

Freeze the selected ECG-AIM checkpoint. Pretrain only the auxiliary branch on the training split with beat-reconstruction L1 plus 0.1 times next-state Smooth-L1; diagnostic labels never update the representation. Select the epoch on validation self-supervised loss. Fit downstream probes on frozen embeddings using training labels, select probe hyperparameters on validation, and evaluate once on the untouched test split.

Models:

1. Current mean/std/max encoder pooling projected to 640 dimensions.
2. Proposed learned-field + LVCG beat encoder + GRU + RR embedding.
3. Capacity control: same branch with Lead I triplicated to three channels and no learned field adapter.
4. Structure control: proposed branch with beat boundaries shuffled within each record while preserving beat count.

Use identical linear probes, standardization fitted on training only, and the same validation selection rule.

## Stage 2: Deletion Tests

Probe full 640-D embedding and three deletions: minus structure, minus dynamics, and minus rhythm. Do not retrain the encoder; retrain only the linear probe for each deletion.

## Outcomes

- Rhythm: AF/AFIB AUROC and AUPRC.
- Morphology: annotation-derived QRS-onset-to-T-offset MAE, RMSE, and concordance; label it explicitly as not clinical QT/QTc.
- Diagnostic breadth: PTB-XL superclass and diagnostic multilabel macro/micro AUROC and AUPRC.
- Representation diagnostics: effective rank, component variance, and component-wise linear CKA.
- Reconstruction: confirm bitwise-identical `y_pred` for the wrapped and unwrapped frozen ECG-AIM on the same batch.

## Statistics

- Patient-disjoint existing splits.
- Paired record bootstrap with 2000 replicates for metric differences.
- Multiplicity correction within the two primary claims.
- Three seeds only after the Stage-1 validation gate passes.

## Run Order and Decision Gates

1. Boundary materialization/audit.
2. Frozen forward equivalence test on 100 validation records.
3. One-seed representation pilot and controls.
4. Stop if C1 morphology fails or rhythm drops by more than 0.02.
5. Component deletions.
6. Only then run seeds 43 and 44 and diagnostic breadth probes.

## Compute Budget

The pilot trains only the auxiliary branch and cached linear probes. It does not retrain the ECG-AIM reconstructor. Record wall time, peak GPU memory, trainable parameters, and preprocessing failures for every run.
