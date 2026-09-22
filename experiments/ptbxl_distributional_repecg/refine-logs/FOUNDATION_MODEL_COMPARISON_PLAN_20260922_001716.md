# Foundation-Model Comparator Plan

**Problem**: Test whether a real, externally pretrained ECG foundation encoder changes the acquisition-configuration conclusion relative to GraphECG and SetOperator.

**Method thesis**: Add only frozen, provenance-audited foundation encoders as comparators; each receives the same PTB-XL labels and patient split, with a newly trained downstream head where that is necessary to make the endpoint commensurate.

## Candidate census and admission

| Candidate | Artifact status | Admissible comparison scope | Decision |
|---|---|---|---|
| ECGFounder 12-lead | Recoverable from the SHA-verified archive: 369,942,585 bytes | Full 12-lead frozen-encoder ceiling only; not a Q8/missing-lead result | Admit conditionally |
| ECGFounder 1-lead | Recoverable from the same verified archive: 369,807,481 bytes | Lead-I frozen-encoder comparator only | Admit conditionally |
| HuBERT-ECG | Source tree present, checkpoint absent from live and archived inventory | None | Exclude |
| EchoNext Minimodel | Recoverable task-specific EchoNext predictor | Not a general ECG representation encoder and not the PTB-XL five-label task | Exclude |

The two ECGFounder files must be extracted only into a new comparison artifact directory after archive SHA verification. Existing project paths will not be overwritten.

## Claim map

| Claim | Minimum convincing evidence | Anti-claim |
|---|---|---|
| C1: Continuous operators retain information under a single physical lead at least as well as a large external ECG foundation encoder. | Same Fold-8 records, labels, Lead-I input, head-selection split, and patient-level paired inference for ECGFounder-1L, GraphECG, and SetOperator. | A result is caused only by using a larger pretrained backbone. |
| C2: Full-lead foundation-model performance is a ceiling, not evidence of configuration invariance. | ECGFounder-12L full-12 score reported separately, with no zero-fill or fabricated subset inputs. | A full-12 number justifies claims about missing leads. |

## Execution plan

### M0 — Repair the currently invalid P07 control

- **Gate**: no-MMD batch-32 control must finish with a new immutable summary. The present run failed at epoch 6 with a non-finite loss and is not evidence.
- **Reason**: final P07 evaluation and the existing strict comparison queue otherwise wait indefinitely.
- **Status**: BLOCKED; do not start a new foundation GPU workload before this is resolved.

### M1 — ECGFounder artifact and leakage admission

1. Verify the archive SHA-256.
2. Extract only the two ECGFounder checkpoint members into a new artifact root.
3. Hash both extracted files and strictly load their state dictionaries.
4. Record model architecture, parameter count, preprocessing, checkpoint digest, and source archive digest.
5. Audit whether PTB-XL was in ECGFounder pretraining. If membership cannot be established, label every result `pretraining-membership-unknown`; it may remain a comparator, but cannot support a strict no-leakage superiority claim.

### M2 — Same-task frozen-head protocol

- Input population: PTB-XL folds 1–7 for head fitting, folds 9–10 for head selection, Fold 8 for final testing.
- Endpoint: the existing five superclass labels; do not use ECGFounder's released 150-task head as the comparison endpoint.
- Training: freeze the ECGFounder encoder; fit only a five-label linear head. Use three head seeds and preserve all predictions/record IDs.
- Preprocessing: implement and hash the authors' filtering, z-score normalization, and resampling contract independently for each model. Native preprocessing is allowed; silent common resampling is not.

### M3 — Configuration cells

| Cell | ECGFounder checkpoint | Comparison interpretation |
|---|---|---|
| Full 12-lead | 12-lead | Ceiling / context only. It is not the GraphECG Q8 cell. |
| Lead I | 1-lead | Admissible primary foundation-model comparator against the Lead-I GraphECG and SetOperator cells. |

No other subset is admitted. In particular, do not zero-fill the 12-lead ECGFounder input, duplicate Lead I into multiple channels, synthesize ICM, or claim exhaustive-Q8 generalization.

### M4 — Statistical and reporting layer

- Save an aligned prediction bundle for every admissible cell.
- Run paired endpoint-level DeLong tests and 2,000-replicate patient-equal bootstrap CIs; apply Holm correction within the predeclared Lead-I family.
- Report macro-AUROC descriptively only; do not create a macro-DeLong p-value.
- Add an explicit eligibility table to the GraphECG/SetOperator comparison document. Excluded cells remain absent rather than receiving proxy values.

## Run order and decision gates

| Run | Goal | GPU cost | Gate |
|---|---|---:|---|
| FM-00 | Archive extraction and strict-load smoke | CPU + brief GPU | Both hashes and state loads pass |
| FM-01 | Frozen ECGFounder-1L, three linear-head seeds | modest | Exact Fold-8 ID/label alignment passes |
| FM-02 | Lead-I paired inference | CPU | All prediction bundles are pairable; ICM absent |
| FM-03 | Frozen ECGFounder-12L, three linear-head seeds | modest | Clearly labelled full-12 ceiling, not Q8 comparison |

## Risks and mitigations

- **PTB-XL may have been used in pretraining**: retain the result but label its membership status and prohibit a strict no-leakage claim.
- **ECGFounder preprocesses at 5,000 samples while current models use a different native path**: compare identical records/splits and document preprocessing; do not call it an identical-input latency test.
- **Missing-lead extrapolation is unsupported by the released 12-lead model**: test only the dedicated 1-lead checkpoint at Lead I.

## Queue state

The admission/evaluation sequence is planned but deliberately not launched. The active P07 chain contains an invalid no-MMD control that must be repaired first; launching now would either contend with training or falsely certify a blocked queue.
