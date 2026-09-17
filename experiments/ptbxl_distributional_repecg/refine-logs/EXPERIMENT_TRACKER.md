# Production-Readiness Tracker

**Database**: `outputs/results.sqlite3`
**Production state**: BLOCKED

| Run ID | Milestone | Purpose | Scope | Priority | Status | Completion evidence |
|---|---|---|---|---|---|---|
| R001 | M0 | strict smoke training | 52 eligible variants, 9 cells each; 3 Paper 4 variants ineligible | MUST | TODO | prior smoke invalidated; corrected target is 468 eligible cells plus explicit Paper 4 ineligibility |
| R002 | M0 | aggregation/reload evaluation | 52 eligible plus 3 ineligible variants | MUST | TODO | rerun only after `outputs/paper_input_reconciliation.json` is complete |
| R003 | M0 | database reconciliation | smoke cells/evaluations | MUST | TODO | database intentionally absent until the corrected smoke starts from a clean state |
| R010 | M1 | mechanism effect audit | Papers 1-6, 11-12, 15 | MUST | IN_PROGRESS | Papers 1 and 3 now have train-only paper-specific representations and controls; inert `training_regime` choices removed; Paper 15 shared control repaired to <1% parameter-count difference; trained effect endpoints still required |
| R011 | M1 | reconstruction auxiliary | Paper 7 | MUST | TODO | current trainer is contract-incompatible: it consumes one Paper-2 tensor instead of sampled `(q,F(q))` sets; requires separate BCE-only and auxiliary continuous/categorical pairs |
| R012 | M1 | KMeans token control | Paper 8 | MUST | DONE | train-only 64-centre MiniBatchKMeans dictionary persisted in checkpoint; 27-cell smoke and checkpoint reload passed |
| R013 | M1 | counterfactual-q control | Paper 9 | MUST | TODO | synthesis endpoint responds to mismatched q |
| R014 | M1 | ERM/IRM/CORAL/CausIRL | Paper 10 | MUST | TODO | objective-specific losses and environment IDs |
| R015 | M1 | phase surgery controls | Paper 13 | MUST | TODO | targeted/random/no-propagation effects |
| R016 | M1 | invariant objectives | Paper 14 | MUST | TODO | environment losses and MMD-only control |
| R020 | M2 | remove fabricated labels | existing OOD builder/archives | MUST | DONE | legacy builder retired fail-closed; source contains no fabricated target construction; archived placeholder outputs remain scientifically invalid |
| R021 | M2 | native-task reconciliation | all datasets | MUST | DONE | `outputs/native_task_reconciliation.json`: 9 datasets, 12 native tasks, split/class denominators and source hashes reconciled |
| R022 | M2 | adapter reconciliation | all eight external datasets | MUST | COMPLETE | exact counts/maps in `outputs/adapter_reconciliation.json`; EchoNext test shapes confirmed and Zhejiang units verified as microvolts |
| R023 | M2 | replace random-model OOD evaluators | Papers 1-15 | MUST | IN_PROGRESS | all 15 unsafe evaluators retired fail-closed; native checkpoint-backed replacements still required |
| R030 | M3 | Kingston channel audit | Kingston-ICU | MUST | DONE | 596 records; 240 Hz; `[I,II,III,V]`; retain I/II only |
| R031 | M3 | zero-padding representation | Kingston-ICU | MUST | DONE | full Train/Test build: 288/293 eligible; exact mask `[1,1,0,0,0,0,0,0]`; transformed missing channels verified zero |
| R032 | M3 | matched masking control | PTB-XL fold 8 | MUST | DONE | 2,173 paired records/1,881 patients; identical IDs/labels; full and `[1,1,0,0,0,0,0,0]` archives share frozen kernel fit |
| R033 | M3 | negative-control metrics | Kingston + matched PTB-XL | MUST | DONE | PTB-XL patient-equal paired deltas with 2,000 patient bootstraps; Kingston native Train/Test metrics with record bootstrap because no patient ID is released; task results stored separately |
| R040 | M4 | full development grid | promoted variants, three seeds | MUST | BLOCKED | M0-M3 gates |
| R050 | M5 | external evaluation | eligible datasets/endpoints | MUST | BLOCKED | M2/M4 gates |
| R051 | M5 | perturbation evaluation | eligible datasets/endpoints | MUST | BLOCKED | complete shared-condition matrix |
| R060 | M6 | integrity/coverage audit | complete program | MUST | BLOCKED | PASS and zero unexplained missing keys |
| R061 | M6 | production promotion | complete program | MUST | BLOCKED | atomic readiness sentinel |

## Current verified notes

- Production launchers fail closed while `outputs/PRODUCTION_READY` is absent.
- Paper 3 `level1` routing was repaired and its strict smoke passed.
- Existing OOD archives are invalid diagnostic evidence because labels were generated as zeros; native labels were found for every dataset and are being wired task-by-task.
- The existing OOD builder covers seven datasets and has no Kingston adapter.
- Kingston is frozen as a negative control and is not pooled into primary OOD performance.
- Kingston's generic `V` channel is not assigned to V1-V6; the frozen canonical mask is `[1,1,0,0,0,0,0,0]`.
- Matched PTB-XL fold-8 masking reduced patient-equal macro AUROC by 0.1347 for kernel (95% patient-bootstrap CI [-0.1461, -0.1233]) and 0.2408 for moments (CI [-0.2578, -0.2251]). This is descriptive selection-fold evidence, not an independent test claim.
- Kingston native SINUS-versus-AFIB/AFLT Test AUROC after Train-only model selection was 0.8786 for kernel and 0.8843 for moments. Kingston has no released patient ID, so its uncertainty is record-bootstrap and is not labeled patient-level.
- M0 engineering smoke completed across all 15 papers. This proves execution
  coverage only; identical outputs exposed inert scientific controls in Papers
  9, 13, and 14, which remain blocked under M1.
- All trainers now accept only the implemented `full_only` training regime;
  previously advertised `mask_aug`, `finetune`, and `scratch` choices were inert
  aliases and have been removed rather than mislabeled as experiments.
- Paper 15's former capacity-matched shared control had 37,628 parameters versus
  130,501 for the full model. The shared mechanism is now widened algebraically,
  leaving the total parameter counts within 1% while preserving exact sharing
  across all 15 transitions.
- Paper 1 now has its own fold-1--7-fitted normalized recurrence artifacts:
  15,244 training and 2,173 fold-8 records, with exact source ID/label alignment,
  cyclic-neighbor exclusion, symmetric finite operators, and a separate
  train-fitted scale for the mean-voltage control. The full and mean-control
  CNNs have identical 9,669-parameter architectures.
- Paper 1's missing falsification arms are now registered and materialized.
  `phase_content_permuted` changes cell content while retaining the original
  exclusion geometry; `cyclic_relabel_sham` applies a record-keyed nonzero
  cyclic conjugation. All records change under the destroyer, while sampled
  sham operators preserve the complete eigenspectrum as required.
- Paper 3 now uses 228-D depth-3 beat-cell log-signatures, a 64-D train-only
  truncated PCA whitening transform, and a passed 128-landmark Nyström map
  (Spearman 0.9106; median relative error 0.0610). Its order destroyer,
  reversal, monotone-warp sham, and unordered-waveform control are materialized
  for 15,244/2,173 records. A separate fold-1--7 PCA reduces the unordered
  control to 128 features so all nonlinear arms use exactly matched phase-CNN
  input capacity.
- Paper 4 is stopped by its frozen approximation gate, not silently repaired:
  128 landmarks produced Spearman 0.649 and median relative error 0.861; the
  frozen maximum of 256 produced 0.671 and 0.816. Both miss the required 0.90
  and 0.15 thresholds. Its three registered variants must be recorded as
  `ineligible_nystrom_fidelity`; no Paper 4 training or task claim is allowed.
