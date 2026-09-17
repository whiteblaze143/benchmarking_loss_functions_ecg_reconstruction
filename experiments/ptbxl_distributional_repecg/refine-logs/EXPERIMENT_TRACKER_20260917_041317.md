# Production-Readiness Tracker

**Database**: `outputs/results.sqlite3`
**Production state**: BLOCKED

| Run ID | Milestone | Purpose | Scope | Priority | Status | Completion evidence |
|---|---|---|---|---|---|---|
| R001 | M0 | strict smoke training | 15 papers, 52 variants, 9 cells each | MUST | RUNNING | 468 exact cell summaries |
| R002 | M0 | aggregation/reload evaluation | all 52 variants | MUST | RUNNING | 52 checkpoint-backed evaluations |
| R003 | M0 | database reconciliation | smoke cells/evaluations | MUST | RUNNING | 52 complete, zero pending/failed |
| R010 | M1 | mechanism effect audit | Papers 1-6, 11-12, 15 | MUST | TODO | matched-state effect tests and endpoint rows |
| R011 | M1 | reconstruction auxiliary | Paper 7 | MUST | TODO | optimized reconstruction loss and held-out metrics |
| R012 | M1 | KMeans token control | Paper 8 | MUST | TODO | frozen train dictionary and distinct token hashes |
| R013 | M1 | counterfactual-q control | Paper 9 | MUST | TODO | synthesis endpoint responds to mismatched q |
| R014 | M1 | ERM/IRM/CORAL/CausIRL | Paper 10 | MUST | TODO | objective-specific losses and environment IDs |
| R015 | M1 | phase surgery controls | Paper 13 | MUST | TODO | targeted/random/no-propagation effects |
| R016 | M1 | invariant objectives | Paper 14 | MUST | TODO | environment losses and MMD-only control |
| R020 | M2 | remove fabricated labels | existing OOD builder/archives | MUST | TODO | no all-zero label construction |
| R021 | M2 | ontology reconciliation | seven primary external datasets | MUST | TODO | class denominators and manifest hashes |
| R022 | M2 | adapter reconciliation | all eight external datasets | MUST | TODO | source/eligible/excluded counts and lead maps |
| R030 | M3 | Kingston channel audit | Kingston-ICU | MUST | TODO | explicit source-to-eight-lead map |
| R031 | M3 | zero-padding representation | Kingston-ICU | MUST | TODO | mask, observed count, transformed-channel zeros |
| R032 | M3 | matched masking control | PTB-XL fold 8 | MUST | TODO | paired full/masked records per Kingston pattern |
| R033 | M3 | negative-control metrics | Kingston + matched PTB-XL | MUST | TODO | patient-equal paired deltas/CIs stored separately |
| R040 | M4 | full development grid | promoted variants, three seeds | MUST | BLOCKED | M0-M3 gates |
| R050 | M5 | external evaluation | eligible datasets/endpoints | MUST | BLOCKED | M2/M4 gates |
| R051 | M5 | perturbation evaluation | eligible datasets/endpoints | MUST | BLOCKED | complete shared-condition matrix |
| R060 | M6 | integrity/coverage audit | complete program | MUST | BLOCKED | PASS and zero unexplained missing keys |
| R061 | M6 | production promotion | complete program | MUST | BLOCKED | atomic readiness sentinel |

## Current verified notes

- Production launchers fail closed while `outputs/PRODUCTION_READY` is absent.
- Paper 3 `level1` routing was repaired and its strict smoke passed.
- Existing OOD archives are invalid diagnostic evidence because labels were generated as zeros.
- The existing OOD builder covers seven datasets and has no Kingston adapter.
- Kingston is frozen as a negative control and is not pooled into primary OOD performance.
