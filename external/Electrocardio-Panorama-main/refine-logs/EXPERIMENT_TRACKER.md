# Experiment Tracker

Statuses: `TODO`, `BLOCKED`, `RUNNING`, `PASS`, `FAIL`. Do not move a run past a failed upstream decision gate.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Artifact | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | materialize and reproduce exact archived baseline | catalog model `factorial_ecg_aim_1010010_s42`, asset 523581550 | existing evaluator/reference panel | SHA256, payload schema, effective observed mask, eight quoted metrics | MUST | TODO | Remote-verified; expected SHA256 `5ac89f3d…ef4f`; catalog training metadata records I, II, V2 |
| R002 | M0 | freeze one-source headline protocol | authoritative fixed-source audit, mask `1010010`, seed 42 | PTB-XL train/val/test | source lead, split hashes, loss/config diff | MUST | TODO | Do not infer source from model ID; inspect checkpoint, evaluator, job config, and logs |
| R003 | M0 | prove refactor parity | original ECG-AIM vs centralized learned mode in working copy | deterministic tiny batch | exact full outputs, positions, state keys | MUST | PASS | Exact full-output parity test passes; original implementation and trainer are unchanged |
| R004 | M1 | implementation safety | all conditioning modes | synthetic + tiny PTB-XL batch | 12 required unit tests | MUST | TODO | Includes mapping, buffers, relative pooling, signs, FiLM identity, adapters/contracts |
| R005 | M1 | optimization sanity | E1 maximum-complexity path | one-batch train/validation smoke | finite checkpoint, provenance, 221 tensors | MUST | PASS | Lead-II E1 GPU smoke completed; mask/seed/source provenance verified |
| R006 | M2 | establish valid headline baseline | A0 learned-ID, seed 42, Lead I + Lead II | fixed-source PTB-XL | validation + eight test endpoints | MUST | RUNNING | Priority queue started with Lead-I A0; mask 1010010; no extra seeds |
| R007 | M2 | test geometry sufficiency | B1 Panorama replacement, seed 42, Lead I + Lead II | fixed-source PTB-XL | primary/secondary + per lead | MUST | TODO | Two queued jobs; frozen angles and correct remap |
| R008 | M2 | test residual geometry | C1 learned + Panorama, seed 42, Lead I + Lead II | fixed-source PTB-XL | primary/secondary + per lead | MUST | TODO | Two queued jobs; spatial gain init 0.1 |
| R009 | M2 | isolate relative geometry | D1 C1 + source→target relative, seed 42, Lead I + Lead II | fixed-source PTB-XL | primary/secondary + per lead | MUST | TODO | Two queued jobs; pool only observed lead |
| R010 | M2 | test repeated conditioning | E1 D1 + zero-init FiLM, seed 42, Lead I + Lead II | fixed-source PTB-XL | primary/secondary + per lead | MUST | TODO | Two queued jobs; lead/time/FFN modulation identity at init |
| R011 | M3 | validate canonical field asset | deterministic asset pipeline | canonical model | metadata, license, signs, units, SHA256 | MUST | BLOCKED | Blocked until documented asset is obtained; never fabricate fields |
| R012 | M3 | validate field features | standard-lead derivation + SVD | canonical model | rank, norms, no NaNs, similarity | MUST | BLOCKED | Depends on R011; K=min(8, rank), retain log magnitude |
| R013 | M3 | test lead-field model | F1 learned + lead field + relative + FiLM, seed 42 | fixed-source PTB-XL | primary/secondary + per lead | MUST | BLOCKED | Depends on R011–R012 |
| R014 | M4 | rule out extra capacity | parameter-matched learned-ID MLP/FiLM control, seed 42 | fixed-source PTB-XL | endpoints + params/FLOPs | MUST | TODO | Match selected spatial model capacity |
| R015 | M4 | rule out feature-distribution effect | fixed permuted spatial labels, seed 42 | fixed-source PTB-XL | endpoints + paired deltas | MUST | TODO | Preserve distribution, destroy lead semantics |
| R016 | M5 | baseline confirmation | A0 seed 123 | fixed-source PTB-XL | eight endpoints + per record | MUST | TODO | Seed 42 reused from R006 |
| R017 | M5 | baseline confirmation | A0 seed 2026 | fixed-source PTB-XL | eight endpoints + per record | MUST | TODO | — |
| R018 | M5 | winner confirmation | selected spatial model seed 123 | fixed-source PTB-XL | eight endpoints + per record | MUST | TODO | Candidate frozen before run |
| R019 | M5 | winner confirmation | selected spatial model seed 2026 | fixed-source PTB-XL | eight endpoints + per record | MUST | TODO | Combine with winner seed 42 screen |
| R020 | M6 | lead-agnostic baseline | A0 seed 42 | variable single-source PTB-XL | macro/worst source + source×target | MUST | TODO | Uniform deterministic source sampling |
| R021 | M6 | lead-agnostic baseline | A0 seed 123 | variable single-source PTB-XL | macro/worst source + source×target | MUST | TODO | — |
| R022 | M6 | lead-agnostic baseline | A0 seed 2026 | variable single-source PTB-XL | macro/worst source + source×target | MUST | TODO | — |
| R023 | M6 | decisive geometry test | selected spatial model seed 42 | variable single-source PTB-XL | macro/worst source + source×target | MUST | TODO | No candidate selection on these test results |
| R024 | M6 | decisive geometry test | selected spatial model seed 123 | variable single-source PTB-XL | macro/worst source + source×target | MUST | TODO | — |
| R025 | M6 | decisive geometry test | selected spatial model seed 2026 | variable single-source PTB-XL | macro/worst source + source×target | MUST | TODO | — |
| R026 | M7 | paired statistics and failures | A0 vs selected model | fixed + variable test | 10k paired bootstrap, full patient deltas | MUST | TODO | Aggregate per record before quantiles |
| R027 | M7 | physical diagnostic | F1 vs A0 | fixed + variable test | field similarity vs performance/Δ | MUST | BLOCKED | Exploratory; only if F1 exists |
| R028 | M7 | produce paper artifacts | all completed runs | frozen outputs | CSVs, plots, hashes, REPORT.md | MUST | TODO | Report every endpoint and failed run |
| R029 | M7 | test attention bias | G1 F1 + lead-axis field bias | fixed-source PTB-XL | endpoints + cost | NICE | BLOCKED | Only if F1 passes; cut if no reproducible added value |
| R030 | M7 | withheld-source stress test | A0 vs selected spatial | one held-out source at a time | per-source/per-target degradation | NICE | BLOCKED | Only after stable B4 win; not true wearable interpolation |
| R031 | M7 | warm-start diagnostic | A0 checkpoint → selected spatial | fixed-source PTB-XL | convergence and endpoints | NICE | TODO | Report separately; never headline |

## Immediate Launch Queue

1. **RUNNING** — `spatial_1lead_a0_1010010_s42_l0`.
2. **NEXT** — Lead-I B1, C1, D1, E1, followed by the same A0–E1 order for Lead II.
3. **AUTOMATIC HANDOFF** — resume the pending 3-lead factorial queue after all ten seed-42 spatial jobs terminate.

## Decision Log

| Date | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-08-22 | One-lead ECG-AIM checkpoints exist for Lead I and Lead II | eight files plus embedded provenance; queue status is stale | use Lead II checkpoint for path/parity; do not claim the one-lead path is missing |
| 2026-08-22 | Exact `factorial_ecg_aim_1010010_s42` artifact is present in the checkpoint catalog | remote-verified asset 523581550; registered and payload SHA256 `5ac89f3d…ef4f` | materialize and evaluate it first; do not retrain merely because local bytes are absent |
| 2026-08-22 | Archived-reference and headline-training roles are distinct | archived metadata records training leads I, II, V2; PRD requires a fixed one-source controlled comparison trained from scratch | reproduce archive, audit effective/source masks, then train all headline rows from scratch under the frozen one-source protocol |
| 2026-08-22 | Run only seed 42 in the active priority screen | user deferred additional seeds until after validation | queue contains 10 A0–E1 jobs across Lead I/II; no 123/2026 jobs |
| 2026-08-22 | Preserve original ECG-AIM files | user supplied `ecg-aimbackup.py` as the modifiable copy | original model/trainer verified unchanged; spatial code and trainer are separate |
| 2026-08-22 | Pause 3-lead queue after `1100012` and resume automatically later | checkpoint validated; `1100013` reset to pending | spatial priority tmux session runs first, then execs the existing queue worker |
| 2026-08-22 | No frontier-necessity experiment | method uses deterministic geometry/standard neural modules | focus paper budget on geometry, capacity controls, and changing-source generalization |
| 2026-08-22 | F1/G1 are asset-gated | PRD forbids invented lead fields and no validated asset is yet documented | run A0–E1 while asset validation proceeds; block F1/G1 if validation fails |
