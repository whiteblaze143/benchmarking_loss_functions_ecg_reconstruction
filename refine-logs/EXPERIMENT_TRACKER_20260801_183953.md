# Experiment Tracker: External Delineation Watcher

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ED-M0-TEST | M0 | matcher/resampler/storage validation | focused pytest suite | synthetic + temp store | one-to-one matching, shape, summary, archive identity | MUST | COMPLETE | 27 combined external-watcher/checkpoint-store tests passed |
| ED-M1-SMOKE | M1 | end-to-end real-data validation | `f_1000000_s42`, 1 record/dataset split, leads III/V3 | LUDB all; ISP train/test; Zhejiang all | boundary F1/timing, Dice, hashes | MUST | COMPLETE | exact checkpoint loaded; all detector rows finite; private cache empty |
| ED-M2-CEILING | M2 | quantify detector/adapter ceiling | source waveform, nine missing-lead positions | all local external records | coverage, one-to-one F1, timing, Dice | MUST | RUNNING | production watcher computes once and reuses by evaluator/data digest |
| ED-M3-S42 | M3 | backfill completed seed-42 checkpoints | all compatible s42 models from SQLite catalog | all external datasets | per-record and aggregate delineation | MUST | RUNNING | one checkpoint at a time after ceiling |
| ED-M4-WATCH | M4 | follow newly archived checkpoints | DB/audit intersection, all seeds | all external datasets | same generation-bound contract | MUST | RUNNING | sleeps 1,200 s only when caught up |
| ED-M5-ANALYSIS | M5 | estimate external loss effects | 160 masks × seeds 42/200/201 | LUDB confirmatory; ISP/Zhejiang exploratory | seed-blocked effects/interactions | MUST | BLOCKED_ON_DATA | launch only after complete identity/evaluation gate |

