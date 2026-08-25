# One-Lead RDB Experiment Tracker

| Run ID | Milestone | Purpose | Population | Split | Priority | Status | Output |
|---|---|---|---|---|---|---|---|
| `RDB1-M0` | M0 | strict loader and scoring smoke | one frozen one-lead checkpoint | 2 held-out records | MUST | PASS | compact schema verified: 1 evaluation, 24 boundary, 12 region, 4 signal rows |
| `RDB1-M1` | M1 | external delineator ceiling | original RDB waveform | 360 test | MUST | PASS | both input-lead groupings complete; primary F1 = 0.7356; 141.6–141.7 s each |
| `RDB1-M2-I` | M2 | calibrate pilot-to-full gate | prespecified Lead-I anchors | 48 + 360 test | MUST | RUNNING | A0 complete: full F1 = 0.6452, PCC p05 = 0.3455; E1 pilot F1 = 0.6221, PCC p05 = 0.3281; E1 full running |
| `RDB1-M2-II` | M2 | calibrate input-lead control gate | matched Lead-II anchors | 48 + 360 test | MUST | QUEUED | pilot/full aggregates |
| `RDB1-M3` | M3 | fit and leave-one-out audit gate | completed anchors | aggregate DB | MUST | TODO | frozen threshold record |
| `RDB1-M4` | M4 | blinded pilot/promotion | remaining frozen 60 | 48 then 360 test | MUST | CHAINED | terminal states |
| `RDB1-M5` | M5 | paired analysis and failure audit | all terminal models | aggregate DB | MUST | TODO | book-ready tables/figures |

Terminal-state accounting target: 60 models total = 30 Lead I primary + 30 Lead II controls. A pruned model is explicitly censored and is not assigned a numerical full-cohort score.

Detached runner: tmux session `onelead_rdb_calibration`. It waits for load <= 3.0, available RAM >= 7 GiB, and free disk >= 8 GiB, then runs calibration followed by the screened 60-model phase with six PyTorch CPU threads.
