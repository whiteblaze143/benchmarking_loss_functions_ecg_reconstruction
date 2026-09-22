# Experiment Tracker

| Run ID | Milestone | System / task | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|
| P07F12-0 | P0 | Real 12-lead target admission | PTB-XL folds 1–7/8 | MUST | TODO | No fold 9/10 or synthetic target. |
| P07F12-1 | P1 | `continuous_full12_aux_mmd` | PTB-XL train/select | MUST | QUEUED_AFTER_BRAID | BCE + IMQ MMD² + 0.1 waveform MSE. |
| P07F12-2 | P2 | `continuous_full12_aux_no_mmd` | PTB-XL train/select | MUST | PENDING | Required before MMD attribution. |
| P07F12-3 | P3 | Frozen Benchmark-2 evaluation | Native cohort splits | MUST | PENDING | Same task-native prediction bundles. |
