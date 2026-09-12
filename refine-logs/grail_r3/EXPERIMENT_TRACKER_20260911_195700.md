# R3 Experiment Tracker

| Run ID | Purpose | Models | Status | Notes |
|---|---|---|---|---|
| R3-E | Frozen folds 1–8 export | UB128, B0, B1, 001, 101, 111 | RUNNING | tmux `grail_r3_audit`; batch 2048 |
| R3-L | Exact per-concept probes | same six | QUEUED | folds 1–6 fit, 7 tune, 8 test |
| R3-PCA | Clinical dimensionality | same six | QUEUED | PCA fit on 1–7 only |
| R3-LS | 20-repeat patient low-shot | same six | QUEUED | shared subsets; 5+/5− eligibility |
| R3-G | Geometry/residual/retrieval/bootstrap | B1, 001, 101, 111 | PLANNED | no Fold-8 alignment fitting |

