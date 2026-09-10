# Experiment Tracker: Predictable Residual Subspace Completion

| Run ID | Milestone | Purpose | System | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| PRS-R001 | M0 | Residual rank curve | PCA oracle k=1..6 | train folds1-8 / val fold9 | MUST | READY | New protocol |
| PRS-R002 | M1 | Geometry/head tests | B1/C1/C2 | synthetic + one batch | MUST | BLOCKED | Wait for K_STAR |
| PRS-R003 | M2 | Direct baseline | B1 seed42 15e | folds1-8/9 | MUST | BLOCKED | Wait for R001/R002 |
| PRS-R004 | M2 | Frozen PCA | C1 seed42 15e | folds1-8/9 | MUST | BLOCKED | Wait for R001/R002 |
| PRS-R005 | M2 | Predictive subspace | C2 seed42 15e | folds1-8/9 | MUST | BLOCKED | Wait for R001/R002 |
| PRS-R006 | M3 | Paired/clinical/latent evaluation | B1/C1/C2 | fold9 | MUST | BLOCKED | Same selected checkpoints |
| PRS-R007 | M4 | Replication | advancing systems seeds43/44 | folds1-8/9 | CONDITIONAL | BLOCKED | Requires R006 |
| PRS-R008 | M5 | Confirmation | final frozen model | fold10/EchoNext | DEFERRED | SEALED | Not current phase |

Frozen gates: FIXED_VCG_SPECIFICITY=NOT_SUPPORTED; dependent-limb sensitivity excludes source residual >0.01 mV; advance requires delta mean independent r >=0.005, paired CI95 low >0, and all harm margins.

