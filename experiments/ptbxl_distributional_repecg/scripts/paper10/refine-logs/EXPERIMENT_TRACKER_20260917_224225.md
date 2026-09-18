# Experiment Tracker: Paper 10

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| P10-R001 | M0 | build/audit paired intervention views | raw transform bank | folds 1-8, no fitting beyond 1-6 | pair completeness, label equality, valid-view denominator | MUST | TODO | prerequisite for every training run |
| P10-R002 | M1 | identify the paired objective | synthetic `S+U_e` | held-out synthetic states | state drift, label error, environment probe | MUST | TODO | include impossible `Y<-e` world |
| P10-R003 | M1 | objective execution | ERM/IRMv1/CORAL/full | synthetic + small real smoke | finite/nonzero distinct terms | MUST | TODO | `causirl` removed |
| P10-R004 | M2 | select full and comparator settings | ERM/IRMv1/CORAL/full | train 1-6, select 7 | robust AUROC, paired stability | MUST | TODO | no fold 8 access |
| P10-R005 | M3 | frozen development comparison | selected systems, 3 seeds | fold 7 decision protocol | patient-equal bootstrap endpoints | MUST | TODO | apply claim-gate matrix |
| P10-R006 | M3 | boundary/collapse audit | full | held-out paired records | `Z_S` rank, `Z_A/Z_S` environment probes | MUST | TODO | restrict final claim to bank |
