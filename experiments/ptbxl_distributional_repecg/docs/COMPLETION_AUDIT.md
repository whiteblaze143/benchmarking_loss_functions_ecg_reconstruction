# Completion Audit

Status: **INCOMPLETE**. This file is a requirement-to-evidence ledger, not a
claim that the program is finished.

| Requirement | Current evidence | State |
|---|---|---|
| New isolated experiment directory | this directory and `README.md` | proven |
| Research-refine pipeline | four review rounds; final score 9.24 and READY in `refine-logs/` | proven |
| Official PTB-XL patient-disjoint folds | real metadata test and cohort audit | proven |
| Fold-10 signal firewall | `PTBXLStore` plus failing-access test | proven for current API |
| Physical/model signal separation | production folds-1--8 Paper-2 cache | proven for Paper 2 |
| Train-only per-lead scaling | 15,245 records and 76,225,000 folds-1--7 samples | proven for Paper 2 development |
| Beat detection and phase normalization | folds 1--7: 15,244/15,245 eligible; fold 8: 2,173/2,173 | proven for Paper 2 development |
| Exact IMQ and Nyström implementation | 1,000-pair audit at 128 and 256 landmarks | 128 failed; 256 passes amended pre-training gate A001 |
| Mathematical cores for Papers 1--8 | branch modules and tests | partial; end-to-end stages missing |
| Independent paper directories | required stage files absent | missing |
| Shared raw baselines | absent | missing |
| Development gates on folds 1--7/8 | Paper-2 Nyström audit: 256 gives `rho=0.957`, median relative error `0.126` | approximation gate passes A001; model gate missing |
| Five-seed final training on folds 1--8/9 | blocked by development gates | missing |
| Locked fold-10 evaluation | correctly unopened | not yet eligible |
| Full-throughput GPU execution | only Paper-2 execution smoke completed | missing |
| Scientific results and interpretation | no eligible result table | missing |

Completion requires evidence for every missing row. Narrow smoke tests cannot
be used to promote a branch or unlock fold 10.
