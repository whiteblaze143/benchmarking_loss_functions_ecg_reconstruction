# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | checkpoint audit | U-Net seven-mask s42 | catalog | coverage/provenance | MUST | DONE | 160/160 |
| R002 | M0 | checkpoint audit | MSVAE seven-mask s42 | catalog | coverage/provenance | MUST | DONE | 155/160 |
| R003 | M0 | checkpoint audit | ECG-AIM seven-mask s42 | catalog | coverage/provenance | MUST | DONE | 123/160 |
| R004 | M0 | clinical audit | `missing_leads_v2` U-Net | PTB-XL | model/endpoint coverage | MUST | DONE | 160 models × 56 targets |
| R005 | M0 | clinical audit | `missing_leads_v2` MSVAE | PTB-XL | model/endpoint coverage | MUST | PARTIAL | 37 models × 56 targets |
| R006 | M0 | clinical audit | `missing_leads_v2` ECG-AIM | PTB-XL | model/endpoint coverage | MUST | BLOCKED | zero rows; evaluation absent |
| R007 | M1 | finish grid | five missing MSVAE cells | validation | finite/provenance | MUST | TODO | do not impute failures |
| R008 | M1 | finish grid | 37 missing ECG-AIM cells | validation | finite/provenance | MUST | TODO | preserve source contract |
| R009 | M2 | clinical completion | remaining 123 MSVAE | test | v2 endpoints | MUST | TODO | same evaluator/version |
| R010 | M2 | clinical completion | 160 ECG-AIM | test | v2 endpoints | MUST | TODO | zero presently evaluated |
| R011 | M3 | paired factorial | eligible architectures | test | reconstruction/clinical | MUST | TODO | digest-keyed ledger |
| R012 | M4 | spatial replication | A0/spatial/permuted | lead I/II | matched deltas/cost | MUST | TODO | ≥3 seeds promoted contrasts |
| R013 | M4 | wavelet increment | A0→wavelet→SSL | lead I/II | recon + six boundaries | MUST | RUNNING | queue is screening |
| R014 | M4 | physiological view | Morlet vs UEG phase | RDB development | T-on/T-off | MUST | RUNNING | matched compute only |
| R015 | M5 | external morphology | promoted models | blinded LUDB/RDB | boundary timing/IoU | MUST | TODO | no peak invention |
| R016 | M5 | robustness appendix | Pareto set | noise/subgroups | degradation | NICE | TODO | after core gates |

