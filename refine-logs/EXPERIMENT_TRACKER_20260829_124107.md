# Experiment Tracker: Current Evidence and Next Gates

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ONE-SCREEN-1110000 | M0 | short mechanism screen | 120 one-lead ECG-AIM cells | PTB-XL val | full 29-field summary | MUST | TERMINAL 115/120 | 2 OOM, 2 transient failures, 1 stuck |
| ONE-SPATIAL-LEGACY | M0 | spatial mechanism screen | 10 variants × 3 masks × 2 leads | PTB-XL val | loss, Pearson | MUST | COMPLETE 60/60 | 24 cancelled queue entries excluded |
| ONE-RDB-SCREENED | M0 | external spatial checkpoint evaluation | 60 models + sentinels, pilot/full | RDB | boundary, region, signal | MUST | COMPLETE | analyze `stage=full`; no peak endpoints |
| ONE-CONV-10E | M0 | longer historical convergence | promoted one-lead variants | PTB-XL val | full 29-field summary | MUST | COMPLETE 23 summaries | do not pool with 3e or 15e |
| ONE-CONV-15E | M1 | final promoted convergence | 22 one-lead jobs | PTB-XL val | full 29-field summary | MUST | ACTIVE | audit snapshot: 8 complete, 1 running, 13 pending |
| ONE-1111002 | M3 | repaired-mask replication | 120 registered jobs | PTB-XL/RDB-derived supervision | reconstruction + six boundaries | MUST | PENDING 120/120 | narrow to survivors if protocol permits |
| ONE-SPATIAL-V1 | M3 | expanded architecture design | 48 registered jobs | PTB-XL val | reconstruction/delineation | NICE | PENDING 48/48 | distinct from completed legacy spatial study |
| THREE-UNET-7MASK | M5 | seven-mask baseline | U-Net seed42 | PTB-XL | validation metrics | MUST | COMPLETE 160/160 | current-contract compatible |
| THREE-MSVAE-7MASK | M5 | cross-architecture parity | MSVAE seed42 | PTB-XL | metrics absent from catalog | MUST | PARTIAL 155/160 | archived generation incompatible with current contract |
| THREE-ECGAIM-7MASK | M5 | novel architecture parity | ECG-AIM seed42 | PTB-XL | metrics absent from catalog | MUST | ACTIVE 123/160 registered | queue audit: 1 running, 34 pending, 2 failed |
| CLINICAL-MISSING-V2 | M5 | biomarker preservation | U-Net/MSVAE/ECG-AIM | PTB-XL | clinical endpoint suite | MUST | PARTIAL 160/37/0 | no ECG-AIM clinical rows |
| THREE-LUDB-BLIND | completed | external delineation | 123 ECG-AIM + original | LUDB | signal + six boundaries | MUST | COMPLETE 124/124 | terminal evaluator |
| THREE-RDB-BLIND | completed | promoted blinded delineation | 31 ECG-AIM + original | RDB | signal + six boundaries/lead group | MUST | COMPLETE 32/32 | terminal evaluator |
| THREE-RDB-ORACLE | completed | rich oracle morphology | 31 ECG-AIM | RDB | waveform, boundary, ST, rhythm strata | MUST | COMPLETE 31/31 | 2,398 records |
| THREE-RDB-EMBED | completed | representation diagnostics | 11 ECG-AIM checkpoints | RDB | probes, similarity, UMAP stability | SUPPORT | COMPLETE 11/11 | two all-split anchors; nine test-only |
| LUDB-SEMISEG | completed | delineator training/evaluation | student/EMA variants | LUDB | semiseg evaluation schema | SUPPORT | COMPLETE | 9 evaluation + 7 training/eval rows |

