# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R000 | M0 | metadata and split audit | shared harness | folds 1--10 metadata only | MUST | TODO | signal access excludes fold 10 |
| R001 | M0 | preprocessing/phase invariants | shared harness | folds 1--7 | MUST | TODO | emit all-row QC |
| R002 | M0 | exact/Nyström witness | shared kernel | folds 1--7/8 | MUST | TODO | 128 then 256 if required |
| R010 | M1 | raw smoke | 8/12-lead ResNet | 128-record dev | MUST | TODO | measure VRAM and throughput |
| R011 | M1 | KME smoke | Paper 2 + moments | 128-record dev | MUST | TODO | identical phase CNN |
| R020 | M2 | raw development | shared baselines | 1--7/8 seed 42 | MUST | TODO | no fold 9/10 |
| R021 | M2 | KME development gate | Paper 2 | 1--7/8 seed 42 | MUST | TODO | terminal PASS/FAIL/INCONCLUSIVE |
| R022 | M2 | recurrence development gate | Paper 1 | 1--7/8 seed 42 | MUST | TODO | direct operator primary |
| R030 | M3 | path gate | Paper 3 | 1--7/8 seed 42 | MUST | TODO | install iisignature in isolated env spec |
| R031 | M3 | Hankel gate | Paper 4 | 1--7/8 seed 42 | MUST | TODO | local dynamics |
| R032 | M3 | Koopman gate | Paper 5 | 1--7/8 seed 42 | MUST | TODO | record-specific eligibility |
| R033 | M3 | conditional gate | Paper 6 | 1--7/8 seed 42 | MUST | TODO | four mandatory models |
| R040 | M4 | operator cost/gate | Paper 7 | 1--7/8 seed 42 | MUST | TODO | separate task/auxiliary pairs |
| R041 | M4 | token cost/gate | Paper 8 | 1--7/8 seed 42 | MUST | TODO | simultaneous pair bounds |
| R070 | M5_Causal | counterfactual cost/gate | Paper 9 | 1--7/8 seed 42 | MUST | TODO | randomized counterfactual angle |
| R071 | M5_Causal | interventional gate | Paper 10 | 1--7/8 seed 42 | MUST | TODO | sham intervention stability |
| R072 | M5_Causal | causal-state gate | Paper 11 | 1--7/8 seed 42 | MUST | TODO | forward temporal linkage |
| R073 | M5_Causal | innovation gate | Paper 12 | 1--7/8 seed 42 | MUST | TODO | causal ordering vs residual |
| R074 | M5_Causal | surgery cost/gate | Paper 13 | 1--7/8 seed 42 | MUST | TODO | surgical masking stability |
| R075 | M5_Causal | IRM gate | Paper 14 | 1--7/8 seed 42 | MUST | TODO | out-of-domain alignment |
| R076 | M5_Causal | factorization gate | Paper 15 | 1--7/8 seed 42 | MUST | TODO | independent mechanism query |
| R080 | M5 | final training | gate-passing branches | 1--8/9 seeds 42--46 | MUST | BLOCKED | blocked on branch gate/freeze |
| R090 | M6 | locked evaluation | frozen final models | fold 10 | MUST | BLOCKED | forbidden until freeze manifest |
| R100 | M7 | multi-dataset OOD | frozen final models | 8 external datasets | MUST | BLOCKED | forbidden until freeze manifest |

