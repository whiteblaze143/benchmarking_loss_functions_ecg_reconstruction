# Experiment Tracker: Predictable Residual Subspace Completion

| Run ID | Milestone | Purpose | System | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| PRS-R001 | M0 | Residual rank curve | PCA oracle k=1..6 | train folds1-8 / val fold9 | MUST | COMPLETE | K_STAR=5; first rank >=90% energy |
| PRS-R002 | M1 | Geometry/head tests | B1/C1/C2 | synthetic + one batch | MUST | COMPLETE | 12 targeted tests; 3 GPU smokes; identical trunk hash |
| PRS-R003 | M2 | Direct baseline | B1 seed42 15e | folds1-8/9 | MUST | COMPLETE | 15/15e; best val r=0.812089 |
| PRS-R004 | M2 | Frozen PCA | C1 seed42 15e | folds1-8/9 | MUST | COMPLETE | 15/15e; best val r=0.806303 |
| PRS-R005 | M2 | Predictive subspace | C2 seed42 15e | folds1-8/9 | MUST | COMPLETE | 15/15e; best val r=0.808812 |
| PRS-R006 | M3 | Paired/clinical/latent evaluation | B1/C1/C2 | fold9 | MUST | COMPLETE | Evaluated; C1-B1 delta=-0.0058, C2-B1 delta=-0.0033, C2-C1 delta=+0.0025; gate failed (no advance) |
| PRS-R007 | M4 | Replication | advancing systems seeds43/44 | folds1-8/9 | CONDITIONAL | STOPPED | Gate rule triggered: neither C1 nor C2 advanced over B1 |
| PRS-R008 | M5 | Confirmation | final frozen model | fold10/EchoNext | DEFERRED | SEALED | Not current phase |

Frozen gates: FIXED_VCG_SPECIFICITY=NOT_SUPPORTED; dependent-limb sensitivity excludes source residual >0.01 mV; advance requires delta mean independent r >=0.005, paired CI95 low >0, and all harm margins.

## Pre-launch evidence

- Full rank curve: 17,418 training ECGs and all 2,183 fold-9 ECGs. K_STAR=5 (93.680985% training residual energy); rank-5 oracle mean/p05 independent r=0.986054/0.958022.
- Source QC automatically flagged exactly ECG 5930 and 15867. They remained in primary independent metrics.
- Production-matched B1/C1/C2 smoke shared-trunk SHA-256: `827f066b39f9467c7c601888452221eff12c40938aa8cda6c7f04cf256168e60`. (The earlier no-fiducial-head smoke used `99b26ca5e0a6ca9eb70bd8a8b4cfc98dc08832350a409c6a08c56d060289cfcf`; production retains the inherited fiducial head.)
- Output implementation: identical seven-signal proposal trunk; B1 identity, C1 fixed rank-5 projection, C2 QR-orthonormal learned rank-5 projection. This isolates output subspace without a decoder-architecture or parameter-budget confound.
- Inherited normalization audit on frozen T_patch10: global-target z-score mean r=0.822248; Lead-I-only=0.822290; raw=0.822289. Primary Pearson is insensitive here. Global z-score is retained for exact training comparability but is not considered deployment-clean for amplitude endpoints.
- Production queue launched 2026-09-10 after all gates. B1 epoch 1 advanced to 23/545 batches at first live check; A100 allocation 31,522 MiB; 7.8 GiB filesystem free. Runs are sequential to bound checkpoint/resume storage.
- B1 epoch 1 completed in 8m40s and checkpointed correctly: patient-weighted validation independent r=0.635632, p05=0.289388. Epoch 2 began immediately. This is an early training milestone, not a model comparison.
- Live check at 2026-09-10 20:55 EDT: B1 epoch 2 reached 323/545 batches (59%); GPU utilization 100%, 31,547/40,960 MiB used, and 7.4 GiB filesystem free. The extra trainer command lines are the two current and two retiring DataLoader worker children of PID 1499750, not duplicate queue launches.
- B1 epoch 2 completed in 8m38s and improved the checkpoint metric from 0.635632 to 0.727541. Its patient-tail p05 improved from 0.289388 to 0.348616 and validation reconstruction loss fell from 9.076115 to 6.813446. Secondary validation metrics also moved in the favorable direction (P/QRS/T IoU 0.060738/0.830535/0.481349; mIoU 0.457541). Epoch 3 started immediately. These within-run learning dynamics show healthy convergence but are not a B1/C1/C2 comparison.
- Live check at 2026-09-11 00:27 EDT: B1 completed 13/15 epochs and epoch 14 reached 89/545 batches. Epoch 13 is currently best by the frozen selection metric: patient-weighted independent r=0.811678, p05=0.526519, validation reconstruction loss=5.282073. Measured wall throughput is 17.42 minutes per completed epoch including validation/checkpointing; estimated B1 completion is about 00:59 EDT and all three sequential seed-42 trainings about 09:42 EDT if C1/C2 have comparable throughput.
- The frozen post-training evaluator is implemented and compiles. Synthetic gates verify paired mean/p05 bootstrap behavior and exact latent-statistic recovery. It will evaluate checkpoint hashes, full metrics, dependent-lead QC sensitivity, clinical harm, oracle-gap closure, C1 coordinate predictability, and C2-to-PCA principal angles after all three runs complete.
