# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| NATIVE-PF | M0 | causal gate | A0/C1/C2 real-record preflight | fold 9, 32 records | identity, hashes, gradients, shape, target independence | MUST | PASS | Exact production batch; no synthetic fallback |
| PANO-MAP | M0 | channel/query provenance | official PanoBench release and Nef-Net v2 code | 48 stored channels | channel identities, formulas, angle tables | MUST | PASS_WITH_QUALIFIER | Channels 0-43 are I, II, and 42 BSPM views; 44-47 are exact derived III/aVR/aVL/aVF. Standard V1-V6 are separate continuous query coordinates, not a 1:1 six-channel subset. Claim only continuous geometric transfer. |
| C0 | M1 | matched continuation | `C0_A0_NATIVE_CONTROL` | folds 1-8 / fold 9 | missing11, chest, II, p05, loss | MUST | COMPLETE | Three epochs complete; fold-9 missing11 Pearson 0.746395, p05 0.403235 |
| C1 | M2 | native Angle+View | `C1_A0_NATIVE_AE_VE` | folds 1-8 / fold 9 | same | MUST | HALTED_HARDWARE | CUDA peer-memory/hardware error at epoch 1 batch 155/545; no optimizer-complete checkpoint; tmux exited |
| C2 | M2 | incremental GeoVT | `C2_A0_NATIVE_FULL` | folds 1-8 / fold 9 | same | MUST | BLOCKED_ON_C1 | Not started; do not relaunch until GPU ECC/hardware health is remediated |
| PAIR | M3 | promotion decision | C1-C0, C2-C0, C2-C1 | fold 9 | paired bootstrap and tail safeguards | MUST | BLOCKED_ON_RUNS | Fold 10 sealed |
| FROZEN-CONTROL | M3 | transfer diagnosis | completed H0/H1/H2 | fold 9 | compatible endpoints | NICE | COMPLETE | Preserve unchanged |
