# Experiment Plan: Complete Factorial ECG Loss Benchmark v2

Version: 2026-07-24T17:33:00Z  
Status: active  
Authoritative queue: `experiment_queue/factorial_v2/manifest.json`

## Controlled design

The controlled registry contains a 3 × 16 grid over U-Net, MultiScale-VAE, and ECG-AIM (`ecgaim`, legacy registry kind `alitok`). Every family is trained at seed 42 with masks `e c m d = 0000` through `1111`. The prespecified poster analysis is the 24-cell `e=1` slice: base MSE is always enabled and the remaining bits form the required \(2^3\) correlation/MMD/derivative factorial. The `e=0` slice is retained as an explicitly supplemental exploratory MSE-toggle analysis; it is not allowed to redefine the primary contrast because U-Net `0000` has zero training loss and VAE base objectives are family-specific. Stable weights are MSE 1.0, correlation 0.5, MMD 0.1, and derivative 0.05.

After validation is locked, each active family repeats base (`1000`), full (`1111`), and its validation-selected best nontrivial `e=1` mask at seeds 1337 and 2026. This is 48 seed-42 registry runs plus 18 confirmation runs. Five separately labeled paper-parity U-Net anchors reproduce M0, full, and the three leave-one-out masks with the documented normalized objective.

All active checkpoints use the same selector: lowest missing-lead validation MSE, then highest missing-lead validation Pearson. Composite training loss and test results are forbidden selectors. Batch sizes are U-Net 256 and MultiScale-VAE/ECG-AIM 32; each active family trains for 10 epochs.

cNVAE is retained in the repository but excluded from the active poster grid after its gate baseline failed scientifically: the four analytic limb leads reached approximately R² 1 while V1/V3/V4/V5/V6 each remained approximately R² 0, producing aggregate Pearson 0.4446 and R² 0.4433. A training-split-only linear sanity baseline from I/II/V2 achieved validation chest-lead Pearson 0.7716, localizing the failure to the current cNVAE path rather than the data. The machine-readable decision is `results/factorial_v2/exclusions/cnvae.json`.

Every registry entry and checkpoint records seed, mask, exact weights, preprocessing, optimizer, scheduler, selector, split inventories, Git revision, and SHA-256 hashes of the exact architecture/loss sources. Model-only checkpoints are the default.

## Gates and phase graph

The generated active queue contains 133 jobs across 18 phases:

1. storage/data/dependency preflight;
2. focused loss, inference, robustness, statistics, and EchoNext tests;
3. generated protocol/registry/source-hash audit;
4. 48-cell active adapter smoke matrix;
5. 48 active registry cells;
6. five paper-parity anchors;
7. primary validation evaluation;
8. validation-only mask selection;
9. 18 confirmation runs with full validation evaluation;
10. leakage-free five-superclass ECG-FM parity-head training;
11. paper-parity anchor evaluation;
12. clean 48-cell PTB-XL evaluation;
13. full 17-condition robustness evaluation;
14. factorial statistics and interim figures;
15. EchoNext acquisition/provenance gate;
16. 48-cell EchoNext evaluation;
17. clean zero-shot PhysioNet smartwatch evaluation;
18. final completeness gate and poster artifacts.

Preflight requires at least 12 GiB free, canonical train/validation/test counts of 17,418/2,183/2,198, zero ECG-ID overlap, ECGFounder assets, and NSTDB BW/EM/MA records. Duplicate PTB-XL trees were byte-inventory verified before consolidation.

## Evaluation protocol

All 48 active registry cells are evaluated on all 2,198 PTB-XL fold-10 records. Primary poster claims and marginal effects use the 24 `e=1` cells. Metrics cover the nine reconstructed leads: MSE, RMSE, MAE, R², Pearson, SNR, derivative MSE, and per-lead metrics. Observed leads I, II, and V2 must be preserved exactly.

Morphology uses reference-detected beats and standardized QRS [−50,+100] ms and ST [+120,+240] ms windows, J-point amplitude error, R-peak timing, regional correlations, diagnostic-class strata, and spectral preservation at 0.5–40, 40–100, and 100–150 Hz. Detector coverage is mandatory.

Diagnostic evaluation uses the frozen 150-task ECGFounder checkpoint for AUROC, AP, F1, sensitivity, specificity, ECE, Brier score, and sex/age gaps. Reconstructed and original-ECG probabilities are retained per record. The legacy five-superclass result is not trusted; a new frozen ECG-FM head is trained using canonical train/validation data only and evaluated as a separate parity endpoint.

Robustness uses identical deterministic samples for every model: Gaussian at 24/12/6/0 dB, synthetic baseline drift, and NSTDB BW/EM/MA at 24/12/6/0 dB. All 2,198 records are used. Fitbit 20 dB, 10 dB, and baseline-wander conditions are supported only after a source-record, extraction-method, artifact-hash provenance bundle passes validation. The local smartwatch simulator tree is not accepted as that evidence.

EchoNext remains mandatory for final poster completion. Its gate requires source/version, exact lead order, units, 250 Hz sampling, and invertible normalization provenance. Resampling is polyphase 250→500 Hz; per-record min–max normalization is prohibited. All 5,442 test records receive clean evaluation, while morphology and the deterministic 17-condition stress suite use 2,198 records per model. The prior 512-stress artifact is labeled as a pilot and cannot satisfy the final gate.

The PhysioNet smartwatch benchmark is clean, single-lead-to-12-lead zero-shot evaluation only. A source audit found that the legacy evaluator labeled repeated clean inference as 17 stress conditions; those duplicate labels were removed. No smartwatch robustness claim is permitted until noise is actually applied and validated.

## Statistics and artifacts

Paired record-level effects use 2,000 BCa resamples. Primary factorial outputs use the `e=1` slice and include the three marginal effects, all pairwise interactions, and the three-way interaction within each family. A separate supplemental table reports all four main effects, six pairwise interactions, four three-way interactions, and the four-way interaction from the full 16-cell family grid. Full-vs-base endpoint tests compare `1111` with `1000`. ECGFounder non-inferiority uses a 0.02 AUROC margin and paired record-level AUROC influence-function BCa intervals. QRS, ST, and diagnostic utility retain family-wise α=0.0167.

Outputs live under `results/factorial_v2`: strict JSON, per-record Parquet, main and supplementary tables, interaction/bootstrap tables, provenance, completeness reports, and poster figures. Required figures are the factorial heatmaps, marginal-effect forest plot, morphology/diagnostic Pareto plot, NSTDB degradation curves, and EchoNext heatmap plus representative reconstructions.

No final poster artifact may claim completeness unless it verifies 48/48 registry cells, the complete 24-cell primary `e=1` slice, 9/9 family-slot confirmation configurations (18 seed-specific runs), 2,198 PTB-XL clean and stress records per cell, exact core 17-condition coverage, complete ECGFounder/morphology outputs, strict serialization, 48 EchoNext cells with 5,442 clean and 2,198 morphology/stress records, and plotted values derived from the current machine-readable tables. cNVAE is reported separately as an excluded collapsed pilot, never as a robustness result.
