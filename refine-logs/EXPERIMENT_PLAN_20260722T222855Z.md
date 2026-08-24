# Experiment Plan: Complete Factorial ECG Loss Benchmark v2

Version: 2026-07-22T22:28:55Z  
Status: active  
Authoritative queue: `experiment_queue/factorial_v2/manifest.json`

## Controlled design

The primary experiment is a 4 × 8 factorial grid over U-Net, cNVAE, MultiScale-VAE, and ECG-AIM (`ecgaim`, legacy registry kind `alitok`). Every family is trained at seed 42 with masks `000` through `111`, where the bits enable correlation, MMD, and derivative losses. Stable extended-grid weights are correlation 0.5, MMD 0.1, and derivative 0.05.

After validation is locked, each family repeats base (`000`), full (`111`), and its validation-selected best nontrivial mask at seeds 1337 and 2026. This is 32 primary plus 24 confirmation runs, or 56 controlled runs. Five separately labeled paper-parity U-Net anchors reproduce M0, full, and the three leave-one-out masks with the documented normalized objective.

All checkpoints use the same selector: lowest missing-lead validation MSE, then highest missing-lead validation Pearson. Composite training loss and test results are forbidden selectors. Batch sizes are U-Net 256, cNVAE 16, MultiScale-VAE 32, and ECG-AIM 32. U-Net, MultiScale-VAE, and ECG-AIM train for 10 epochs; cNVAE receives at most 30 epochs with validation early stopping.

Every registry entry and checkpoint records seed, mask, exact weights, preprocessing, optimizer, scheduler, selector, split inventories, Git revision, and SHA-256 hashes of the exact architecture/loss sources. Model-only checkpoints are the default.

## Gates and phase graph

The generated queue contains 107 jobs across 20 phases:

1. storage/data/dependency preflight;
2. eight-mask gradient-routing tests;
3. 32-cell adapter smoke matrix;
4. full cNVAE `000` baseline;
5. cNVAE validation evaluation;
6. cNVAE validity gate;
7. 24 non-cNVAE primary cells;
8. seven gated cNVAE primary cells;
9. five paper-parity anchors;
10. primary validation evaluation;
11. validation-only mask selection;
12. 24 confirmation runs with full validation evaluation;
13. leakage-free five-superclass ECG-FM parity-head training;
14. paper-parity anchor evaluation;
15. clean 32-cell PTB-XL evaluation;
16. full 17-condition robustness evaluation;
17. factorial statistics and interim figures;
18. EchoNext acquisition/provenance gate;
19. 32-cell EchoNext evaluation;
20. final completeness gate and poster artifacts.

The non-cNVAE grid can progress if the cNVAE gate fails, while all cNVAE masks beyond `000` remain blocked. The cNVAE gate requires finite inference, validation Pearson ≥0.70, R² ≥0.50, and at least one matched R peak. A failure triggers architecture/preprocessing diagnosis; it is never relabeled as robustness.

Preflight requires at least 12 GiB free, canonical train/validation/test counts of 17,418/2,183/2,198, zero ECG-ID overlap, ECGFounder assets, and NSTDB BW/EM/MA records. Duplicate PTB-XL trees were byte-inventory verified before consolidation.

## Evaluation protocol

All 32 primary cells are evaluated on all 2,198 PTB-XL fold-10 records. Primary metrics cover the nine reconstructed leads: MSE, RMSE, MAE, R², Pearson, SNR, derivative MSE, and per-lead metrics. Observed leads I, II, and V2 must be preserved exactly.

Morphology uses reference-detected beats and standardized QRS [−50,+100] ms and ST [+120,+240] ms windows, J-point amplitude error, R-peak timing, regional correlations, diagnostic-class strata, and spectral preservation at 0.5–40, 40–100, and 100–150 Hz. Detector coverage is mandatory.

Diagnostic evaluation uses the frozen 150-task ECGFounder checkpoint for AUROC, AP, F1, sensitivity, specificity, ECE, Brier score, and sex/age gaps. Reconstructed and original-ECG probabilities are retained per record. The legacy five-superclass result is not trusted; a new frozen ECG-FM head is trained using canonical train/validation data only and evaluated as a separate parity endpoint.

Robustness uses identical deterministic samples for every model: Gaussian at 24/12/6/0 dB, synthetic baseline drift, and NSTDB BW/EM/MA at 24/12/6/0 dB. All 2,198 records are used. Fitbit 20 dB, 10 dB, and baseline-wander conditions are supported only after a source-record, extraction-method, artifact-hash provenance bundle passes validation. The local smartwatch simulator tree is not accepted as that evidence.

EchoNext remains mandatory for final poster completion. Its gate requires source/version, exact lead order, units, 250 Hz sampling, and invertible normalization provenance. Resampling is polyphase 250→500 Hz; per-record min–max normalization is prohibited.

## Statistics and artifacts

Paired record-level effects use 2,000 BCa resamples. Factorial outputs include the three marginal effects, all pairwise interactions, and the three-way interaction within each family. ECGFounder non-inferiority uses a 0.02 AUROC margin and paired record-level AUROC influence-function BCa intervals. QRS, ST, and diagnostic utility retain family-wise α=0.0167.

Outputs live under `results/factorial_v2`: strict JSON, per-record Parquet, main and supplementary tables, interaction/bootstrap tables, provenance, completeness reports, and poster figures. Required figures are the factorial heatmaps, marginal-effect forest plot, morphology/diagnostic Pareto plot, NSTDB degradation curves, and EchoNext heatmap plus representative reconstructions.

No final poster artifact may claim completeness unless it verifies 32/32 primary cells, 12/12 family-slot confirmation configurations (24 seed-specific runs), 2,198 clean and stress records per cell, exact core 17-condition coverage, complete ECGFounder/morphology outputs, strict serialization, 32 EchoNext cells, and plotted values derived from the current machine-readable tables.

