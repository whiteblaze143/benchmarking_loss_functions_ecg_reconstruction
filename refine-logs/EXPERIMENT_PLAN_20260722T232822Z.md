# Experiment Plan Snapshot: factorial_v2 Active Three-Family Grid

Version: 2026-07-22T23:28:22Z  
Status: active, immutable snapshot  
Authoritative queue: `experiment_queue/factorial_v2/manifest.json`

## Controlled design

The active poster benchmark is a complete 3 × 8 factorial over U-Net,
MultiScale-VAE, and ECG-AIM. The three bits enable correlation, MMD, and
derivative losses with weights 0.5, 0.1, and 0.05. All 24 primary cells use
seed 42. After validation-only selection, base, full, and the best nontrivial
mask per family are repeated at seeds 1337 and 2026, giving 18 confirmation
runs and 42 active controlled runs in total.

Every checkpoint is selected by lowest missing-lead validation MSE, with
missing-lead Pearson as the tie-breaker. U-Net uses batch 256; MultiScale-VAE
and ECG-AIM use batch 32. All train for ten epochs. Five separately labeled
paper-parity U-Net anchors reproduce M0, full, and the three leave-one-out
objectives using the paper-normalized definitions.

cNVAE remains implemented but is excluded from the active grid. Its gate run
reached aggregate Pearson 0.4446 and R² 0.4433 because four analytically
derived limb leads were nearly exact while all five missing chest leads stayed
near R² 0. A train-only linear I/II/V2 sanity model reached validation chest
Pearson 0.7716, localizing the failure to cNVAE. Evidence is stored in
`results/factorial_v2/exclusions/cnvae.json`.

## Queue and evaluation

The generated queue contains 84 jobs in 17 dependency-gated phases: preflight,
tests, protocol audit, 24 smoke cells, 24 primary runs, five paper anchors,
validation selection, 18 confirmations, parity-classifier training, clean and
robust PTB-XL evaluation, statistics, EchoNext, and final completeness.

Every active primary cell is evaluated on all 2,198 PTB-XL test records for
missing-lead signal metrics, per-lead metrics, QRS/ST/J-point/R-peak morphology,
regional and class-stratified morphology, three spectral bands, frozen
150-task ECGFounder utility/calibration/fairness, and the deterministic core
17 stress conditions. Fitbit-derived noise is admitted only with validated
source and extraction provenance.

Statistics use 2,000 paired record-level BCa resamples, all factorial main and
interaction effects, ECGFounder AUROC non-inferiority margin 0.02, and
α=0.0167 for QRS, ST, and diagnostic utility. Poster figures are gated on
machine-readable coverage and provenance checks.

EchoNext remains required for final poster completion and must pass its lead
order, units, 250 Hz sampling, invertible normalization, and acquisition gates.
Resampling is polyphase 250→500 Hz; per-record min–max normalization is banned.

## Completion contract

Completion requires 24/24 active primary cells, 9/9 family-slot confirmation
configurations (18 runs), 2,198 clean and stress records per cell, exact
17-condition coverage, complete morphology and ECGFounder outputs, five paper
anchors, 24 EchoNext cells, strict JSON/Parquet artifacts, and figures whose
numbers hash back to the current machine-readable tables. cNVAE is reported as
an excluded collapsed pilot rather than a robustness result.
