# Experiment Tracker: factorial_v2

Updated: 2026-07-24T17:33:00Z  
Queue session: `factorial_v2_queue`  
State: `experiment_queue/factorial_v2/queue_state.json`

## Current state

- Active design: 48 seed-42 registry cells across U-Net, MultiScale-VAE, and ECG-AIM. The prespecified poster analysis is the 24-cell MSE-on \(2^3\) correlation/MMD/derivative slice; the MSE-off slice is supplemental. There are 18 confirmation runs and five paper-parity anchors.
- Queue definition: 133 jobs, 18 dependency-gated phases, one A100 GPU. Only corrected EchoNext evaluation and finalization remain.
- Preflight: passed with canonical 17,418/2,183/2,198 splits and zero overlap.
- Focused tests: 34/34 pass locally, including all 16 gradient masks, primary and supplemental factorial contrasts, paired statistics, deterministic stress pairing, streamed EchoNext preprocessing, and BCa edge cases.
- Generated protocol audit: the 48-cell grid, exclusion, dependency, throughput, split, and source-hash checks passed before launch.
- Smoke matrix: all 48/48 registry cells passed. The cNVAE fixtures are excluded from the scientific grid.
- Unified adapter invariant: passed cells produce finite `[1,12,5000]` and exact observed-lead MSE 0.
- cNVAE: excluded and deferred. At epoch 2, the four derived limb leads were approximately R² 1 while V1/V3/V4/V5/V6 were approximately R² 0; aggregate Pearson 0.4446 and R² 0.4433 fail the gate. A train-only linear I/II/V2 sanity model reached chest-lead validation Pearson 0.7716, confirming an architecture/optimization failure rather than absent signal. Evidence is preserved in `results/factorial_v2/exclusions/cnvae.json`.
- Primary/confirmation training: 48/48 seed-42 registry cells and 18/18 confirmation runs completed.
- PTB-XL: 48/48 cells completed on 2,198 clean records and all 17 deterministic stress conditions; per-record signal, morphology, ECGFounder, fairness, and paper-parity artifacts are present.
- Statistics: primary 3-factor and supplemental 4-factor analyses implemented; 48 ECGFounder non-inferiority rows and nine full-vs-base endpoint tests complete. Final rerun uses `1111` versus `1000`.
- ECGFounder checkpoint: strict-load verified, 150 tasks; per-record reference probabilities are part of the evaluator.
- Five-superclass parity head: trained and evaluated without test leakage.
- NSTDB: BW/EM/MA available; deterministic core 17-condition tests pass.
- Fitbit-derived noise: blocked from inclusion because the original `Data.csv` and extraction provenance/hash bundle are absent. Smartwatch simulator files are not substituted.
- EchoNext: provenance gate passed. The corrected 48-model run is active with 5,442 clean records and 2,198 morphology/stress records required per model. Batch 64 with all 17 conditions grouped uses about 8.9 GiB and has observed 87% utilization.
- Smartwatch: 48-model clean zero-shot output retained. 3,264 invalid legacy “stress” entries were removed after source audit showed no noise was applied.
- Storage: old W&B binary bundles and superseded pilot weights were removed; the duplicate legacy PTB-XL path was hash-verified before removal. Canonical data and active/confirmation checkpoints remain.

## Implementation checkpoints

- Factorial loss routing and paper-parity objective implemented.
- Canonical cNVAE split/scaling and objective-weight bugs repaired.
- Neutral checkpoint selection implemented for all active families; cNVAE support remains available for later rework.
- Exact source SHA-256 provenance added to registries/checkpoints.
- Clean, 17-condition robustness, morphology, ECGFounder, fairness, and per-record outputs implemented.
- Leakage-free five-superclass parity classifier implemented.
- Confirmation jobs now evaluate all 2,183 validation records.
- ECGFounder AUROC non-inferiority, confirmation tables, and EchoNext poster figures implemented.
- Generated queue state synchronization and selective requeue utilities implemented.
- Queue state repaired against verified artifacts; stale ad-hoc tmux sessions were removed.

Timestamped copies are historical snapshots. This fixed tracker is updated as the detached queue advances.
