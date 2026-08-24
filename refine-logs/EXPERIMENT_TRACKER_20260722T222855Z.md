# Experiment Tracker: factorial_v2

Snapshot: 2026-07-22T22:28:55Z  
Queue session: `factorial_v2_queue`  
State: `experiment_queue/factorial_v2/queue_state.json`

## Current state

- Generated design: 32 primary cells, 24 confirmation runs, five paper-parity anchors.
- Queue definition: 107 jobs, 20 dependency-gated phases, one A100 GPU.
- Preflight: passed with canonical 17,418/2,183/2,198 splits and zero overlap.
- Loss tests: 20 focused tests pass locally; the queue’s 14 mask/gradient tests passed.
- Smoke matrix: U-Net 8/8 passed; corrected cNVAE 8/8 passed; MultiScale-VAE is in progress; ECG-AIM is pending.
- Unified adapter invariant: passed cells produce finite `[1,12,5000]` and exact observed-lead MSE 0.
- cNVAE full validity gate: pending. Smoke validation is approximately Pearson 0.444 and R² 0.382, so the 30-epoch baseline must improve materially or architecture diagnosis is required.
- Primary/confirmation training: pending smoke and cNVAE gates.
- Clean, robustness, statistics, and poster evaluation: pending training.
- ECGFounder checkpoint: strict-load verified, 150 tasks; per-record reference probabilities are part of the evaluator.
- Five-superclass parity head: implementation complete; training pending.
- NSTDB: BW/EM/MA available; deterministic core 17-condition tests pass.
- Fitbit-derived noise: blocked from inclusion because the original `Data.csv` and extraction provenance/hash bundle are absent. Smartwatch simulator files are not substituted.
- EchoNext: blocked on acquisition and `data/echonext/PROVENANCE.json`; final completion remains impossible until supplied.
- Storage: near the 12 GiB minimum and monitored before every major phase.
- GPU: A100 40 GiB; U-Net reached 99% utilization in smoke profiling. Per-family batch sizes remain the memory-safe planned values.

## Implementation checkpoints

- Factorial loss routing and paper-parity objective implemented.
- Canonical cNVAE split/scaling and objective-weight bugs repaired.
- Neutral checkpoint selection implemented for all four families.
- Exact source SHA-256 provenance added to registries/checkpoints.
- Clean, 17-condition robustness, morphology, ECGFounder, fairness, and per-record outputs implemented.
- Leakage-free five-superclass parity classifier implemented.
- Confirmation jobs now evaluate all 2,183 validation records.
- ECGFounder AUROC non-inferiority, confirmation tables, and EchoNext poster figures implemented.
- Generated queue state synchronization and selective requeue utilities implemented.

This snapshot is historical. The fixed tracker is updated as the detached queue advances.

