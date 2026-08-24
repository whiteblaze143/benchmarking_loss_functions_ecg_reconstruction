# Experiment Tracker: factorial_v2

Updated: 2026-07-25T00:50:29Z  
Queue session: finished and closed (`factorial_v2_queue`)  
State: `experiment_queue/factorial_v2/queue_state.json`

## Current state

- Active design: 48 seed-42 registry cells across U-Net, MultiScale-VAE, and ECG-AIM. The prespecified poster analysis is the 24-cell MSE-on \(2^3\) correlation/MMD/derivative slice; the MSE-off slice is supplemental. There are 18 confirmation runs and five paper-parity anchors.
- Queue definition: all 133 jobs and all 18 dependency-gated phases completed on one A100 GPU.
- Preflight: passed with canonical 17,418/2,183/2,198 splits and zero overlap.
- Focused tests: 37/37 pass locally, including all 16 gradient masks, primary and supplemental factorial contrasts, paired statistics, deterministic stress pairing, streamed EchoNext preprocessing, BCa edge cases, and stale-output/dependency queue regressions.
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
- EchoNext: provenance gate passed and the corrected run completed 48/48 models: U-Net 16/16, MultiScale-VAE 16/16, and ECG-AIM 16/16. Each cell has 5,442 clean records plus 2,198 morphology/stress records and all 17 deterministic stress conditions. U-Net used batch 64; MultiScale-VAE/ECG-AIM used batch 32 with at most six grouped stress conditions, reaching 100% utilization at about 18 GiB without OOM.
- Smartwatch: 48-model clean zero-shot output retained. 3,264 invalid legacy “stress” entries were removed after source audit showed no noise was applied.
- Final analysis: the finalizer was rerun directly after the scheduler incorrectly accepted a stale report. It exited 0, regenerated all tables and poster plots, and produced a 27/27 complete machine-readable coverage gate. All source hashes in `figure_provenance.json` independently match.
- Storage: old W&B binary bundles and superseded pilot weights were removed; the duplicate legacy PTB-XL path was hash-verified before removal. A 400,569,587-byte migrated `bridge.pt` worktree duplicate was SHA-256 verified and hard-linked in place, preserving both paths while reclaiming one physical copy. Canonical data and active/confirmation checkpoints remain. Free disk is 4.1 GiB; the remaining dominant consumers are protected Git history, canonical datasets, model checkpoints, and active environments.

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
- Queue completion detection now ignores pre-existing outputs while a process is live, removes stale exit-code sentinels before launch, requires a fresh expected output from the current attempt, and blocks downstream phases when a dependency is stuck.
- Strict JSON parsing passes for the result set, registry, and queue state; all recorded values are finite.

Timestamped copies are historical snapshots. This fixed tracker is updated as the detached queue advances.
