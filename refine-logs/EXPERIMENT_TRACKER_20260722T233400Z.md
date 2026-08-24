# Experiment Tracker Snapshot: factorial_v2

Updated: 2026-07-22T23:34:00Z  
Queue session: `factorial_v2_queue`  
State: `experiment_queue/factorial_v2/queue_state.json`

## State at snapshot

- Active design: 24 primary cells, 18 confirmation runs, five paper-parity anchors.
- Generated queue: 84 jobs across 17 phases on one A100 40 GB GPU.
- Preflight passed: canonical 17,418/2,183/2,198 splits with zero overlap and at least 12 GiB free at launch.
- Focused suite passed: 23/23 tests covering all eight masks, statistics, deterministic stress pairing, and streamed EchoNext preprocessing.
- Revised generated-protocol audit passed all 13 active-grid, exclusion, dependency, source-hash, split, and throughput checks.
- Active smoke coverage passed 24/24 with finite `[1,12,5000]` output and exact observed-lead preservation.
- cNVAE was stopped and excluded after its gate diagnosis; evidence is in `results/factorial_v2/exclusions/cnvae.json`.
- U-Net `000` primary was active and its validation MSE improved from 0.0481 at epoch 1 to 0.0333 at epoch 3.
- U-Net training reached 97% GPU utilization at batch 256 while using about 7.7 GB.
- ECGFounder 150-task evaluation, the leakage-free five-superclass parity head, 17-condition robustness, paired statistics, and completeness-gated figures are implemented and queued.
- Fitbit-derived noise remains provenance-gated and excluded until the original extraction bundle is supplied.
- EchoNext remains acquisition/provenance-gated and is mandatory for final external-validation completion.

The fixed-name `EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md` continue to
track the live queue; this file preserves the decision-state snapshot.
