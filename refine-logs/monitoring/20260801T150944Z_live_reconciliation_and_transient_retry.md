# Live reconciliation and transient retry — 2026-08-01 15:09 UTC

- Queue: 33 completed, 1 running, 446 pending after requeueing one transient CUDA/NVLink abort, and 0 terminally stuck jobs.
- Historical-count correction: the recovery ledger proves that the earlier 131 rows labeled completed contained 74 exit-zero runs and 57 partial/nonzero generations. A separate current-log census found 235 exit-code-1 markers: 218 missing `ptbxl_database.csv`, 11 violating the earlier tensor-inventory contract, and 6 runtime-finalization failures. These episodes must not be conflated, and none of those nonzero attempts is a current-contract model completion.
- Failure diagnosis: `f_1000113_s42` trained through epoch 7 and aborted during epoch 8 with exit 134 and `Invalid access of peer GPU memory over nvlink or a hardware error`.
- Repair: the queue manager now recognizes only a narrow allowlist of transient GPU/driver faults, delays and retries them with a manifest-bound maximum of two total attempts, and continues to park device-side assertions, illegal accesses, and generic nonzero exits for inspection.
- Continuity check: the active `f_1001012_s42` Python process and detached screen survived the scheduler-only restart; the replacement scheduler is live in the `factorial_queue` tmux session.
- Precision: training already uses CUDA autocast plus `GradScaler`; the exact archived state dictionaries are explicitly converted to float16 and are approximately 41.0 MB per model.
- Archive: 33 compatible models are inference-addressable (1.353 GB logical), while only one 41.0 MB checkpoint is locally materialized under the bounded cache policy.
- Inference: all 33 compatible models strictly loaded and produced finite `[1, 12, 5000]` output for `/home/mithunmanivannan/data/ptb_xl/tensors/test/100.pt`; retained audit-cache bytes were zero.
- Temporal morphology: 23/33 compatible models pass the current generation-bound evaluator; the 10-model backlog has a conditional median-rate catch-up estimate of 17.4 hours.
- Disk: 13 GiB remained free versus the queue's 5 GiB launch floor.

These counts are operational snapshots, not factorial results. The legacy 420-row `temporal_mmd_evaluation.csv` remains generation-unbound and is excluded from current-grid claims.

## Verified transition at 2026-08-01 15:26 UTC

The bounded retry completed its state-machine transition under live load. `f_1001012_s42` exited zero and moved the queue to 34 completed jobs. On the next scheduler step, `f_1000113_s42` moved from retry-queued to running with `attempts=2`, a new start timestamp, a new detached screen/Python process, and 96% observed GPU utilization. Its new log was truncated at the attempt boundary and began a fresh W&B run before Epoch 1, so the old exit-134 diagnostic cannot falsely retrigger the classifier. No terminally stuck job remains.

## Downstream catch-up at 2026-08-01 15:30 UTC

- The retried `f_1000113_s42` completed Epoch 1 and entered Epoch 2 without a repeated CUDA/NVLink fault. Its log records CUDA execution and a newly selected checkpoint explicitly saved as float16, confirming both mixed-precision execution and lossless FP16 state storage on the retry path.
- The compatibility/archive watcher now exposes 34 inference-addressable checkpoints: 33 are remotely verified and the currently active 41,010,928-byte checkpoint is the sole local cache entry. The logical inference-addressable cohort is 1,394,372,128 bytes, while local checkpoint use remains bounded to one model.
- The inference-readiness audit strictly loaded all 34 compatible checkpoints and obtained finite `[1, 12, 5000]` outputs for the pinned real PTB-XL input. It retained zero audit-cache bytes after completing and reports a 0.141 s cross-model median CPU forward time; this is an operational readiness measurement, not a hardware-normalized model comparison.
- The generation-bound temporal evaluator has accepted 25 of 34 eligible checkpoints. Its current observed medians are 19.08 training minutes/model and 16.22 evaluation minutes/model, corresponding to 3.15 and 3.70 models/hour. The resulting 0.55 models/hour net backlog-drain estimate says the evaluator keeps pace under the stated serial, uninterrupted-runtime assumptions; the nine-model backlog has a conditional 16.2-hour catch-up estimate.
- Queue state is 34 completed, one running, 445 pending, zero terminal failures/stuck jobs, and zero retry-queued jobs. Disk headroom remains 13 GiB and the active GPU shows sustained utilization.

These downstream values supersede the 33-model operational snapshot above but do not alter its historical diagnosis or the exclusion of generation-unbound results.
