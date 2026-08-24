# Factorial Training Integrity Snapshot

Recorded at 2026-08-01T05:26:29Z.

## Authoritative state

- Queue: 6 completed, 1 running, 473 pending.
- Running model: `f_1011103_s42` with batch size 1024.
- Checkpoint compatibility: 6 compatible, 0 incompatible after quarantine and requeue.
- Checkpoint catalog: 6 remotely verified current models, 140 historical/error identities, 0 local checkpoint bytes after cache pruning.
- Disk: 96 GiB filesystem, 85 GiB used, 11 GiB available (89% used).
- GPU: active trainer on the A100 using approximately 30.6 GiB; CPU inference did not displace it.

## Repair performed

Historical queue completions were not scientifically compatible with the restored training contract. Forty-nine checkpoint identities were quarantined/requeued for source or regime incompatibility, and eighteen false completed identities without a recoverable current checkpoint were requeued. Recovery records are preserved in:

- `refine-logs/queue/recovery/20260801T051430Z_post_restore_source_batch_contract_requeue/`
- `refine-logs/queue/recovery/20260801T051528Z_unrecoverable_post_restore_completed_requeue/`

The queue manager and checkpoint archiver were restarted in tmux. Current checkpoints use approved source bundle `6e262086df8e995d90cea19208f189d78658959641069e8e958bcaa7af83e6bd` and batch size 1024. Older batch-size-256 checkpoints are not pooled with this generation.

## Real-data inference proof

Model `f_1011102_s42` was materialized from the verified remote archive and run on `/home/mithunmanivannan/data/ptb_xl/tensors/test/382.pt` using CPU. The input was reduced to leads I, II, and V2 and padded from 5000 to 5024 samples for the network. Inference returned a finite tensor of shape `[1, 12, 5000]`, mean `-0.00041897`, and standard deviation `0.19622873`. Checkpoint SHA-256 was `ddca24dcf77a6914e1f0de54d6ab8ca2ea68449ad0bbf9a7e1bbd088379e2b0e`. The 41,010,928-byte materialized cache file was pruned afterward, returning local checkpoint storage to zero bytes.

## Evaluation warning

`results/factorial_v4/temporal_mmd_evaluation.csv` contains 420 rows for 35 masks but omits model ID, seed, checkpoint digest, source bundle, and data roots. Only one mask overlaps the six compatible masks, and mask overlap does not prove generation identity. The file is retained as a legacy forensic artifact and excluded from current-grid claims.
