# Three-architecture queue and biomarker recovery audit

Observed at 2026-08-10 15:10 EDT (19:10 UTC).

## Authoritative pre-reconciliation state

- `refine-logs/queue_3arch/queue_state.json` is valid JSON with 319 unique jobs,
  but the disk-full repair truncated the final intended cell:
  `ecg_aim_f_1111114_s42`. The reconciliation script restores it, yielding the
  intended paired 160 MSVAE + 160 ECG-AIM grid (320 jobs).
- Recorded states: 65 completed, 91 failed, 2 running, and 161 pending.
- Only `msvae_f_1111112_s42` has a live trainer. The other running row,
  `msvae_f_1111111_s42`, is an orphan left when the previous tmux worker was
  interrupted at 14:47 EDT.
- The live job is progressing normally: validation completed through epoch 4,
  epoch 5 is active, GPU utilization is 80--100%, and GPU memory is about
  23.7/41.0 GiB.
- W&B output proves the first failed cohort hit the since-fixed missing
  `logging` import. Later failures reached validation and then failed in
  `torch.save` with `PytorchStreamWriter failed writing`, matching the disk-full
  event. These are infrastructure/implementation failures, not completed runs.

## Checkpoint availability correction

The 65 recorded completed MSVAE rows are not inference-addressable: no MSVAE
entry exists in `results/checkpoint_store/catalog.sqlite`, and none of their
checkpoint files remains locally. They must be retrained unless an exact
checkpoint is recovered from an external source. W&B stores metrics and logs,
not the missing state dictionaries.

The checkpoint store now recognizes `factorial_msvae_*` and
`factorial_ecg_aim_*`, maps them to the corresponding queue IDs, validates a
strict architecture-specific state load, records state-schema identity,
performs a byte-identical remote round trip, and permits eviction only after
that verification. The watcher is running in tmux session
`checkpoint_archiver_3arch` with a 60-second poll interval.

The trainer now writes structured FP16 checkpoints with normalized (uncompiled)
state keys, architecture metadata, exact source provenance, factorial identity,
preprocessing metadata, and best-validation selection metadata. The currently
running process predates this edit, so its checkpoint will be accepted as a
strictly loadable legacy raw state; subsequent processes use the structured
format.

## Queue safety correction

`scripts/run_3arch_queue.py` now provides atomic/fsynced state writes, a
single-worker lock, orphan recovery, a 4-GiB free-space launch gate, per-job
logs, attempt accounting, output streaming, and post-exit checkpoint loading
validation. `scripts/reconcile_3arch_queue.py` fail-closes while any trainer is
live and will reset historical completed/failed/orphan rows to pending unless
their exact checkpoints pass the inference-addressable archive gates. The
handoff is deliberately deferred until the active trainer exits so its work is
not discarded.

## Clinical biomarker evaluator

The former evaluator was not live. It had terminated at Sunnybrook with
`NameError: gc is not defined`; its earlier log also contained disk-full SQLite
errors and a compiled-state `_orig_mod.` key mismatch. The database passed
`PRAGMA quick_check` and retained 12,710 unique rows before recovery.

The evaluator was fixed and relaunched through the continuous watcher in tmux
session `eval_daemon`:

- GPU is hidden (`CUDA_VISIBLE_DEVICES=''`).
- CPU affinity is restricted to cores 4 and 5 at nice level 19 and idle I/O
  priority.
- Two single-threaded process workers replace the previous five-worker pool.
- Compiled and normalized checkpoint keys both undergo strict loading.
- Resume uses a deterministic final-metric sentinel rather than treating any
  partial row as model completion.
- SQLite errors now fail closed so the outer watcher retries instead of silently
  accepting missing rows.
- Active checkpoints are excluded from the generated registry until training
  has completed.

At the snapshot, the evaluator was actively repairing the incomplete
`f_1000000_s42` PTB-XL result and using about 1.5 CPU cores. Available RAM was
8.8 GiB, load average was 2.78 on eight logical CPUs, and root storage had about
20 GiB free.

The repair subsequently completed at 15:16 EDT: `f_1000000_s42` advanced from
32 to the full 58 PTB-XL metric rows, committed the `ST_Lead_V6` completion
sentinel, and the SQLite database again passed `PRAGMA quick_check`. The same
process then skipped the already-complete EchoNext cohort and resumed
Sunnybrook at model 10/160, proving restart continuity rather than a fresh
duplicate pass.

## Verification completed

- Python compilation passed for all modified queue, storage, training,
  registry, and evaluator scripts.
- Focused checkpoint-store, external-watcher, clinical-metric, and clinical
  queue tests passed: 39 tests.
- Additional post-change checkpoint-store and clinical tests passed: 30 tests.
- Checkpoint filename/queue identity mapping was verified for UNet, MSVAE, and
  ECG-AIM.
- A CPU-only semantic storage smoke test loaded a synthetic compiled-prefix
  MSVAE FP16 state strictly into `WearECGVAE`, validated all 238 tensors, and
  recorded schema SHA-256
  `16cd75683ac5deb65ac62afd6570cf2c102cd972a7909388546e13825e7b2145`.
- The final combined queue/storage/clinical/external-watcher suite passed 44
  tests; `git diff --check` also passed.

## Remaining live gates

1. Let `msvae_f_1111112_s42` exit without interruption.
2. Verify its first architecture-aware remote archive round trip and local
   eviction.
3. Stop the legacy queue worker between jobs, run fail-closed reconciliation,
   and launch the crash-safe worker.
4. Prove the corrected state advances by at least one completed-and-archived
   job, then materialize that checkpoint and run strict inference.

At 15:16 EDT the legacy controller PID 867656 was placed in `SIGSTOP` state to
make this handoff race-free. Its child shell and trainer remain live and the GPU
continued at 92% utilization. This prevents the old non-atomic controller from
launching the next cell during the interval between current-job exit and state
reconciliation.
