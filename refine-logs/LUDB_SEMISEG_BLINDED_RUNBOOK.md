# Learned SemiSeg blinded LUDB protocol

This is a new protocol and database. It does not reuse the DWT/prominence
snapshot. The validation-selected `model_ema` state from `best-MeanIoU.pth`
receives only each waveform; annotations are scoring-only. Evaluation is
restricted to the 40 official untouched LUDB test subjects.

The production launcher uses six CPU cores and no GPU:

```bash
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
scripts/run_ecgaim_ludb_semiseg_blinded.sh
```

It creates tmux session `ecgaim_ludb_semiseg_blinded`, writes its log to
`results/ecgaim_ludb_semiseg_blinded/daemon.log`, and resumes the compact
SQLite database at `results/ecgaim_ludb_semiseg_blinded/compact.sqlite`.

Monitor without attaching:

```bash
tmux capture-pane -pt ecgaim_ludb_semiseg_blinded -S -30
tail -30 results/ecgaim_ludb_semiseg_blinded/daemon.log
sqlite3 results/ecgaim_ludb_semiseg_blinded/compact.sqlite \
  'select status,count(*) from evaluations group by status;'
```

To stop, send `SIGTERM` to the evaluator and let the current model transaction
finish or record an error. Restart with the same launcher after the tmux session
has exited. A model is skipped only when its status, reconstruction checkpoint,
protocol, delineator checkpoint/state, dataset, and frozen split identities all
match.

The database contains only metadata, one row per evaluated reconstruction model
(plus one original-signal delineator ceiling), and six aggregate boundary rows
per model. It stores no raw signals,
predictions, per-record metric rows, or CSV exports. The disk gate requires at
least 8 GiB free before claiming each model.
