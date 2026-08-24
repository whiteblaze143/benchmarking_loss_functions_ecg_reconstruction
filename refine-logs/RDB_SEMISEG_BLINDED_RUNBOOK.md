# Learned SemiSeg blinded RDB runbook

This secondary external-domain protocol applies the validation-selected LUDB
Mean-Teacher EMA delineator to RDB waveforms without annotations as input. RDB's
12 lead-specific P/QRS/T regions are used only afterward for six-boundary
scoring. It evaluates the same 31 reconstruction models frozen by the LUDB
oracle `signal_pearson_p05 >= 0.50` rule.

The compact SQLite database stores one evaluation row and 18 aggregate boundary
rows per model (six boundaries for all missing, primary missing precordial, and
derived-limb control groups), plus the original-signal ceiling. It stores no raw
signals, predictions, per-record rows, or CSV exports.

The durable handoff is:

```bash
scripts/run_rdb_semiseg_after_oracle.sh
```

Session `ecgaim_rdb_semiseg_handoff` polls the fixed-region RDB oracle database.
Only after exactly 31 models are complete, none are running/failed, and SQLite
integrity passes does it stop the idle oracle poller and start the learned run on
CPU cores 0-5. Any premature oracle exit fails closed.

Monitor:

```bash
tail -f results/ecgaim_rdb_semiseg_blinded/supervisor.log
sqlite3 results/ecgaim_rdb_semiseg_blinded/compact.sqlite \
  'select status,count(*) from evaluations group by status;'
```

The production database is
`results/ecgaim_rdb_semiseg_blinded/compact.sqlite`. The protocol binds the RDB
content identity, the 31-model selection manifest, evaluator/loader/registry
code, and the learned checkpoint SHA/state. It resumes only exact matches and
pauses below 8 GiB free disk.
