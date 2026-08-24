# Storage audit — 2026-08-22

## Canonical results and CSV verification

- The LUDB and RDB evaluators use SQLite as their canonical store and generate CSV files with `write_exports(...)` queries.
- There are 57 result CSV files totaling 77,389,554 bytes (73.8 MiB).
- SQLite files and their live WAL files total approximately 2.62 GiB.
- Deleting the live CSV exports is low impact and temporary: the daemons rewrite them after each completed model. Keep them while evaluation is active; they can be compressed or regenerated after the databases become quiescent.

## Completed safe cleanup

| Candidate | Size before cleanup | Evidence | Action |
|---|---:|---|---|
| `/tmp` evaluation snapshots, smoke checkpoints, and CSV-size probes | about 1.2 GiB | Exact paths inspected; no open files; names and contents identify reproducible smoke/snapshot artifacts | Removed |
| Three-lead inference cache plus stale cache bookkeeping | 189,666,614 bytes locally, 121 stale catalog paths | All rows had complete remote and payload verification; remaining local bytes matched catalog SHA-256 | Local cache evicted and all 588 rows reconciled. The live LUDB evaluator may temporarily materialize one verified model (~181 MiB) and prunes it after evaluation |

The active `/tmp/torchinductor_mithunmanivannan` tree was explicitly excluded because the running one-lead trainer has mapped compiled objects from it.

## Completed archive

| Candidate | Size | Evidence | Safe handling |
|---|---:|---|---|
| 24 legacy seed-201 one-lead checkpoints | 2,398,972,064 bytes (2.234 GiB) | Every payload contains matching architecture, factorial mask, seed, observed-lead preprocessing, and a structured state dictionary | Strict architecture load, private-release upload, independent SHA-256 download verification, and local eviction completed; catalog now contains 34 `remote_verified` models and zero local bytes |

## Candidate list

| Priority | Candidate | Reclaim estimate | Classification / prerequisite |
|---:|---|---:|---|
| 1 | Legacy one-lead checkpoints | 2.234 GiB | Completed and reclaimed |
| 2 | TorchInductor cache | 0.691 GiB | Safe only after the current compiled trainer exits; do not delete while mapped |
| 3 | Git LFS local cache | up to 4.9 GiB | Not safe yet. `git lfs fsck` passes, but `git lfs prune --verify-remote` found two objects missing remotely: the one-lead catalog DB and live blinded-LUDB DB. Push/archive quiescent snapshots first |
| 4 | Git object database | 1.62 GiB loose + 3.22 GiB packed; 76.84 MiB explicit garbage | Run conservative Git maintenance only after experiments stop and sufficient temporary space exists; preserve the dirty worktree |
| 5 | Completed queue logs | 98.28 MiB | Compress only logs whose jobs are terminal; exclude active tmux/tee targets |
| 6 | Local W&B run trees | 211.07 MiB | Remove only after verifying each run is synced and not active |
| 7 | `results/checkpoint_store/catalog.jsonl` | 101.97 MiB | Derived recovery export; compact provenance or compress after the three-architecture watcher no longer rewrites it |
| 8 | Quarto/Jupyter rendered caches | about 27.6 MiB | Reproducible and safe when no render is active; low impact |
| 9 | Result CSV exports | 73.8 MiB | Already backed by SQLite and reproducible, but active evaluators rewrite them; handle after completion |

## Keep as canonical

- `results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite`
- `results/ecgaim_ludb/ecgaim_ludb_blinded.sqlite`
- `results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite`
- `results/checkpoint_store/catalog.sqlite`
- `results/onelead_checkpoint_store/catalog.sqlite`

Do not archive or copy a live SQLite database by reading the main file alone. Use the SQLite backup API or stop/checkpoint the writer first so WAL contents are included consistently.
