# Storage candidate audit — 2026-08-22

The initial inventory was read-only. The user subsequently approved the Tier 1
LFS temporary files, cached-extension trash, and two old ChatGPT extensions for
deletion. That cleanup reclaimed 2,869,415,936 bytes. No experiment artifact,
checkpoint, database, queue, or running process was removed.

A second user-authorized cleanup removed four obsolete VS Code CLI servers,
the unused legacy Antigravity server, Pip/Jedi caches, Git-LFS-prunable local
objects, four failed MSVAE checkpoint payloads after verified archival, and
API-verified inactive W&B run trees. Across the second cleanup, net free space
increased from about 6.9 GB to 21.0 GB; root utilization fell to 80%.

## Current constraint

- Root filesystem: 102,888,095,744 bytes total; 98,805,825,536 bytes used;
  4,065,492,992 bytes available (97% used).
- The blinded-LUDB/RDB handoff requires at least 5 GiB free.

## Tier 1 — verified reproducible and currently unused

Cleanup status: completed for the first three rows below. Parent temporary/trash
directories were preserved, as was the live ChatGPT extension `26.818.41509`.

| Candidate | Bytes | Evidence |
|---|---:|---|
| `.git/lfs/tmp/*` | 1,044,385,792 | Four incomplete transfer files; no open file and no Git/LFS transfer process. Re-downloadable. |
| Antigravity cached-extension trash | 461,696,194 | Explicit `.trash` directory; no open file. |
| Old ChatGPT Antigravity extensions `26.818.22352` and `26.818.31338` | 1,095,605,094 | Current live processes use only `26.818.41509`; old versions have no live executable. |
| Git LFS prune candidate `bada2988…` | 105,795,584 | `git lfs prune --dry-run --verbose` identifies it; remote-verification pass did not list it among the two missing remote objects. Use the Git LFS prune command, never manual object deletion. |
| Pip cache | 88,741,616 | Reproducible download cache. |
| Jedi cache | 111,429,407 | Reproducible code-analysis cache. |

The first three candidates alone total 2,601,687,080 bytes (2.42 GiB), enough
to move free space above the 5 GiB experiment gate at the current filesystem
state.

## Tier 2 — reproducible, but preserve a current fallback

Cleanup status: the four older VS Code servers, legacy Antigravity server, and
Pip/Jedi caches were removed. Newest VS Code server `Stable-110a…` was retained.

| Candidate | Reclaim estimate | Evidence / condition |
|---|---:|---|
| Four older VS Code CLI server versions, retaining newest `Stable-110a…` | 2,850,114,001 bytes | No process executes from any CLI-server version; the active VS Code process uses the separate `~/.vscode-server/code-125d…` binary. Versions are re-downloadable. |
| Legacy `~/.antigravity-server` installation | 1,416,007,448 bytes | Last installation activity was May 2026 and no live executable uses it; current processes use `~/.antigravity-ide-server`. Re-downloadable if that older client is used again. |
| Hugging Face cache | 438,886,478 bytes | Re-downloadable, but retain if MedCPT or another cached model is needed offline. |
| APT package/list caches | about 316 MiB | Re-downloadable; cleanup may require elevated privileges. |

## Tier 3 — archive or verify before removal

Cleanup status:

- The four failed MSVAE artifacts were strict-loaded, archived in the existing
  private draft release `factorial-checkpoints-v1`, downloaded independently,
  matched byte-for-byte by SHA-256, recorded in
  `results/msvae_failed_checkpoint_store/catalog.sqlite`, and locally evicted.
- W&B API verification found all 114 home runs and 350/356 project runs remote.
  The lone offline run was synchronized first. All verified inactive run trees
  were removed. Active run `ag5nvzws` and six remote-missing runs were retained.
- Standard Git LFS pruning removed eligible unreferenced local cache objects.
  The cache fell to 622 MiB and `git lfs fsck` passes. Canonical working SQLite
  databases remain present.

| Candidate | Reclaim estimate | Requirement |
|---|---:|---|
| Four unarchived `factorial_msvae_*_s42.pt` checkpoints | 276,752,416 bytes | None matches the checkpoint catalog. Archive and round-trip verify before local eviction. |
| Local W&B trees | about 754 MiB across project and home | Verify every run is synced and inactive first. |
| Result CSV exports | 77,389,554 bytes in the prior audit | Canonical SQLite-backed and regenerable, but low-impact; remove only quiescent exports. |
| `results/checkpoint_store/catalog.jsonl` | 106,924,204 bytes | Derived recovery export, but the active archiver rewrites it. Compress/remove only after the watcher stops or change the exporter. |
| Completed queue logs | about 98 MiB in the prior audit | Compress only terminal-job logs and exclude active tee targets. |

## Explicitly retain

- `results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite` (1.84 GB): canonical,
  compact, zero freelist pages.
- `results/ecgaim_ludb/ecgaim_ludb_blinded.sqlite` (1.05 GB): canonical partial
  run, compact, zero freelist pages.
- `.git/lfs/objects/58/59/5859418…` (1.05 GB): currently missing remotely and
  represents a blinded-LUDB snapshot; do not prune.
- `.git/lfs/objects/0a/0b/0a0bd5…`: missing remotely; retain.
- `/tmp/torchinductor_mithunmanivannan` (about 802 MiB): mapped by active
  one-lead trainer PID 3822078.
- Current spatial one-lead checkpoint and all active queue/job state.
- The broader 6.64 GB of LFS objects not referenced by current Git refs must not
  be manually deleted: recency, dirty-worktree, and remote-availability gates
  still apply.

## Instrumentation changelog

| File | Change type | What was added/modified | Lines |
|---|---|---|---|
| `profile_output/storage_candidate_audit_20260822.md` | created | Read-only storage profile and candidate classification | — |
| `profile_output/storage_candidate_audit_20260822.md` | modified | Recorded the user-authorized Tier 1 cleanup and measured reclaim | — |
| `profile_output/storage_candidate_audit_20260822.md` | modified | Recorded Tier 2/3 cleanup, archive identities, W&B retention gates, and final disk state | — |

No target code was instrumented or modified.
