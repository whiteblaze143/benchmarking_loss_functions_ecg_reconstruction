# Verified Git temporary-garbage cleanup

Timestamp: 2026-08-01T08:31:00Z

- Disk attribution showed that active checkpoints occupied only 81 MB, whereas
  Git contained five unindexed `tmp_pack_*`/`tmp_obj_*` files totaling
  692.07 MiB.
- `git count-objects -vH` classified exactly those files as garbage. No Git
  process or open file handle referenced them, the two large temporary packs
  were byte-identical, and `git fsck --full` completed successfully before
  removal.
- Only the five named temporary garbage files were removed. They are not
  recoverable. No indexed pack, normally hash-named loose object, checkpoint,
  source file, or worktree change was removed.
- Post-cleanup Git accounting reports three valid packs, zero garbage files,
  zero garbage bytes, and successful connectivity verification. Dangling
  hash-addressed objects were deliberately retained because they can represent
  recoverable user history.
- Filesystem free space increased from approximately 11 GB to 12 GB. The
  checkpoint lifecycle remains bounded to the active training checkpoint plus
  at most one verified evaluator/inference cache checkpoint.
