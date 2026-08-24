# Exact checkpoint storage and inference

The factorial queue produces hundreds of dense model checkpoints. Storing the
same bytes inside SQLite does not save space, and lossy quantization would
change the scientific artifact. The repository therefore uses two layers:

- `results/checkpoint_store/catalog.sqlite` stores model ids, masks, seeds,
  byte counts, SHA-256 digests, provenance metadata, and remote asset ids.
- The exact `.pt` bytes live as assets in the private draft GitHub release
  `factorial-checkpoints-v1`.
- `results/checkpoint_store/catalog.jsonl` is a portable recovery index. The
  archiver periodically snapshots it into the same private release.

A completed local checkpoint is deleted only after the store has:

1. computed its local SHA-256;
2. uploaded it to the private release;
3. checked the remote asset state, byte count, and GitHub-computed digest;
4. downloaded the asset independently; and
5. reproduced the original SHA-256;
6. structured-loaded the payload; and
7. matched its embedded mask and seed to the queue/catalog identity;
8. strictly loaded every key and shape into `MCMAModel`; and
9. rejected unsupported dtypes or non-finite state tensors and recorded a
   deterministic state-schema SHA-256.

The background `checkpoint_archiver` tmux session repeats this process for each
newly completed queue job. It keeps standalone metadata sidecars locally
because they are small and useful for audits, and embeds their complete JSON in
the remotely snapshotted recovery index.

Training never writes a best checkpoint directly over its canonical name.
`train_factorial.py` writes and flushes a same-filesystem temporary file,
calls `fsync`, atomically renames it into place, and then flushes the parent
directory. The final JSON sidecar uses the same protocol. A killed writer can
therefore leave an ignorable dot-prefixed temporary file, but cannot expose a
partially written canonical checkpoint or sidecar.

The scientific release policy is stricter than byte integrity. New runs must
match `refine-logs/factorial_training_contract.json`: one approved source
bundle, the declared PTB-XL split/preprocessing contract, and the float16 state
schema. The split contract includes a manifest of every tensor's record ID,
byte count, and SHA-256, with separate train/validation/test content roots.
The tensor manifest, PTB-XL metadata tables, and manifest-producing script are
also SHA-bound. Each process verifies the full tensor corpus before Epoch 1,
before each best-checkpoint write, and before its final sidecar; it captures
its source bundle once at startup and embeds the exact
base64 source bytes and binary diff in checkpoint provenance, and refuses a
later save if any source file changed. Run the fail-closed audit with:

```bash
/home/mithunmanivannan/.venv/bin/python \
  scripts/audit_factorial_compatibility.py
```

The legacy lead-loss implementation is source-compatible only for masks where
the lead digit is zero and its broken branch is dormant. Those legacy runs
still lack the final tensor-content roots, so none is release-compatible under
the complete contract. Historical generations remain exact private
`QUARANTINED_release_policy_...` assets; they are not inference candidates or
results, and all jobs are retrained under the content-pinned contract.

## Inference on any release-compatible model

Use the project virtual environment. The command below retrieves the exact
checkpoint if it is not cached, verifies its SHA-256, loads it strictly, and
runs the training-consistent I/II/V2-to-12-lead path on a real PTB-XL tensor.
By default, a `finally` cleanup bounds the cache to zero even if loading,
forward inference, or output writing fails:

```bash
/home/mithunmanivannan/.venv/bin/python \
  scripts/infer_factorial_checkpoint.py \
  f_1000000_s42 \
  data/ptb_xl/tensors/test/100.pt \
  --output results/inference/f_1000000_s42_ecg100.pt
```

The `--output` argument is optional. Omit it when the caller only needs the
machine-readable report printed to standard output and does not want to retain
the reconstructed tensor:

```bash
/home/mithunmanivannan/.venv/bin/python \
  scripts/infer_factorial_checkpoint.py \
  f_1000000_s42 \
  data/ptb_xl/tensors/test/100.pt
```

This is the lowest-retention mode: one exact checkpoint is materialized and
verified, the forward pass runs, no reconstruction file is written, and the
checkpoint cache returns to zero. When outputs are required, place them on a
deliberately managed results volume and treat their retention separately from
checkpoint retention.

When `--output` is supplied, the CLI does not write directly over the final
path. It serializes to a same-directory hidden temporary file, flushes and
`fsync`s the file, atomically renames it, and flushes the parent directory. An
ordinary exception therefore removes the temporary file and never exposes a
partial reconstruction under the requested final name. Input shape and channel
validation also occurs before remote checkpoint materialization, avoiding a
download for a malformed request.

Python callers can load a checkpoint payload directly:

```python
from scripts.checkpoint_store import load_checkpoint

checkpoint = load_checkpoint("f_1000000_s42", map_location="cpu")
```

The low-level Python loader intentionally leaves its verified materialization
in `checkpoints/cache` so a caller can reuse it. Long-running Python workflows
must call the store's `prune_cache(...)` in a `finally` block, or invoke the
inference CLI in a subprocess. The CLI is the space-safe default because it
performs this cleanup automatically on success and failure.

Materialize without loading:

```bash
/home/mithunmanivannan/.venv/bin/python \
  scripts/checkpoint_store.py materialize f_1000000_s42
```

The verified local cache defaults to `checkpoints/cache`. A single inference
defaults to `--max-cache-gib 0`; retain an operator-managed 2 GiB cache instead
with:

```bash
/home/mithunmanivannan/.venv/bin/python \
  scripts/infer_factorial_checkpoint.py \
  f_1000000_s42 data/ptb_xl/tensors/test/100.pt \
  --max-cache-gib 2
```

For a strictly streaming inference loop, the default zero bound removes only
verified cache copies after every invocation; the exact private-release asset
and catalog row remain:

```bash
mapfile -t model_ids < <(
  /home/mithunmanivannan/.venv/bin/python \
    scripts/infer_factorial_checkpoint.py --list-ready
)
for model_id in "${model_ids[@]}"; do
  /home/mithunmanivannan/.venv/bin/python \
    scripts/infer_factorial_checkpoint.py \
    "${model_id}" data/ptb_xl/tensors/test/100.pt \
    --output "results/inference/${model_id}_ecg100.pt"
done
```

This uses space proportional to one checkpoint rather than all 480, including
on failed inference attempts. The example intentionally retains one output per
model; omit each `--output` argument for a zero-retention smoke test, or consume
and remove outputs incrementally in a downstream workflow. An eligible model
can be fetched again at any time; `materialize` rechecks the exact catalog
SHA-256 before returning its path. Both `--list-ready` and ordinary inference fail closed unless the
compatibility audit is bound to the current training contract and approved
source bundle. Ordinary inference also requires the catalog digest to equal
the digest in that audit, so a replaced or stale model id cannot silently load.

Audit every currently compatible archived model on the same real PTB-XL test
record with a dedicated zero-retention cache:

```bash
taskset -c 6 nice -n 19 ionice -c 3 \
  /home/mithunmanivannan/.venv/bin/python \
  scripts/benchmark_factorial_inference_readiness.py --repeats 3
```

The command publishes only after every current compatible identity passes
strict state loading, digest matching, expected output-shape checking, and a
finite-output check. Its CSV and JSON summary are written under
`results/factorial_mixed_level/inference_readiness`; the summary binds the
input tensor, compatibility audit, benchmark code, training contract, and CSV
by SHA-256. It does not retain reconstructed tensors or checkpoint cache files.

Inspect archive coverage and logical/local byte counts:

```bash
/home/mithunmanivannan/.venv/bin/python scripts/checkpoint_store.py status
```

The status output separates `inference_addressable_bytes` from
`error_generation_bytes`. The former is the exact remote model volume that can
be materialized through the inference API; the latter is historical or failed
generation volume retained for audit but rejected by inference. `local_bytes`
is the catalogued bulky volume currently materialized on this machine. These
are deliberately separate so a large remote quarantine history is not mistaken
for current filesystem usage.

Do not manually remove a checkpoint that is only in `local` or `error` status.
`remote_verified` means its exact bytes passed the independent round trip and
semantic payload validation. `cached` means a verified remote copy exists and
a local materialization is currently present. `error` is fail-closed: inspect
the catalog error and recovery audit rather than inferring that the model is
usable.

## Disk-pressure triage outside the checkpoint store

Checkpoint cache size and filesystem pressure are different quantities. Inspect
both before deleting anything:

```bash
/home/mithunmanivannan/.venv/bin/python scripts/checkpoint_store.py status
du -x -h --max-depth=1 . | sort -h
git count-objects -vH
```

The checkpoint status reports logical addressable bytes separately from bulky
local bytes. Git may also report interrupted `tmp_pack_*` or `tmp_obj_*` files
as `garbage`; remove such files only after confirming the exact paths with
`git count-objects`, confirming that no Git process or open handle uses them,
and completing `git fsck --full`. Do not equate `dangling` hash-addressed
objects with temporary garbage: dangling objects can contain recoverable user
history and should not be pruned merely to gain space.

Historical nonzero-exit partials are retained under
`QUARANTINED_exit<code>_..._sha<digest>.pt` names. They cannot be materialized
through the inference API and their jobs remain pending until a new exit-zero
generation is trained. When a replacement digest or size appears, the catalog
atomically clears every old asset, timestamp, sidecar-derived, and semantic
validation field before fresh validation.

## One-generation release binding

The scientific results chain is keyed by both `model_id` and the exact
`checkpoint_sha256`, not by a reusable filename alone. The compatibility audit
records the checkpoint digest and byte count from the verified catalog
generation. Every per-record Parquet repeats one constant `model_id` and
`checkpoint_sha256` across its 2,198 rows. The 480-row summary repeats the same
digest and byte count and its headline metrics must equal aggregates recomputed
from that model's Parquet. The final writer joins all four layers—compatibility
audit, SQLite catalog, per-record artifact, and summary—and rejects missing,
duplicated, stale, non-finite, or cross-generation records.
