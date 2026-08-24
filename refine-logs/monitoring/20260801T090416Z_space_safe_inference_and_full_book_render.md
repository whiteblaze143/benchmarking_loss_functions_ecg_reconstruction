# Space-safe archived inference and unified-book render

Timestamp: `2026-08-01T09:04:16Z`

## Checkpoint-store state

- The SQLite catalog contains 146 identities across current and historical
  generations. Its 8,476,831,102 logical bytes are addressable artifact bytes,
  not bytes retained on the working filesystem.
- The current compatibility cohort contains 16 models. All 16 are
  `remote_verified`, with 656,175,424 logical checkpoint bytes and zero bulky
  checkpoint-store bytes retained locally after the inference check.
- The 130 catalog `error` rows are quarantined historical generations. They are
  not live queue errors, are not returned by `--list-ready`, and cannot be
  materialized through the inference CLI.
- Filesystem free space is 12 GiB. Git temporary garbage remains zero; dangling
  Git objects were deliberately not pruned because they may contain recoverable
  user history.

## Real archived-model inference proof

The project venv executed `f_1000013_s42` against the actual tensor resolved
from `data/ptb_xl/tensors/test/100.pt`:

- checkpoint SHA-256:
  `025ac2aefc16518d0b0cf965f9bd5946c33e1075280c05a3eb1212d774dcd708`;
- exact checkpoint size: 41,010,928 bytes;
- prepared input shape: `[1, 3, 5024]`;
- reconstruction shape: `[1, 12, 5000]`;
- all reconstruction values finite;
- strict contract/source/content-root provenance present;
- two temporary cache artifacts pruned in the CLI `finally` path;
- zero checkpoint cache files remained after verification.

A subsequent implementation audit made requested reconstruction outputs
failure-safe as well: output bytes now go to a same-directory temporary file,
are flushed and `fsync`ed, and are atomically renamed. An injected-save-failure
test proved that neither a final file nor a hidden temporary survives the
exception. A second actual-data inference proved the successful path left zero
hidden output temporaries and zero checkpoint cache files. Invalid tensor
shapes are now rejected before any remote checkpoint materialization.

The temporary reconstruction and report under `/tmp` were inspected and then
removed. Exact checkpoint recovery remains available from the private release.
The operator documentation now distinguishes checkpoint retention from output
retention and documents the zero-output-file invocation obtained by omitting
`--output`.

## Queue health

The live queue was not stuck: it reported 16 completed, one running, and 463
pending jobs. `f_1000014_s42` advanced through epoch 6 and entered epoch 7/10
during this audit. Its validation MSE improved from 0.1620 at epoch 1 to 0.0551
at epoch 6. The GPU snapshot showed 83% utilization and 28,682 MiB allocated.

The monitor remained attached through completion. `f_1000014_s42` finished
exit-zero with final validation MSE 0.0418, passed upload/download SHA-256 and
strict semantic loading, and was evicted locally. The queue advanced to
`f_1000100_s42`, leaving 17 completed, one running, and 462 pending. The
cohort-aware real-data readiness sweep then executed all 17 compatible models,
produced 153 per-model-per-missing-lead rows, and retained zero cache bytes.
Training diagnostics now contain 170 epoch rows, and the final morphology
generation has two accepted model artifacts (24 lead-feature rows) among 17
eligible checkpoints.

## Render and checks

- Full `quarto render` completed all 16 connected book pages with the project
  `venv-kernel`; output is `book/_book/index.html`.
- The actual-data EDA is the first chapter.
- EchoNext executed all 12 cells; the engineering/storage chapter executed all
  22 cells.
- Post-render normalization found 687 local references, zero missing targets,
  and zero embedded runtime-error markers.
- The rendered engineering chapter contains the new zero-retention inference
  guidance.
- A final coherent full-book render binds the live tables to 17 compatible
  inference-ready models and two accepted temporal-morphology artifacts.
- Focused checkpoint/inference tests after the atomic-output change: 27 passed.
- The broader evaluator/storage suite immediately preceding this snapshot:
  45 passed with one third-party SciPy deprecation warning.

The requested cross-family reviewer bridge is unavailable in this environment.
These checks are same-agent structural and executable validation and are not
represented as an independent review score.
