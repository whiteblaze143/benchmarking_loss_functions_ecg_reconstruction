# Unified Book and First Strict Diagnostic Snapshot

Timestamp: 2026-08-01T06:33:00Z

## Queue and storage

- Queue: 9 completed, 1 running (`f_1000002_s42`), 470 pending.
- Compatibility audit: 9 compatible, 0 incompatible for the approved current contract.
- Archive: nine current-generation checkpoints are remote-verified; historical error/quarantine rows are not inference candidates.
- Active local storage is bounded to the training checkpoint plus at most one evaluator/inference cache checkpoint. The exact remote blob, not SQLite, holds large weights; SQLite holds identity, digest, provenance, and locator metadata.
- Exact current-generation inference was rerun for `f_1000000_s42` on PTB-XL ECG 100. Checkpoint SHA-256 is `ea2df9939d4438de51c702ddd887edf2ccd57b8e2c094737f9207bffc9070349`; output shape is `[1,12,5000]`, all values are finite, and the zero-GiB cache policy pruned the verified materialization.

## Generation-bound temporal diagnostic

- Target feature cache: 2,198 PTB-XL records, 291,708 finite feature rows, Parquet SHA-256 `057765b02da864d334dd7a65c5440b96126ad4354986c2f4e1650035c3ff98b0`.
- Current evaluator SHA-256: `7f5323ecdfcf6601d5972d804864498db55425f86f4d05cbd644393195676fbe`.
- Accepted: 1 model, 12 V3/V6 feature rows. Excluded: one older-evaluator artifact.
- First accepted model: `f_1000000_s42`, checkpoint `ea2df...`, runtime 938.4 seconds, zero reconstruction-batch failures.
- Record-pair coverage ranges 94.54%–99.95%. Eleven of 12 detected feature/lead variance ratios are below one, but V6 QT is above one (1.52); this is heterogeneous partial evidence, not a universal MSE-smoothing conclusion.
- Current pairing truncates by within-record detection order. Confirmatory work still requires time-tolerant fiducial matching and unmatched-event accounting.

## Unified HTML book

- Full venv render completed successfully with 16 HTML pages.
- Output: `book/_book/index.html`.
- Validator: 735 local references checked, zero missing; zero execution-error markers; zero unresolved Quarto cross-references.
- All 16 source QMD files are listed exactly once in `_quarto.yml`.
- Narrative begins with real EDA and dataset contracts. Zhejiang is now connected to Part I. The PTB-XL, architecture, loss, smartwatch, EchoNext, engineering, and cath-lab chapters distinguish observed data, locked results, partial diagnostics, protocols, and simulations.

## Review status

REVISE / IN PROGRESS. The requested Claude-family bridge remains unavailable, so the same-agent integrity review is explicitly non-independent and no numerical reviewer score is fabricated.
