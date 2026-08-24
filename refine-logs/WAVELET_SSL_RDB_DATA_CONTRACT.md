# Wavelet SSL RDB data contract

Updated: 2026-08-22 America/Toronto

This amendment records the user's instruction to use RDB rather than ISP for
the wavelet SSL + delineation screen. No queued command or sweep manifest may
reference `data/delineation_cache` (the earlier ISP cache).

## Source identity and preprocessing

- Source: `data/rdb`, 2,399 released RDB IDs mapped to 2,398 unique Chapman
  records by `data/rdb/rdb_chapman_mapping.xlsx`.
- Mapping-flagged duplicate `SI0211` is excluded; no duplicated Chapman source
  crosses a split.
- Content identity: `81d69552522ae2b86116e33f578cbdf5965214f284c5092523d19660acefda83`.
- Signals remain at the released 500 Hz / 5,000-sample clock and are divided by
  1,000 to obtain mV, following the existing RDB oracle contract.
- Labels use each of the 12 lead-specific annotation streams. Inclusive
  P/QRS/T regions map to classes 1/2/3; background is 0. Consensus `.all`
  annotations are audited but never substituted for a lead-specific stream.
- RDB supplies regions, not peaks. The fiducial head is disabled.

## Frozen split

Split seed `20260822` uses rhythm-stratified SHA-256 ordering of unique Chapman
record IDs:

- train: 1,678
- validation: 360
- untouched test: 360

The test split is excluded from architecture selection and training. Because
RDB is now a training source, the entire RDB cohort is no longer valid as an
external evaluation dataset; only the frozen 360-record test partition can be
used for held-out RDB evaluation of this model family.

## Label quarantine

- 502 lead streams containing an invalid released annotation row are excluded
  from supervised segmentation by setting their dense labels to `-1` and
  `seg_valid=false`; their waveforms remain usable for reconstruction.
- The sole valid-row cross-class collision comprises 43 samples in `SB0101`.
  Only those ambiguous pixels are marked invalid.
- Train and validation audits found zero non-finite waveforms, zero missing
  patient IDs, and exact lead-specific label provenance.

The authoritative materialized cache is
`data/rdb_wavelet_delineation_cache`; its `manifest.json` binds every source
bundle and every generated tensor by SHA-256.
