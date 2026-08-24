# RDB Oracle-Region ECG-AIM Evaluation Plan

Updated: 2026-08-22 01:01 America/Toronto

## Question and claim boundary

Select ECG-AIM loss configurations for extreme reconstruction robustness using the richer RDB rhythm-diverse, lead-specific cardiologist regions. The primary claim is waveform fidelity on genuinely missing precordial leads V1/V3–V6. Observed I/II/V2 and algebraically derivable III/aVR/aVL/aVF are controls.

This is not a segmentation-model benchmark. ECG-AIM emits reconstructed voltage, not a P/QRS/T mask. Therefore Dice, timing F1, sensitivity, and PPV are not defined in the primary fixed-region track. A later frozen-segmenter track may report those as secondary evidence with the paper's 150 ms matching rule, but must never replace the fixed-label fidelity result.

## Dataset and mapping contract

- Signal/annotation root: `data/rdb`.
- Authoritative record crosswalk: `data/rdb/rdb_chapman_mapping.xlsx` (2,399 RDB IDs, 2,398 unique Chapman IDs).
- Primary cohort: 2,398 unique-source records. Mapping-flagged `SI0211` is excluded because its signal is byte-identical to `SI0109`; its independently different annotation set would otherwise pseudoreplicate one ECG.
- Released-to-canonical mapping codes: AF→AF, AFIB→AFIB, SB→SB, SR→SR, SI→SA, ST→ST, VT→SVT, AT→AT. Store both fields; never relabel `VT` as ventricular tachycardia.
- Rhythm descriptions remain code-level in outputs because the paper's AF/AFL naming and the released AF/AFIB filenames do not align unambiguously. The mapping spreadsheet, not filename prose, controls subgroup membership.
- Every primary metric uses all 12 lead-specific annotation files. The `.all` file is loaded and audited but never substituted for a lead.
- Signals are 5,000×12 at 500 Hz. Released numeric amplitude is divided by 1,000 before ECG-AIM. Evidence: raw 0.1%/99.9% values near −2,069/+2,103, 4.88-unit quantization, and limb-law residuals become physiologic only under this scaling; the README's `mV` wording conflicts with the actual values.
- Dataset identity hashes the README, mapping, every selected signal, and all 13 annotation streams per record.

## Annotation resolution and quarantine

- Parse TYPE/START/END as finite numeric values; TYPE 0/1/2 maps to P/QRS/T.
- Resolve nonnegative coordinates to the 500 Hz grid by deterministic half-up rounding (`floor(x+0.5)`). Maximum observed adjustment: 0.499847 sample (<1 ms). Do not claim sub-sample timing precision.
- END is inclusive. Zero-length, reversed, nonfinite, malformed, or out-of-bounds regions are excluded, never swapped, clipped, or expanded.
- Full preflight audit: 1,042,717 lead-specific rows plus 86,887 consensus rows. The 12 lead streams retain 192,131 P, 427,892 QRS, and 422,192 T valid regions. Across lead plus consensus streams there are 34,037 fractional rows, 526 zero-length rows, one reversed row, and 13 out-of-bounds rows.
- RDB has regions but no labeled peaks. Do not invent peak errors. Exact onset/offset voltage errors are valid because those samples are labeled region boundaries.

## Fixed-region endpoints

- Whole signal per record/lead: Pearson, MSE, MAE, derivative MSE.
- Each P/QRS/T inclusive region: onset and offset signed/absolute voltage error; fixed-window correlation, RMSE, MAE, max absolute error; linear-endpoint-baseline-corrected signed and absolute area error.
- QRS/ST: QRS offset voltage at inclusive END; J/ST-start voltage at END+1; J+20/40/60/80 ms when before the next T onset; ST mean, area, correlation, and RMSE.
- Primary record-level Pareto panel: signal Pearson p05, signal MSE p95, boundary-voltage error p95, QRS/T RMSE and area-error p95, and J error p95 on V1/V3–V6.
- Rhythm-tail table: the same record-aggregated endpoints for every canonical Chapman rhythm code.
- Supporting analyses: oracle-only Pareto front; transparent equal-endpoint minimax sensitivity ordering; factorial matched-cell main effects, MMD contrasts, and interactions including correlation×VCG; paired 2,000-resample record bootstrap against `1000000`.
- Do not select a final winner until all 160 cells are complete. All cells are seed 42, so finalists require confirmatory seeds.

## Storage and execution contract

- CPU only: `CUDA_VISIBLE_DEVICES=''`; six Torch/BLAS threads; current allowed CPU affinity inherited dynamically.
- Compact SQLite stores record-role aggregates plus lead/rhythm summaries. It deliberately omits per-interval model rows; raw annotations remain bound by dataset SHA-256.
- A worst-case synthetic schema fill measured 5.55 MB/model and projects the 160-model SQLite DB to 0.83 GiB. Record-level tables remain only in SQLite; automatic CSV exports are limited to evaluations and compact lead/wave summaries. Budget roughly 1–1.5 GiB total. Production pauses below 5 GiB free.
- LUDB's median 173 seconds per 200-record model implies about 35 minutes/model by record scaling; budget 40–60 minutes/model and roughly 4–7 days for 160 models.
- Production DB: `results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite`.
- Production tmux session: `ecgaim_rdb_oracle_eval`.
- The guarded launcher refuses production unless `RDB_ORACLE_CONFIRM_PRODUCTION=I_UNDERSTAND_RDB_PRODUCTION` is explicitly set and refuses a duplicate tmux session.
- Per user instruction, do not launch RDB until the current LUDB evaluation is finished/stopped.

## Acceptance gates

1. Full 2,398-record preflight completes with stable audit counts and dataset hash.
2. Unit tests cover rounding, exact inclusive boundaries, J=END+1, lead roles, DC-invariant areas, compact schema, and absence of invented Dice/F1/timing columns.
3. One-record/one-model CPU smoke completes in a temporary DB; SQLite integrity and foreign keys pass; exports and analysis run.
4. Fresh agent executes the documented smoke invocation verbatim and reports any divergence.
5. Production DB and `ecgaim_rdb_oracle_eval` tmux session remain absent until explicitly launched later.
6. After production completion, run the independent postrun experiment-integrity audit before making loss claims.
