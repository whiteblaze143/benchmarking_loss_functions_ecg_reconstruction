# Factorial v3 Clinical Artifact Manifest

Status: paused after 10/48 EchoNext cells because the frozen legacy registry's
MMD term failed a full-dimensional gradient audit. The partial is audit-only.
Clinical evaluation resumes with the corrected `factorial_v4` registry;
completion remains governed by `completeness.json`.

## Control and audit

- `../../EXPERIMENT_AUDIT.md` and `../../EXPERIMENT_AUDIT.json` — independent
  pre-repair snapshot audit.
- `../../refine-logs/EXPERIMENT_PLAN.md` — active v3 protocol.
- `../../experiment_queue/factorial_v3_clinical/manifest.json` — generated
  8-phase/15-job repair graph.
- `preflight.json` — frozen model/data/header/provenance gate.

## PTB-XL

- `ptbxl_clinical_metrics.json` — full nested task metrics for 48 models.
- `ptbxl_task_metrics.parquet` and `.csv` — normalized 14,640-row database.

## EchoNext

- `echonext_results.json` — expected official 12-task clean/stress result.
- `echonext_reference_shd.parquet` — expected original-ECG predictions/labels.
- `echonext_per_record/` — expected clean and 17-condition SHD predictions.

## Smartwatch simulator

- `smartwatch_results.json` — expected repaired lead-II benchmark.
- `smartwatch_per_record/` — expected paired record and 150-task probability
  artifacts.

## Combined databases

- `classification_task_metrics.{parquet,csv}` — expected real-GT PTB-XL and
  EchoNext classification table with explicit stress taxonomy.
- `smartwatch_task_fidelity.{parquet,csv}` — expected proxy-only 150-task table.
- `smartwatch_protocol_metrics.{parquet,csv}` — expected calibrated simulator
  device table.
- `clinical_database_manifest.json` — expected row counts and taxonomy.

## Poster figures

Every base name under `figures/` requires PDF, SVG, PNG, and provenance JSON:

- `ptbxl_ecgfounder_task_rank`
- `ptbxl_five_superclasses`
- `echonext_shd_tasks`
- `echonext_shd_stress`
- `smartwatch_radar`
- `smartwatch_protocol_accuracy`
- `smartwatch_task_rank`
- `echonext_shd_calibration`

`FIGURE_REVIEW.md` and `FIGURE_REVIEW.json` record the required
original-resolution visual inspection separately from existence and hash
checks. The machine-readable review is part of the strict completion gate.

## Completion

- `completeness.json` — expected strict machine-readable gate.
- `COMPLETENESS_REPORT.md` — expected human-readable gate.

The two existing PTB-XL figures are preliminary rendered outputs from the
completed backfill. The remaining entries are expected artifacts and are not
evidence until the completion gate passes.
