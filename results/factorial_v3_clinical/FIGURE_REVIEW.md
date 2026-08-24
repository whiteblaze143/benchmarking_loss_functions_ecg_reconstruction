# Factorial v3 Clinical Figure Review

This ledger records visual inspection separately from file existence and hash
validation. A figure is accepted only after its source-bound PDF/SVG/PNG
triplet exists and the PNG has been inspected at original resolution.

| Figure | Data/hash gate | Visual review | Notes |
|---|---|---|---|
| `ptbxl_ecgfounder_task_rank` | PASS | PASS | Axes, logarithmic scales, supported-task filter, family colors, and legend are readable; no clipping or overlap. |
| `ptbxl_five_superclasses` | PASS | PASS | All nine rows and five superclass columns are legible; annotations and colorbar are readable; no clipping. |
| `echonext_shd_tasks` | PENDING | PENDING | Awaiting complete EchoNext database. |
| `echonext_shd_stress` | PENDING | PENDING | Awaiting complete EchoNext database. |
| `smartwatch_radar` | PENDING | PENDING | Awaiting repaired smartwatch results. |
| `smartwatch_protocol_accuracy` | PENDING | PENDING | Awaiting repaired smartwatch results. |
| `smartwatch_task_rank` | PENDING | PENDING | Awaiting repaired smartwatch fidelity database. |
| `echonext_shd_calibration` | PENDING | PENDING | Awaiting complete EchoNext per-record predictions. |

Final review must also confirm that plotted values agree with their
machine-readable source tables and that every provenance input/output hash
passes the strict completeness gate.
