# Factorial v4 Clinical/Postprocessing Manifest

Status: complete (13/13 managed jobs; strict clinical completeness PASS 16/16).

Control graph:
`../../experiment_queue/factorial_v4_clinical/manifest.json`.

This follow-on queue consumes only:

- `../../experiment_queue/factorial_v4/model_registry.json`;
- `../factorial_v4/comprehensive_results.json`;
- `../factorial_v4/echonext_results.json`;
- `../factorial_v4/smartwatch_results.json`;
- `../factorial_v4/selected_masks.json`.

It produces the 150-task ECGFounder and five-superclass PTB-XL backfill,
normalized real-ground-truth EchoNext database, smartwatch probability-fidelity
and calibrated-protocol databases, and eight PDF/SVG/PNG poster figures.
Validation-best figures hash the v4 selection file in their provenance.

The strict gate is incomplete until all eight figures are inspected at original
resolution and recorded as PASS in `FIGURE_REVIEW.json`, followed by a fresh
independent experiment audit.

All eight clinical figures have now passed original-resolution inspection.
Their provenance records bind current figure scripts, the shared figure helper,
data inputs, and all PNG/SVG/PDF outputs with SHA-256 hashes. The fresh
post-run audit is the remaining final gate.

`../../scripts/check_factorial_v4_final.py` is the final combined fail-closed
gate. It requires both managed queues, both completeness reports, all fifteen
visual reviews, the corrected protocol/MMD evidence, and
`../../EXPERIMENT_AUDIT_POSTRUN.json` to pass.
