# Factorial v4 Corrective Run

Status: training and evaluation complete (91/91 managed jobs). Queue state is
`../../experiment_queue/factorial_v4/queue_state.json`.

This run supersedes the extended v2 grid for correlation/MMD/derivative
factorial inference. It retrains all 48 seed-42 cells across U-Net,
MultiScale-VAE, and ECG-AIM with
`adaptive_multiscale_rbf_mean_squared_distance_v2`, then performs
validation-only selection, 18 confirmation runs, clean and 17-condition
PTB-XL evaluation, EchoNext SHD evaluation, and the calibrated smartwatch
benchmark.

Required controls:

- `preflight.json`: at least 12 GiB free, canonical split hashes, zero overlap.
- `protocol_audit.json`: exact 48-cell registry and source hashes.
- `smoke/*.json`: MMD-path smoke for MSE-off/MSE-on configurations in all
  three families.
- `MMD_REPAIR_VALIDATION.json`: measured live-gradient, six-cell smoke, and
  A100 throughput/memory evidence.
- `../factorial_v3_clinical/PAUSED_INVALID_MMD.md`: reason the legacy
  evaluation was stopped.
- `../../tests/test_factorial_losses.py`: full-dimensional MMD
  gradient-liveness regression.

The empirical tables are complete and may be used for analysis, but no v4
result is promoted to a publication-level poster claim until the fresh post-run
integrity audit and result-to-claim gate return verdicts.

Core completeness is now `complete`, clinical completeness is PASS (16/16),
all seven core figure reviews are PASS,
and figure provenance includes SHA-256 hashes for the locked inputs, current
generator script, and all PNG/SVG/PDF outputs. The final checker is 9/10; the
fresh post-run audit is the remaining gate. Reviewer infrastructure returned
`REVIEW_UNAVAILABLE`, which is retained honestly rather than converted to PASS.

`RESULTS_INTERPRETATION.md` is the poster-planning interpretation of the locked
empirical tables. It is deliberately separated from `CLAIMS_FROM_RESULTS.md`,
whose claim verdict remains unavailable.

`poster_evidence/` is the machine-verifiable layout handoff. Its source and
output hashes pass `scripts/build_poster_evidence_package.py --verify-only`;
the package includes all 15 reviewed figure paths and recommended poster roles.

The MultiScale-VAE MSE-off/on toggle is quarantined from the exploratory
\(2^4\) supplement: detached historical decoder loss caused all eight matched
checkpoint pairs to be tensor-identical. The primary MSE-on \(2^3\) grid is
unaffected. Evidence and claim policy are in
`exclusions/msvae_mse_toggle.json` and `CLAIM_SCOPE.md`.

`FIGURE_REVIEW.json` separately requires source-hash and original-resolution
visual PASS for the seven core factorial figures. Together with the eight
clinical figures, the final review covers fifteen poster assets.
