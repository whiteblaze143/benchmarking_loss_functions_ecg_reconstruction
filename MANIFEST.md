# Factorial Benchmark Artifact Manifest

Updated: 2026-07-26T20:41:00Z

> **The v2 grid is invalid for MMD effects and interactions.** A
> full-dimensional audit found effective gradient underflow in its fixed-width
> MMD. The corrective v4 graph is complete (91/91 core jobs and 13/13 clinical
> jobs). v4 is the authoritative evidence package.

## Authoritative control

- `experiment_queue/factorial_v2/manifest.json` — generated phase/job graph.
- `experiment_queue/factorial_v2/queue_state.json` — live queue state.
- `experiment_queue/factorial_v2/model_registry.json` — 48 seed-42 model specifications.
- `experiment_queue/factorial_v4/manifest.json` — active 48-cell corrective
  training/evaluation graph.
- `experiment_queue/factorial_v4/model_registry.json` — corrected adaptive-MMD
  registry and source hashes.
- `experiment_queue/factorial_v4_clinical/manifest.json` — corrected-registry
  individual-task database and eight-figure follow-on graph.
- `scripts/launch_dependent_queue.py` — fail-closed 91/91 core-to-clinical
  handoff used by detached session `factorial_v4_clinical_queue`.
- `tests/test_launch_dependent_queue.py` — exact-count, status, and required
  output regression tests for the cross-manifest handoff.
- `refine-logs/EXPERIMENT_PLAN.md` — current protocol.
- `refine-logs/EXPERIMENT_TRACKER.md` — current execution status.

## Authoritative v4 evidence

- `results/factorial_v4/comprehensive_results.json` — corrected PTB-XL clean,
  morphology, ECGFounder, common-task-set fairness, and 17-condition results.
- `results/factorial_v4/{statistics.json,factorial_effects_bca.parquet}` —
  patient-cluster BCa factorial inference with explicit seed/claim scope.
- `results/factorial_v4/{echonext_results.json,smartwatch_results.json}` —
  external and calibrated wearable evaluations.
- `results/factorial_v4/figure_provenance.json` and
  `results/factorial_v4/FIGURE_REVIEW.json` — hashed seven-figure core suite.
- `results/factorial_v4/RESULTS_INTERPRETATION.md` — locked empirical
  interpretation for poster planning, with claim boundaries.
- `results/factorial_v4/poster_evidence/` — reproducible poster handoff:
  selected architecture/component/robustness/device tables, five qualified
  empirical statements, all 15 figure hashes/roles, and a PASS verifier.
- `scripts/build_poster_evidence_package.py` — fail-closed generator and
  verifier for the poster handoff.
- `results/factorial_v4_clinical/` — task databases, eight clinical figure
  triplets, provenance, and strict completeness report.
- `EXPERIMENT_AUDIT_POSTRUN.{json,md}` — explicit `REVIEW_UNAVAILABLE`
  status and preserved reviewer-attempt traces; not an integrity PASS.
- `results/factorial_v4_clinical/FINAL_COMPLETION.{json,md}` — final
  fail-closed gate currently 9/10, with only the independent audit unresolved.

## Historical v2 evidence (pilot/regression only)

- `results/factorial_v2/comprehensive_results.json` — PTB-XL clean, morphology, ECGFounder, fairness, and 17-condition results.
- `results/factorial_v2/legacy_checkpoint_metadata_20260725.tar.gz` — archived
  JSON metadata for deleted invalid extended-grid weights.
- `results/factorial_v2/paper_parity_results.json` — five U-Net paper-parity anchors.
- `results/factorial_v2/confirmation/` — 18 seed-confirmation markers and evaluations.
- `results/factorial_v2/statistics.json` — primary \(2^3\) and supplemental \(2^4\) statistical output.
- `results/factorial_v2/factorial_effects_bca.parquet` — prespecified MSE-on effects.
- `results/factorial_v2/factorial_effects_4factor_supplementary.parquet` — exploratory MSE-toggle effects.
- `results/factorial_v2/ecgfounder_noninferiority.parquet` — paired AUROC non-inferiority.
- `results/factorial_v2/familywise_endpoint_tests.parquet` — QRS/ST/diagnostic endpoint tests.
- `results/factorial_v2/main_48_cell_table.parquet` — compact registry table.
- `results/factorial_v2/main_primary_24_cell_table.{csv,parquet,html,tex}` — prespecified MSE-on poster table.

## External evidence

- `data/echonext/PROVENANCE.json` — EchoNext acquisition and preprocessing provenance.
- `results/factorial_v2/echonext_preflight.json` — EchoNext provenance gate.
- `results/factorial_v2/echonext_results.json` — final external result; generated only after 48-cell coverage.
- `results/factorial_v2/echonext_results__pilot_stress512.json` — retained pilot, never used for final completeness.
- `results/factorial_v3_clinical/echonext_results.json.partial` — ten-cell
  atomic audit-only partial retained at the pause boundary.
- `results/factorial_v2/echonext_shd_label_audit.json` — 12-label presence/identity audit; explicitly records why SHD prediction metrics are not claimed.
- `results/factorial_v2/smartwatch_results.json` — clean-only PhysioNet smartwatch zero-shot benchmark.

## Exclusions and reports

- `results/factorial_v2/exclusions/cnvae.json` — cNVAE collapse/exclusion decision.
- `results/factorial_v2/completeness.json` — machine-readable final coverage gate.
- `results/factorial_v2/COMPLETENESS_REPORT.md` — human-readable final coverage report.
- `results/factorial_v2/figure_provenance.json` — source hashes for generated tables/figures.
- `results/factorial_v2/plots/` — completeness-gated poster plots.

## Historical v2 verification (withdrawn for MMD claims)

- Queue: 133/133 jobs and 18/18 phases completed.
- Coverage gate: 27/27 checks pass in `results/factorial_v2/completeness.json`.
- EchoNext: 48/48 cells; 5,442 clean records per cell and 2,198 morphology/stress records per cell.
- Tests: 37/37 focused loss, statistics, streaming, and queue-completion tests pass.
- Scheduler safety: `.agents/skills/experiment-queue/scripts/queue_manager.py` now requires the current attempt's exit code and a freshly written expected output, preventing stale artifacts from certifying completion.
- Storage: the migrated 400,569,587-byte `bridge.pt` duplicate was hash-verified and hard-linked in place; generated Python/test caches were removed.

Historical timestamped plans, stale pilot reports, and ad-hoc figures are not authoritative unless referenced above.

## Quarto analysis book and review provenance (2026-08-24)

- `book/_quarto.yml` and the 19 configured QMD chapters — executable analysis
  book, separated into three-lead ECG-AIM and one-lead wavelet/SSL evidence.
- `book/requirements-book.txt` — pinned book analysis environment contract.
- `book/real_dataset_umap.py` — bounded, pseudonymized 96-feature waveform
  embedding loaders for PTB-XL, EchoNext, LUDB, ISP, Sunnybrook, Zhejiang, and
  RDB, with trustworthiness and three-seed projection-stability diagnostics.
- `scripts/{render_quarto_chapters,snapshot_book_inputs,vendor_book_runtime,audit_quarto_book,build_book_release_manifest}.py`
  — sequential source-bound rendering, immutable SQLite snapshots, local runtime
  vendoring, fail-closed audit, and hash-bound release decisions.
- `review-stage/BOOK_{FOUNDATIONS,BENCHMARKS,LIVE_SYSTEM}_REVIEW_20260824.*` —
  first independent review round; timestamped raw Markdown and JSON findings.
- `review-stage/BOOK_{FOUNDATIONS,BENCHMARKS,LIVE_SYSTEM}_REVIEW_ROUND2_20260824.*`
  — fresh-context second review round and machine-readable findings.
- `review-stage/CHAPTER_IMPROVEMENT_PLAN.md` — chapter-by-chapter remediation
  plan; `review-stage/render-round*/` — resumable executable render evidence.
- `review-stage/BOOK_MECHANICAL_AUDIT.{md,json}` — current mechanical release
  gate; `.aris/traces/book_overhaul_20260824/README.md` — workflow trace.
- `book/_book/RELEASE_MANIFEST.json` — generated only after a clean render and
  audit; remains explicitly provisional/nondeployable while the independent
  scientific completion gate is not PASS.
