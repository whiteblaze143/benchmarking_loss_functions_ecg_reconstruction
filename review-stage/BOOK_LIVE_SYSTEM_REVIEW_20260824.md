review_independence: same-family  
acceptance_status: provisional  
audit snapshot: 2026-08-24 01:34 EDT  
edits made: none

## Executive verdict

Overall score: **4/10 — reject release in current form.**

The strongest aspect is clinical restraint: Chapter 14 clearly prohibits clinical deployment and Chapters 13/16 repeatedly qualify exploratory findings. I found no direct claim that the reconstructed ECG is ready for diagnosis or cath-lab use.

The release nevertheless has two critical blockers:

1. **Chapter 13 is not executable and its HTML hides that failure.** The current render fails at `book/13_engineering_ml_visual_atlas.qmd:117` because `results/factorial_mixed_level/inference_readiness/summary.json` is absent. Later required training and temporal-result roots are absent as well. The checked-in HTML contains 28 code cells but zero outputs, tables, or figures; for example, `book/_book/13_engineering_ml_visual_atlas.html:382` proceeds directly from source code to prose at line 409.
2. **The documented build is not the build that produced the site.** `book/README.md:14-18` pins Quarto 1.10.18, but all five audited HTML pages report `quarto-1.4.555` at HTML line 5. This invalidates the claimed reproducible release path.

| Chapter | Score | Executable status | Readiness |
|---|---:|---|---|
| 13 Engineering atlas | 2/10 | **FAIL**; missing required artifacts | Release-blocking |
| 14 Clinical trust | 8/10 | Static; renders | Publishable as a safety/protocol chapter only |
| 15 Observatory | 6/10 | PASS at audited snapshot | Internal monitoring beta |
| 16 Three-lead ECG-AIM | 4/10 | PASS, but most production analyses pending | Exploratory only; not results-ready |
| 17 One-lead wavelet/SSL | 3/10 | PASS, but 10/120 runs, one seed/lead | Screening only |
| Whole book | 4/10 | In-scope release gate fails | Not deploy-ready |

## Chapter 13 — Engineering, Storage, and Inference Diagnostics

Score: **2/10**

### CRITICAL

- **The current chapter cannot execute.** It unconditionally reads the inference-readiness bundle at `book/13_engineering_ml_visual_atlas.qmd:112-127`; the first missing file triggers a `FileNotFoundError` at line 117. The referenced training-diagnostic artifacts at lines 255-270 and temporal artifacts at lines 361-368 and 442-453 are also absent in the current workspace.
  
  Minimum fix: either restore and checksum-bind every declared artifact before rendering, or replace the affected sections and all downstream numerical prose with an explicit “artifact unavailable” state. A release gate must execute the chapter and require its expected output inventory.

- **The deployed HTML masks the failed execution.** All 28 cells show source only, with zero output blocks, tables, or figures. `book/_book/13_engineering_ml_visual_atlas.html:382-409` is the first clear example. Nevertheless, prose asserts concrete measurements at `book/13_engineering_ml_visual_atlas.qmd:152`, `197`, `749`, and `1021-1025`.
  
  Minimum fix: never retain an old/no-execute HTML after a failed render. Render to a staging directory and atomically promote only if every required cell executes and an expected-output manifest passes.

### MAJOR

- **“Current local bytes” are not actually measured.** Lines 83-86 count a catalog row’s nominal size whenever `local_path` is nonempty; they do not test whether the file exists, inspect its actual size, or account for cache/WAL files. The label at line 97 is therefore stronger than the computation.
  
  Minimum fix: resolve each path, require existence, use `stat().st_size`, deduplicate by content hash, and separately report base DB, WAL, cache, and retained-output usage.

- **Live input reads are not snapshot-consistent.** Queue JSON, audit JSON, JSONL catalog, generated CSVs, and metadata are read separately at lines 15-24 and throughout the chapter. Concurrent updates can mix generations or expose a partially appended JSONL line.
  
  Minimum fix: build an immutable release manifest or stage all inputs under one snapshot identifier before execution.

- **The optional target-EDA gate is internally incomplete.** Lines 369-371 test only three paths, but line 373 immediately reads `target_cache_metadata_path`, which was omitted from the existence test.
  
  Minimum fix: include every consumed path in the gate and report exactly which prerequisite is missing.

- **Other “live” sections are unconditional.** `accepted_summary.json` and its tables are read at lines 442-453 without a pending-state guard. That contradicts the otherwise fail-closed language.
  
  Minimum fix: decide per section whether missing data is a hard release error or a visibly pending analysis; implement that contract consistently.

- **Hard-coded numerical prose will drift.** The concrete values at lines 749 and 1021-1023 are not generated from the displayed data and remain visible even when outputs disappear or the accepted generation changes.
  
  Minimum fix: generate the narrative values from checked tables, or label them with a fixed artifact hash and as-of timestamp.

- **The inferential evidence is still one-seed, detector-conditioned, and multiplicity-unadjusted.** The chapter acknowledges this at lines 691, 1025, and 1027. These are useful diagnostics but not loss-function effects.
  
  Minimum fix: complete prespecified seed blocks, use patient-clustered paired contrasts, adjust the endpoint family, and model differential detection failure.

- **The “visual atlas” lacks the clinically necessary visual object: ECG waveforms.** Even if execution were restored, it mainly supplies tables, heatmaps, and aggregate feature plots. There are no measured-versus-reconstructed waveform overlays, fiducial markers, tail-error cases, or detector-failure examples.
  
  Minimum fix: add digest-bound representative and worst-case waveform panels selected by a prespecified rule.

### MINOR

- `book/13_engineering_ml_visual_atlas.qmd:1031` says inference latency is missing, while lines 146-152 already report CPU forward timings. Clarify that robust end-to-end latency, tail latency, and hardware-normalized throughput remain missing.
- The anchor model is selected lexicographically at line 636 rather than by a registered control. As accepted models change, the displayed anchor could change silently.
- Provenance checks rely on `assert`; use explicit validation exceptions so optimization flags cannot disable gates.

Executable status: **FAIL**. Python syntax is valid, but required artifacts are missing and the current functional render stops at cell 4.

Readiness: **Not publishable as evidence.**

## Chapter 14 — Clinical Trust, Cath-Lab Safety, and Validation Requirements

Score: **8/10**

This is the best audited chapter. The boundary at `book/14_cath_lab_clinical_trust_atlas.qmd:3-8` is explicit and appropriate. The distinction among signal, morphology, diagnosis, and workflow evidence at lines 13-22 is especially valuable.

### MAJOR

- **The ischemia plan is not yet a complete statistical analysis plan.** Lines 24-28 omit a primary estimand, recruitment design, ECG-to-reference time window, label adjudication procedure, missing/indeterminate-label handling, sample-size/power calculation, and site-level heterogeneity model.
  
  Minimum fix: add these items and identify one primary endpoint plus a hierarchy of secondary endpoints.

- **The non-inferiority notation is underspecified.** `D` at lines 32-36 has no defined direction, scale, paired estimator, confidence level, alpha, or justification for the margin.
  
  Minimum fix: provide a worked endpoint-specific formulation and power calculation.

- **Subgroup and endpoint multiplicity are named but not operationalized.** Lines 40-47 and 75-81 do not specify minimum support, pooling/hierarchical methods, or a multiplicity family.
  
  Minimum fix: prespecify subgroup support rules and a hierarchical/Holm/FDR strategy.

- **The reader-study plan lacks an executable MRMC design.** Lines 69-71 do not state reader/case sample size, crossover analysis, carryover handling, or the statistical model.
  
  Minimum fix: add a concrete multi-reader multi-case estimand and power design.

### MINOR

- Add references to the clinical/statistical standards underlying the protocol.
- The chapter is called an atlas but has only one table. A claim ladder or decision-flow diagram would make the “signal → morphology → diagnostic → workflow” boundary easier to use.
- The release checklist should add dataset shift monitoring, model-change control, privacy/security review, post-deployment surveillance, and rollback criteria.

Executable status: **Static render passes.**

Readiness: **Suitable for publication as a safety and validation-requirements chapter, not as evidence that validation has occurred.**

## Chapter 15 — Live Experiment Observatory

Score: **6/10**

### MAJOR

- **The “front door” inventories eight sources but summarizes only four streams.** Sources are listed at `book/15_live_results_observatory.qmd:20-29`; status rows at lines 40-53 omit clinical evaluation, both checkpoint catalogs, and the running checkpoint-representation study.
  
  Minimum fix: report every declared source’s lifecycle state, completeness numerator/denominator, last successful write, and error/heartbeat status.

- **No coherent database snapshot is established.** Separate helper calls open independently changing SQLite files at lines 40-43. The rendered values are individually valid but need not represent the same instant.
  
  Minimum fix: create read-only SQLite backups/snapshots first, assign one render ID, and query only those snapshots.

- **The advertised modification time is wrong for WAL-backed activity.** The rendered one-lead page reports the base database updated at `book/_book/17_one_lead_wavelet_ssl_live.html:433`, yet jobs completed much later at lines 610 and 622. The inventory is looking at the base file while current writes live in the WAL.
  
  Minimum fix: report base plus `-wal` mtimes/sizes, SQLite `data_version`, and a transactionally recorded logical-update timestamp.

- **Critical helper behavior is hidden behind a wildcard import.** `book/15_live_results_observatory.qmd:18` imports `live_results.*`, including `ROOT`, metric direction rules, read-only connection behavior, and embedding behavior, without displaying a helper version or digest.
  
  Minimum fix: use explicit imports and display/check the helper source digest and schema version.

- **A missing or not-yet-created database is not handled gracefully.** `source_inventory` may show `exists=False`, but lines 40-43 immediately load four databases.
  
  Minimum fix: convert missing sources into explicit blocked/pending rows or make them a deliberate release failure with a clear message.

- **The rerun command is inconsistent with the README.** Lines 74-78 omit the `PATH` setting that `book/README.md:20` says is required.
  
  Minimum fix: expose one canonical script/Make target and reuse it everywhere.

### MINOR

- Normalize `completed` and `complete`; the current summary produces separate columns and floating-point counts.
- “Newly committed DB rows” at lines 57-58 is ambiguous because the authoritative DBs are not Git-committed. Use “transactionally persisted.”
- Add a site-wide “snapshot as of” banner and render ID.

Executable status: **PASS** at the audited snapshot; two executed cells, two outputs, no rendered errors.

Readiness: **Useful internal dashboard, not yet an authoritative or atomic observatory.**

## Chapter 16 — Three-Lead ECG-AIM: Live Results

Score: **4/10**

The chapter is unusually careful about interpretation, but the deployed “results” are mostly preliminary or pending.

### MAJOR

- **The production representation analysis is incomplete.** The rendered study has one running checkpoint with 896/2,398 records at `book/_book/16_three_lead_ecgaim_live.html:1281-1292`. Probe, rigorous held-out, probability, paired-checkpoint, repolarization, multiclass, QC, and geometry sections all render pending states; examples appear at HTML lines 1365, 1414, 1475, and 1518.
  
  Minimum fix: place the entire production subsection behind a prominent “analysis incomplete; no results” gate until required checkpoints and downstream tables are complete.

- **The raw leaderboard ranks semantically inapplicable values.** `book/16_three_lead_ecgaim_live.qmd:90-92` ranks every chosen metric within every dataset/target. The rendered leaderboard declares multiple `mae=0` entries for EchoNext `QRS_Overall` at HTML lines 664-702. The modular scatter at lines 101-108 likewise mixes endpoints and units and contains large degenerate zero/one strata.
  
  Minimum fix: introduce a typed endpoint–metric applicability registry, represent non-applicable cells as missing, reject sentinel zeros, and facet only comparable units/endpoints.

- **The headline clinical views exclude ECG-AIM.** Lines 79-85 classify `factorial_ecg_aim_*` as `threelead_ecgaim_external`, but the leaderboard and scatter filter only `canonical_160` at lines 91 and 102. Thus the opening “ECG-AIM” result views primarily show `f_*` models.
  
  Minimum fix: show a registered ECG-AIM-versus-control comparison or rename the sections to reflect the actual population.

- **“Held-out” is not yet patient-held-out.** Lines 341-343 admit that one cached row per ID does not establish absence of repeated persons, while the uncertainty uses record-level stratified bootstrap.
  
  Minimum fix: verify patient identity, split by patient, and bootstrap/group-resample patients. Do not label the analysis held-out until this gate passes.

- **The broad paired table invites multiplicity-driven discovery.** Lines 617-620 sort a large collection by unadjusted p-value and display the first 100. The smallest values are extreme and span many models/endpoints.
  
  Minimum fix: show only prespecified contrasts, organize hypothesis families, report adjusted values, and prioritize effect sizes with clinically meaningful units.

- **Live reads can combine inconsistent database states.** Jobs, probes, ORs, extended results, predictions, and comparisons are each queried separately at lines 285-287, 309-312, 347, 376, 401, and 419. A concurrent post-analysis write can make the rendered page internally inconsistent.
  
  Minimum fix: render from one SQLite backup or hold one read transaction/connection for the entire chapter.

- **Several visualizations are non-reproducible or incorrect.**
  
  - Lines 487 and 545 select `.iloc[-1]` from unordered SQL results, making the displayed checkpoint arbitrary.
  - Lines 531-535 connect UMAP stability observations by job but not seed, so separate seed replicates can be connected as one line.
  - Lines 457-464 call the plot Bland–Altman but show only the zero line, not bias or limits of agreement.
  - Lines 588-595 promise nearest-neighbor waveform inspection but contain no executable inspection output.
  
  Minimum fix: expose an explicit checkpoint control, order SQL deterministically, use seed in the line group, draw bias/LoA, and implement the promised waveform panel.

- **There is a latent empty-table crash.** If `multiclass_results` exists but `confusion_matrix` is empty, lines 483-488 call `cm.model_id.iloc[-1]`.
  
  Minimum fix: guard both tables and validate one complete matrix per chosen model.

- **Clinical/reconstruction examples are missing.** The chapter has six plots but no measured-versus-reconstructed ECG overlays, no ST/QRS/T boundary examples, and no worst-tail failures.
  
  Minimum fix: add prespecified waveform and detector-failure panels, with measured input leads clearly distinguished from reconstructed leads.

- **Public output may leak operational or linkage identifiers.** Raw `error` fields are displayed at lines 293-295; patient IDs appear in the optional nearest-neighbor and UMAP outputs at lines 217-219 and 548-549.
  
  Minimum fix: sanitize errors and remove/hash identifiers before public HTML generation.

### MINOR

- GPU use can be enabled by editing `CHECKPOINT_DEVICE` at line 22 without an automatic training-GPU lock.
- The LUDB heatmap grows to roughly 2,232 px for 124 models under lines 641-644; provide filtering or pagination.
- The generic labels “clinical leaders” and “clinical endpoints” should distinguish measured clinical outcomes from model-mediated preservation proxies.

Clinical overclaim assessment: **No direct unsafe clinical claim.** Lines 302-307, 434-442, 494-502, and 603-611 contain strong caveats. The remaining problem is that section titles and visuals can still make proxy and incomplete analyses look more mature than they are.

Executable status: **PASS** at the audited snapshot; 23 cells, 25 output blocks, no rendered execution errors. Scientific completion is a separate failure.

Readiness: **Exploratory dashboard only; not claim-ready and not clinically deployable.**

## Chapter 17 — One-Lead Wavelet and SSL: Live Results

Score: **3/10**

### MAJOR

- **The experiment is only 10/120 complete.** The rendered snapshot has 109 pending, 10 completed, and one running job at `book/_book/17_one_lead_wavelet_ssl_live.html:465-478`. Every completed configuration is lead I, seed 42; the replication table reports `n_seeds=1`.
  
  Minimum fix: place leaderboards and embeddings behind a screening-only banner and prevent confirmatory language until the registered seed/lead cells complete.

- **The advertised SSL comparison does not yet exist.** The registered hierarchy at `book/17_one_lead_wavelet_ssl_live.qmd:42-54` requires `A0 → wavelet → wavelet+SSL`, but the completed run names are raw, `noSSL`, or phase/mechanism configurations. No completed SSL replication supports the chapter title.
  
  Minimum fix: explicitly display which registered contrasts are estimable and suppress unavailable comparisons.

- **There is no valid inferential unit.** Lines 60-65 compare one aggregate result per run; lines 103-107 compute mean/std with one seed. No patient/record-level paired uncertainty or multiplicity control is provided.
  
  Minimum fix: use identical test records, patient-clustered paired deltas, at least the registered seed count, and adjusted endpoint families.

- **Validation/smoke metrics are being ranked as results.** Lines 70-76 include `boundary_f1_smoke`, validation Pearson, and validation loss. This is suitable for screening, not final comparative evidence.
  
  Minimum fix: separate optimization/selection metrics from a locked external test table.

- **The improvement heatmap averages one observation.** Lines 89-96 compute a mean across cells even though every current group has one seed.
  
  Minimum fix: show raw per-seed deltas until replication exists; only add means and intervals after the minimum seed gate.

- **The performance UMAP is decorative at the current sample size.** Lines 118-127 embed only ten completed runs across numerous correlated metrics; the rendered title uses `n_neighbors=9`. There is no trustworthiness or seed/neighborhood sensitivity.
  
  Minimum fix: suppress it until materially more runs exist, or use a transparent PCA/biplot and show loading/sensitivity diagnostics.

- **The mechanism claim is too specific.** Lines 50-54 say a T-on/T-off-localized gain would “specifically support” the physiological second-view hypothesis. Detector artifacts, selection, leakage, or regularization could produce the same pattern.
  
  Minimum fix: say “would be consistent with,” and require phase-scrambling, matched-capacity, detector, and negative-control experiments.

- **Dataset and checkpoint integrity is reduced to counts.** Lines 137-143 count `.pt` files and sum catalog bytes but do not verify record manifests, patient uniqueness, file hashes, checkpoint contract, or readability.
  
  Minimum fix: show manifest hashes, expected-versus-observed identities, integrity failures, and a load smoke test.

- **Resource and resume health are inadequate for a live queue.** The page exposes status/timestamps/errors but no heartbeat age, elapsed time, retry count, ETA, GPU/RAM/disk state, or stuck-job rule.
  
  Minimum fix: add a queue-health table and sanitize raw error text.

### MINOR

- The source inventory’s base-file time and size ignore WAL activity, as demonstrated by HTML lines 429-435 versus later job completions.
- Add training curves and measured/reconstructed waveform/delineation overlays.
- Include an explicit statement that current ranking is adaptive screening and cannot be used for final model selection.

Executable status: **PASS** at the audited snapshot; nine cells, 13 outputs, no rendered errors.

Readiness: **Early screening only.**

## Whole-book navigation and deployment integrity

### What passes

- The declared live pages are first in navigation: `_quarto.yml:17-23`.
- Next/previous links correctly traverse index → 15 → 16 → 17 → Chapter 9.
- All 19 root HTML pages exist.
- I found **zero missing local `src`/`href` targets** across the current rendered book.
- `.nojekyll`, `search.json`, and `site_libs/` are present.
- The five HTML pages embed source matching the current QMD text.

### CRITICAL / MAJOR deployment problems

- **Renderer mismatch:** `book/README.md:13-18` declares Quarto 1.10.18, but each audited HTML page says Quarto 1.4.555 at line 5.
  
  Minimum fix: make the renderer refuse any version other than the pinned version and write Quarto/Python/package hashes into a release manifest.

- **The in-scope release gate fails:** Chapter 13 cannot render, directly violating `book/README.md:84`, which says any failed Python cell is a failed release.
  
  Minimum fix: do not deploy or commit `_book` until a clean staged render completes.

- **The deployment is not atomic.** `book/README.md:29-43` encourages independent page refreshes while databases are changing. A failed render leaves old HTML in place, exactly what happened for Chapter 13.
  
  Minimum fix: render all pages into a temporary output directory from immutable DB snapshots, run validation, then atomically swap the directory.

- **The Pages artifact is not self-contained.** `_quarto.yml:64-66` and `book/README.md:22-27` emphasize shared local resources, but the selected pages still make nine Plotly CDN requests plus external jQuery, RequireJS, MathJax, and polyfill requests. Example: `book/_book/16_three_lead_ecgaim_live.html:60` and Plotly at line 806.
  
  Minimum fix: vendor runtime JS locally, or explicitly document and test the network/CSP requirement.

- **The build environment is machine-specific.** Absolute user paths appear throughout `book/README.md:14-17` and `39-41`; `_quarto.yml:80` depends on a named kernelspec.
  
  Minimum fix: provide a repository-relative build entry point, environment lock, and kernel bootstrap/check.

- **Warnings are globally suppressed.** `_quarto.yml:72-78` disables warnings/messages during live statistical computation.
  
  Minimum fix: capture warnings in release logs and fail on selected numerical, schema, deprecation, and convergence warnings.

- **There is no release manifest binding source to data.** Source inventory shows paths, mtimes, sizes, and row counts, but not one site-wide set of QMD hashes, helper hashes, DB snapshot hashes, schema versions, environment versions, and build time.
  
  Minimum fix: generate and display a signed/hash-bound manifest.

- **The checked-in snapshot is not independently reproducible from the repository.** `book/README.md:70-73` states authoritative databases/checkpoints are ignored.
  
  Minimum fix: publish deidentified immutable release databases or a citable artifact bundle plus manifest, while keeping full checkpoints private if necessary.

- **The release checklist is incomplete.** `book/README.md:76-84` does not check Chapter 13 output presence, pending-result gates, renderer identity, external-CDN availability, HTML/JS smoke behavior, source/render hashes, or atomic snapshot consistency.
  
  Minimum fix: automate these checks and make deployment conditional on them.

### Navigation/documentation issues

- Inserting three live pages at the front shifts source file 13 to rendered Chapter 17 and source file 14 to Chapter 18. `book/README.md:82` says “Chapters 9–11,” which is ambiguous between filenames and displayed chapter numbers.
- `_quarto.yml:11` retains a July 31 date while deployed live snapshots are from August 24. Add a separate build/snapshot timestamp.
- Calling static GitHub Pages “live” is acceptable only if every page prominently states that it is a render-time snapshot and gives its age.

## Minimum release package

1. Restore or remove Chapter 13’s missing artifact-dependent claims; obtain a clean executed render with expected tables/figures.
2. Render with the pinned Quarto version into a staging directory from immutable SQLite snapshots.
3. Add a release manifest and automated gates for renderer, source/data hashes, output counts, cell failures, local/external resources, and pending-state policy.
4. Gate Chapter 16’s incomplete representation analysis and remove semantically invalid zero-valued leaderboard rows.
5. Gate Chapter 17 as single-seed screening until registered SSL, lead, and seed cells complete.
6. Add patient-clustered uncertainty, prespecified multiplicity families, and patient-disjoint split verification.
7. Add clinically interpretable waveform/fiducial/failure-case figures.
8. Vendor JS dependencies and sanitize public IDs/errors.

Final disposition: **not ready for public scientific release, not ready for confirmatory claims, and explicitly not ready for clinical use.** Chapter 14 can remain as the governing safety boundary while the executable/result chapters are repaired.
