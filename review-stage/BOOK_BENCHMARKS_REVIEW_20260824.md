# Zero-context senior review

Overall verdict: **FAIL / not publication-ready**. Mean score: **3.7/10**.

I reviewed all prose and all 63 Python chunks in the six QMDs, plus the corresponding HTML. No files were edited.

All Python chunks parse syntactically. The decisive execution problem is that chapters 07, 08, 10, and 12 have **zero rendered cell outputs**; only 09 and 11 show executed cells. The entire `results/comprehensive_latest_48_models/` directory referenced by 07–10 is currently missing.

| Chapter | Score | Executable/render status | Readiness |
|---|---:|---|---|
| 07 smartwatch | 3/10 | 6 valid Python chunks; current inputs missing; HTML has 0 outputs | Not ready |
| 08 factorial | 2/10 | 12 valid chunks; fails at missing master table; HTML has 0 outputs | Not ready |
| 09 real-data EDA | 5/10 | 18/18 chunks rendered successfully, but one evidence gate falls back to “unavailable” | Major revision |
| 10 EchoNext validation | 2/10 | 12 valid chunks; current tables/archive missing; HTML has 0 outputs | Not ready |
| 11 dataset atlas | 6/10 | 13 chunks executed; no runtime errors | Atlas usable after corrections; not benchmark-ready |
| 12 exhaustive analysis | 4/10 | 2 chunks run read-only, but HTML has 0 outputs and status sources are asynchronous | Specification only |

## Cross-cutting artifact/render audit

- All HTML files are newer than their QMD sources, so they are not source-timestamp stale.
- Chapters 07, 08, 10, and 12 were nevertheless rendered without execution. Their HTML cell containers lack `data-execution_count` and contain no `cell-output` or Plotly graph nodes: e.g. `book/_book/07_smartwatch_reconstruction_case_study.html:426`, `book/_book/08_factorial_loss_matrix_benchmarks.html:473`, `book/_book/10_echonext_classifier_external_validation.html:561`, `book/_book/12_exhaustive_model_analysis.html:414`.
- Missing current artifacts include:
  - `results/comprehensive_latest_48_models/` in its entirety;
  - `results/factorial_v4/temporal_mmd_evaluation.csv`;
  - `results/factorial_mixed_level/per_record_manifest.json`;
  - `results/factorial_mixed_level/release_report.json`;
  - `results/factorial_mixed_level/summary.csv`.
- Present mixed-level sources are from incompatible snapshots:
  - `queue_state.json`: 2026-08-13;
  - `compatibility_audit.json`: 2026-08-22;
  - `catalog.sqlite`: 2026-08-24 01:19, newer than the chapter-12 HTML rendered at 00:54.
- Read-only execution of chapter 12 currently gives 394 completed, 1 running, 85 pending. The audit has 588 rows: 310 current-manifest compatible IDs plus 278 historical extra IDs, with 170 manifest IDs absent from the audit. This materially affects the release-gate findings below.

# Chapter 07 — Four-Device Smartwatch Transfer Case Study

Score: **3/10**. Verdict: **not ready**.

### CRITICAL

1. **The evidence bundle is absent and the HTML contains no computed outputs.**

   Evidence: inputs are read at `book/07_smartwatch_reconstruction_case_study.qmd:34-44`; the six HTML cells begin at `book/_book/07_smartwatch_reconstruction_case_study.html:426`, `:464`, `:489`, `:510`, `:531`, and `:558`, but none has execution output.

   Impact: every numerical prose claim—192 rows, paired-record counts, device distributions, anchors, calibration—is currently a static assertion.

   Minimum fix: restore a hash-pinned public/reviewer-accessible bundle, fail render if absent, execute all cells, and show artifact digest and generation timestamp.

### MAJOR

2. **The “full Cartesian product” check is not a real completeness check.**

   Evidence: `qmd:54-55` merely displays row count and `n_models × n_devices`; it does not assert unique `(model_id, device)` keys, exact expected IDs, duplicates, or missing cells. `qmd:94-107` uses `paired_records=("n_paired_records","first")`, which can hide inconsistent counts across models.

   Minimum fix: assert 192 unique model-device keys, exact 48-model and four-device sets, no duplicates, and constant per-device record identities/counts across models.

3. **The anchor table can silently be incomplete.**

   Evidence: fixed IDs are filtered at `qmd:137-153` without asserting that all six anchors occur for all four devices.

   Minimum fix: assert all 24 anchor-device rows and exact mask/family mapping before display.

4. **No uncertainty, record-level pairing, or multiplicity control is executed.**

   Evidence: the chapter explicitly concedes this at `qmd:156`, but all current displays remain aggregate model rows. Four devices, multiple tasks, six anchors, and multiple fidelity endpoints are repeatedly inspected.

   Minimum fix: publish per-record paired outputs; use patient/record-cluster bootstrap or an explicitly specified hierarchical model; define endpoint families and multiplicity adjustment.

5. **The required failure analysis is entirely prospective.**

   Evidence: baselines, per-lead analysis, alignment sensitivity, condition strata, and discordant-record analysis are only listed at `qmd:183-191`.

   Minimum fix: execute at least the simple baselines, all missing leads separately, before/after alignment, four protocol strata, and algorithmically selected failure cases.

### MINOR

6. **The smartwatch output contract is ambiguous across chapters.**

   Chapter 08 says observed I, II, and V2 are copied (`book/08_factorial_loss_matrix_benchmarks.qmd:263`), while chapter 07 zero-fills I and V2 and scores eleven leads (`qmd:17-21`). It is unclear whether I/V2 smartwatch “predictions” are learned outputs or deterministic copied zeros.

   Minimum fix: state the exact output postprocessing and score mask for the smartwatch adapter.

# Chapter 08 — Interim 48-Cell and Running Mixed-Level Benchmark

Score: **2/10**. Verdict: **not ready**.

### CRITICAL

1. **The locked 48-cell package is absent, and all dynamic results/gates are missing from HTML.**

   Evidence: the master is loaded at `qmd:51-53`; family-wise tests, effects, and seed files at `qmd:236-239`, `:283-286`, and `:304-307`. Those paths do not exist. HTML cells begin at `book/_book/08_factorial_loss_matrix_benchmarks.html:473` onward with no outputs.

   Minimum fix: restore and checksum the complete locked package, then render with execution enabled.

2. **The purported “ANOVA” is hard-coded, not calculated, and includes a nonexistent interim factor.**

   Evidence: `qmd:163-180` hard-codes `[0.0378, 0.0124, 0.0085, 0.0042]`; the factors include `VCG (V)`, although the actual 48-cell factors at `qmd:19-24` are E, C, MMD, and derivative. No ANOVA model is fit. The “Pareto frontier” at `qmd:147-161` is only a scatter plot with no dominance calculation.

   Impact: this conflates the later mixed-level VCG design with the locked four-toggle study and mislabels descriptive constants as inferential effects.

   Minimum fix: delete the hard-coded panel; compute marginal effects and intervals from the authoritative artifact; call them ANOVA only if a model is actually fit; calculate and mark nondominated points explicitly.

3. **The release compatibility gate is permanently contaminated by historical audit rows.**

   Evidence: it uses global audit counts and all audit IDs at `qmd:906-919`, requiring `compatible == 480`, `incompatible == 0`, and all audit IDs equal current IDs. Current audit state contains 310 current compatible rows plus 278 historical incompatible rows. The prose says quarantined histories are preserved (`qmd:1154-1167`), so the unfiltered gate cannot pass even after all current jobs complete.

   Minimum fix: filter audit rows to the exact manifest-ID set before current-release counts; report historical/quarantined rows in a separate table.

### MAJOR

4. **Queue, manifest, audit, and catalog are not bound into one coherent snapshot.**

   Evidence: `expected_job_ids` comes from live queue state at `qmd:757`, while scheduled IDs come from a different manifest; no equality assertion joins the two before release. The compatibility audit and SQLite catalog are independently loaded at `qmd:497-515` and `:906-919`, without freshness or shared snapshot digest.

   Current data happen to have equal queue/manifest ID sets, but the gate does not enforce that.

   Minimum fix: assert exact ID equality and a shared manifest/contract digest and snapshot generation across all inputs.

5. **Rendering mutates scientific result state.**

   Evidence: the chapter writes and atomically replaces `release_report.json` during render at `qmd:1085-1129`.

   Impact: rendering is no longer read-only or reproducible; `generated_at` changes on every run, and a book build can alter release state.

   Minimum fix: generate the release report in a dedicated validated pipeline; the QMD must read it only.

6. **The 48-cell inferential claim is seed-specific.**

   Evidence: seed 42 is the primary grid (`qmd:3`, `:25`); `qmd:295` and `:299-324` acknowledge that only selected masks have supplementary validation seeds. Yet the conclusion at `qmd:355` reads as a general loss result.

   Minimum fix: qualify all 48-cell effects as seed-42 results or complete the full multi-seed test evaluation.

7. **Multiplicity scope is underspecified.**

   Evidence: `qmd:215` uses α=0.0167 for three endpoints within each architecture, but the prose generalizes across three architectures while numerous other cells/endpoints and maxima are inspected.

   Minimum fix: state the exact confirmatory family; control across all claims in that family; label all maxima and transfer panels exploratory.

8. **The smoke test verifies the report more strongly than the output.**

   Evidence: `qmd:1324-1340` checks only that the output path exists and trusts report fields for finiteness/shape; it does not hash or reload the output tensor.

   Minimum fix: hash and reload the produced tensor and independently verify shape/finiteness against the report.

### MINOR

9. The inference command is machine-specific (`qmd:1295`) and not portable.

10. The evidence callout says all numbers below come from the locked package (`qmd:14`), but the running-study counts later come from queue/catalog artifacts.

# Chapter 09 — Real-Cohort EDA and Diagnostics

Score: **5/10**. Verdict: **major revision**.

### CRITICAL

1. **The rendered chapter says EchoNext endpoint-support evidence is unavailable, while the prose still makes exact support and identity claims.**

   Evidence:
   - exact support claim at `qmd:541-547`;
   - fallback at `qmd:552-564`;
   - rendered table explicitly says `unavailable` and “No endpoint-support result is claimed” at `book/_book/09_real_data_eda_and_diagnostics.html:2647-2662`;
   - shared 48-table label-hash claim at `qmd:609`;
   - significant current-result claim at `qmd:726`.

   Minimum fix: restore the label audit or compute support and label hash directly from currently available record-level data; suppress all dependent prose when unavailable.

### MAJOR

2. **Missing endpoint flags are converted to negatives despite an explicit warning not to do so.**

   Evidence: `qmd:530` says missing measurements must never be naively converted to zero, but `qmd:583` applies `.fillna(0).astype(bool)` to all component flags.

   Minimum fix: assert endpoint flags are complete; otherwise use explicit missingness and complete-case denominators per pair.

3. **The “every tensor” audit is a cached artifact, not a live bound audit.**

   Evidence: `qmd:289-299` only loads precomputed Parquet/JSON, after which `qmd:333` claims all 21,799 current tensors were checked. The content manifest at `qmd:347-358` is displayed but not cross-validated against the cached inventory or current files.

   Minimum fix: store and verify data-root, script hash, audit timestamp, row hash, and summary hash; refuse stale/mismatched audit artifacts.

4. **Several “actual full-array” EchoNext values are hard-coded rather than produced by the live block.**

   Evidence: the executable audit samples 256 records at `qmd:633-662`, but full-array clipping percentiles are static at `qmd:665-678`, and full-array 99th-percentile lead-law residuals are asserted at `qmd:689`.

   Minimum fix: load a hash-pinned full-array audit artifact and display it programmatically, or compute the claimed summaries live.

5. **The advertised deep diagnostics remain a checklist.**

   Evidence: spectral/morphology and cross-lead sections at `qmd:691-697` prescribe analyses but show none; reconstruction diagnostics at `qmd:699-716` are a rubric rather than executed analysis.

   Minimum fix: execute the promised spectra, fiducials, correlation matrices, condition numbers, failure rankings, and reconstruction-level diagnostics.

### MINOR

6. `qmd:16` claims every executable audit reads under `../data/` and prints the resolved path; multiple blocks instead read `results/` and `refine-logs/`, and most do not print resolved paths.

7. `qmd:69` counts `~age.between(0,120)`, which also counts missing ages as “outside” if missingness appears later. Use `age.notna() & ~age.between(...)`.

8. The source calls this the “first numbered chapter” (`qmd:3`), but rendered HTML labels it chapter 4 (`book/_book/09_real_data_eda_and_diagnostics.html:585`).

# Chapter 10 — EchoNext Classifier External Validation

Score: **2/10**. Verdict: **not ready**.

### CRITICAL

1. **All result inputs are absent and none of the 12 HTML cells executed.**

   Evidence: per-task CSV at `qmd:99-100` and `:169-171`; archive at `qmd:242-245`; HTML cells start at `book/_book/10_echonext_classifier_external_validation.html:561`, `:703`, `:841`, etc., with no outputs.

   Impact: endpoint metrics, rare-task examples, reference performance, drift tails, calibration, subgroup findings, and worst records are static, currently unverifiable claims.

   Minimum fix: restore the exact archive/table and render every output with artifact digests.

### MAJOR

2. **Reconstructed clinical metrics are not recomputed from the same probabilities used for fidelity.**

   Evidence: at `qmd:309-326`, AUROC/AP/Brier/ECE are copied from the loose `tasks` table; only fidelity is computed from archived probabilities at `qmd:313` and `:328-332`.

   Impact: a stale or mismatched summary CSV can disagree with the archive without detection.

   Minimum fix: recompute every reconstructed per-task metric from archived clean probabilities and reference labels; compare against the CSV and fail on mismatch.

3. **Archive integrity checks are incomplete.**

   Evidence: `qmd:451-465` audits reference rows only. Later `qmd:495-500` checks reconstructed row count and ECG-key order, but there is no assertion that probabilities are finite/in [0,1], labels are binary/equal to reference, row indices are unique, all 12 tasks are present, or exactly 18 conditions exist.

   Minimum fix: add a full schema/value/condition/label identity gate for every anchor, ideally all 48 models.

4. **Uncertainty is missing for the clinically relevant endpoints.**

   Evidence: only mean probability drift receives a 1,000-resample percentile interval (`qmd:538-558`). AUROC/AP, calibration, task deltas, full-vs-MSE contrasts, subgroup differences, and rare-task metrics have no intervals. The statistical-inference section at `qmd:807-815` is prospective.

   Minimum fix: paired patient bootstrap/DeLong as appropriate for AUROC deltas; bootstrap AP/calibration/drift contrasts; report rare-task intervals and prespecified multiplicity.

5. **External-test exploration is extensive and remains vulnerable to test-set selection.**

   Evidence: top-12 macro ranking at `qmd:217-225`, highest-drift subgroup selection at `qmd:693-700`, worst-record selection at `qmd:730-735`, and six nominal correlations at `qmd:781-802`. Caveats are present, but no locked selection artifact is shown.

   Minimum fix: make all such panels explicitly exploratory and keep confirmatory claims limited to versioned prespecified anchors/endpoints.

6. **The promised aggregate/clinical diagnostics are incomplete.**

   Evidence: required views are listed at `qmd:205-213`, but there is no 48×12 heatmap, factorial-effect analysis, prevalence-performance plot, rare-task calibration panel, or waveform overlay. `qmd:756` explicitly says overlays are unavailable. Morphology discordance at `qmd:767` promises QRS/ST/amplitude comparisons but executes only Pearson-versus-drift.

   Minimum fix: provide those outputs or narrow the chapter title and claims.

### MINOR

7. “Official frozen minimodel” is trusted from summary metadata; no weight/architecture/adapter digest is verified in this chapter (`qmd:40-51`).

8. The AUROC explanation at `qmd:63-67` omits the half-credit tie term and should say “can be misleading,” rather than implying class imbalance itself changes AUROC.

# Chapter 11 — Dataset Atlas

Score: **6/10**. Verdict: **usable descriptive atlas after major corrections; not ready as a benchmark foundation**.

### MAJOR

1. **Executed ISP integrity output contains invalid targets, but prose does not disclose or resolve them.**

   Evidence: audit code at `qmd:321-348`; rendered output reports one invalid-width and one out-of-window test record, plus two out-of-window training records at `book/_book/11_dataset_atlas_ludb_isp_sunnybrook.html:1768-1782`. The following prose at `qmd:351-353` does not identify, exclude, or adjudicate them.

   Minimum fix: list affected records/intervals, define a versioned repair/exclusion policy, recompute counts and widths after that policy, and prevent model use until resolved.

2. **The claimed one-to-one ISP linkage is not actually tested.**

   Evidence: `qmd:220` claims one-to-one linkage; `qmd:259-273` only checks that each CSV-referenced `.hea`/`.dat` exists. It does not detect duplicate `file_name`s, orphan waveform files, cross-split duplicates, or subject duplication.

   Minimum fix: assert unique filenames per split, exact referenced-vs-present set equality, cross-split content-hash disjointness, and subject independence when identifiers become available.

3. **ISP/Sunnybrook governance gaps block benchmark-grade use.**

   Evidence: unresolved source/license/class semantics at `qmd:27`, `:33-34`, `:351`; nevertheless the role matrix calls ISP suitable for “delineation learning/audit” at `qmd:463`.

   Minimum fix: treat ISP model training as blocked until source, license, version, class mapping, and subject unit are documented; treat Sunnybrook as local exploratory data only.

4. **The “fully measured limb leads” inference is physiologically/methodologically misleading.**

   Evidence: `qmd:442` says fully measured limb leads can expose models exploiting algebraic identities. The rendered Sunnybrook table shows `III-(II-I)` residuals near numerical zero, while augmented-lead residuals are only a few µV (`book/_book/11_dataset_atlas_ludb_isp_sunnybrook.html:2085` onward), consistent with derived limb leads. Independently acquired morphology is in the precordial/electrode potentials, not an independent III/aVR/aVL/aVF ground truth.

   Minimum fix: distinguish measured electrodes from derived lead channels and do not use derived limb leads as independent validation targets.

5. **Sunnybrook transient reporting is incomplete.**

   Evidence: `qmd:452` highlights 20.44 mV V5 in ECG010 and 6.66 mV V4 in ECG008, but the rendered EDA also shows ECG010 at 20.05 mV in V4 and 16.605 mV in V6 (`HTML:2043-2077`).

   Minimum fix: algorithmically enumerate every lead/record exceeding a prespecified threshold, not selected examples.

6. **The stated ~18 µV Einthoven residual is unsupported by the rendered table.**

   Evidence: `qmd:454`; rendered III residual is essentially zero and augmented-lead residuals shown are generally about 1.7–5 µV.

   Minimum fix: cite and display the exact statistic/definition producing 18 µV or remove it.

### MINOR

7. LUDB p99 flat-difference fractions are approximately 0.46–0.58 in the rendered table, but `qmd:180-204` does not explain that these likely reflect quantization rather than flatline duration. Define a clinically meaningful flatline detector.

8. ISP interval-width numbers at `qmd:218` are not computed by the displayed code, and currently include invalid intervals.

9. LUDB and ISP “diagnostics enabled” sections remain prospective; no reconstruction/delineation checkpoint is evaluated (`qmd:206-210`, `:351`).

# Chapter 12 — Generation-Bound Exhaustive Model Analysis

Score: **4/10**. Verdict: **acceptable as an analysis specification, not as exhaustive analysis or live status evidence**.

### MAJOR

1. **The rendered HTML contains no live inventory output.**

   Evidence: `book/_book/12_exhaustive_model_analysis.html:414` and `:453` show code-only cells. Thus “Training is active” at `qmd:8` and every status count are unsupported in the rendered artifact.

   Minimum fix: execute the chapter and visibly timestamp its snapshot, or rename it as a static protocol chapter.

2. **The displayed compatibility counts pool current and historical generations.**

   Evidence: `qmd:22-25` loads the audit; `qmd:43-46` displays global counts; `qmd:50` claims earlier quarantined generations are not pooled. Current read-only evaluation shows 310 current compatible IDs and 278 historical extra/incompatible IDs, so the displayed “contract-incompatible” count does pool historical rows.

   Minimum fix: filter to exact manifest IDs, show 310/480 current coverage and 170 not-yet-audited separately, and report historical quarantine counts in another table.

3. **The “live” inputs are asynchronous and stale relative to one another.**

   Queue state is 11 days older than the current date, the audit is two days newer than queue state, and the catalog is newer than the HTML. No code checks their shared manifest digest or snapshot time.

   Minimum fix: produce an atomic status snapshot or require matching manifest/contract hashes and report each source timestamp prominently.

4. **The source does not assert per-seed full-grid identity.**

   Evidence: `qmd:35-41` verifies global counts and allowed character sets, but not that each seed’s exact mask set equals the 160-condition expected set. The current manifest does happen to have 160 unique masks per seed, but the executable gate should enforce it.

   Minimum fix: construct the expected set and assert equality for each seed.

5. **There is no model/result analysis yet.**

   Evidence: all evidence fields are future-tense at `qmd:88-96`; statistical analysis is planned at `qmd:112-116`.

   Minimum fix: keep the title/status explicitly “protocol/specification” until the release gate passes and model-level results exist.

### MINOR

6. The weights/normalizers at `qmd:74-82` are not linked to or verified against an executable configuration digest in this chapter.

# Contradictions across the group

1. **Artifact availability:** chapter 09’s rendered gate says the EchoNext label audit is unavailable (`HTML:2647-2662`), while chapter 10 says supports are identity-checked (`qmd:21`) and lists that missing audit as a reproduction artifact (`qmd:854`). Chapter 08 likewise says the locked package exists and includes per-record/verification artifacts (`qmd:340-349`), but that package is absent.

2. **Executed-versus-static evidence:** chapters 09 and 11 present live executed outputs; 07, 08, 10, and 12 present code and static prose only. Readers are given no clear visual distinction.

3. **Factor-set contamination:** the 48-cell study has E/C/MMD/D (`08:qmd:19-24`), but its hard-coded “ANOVA” introduces VCG (`08:qmd:163-165`), which belongs to the later mixed-level design (`08:qmd:361-369`; `12:qmd:77-82`).

4. **Current-generation scoping:** chapter 12 says historical checkpoints are not pooled (`12:qmd:50`), but its displayed audit counts pool 278 historical IDs. Chapter 08’s release gate then uses those same unfiltered rows and becomes fail-closed for the wrong reason.

5. **Smartwatch output semantics:** chapter 08 says I/II/V2 are copied, while chapter 07 zero-fills I/V2 and includes them among eleven scored leads. The exact inference/postprocessing contract must be reconciled.

6. **Chapter status/order:** chapter 09 calls itself the first numbered chapter, but current HTML labels it chapter 4; the six reviewed files render as chapters 4, 5, 13, 14, 15, and 16 rather than filename order.

## Minimum group-level release gate

Before these chapters can be treated as current results:

1. Restore or publish the hash-pinned locked 48-cell package.
2. Render all six chapters with execution enabled and fail on missing claim-bearing outputs.
3. Remove the hard-coded pseudo-ANOVA.
4. Scope all mixed-generation audits to the exact manifest IDs and one atomic snapshot.
5. Recompute EchoNext metrics from record-level probabilities with paired uncertainty.
6. Resolve invalid ISP intervals and governance/class semantics.
7. Add explicit artifact hashes beside every static numerical table.
8. Separate confirmatory, exploratory, operational, simulator, and placeholder evidence visually and statistically.
