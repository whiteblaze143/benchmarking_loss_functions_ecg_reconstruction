# Zero-context senior ML/clinical-methods audit

Scope inspected in full: all prose sections, all code blocks, and the corresponding rendered HTML for the eight requested chapters. No files were changed or created.

## Overall verdict

| Chapter | Score | Current-disk execution | Readiness |
|---|---:|---|---|
| `book/index.qmd` | 5/10 | Static page renders; links resolve | HOLD |
| `book/01_dataset_and_preprocessing.qmd` | 7/10 | Yes with the registered venv kernel | CONDITIONAL / near-ready |
| `book/02_network_architectures.qmd` | 3/10 | Code runs, but only via an invalid fallback for the intended evidence | BLOCKED |
| `book/03_loss_function_formulations.qmd` | 5/10 | Python runs; one displayed equation is broken | BLOCKED |
| `book/04_regression_to_mean.qmd` | 2/10 | Yes, as a synthetic demonstration | NOT READY |
| `book/05_clinical_metrics_and_delineation.qmd` | 4/10 | Yes, but nondeterministic and didactic only | NOT READY |
| `book/05_zhejiang_delineation_tutorial.qmd` | 4/10 | Yes, but its central sanity gate visibly fails | BLOCKED |
| `book/06_fairness_robustness_sqi.qmd` | 3/10 | Yes, as a synthetic demonstration | NOT READY |

The strongest chapter is the PTB-XL contract chapter. The main publication blockers are:

1. the architecture chapter silently substitutes a live 588-checkpoint catalog for a missing locked 48-cell registry;
2. the Zhejiang `ground_truth_ceiling` row is nowhere near perfect;
3. the regression-to-mean chapter contains false clinical and mathematical guarantees;
4. the loss chapter has a broken rendered energy-distance equation and does not faithfully describe/test the active IMQ implementation;
5. the clinical and fairness chapters are tutorials, not evidence chapters.

## Execution and rendered-artifact audit

- All eight HTML files are newer than their corresponding QMD sources.
- No rendered page contains a captured Python traceback or failed-cell output.
- All referenced local CSS, JS, and image assets exist.
- The registered `venv-kernel` points to `/home/mithunmanivannan/.venv/bin/python`, where NumPy, pandas, PyTorch, Plotly, SciPy, Matplotlib, IPython, and NeuroKit2 are installed.
- A direct render from the current shell is not turnkey: both installed Quarto binaries detect `/usr/bin/python3`, which lacks Jupyter, pandas, PyTorch, Plotly, and NeuroKit2. The venv must be activated or placed first on `PATH`.
- Interactive plots are not archival/offline-complete: rendered pages load Plotly from a CDN, e.g. `book/_book/03_loss_function_formulations.html:1022`. A network-free copy loses those figures. Minimum fix: ship a pinned local Plotly asset for archival builds.
- The existing HTML proves prior successful execution, not current semantic validity. In particular, the architecture fallback and Zhejiang sanity failure executed successfully while producing invalid evidence.

---

# Chapter-by-chapter findings

## `book/index.qmd` — 5/10 — HOLD

### MAJOR — “Available” evidence is not tied to an executable artifact gate

Evidence:

- `book/index.qmd:37` presents “locked 48-cell evidence” as part of the current study.
- `book/index.qmd:52-55` hard-codes several evidence streams as “Available” or running.
- The canonical registry used by `book/02_network_architectures.qmd:22`, `results/comprehensive_latest_48_models/provenance/model_registry.json`, is absent from current disk.
- The rendered architecture page therefore falls back to a different 588-row catalog at `book/_book/02_network_architectures.html:693-696`.

This does not prove no legacy results exist—`results/factorial_v2/` exists—but the executive page does not reconcile that directory with the canonical generation/path claimed by the book.

Minimum fix: generate the status table from a manifest that validates artifact existence, generation ID, schema, digest, expected row count, checkpoint coverage, and rendered timestamp. Fail closed or label the stream “unavailable/unverified” if its canonical bundle is missing.

### MAJOR — “Live/current” status is static prose

Evidence:

- `book/index.qmd:24-28` says live pages query databases on every render.
- `book/index.qmd:47-55` is a manually written table with no executable cell.

The book’s front door can therefore remain stale even if downstream live chapters update.

Minimum fix: make the front-page status table query the same read-only databases/manifests and display `as_of`, generation digest, completed/expected cells, and gate status.

### MAJOR — The opening comparison diagram visually asserts the hypothesis as a result

Evidence:

- `book/index.qmd:13-20` places “Flat P-waves / Flattened ST” and “Damped Peak Amplitudes” under MSE, opposite “Candidate shape preservation” under “Combinatorial Composite Loss (Ours).”
- This conflicts with the explicit caveat at `book/index.qmd:9-11` that attenuation and clinical endpoints must be measured.

Even with the word “candidate,” the diagram primes the reader toward a favorable method conclusion.

Minimum fix: replace it with a neutral estimand diagram: observed leads → model/objective → missing-lead prediction → prespecified amplitude/delineation/agreement tests. If the comparison is retained, label both sides “hypothesized behavior” and link to actual results.

### MAJOR — Bibliographic provenance is absent

Evidence:

- `book/index.qmd:7-9` names Presacan et al. (2025), but none of the eight pages has a rendered bibliography or formal citation anchor.

Minimum fix: add an actual CSL citation and bibliography entry with DOI/URL; distinguish the prior paper’s setting from this repository’s three-lead setting.

### MINOR — Task definition is initially imprecise

Evidence:

- `book/index.qmd:5` says “estimate a complete 12-lead representation.”
- The actual evaluation policy is three acquired leads copied exactly plus nine reconstructed leads (`book/02_network_architectures.qmd:9`).

Minimum fix: state this deployment/evaluation distinction in the first paragraph.

### Readiness

Usable as navigation, but not ready as an executive evidence summary until status is machine-derived and the opening visual is neutralized.

---

## `book/01_dataset_and_preprocessing.qmd` — 7/10 — CONDITIONAL

### MAJOR — The “live contract” displays declared hashes rather than validating the data

Evidence:

- `book/01_dataset_and_preprocessing.qmd:33-34` loads both the contract and content manifest.
- `book/01_dataset_and_preprocessing.qmd:34` assigns `content_manifest` but never uses it.
- `book/01_dataset_and_preprocessing.qmd:45-53` checks file counts and prints the contract-declared content root; it does not recompute or compare the current tensors’ content root.
- `book/01_dataset_and_preprocessing.qmd:64` hashes only the contract file itself.

A same-count, same-name tensor mutation would render as apparently valid.

Current-disk note: independent read-only checking found that the manifest and metadata hashes currently match the contract. The problem is that this chapter does not establish that fact itself.

Minimum fix: recompute name/size and byte-content roots, compare every pinned data/source artifact against the contract, display observed versus expected values, and raise on mismatch.

### MAJOR — Claimed fold and patient split contract is not tested in this chapter

Evidence:

- `book/01_dataset_and_preprocessing.qmd:9` declares train folds 1–8, validation fold 9, and test fold 10.
- `book/01_dataset_and_preprocessing.qmd:119-120` reads metadata for only one validation record.
- `book/01_dataset_and_preprocessing.qmd:134` acknowledges repeated patients and delegates split integrity elsewhere.

A chapter titled “Tensor, Lead, and Preprocessing Contract” should not merely inherit its most important leakage gate from another chapter.

Minimum fix: join every tensor filename to `ecg_id`, verify expected `strat_fold`, assert no record overlap and no patient overlap across splits, and report record and unique-patient counts.

### MAJOR — Units and source provenance are declared but not directly cited

Evidence:

- `book/01_dataset_and_preprocessing.qmd:9` declares mV and 500 Hz from the contract.
- `book/01_dataset_and_preprocessing.qmd:221` says authoritative release metadata define units and sample rate, but does not identify or cite that release metadata.

Minimum fix: cite the exact PTB-XL release/version and local source file, include dataset license/version, and distinguish “declared unit” from any empirically checked amplitude scale.

### MINOR — Empty-split handling is inconsistent

Evidence:

- `book/01_dataset_and_preprocessing.qmd:45-53` dereferences `files[0]` and `files[-1]` before any non-empty assertion.
- The later loader correctly raises at `book/01_dataset_and_preprocessing.qmd:101-102`.

Minimum fix: add an explicit empty-split failure before forming the inventory table.

### MINOR — Plot caption and actual color encoding disagree

Evidence:

- `book/01_dataset_and_preprocessing.qmd:197` calls the filtered trace “green.”
- `book/01_dataset_and_preprocessing.qmd:203-211` does not set either line color, so Plotly uses its default palette; the second trace is not guaranteed green.

Minimum fix: set explicit, accessible colors or remove the color reference.

### MINOR — Visualization coverage is too narrow for a preprocessing contract

Evidence:

- `book/01_dataset_and_preprocessing.qmd:140-164` shows one record and only its first 3 seconds.
- `book/01_dataset_and_preprocessing.qmd:170-189` shows one-record Lead-I spectral behavior.

The prose correctly labels these as record-level illustrations, but the chapter lacks cohort distributions for amplitude, clipping, baseline offset, and PSD.

Minimum fix: add links or compact cohort summaries from the opening EDA, with distributions by split/lead rather than one example.

### Execution

The five cells can execute with the intended venv kernel; all required files exist and the current HTML contains three tables and three rendered figures. Direct current-shell Quarto execution needs venv activation.

### Readiness

Near-ready as a tutorial/contract chapter after fail-closed hash and patient-split checks are added.

---

## `book/02_network_architectures.qmd` — 3/10 — BLOCKED

### CRITICAL — Missing locked registry is silently replaced by an unrelated/incomplete live catalog

Evidence:

- `book/02_network_architectures.qmd:22` expects `results/comprehensive_latest_48_models/provenance/model_registry.json`; this file is absent.
- `book/02_network_architectures.qmd:53-66` silently falls back to `results/checkpoint_store/catalog.sqlite`.
- The rendered fallback reports 588 checkpoints, not 48: `book/_book/02_network_architectures.html:693-696`.
- The rendered family rows are U-Net 310 cells/two seeds, MSVAE 155 cells/one seed, and ECG-AIM 123 cells/one seed, rather than 16 masks × three families at seed 42.

The page executes, but the output is not evidence for the prose claim at `book/02_network_architectures.qmd:7`.

Minimum fix: fail closed when the locked registry is absent. If the fallback is intentionally shown, place it in a separate “current queue status” panel keyed by study/generation and never use it to support the locked architecture description.

### MAJOR — Registry-declared MSVAE and ECG-AIM details are currently unverifiable from the chapter

Evidence:

- `book/02_network_architectures.qmd:142-144` claims registry-declared VAE facts.
- `book/02_network_architectures.qmd:148` claims exact ECG-AIM width/depth/head/patch values.
- The expected registry is absent, and the fallback code emits no architecture fields.

Minimum fix: restore the content-pinned registry or point to an exact alternative manifest with digest checks. Suppress these facts if the source artifact is unavailable.

### MAJOR — The fallback mixes studies without any generation gate

Evidence:

- `book/02_network_architectures.qmd:54-61` groups every catalog row by a `family` extracted from arbitrary `metadata_json`.
- It does not filter by study ID, contract ID, source bundle, checkpoint status, or expected seed/mask set.
- `unknown` is accepted as a family instead of being a render failure.

Minimum fix: require a named generation and schema; assert expected families, exact cell IDs, exact seed/mask coverage, source-bundle digest, and no unknown metadata.

### MAJOR — The chapter claims identity guarantees it does not demonstrate

Evidence:

- `book/02_network_architectures.qmd:156-164` gives a strong minimum identity list.
- `book/02_network_architectures.qmd:166` states that both studies satisfy it.
- No code on the page verifies adapter digest, checkpoint digest, state schema, precision, pass-through policy, or per-cell completeness for either study.

Minimum fix: render a per-study identity/gate table derived from manifests and fail the assertion if any field or cell is absent.

### MINOR — Forward validation does not exercise the production path

Evidence:

- `book/02_network_architectures.qmd:121-125` tests a synthetic length-320 tensor.
- `book/02_network_architectures.qmd:129-131` merely prints the claimed 5120→5000 behavior; the code does not run padding, full-length inference, cropping, acquired-lead pass-through, or finite-output checks.

Minimum fix: run one real 5000-sample tensor through the actual evaluator adapter, assert padded/cropped shapes and finite outputs, and verify exact acquired-lead restoration.

### MINOR — No architecture visualization or effective-context audit

The chapter is prose plus tables; it lacks a tensor-shape diagram, skip-path diagram, receptive-field/effective-context calculation, activation-memory estimate, or consistent comparison matrix across families.

Minimum fix: add a source-derived architecture diagram and a table that clearly marks unavailable fields rather than inventing them.

### Execution

The page technically executes because the fallback exists. The intended locked-registry audit cannot execute from current disk. This is a semantic execution failure and should block release.

---

## `book/03_loss_function_formulations.qmd` — 5/10 — BLOCKED

### MAJOR — Energy-distance equation is corrupted in source and rendered HTML

Evidence:

- `book/03_loss_function_formulations.qmd:149` contains a form-feed/control character: `rac1C`.
- It survives unchanged at `book/_book/03_loss_function_formulations.html:840`, so MathJax cannot render the intended leading factor.

Minimum fix: replace with `\frac{1}{C}` and add a render check for control characters/MathJax parse failures.

### MAJOR — The IMQ prose does not match the imported implementation

Evidence:

- `book/03_loss_function_formulations.qmd:188` says the IMQ scale is based on the target median distance.
- `scripts/common_loss.py:192-219` derives the scale from the function’s second argument.
- `scripts/common_loss.py:257-261` calls that helper as `(pred,pred)`, `(target,target)`, and `(pred,target)`, so `k_xx` uses a prediction-derived bandwidth while `k_yy` and `k_xy` use target-derived bandwidths.

Those are not evaluations under one shared kernel. Therefore the expression called MMD may lose standard MMD interpretation/non-negativity. The same helper is used in temporal K-means MMD.

Minimum fix: choose one detached shared bandwidth per block/cluster, use it for `k_xx`, `k_yy`, and `k_xy`, unit-test identity/symmetry/non-negativity within tolerance, and update the prose only after implementation and tests agree.

### MAJOR — The “actual training batch” probe is not an actual training batch

Evidence:

- `book/03_loss_function_formulations.qmd:200` calls the probe an actual PTB-XL training batch.
- `book/03_loss_function_formulations.qmd:210-213` loads raw 5000-sample tensors directly.
- The active training loader pads both inputs and targets to 5120 (`scripts/train_mcma_3lead.py:161-170`), and `train_factorial.py:193-198` uses that loader.

The displayed component scales exclude the 120-sample padded tail used during optimization.

Minimum fix: instantiate the production dataset/loader or apply the exact padding contract; report both full training-tensor and cropped missing-lead diagnostic scales.

### MAJOR — Manifest checks do not establish a complete balanced factorial design

Evidence:

- `book/03_loss_function_formulations.qmd:38-40` extracts masks and seeds with regex.
- `book/03_loss_function_formulations.qmd:60-65` checks only global job count, unique-mask count, seed set, MSE count, and MMD-level set.

A manifest with missing mask–seed cells plus duplicate compensating jobs could pass these summaries.

Minimum fix: assert the exact Cartesian product, exactly one job per mask–seed pair, valid mask regex `1[01]{5}[0-4]`, exact factor balance/orthogonality, unique checkpoint/run names, and no duplicate command targets.

### MAJOR — Loss-weight/normalizer selection provenance is missing

Evidence:

- `book/03_loss_function_formulations.qmd:45-58` reports exact effective coefficients.
- `book/03_loss_function_formulations.qmd:81` describes them only as scale normalizers.

There is no account of the calibration sample, statistic, selection criterion, whether labels/test data were consulted, or sensitivity to these values.

Minimum fix: document a training-only calibration protocol and add coefficient-sensitivity analyses. Treat the weights as hyperparameters, not neutral normalization constants.

### MAJOR — Fragile branches are not exercised by the executable diagnostic

Evidence:

- `book/03_loss_function_formulations.qmd:216` uses mask `1111111`, which exercises only global RBF MMD.
- Levels 2, 3, and 4—especially randomized batch-dependent K-means—receive no forward/backward finite-gradient or determinism check.
- `book/03_loss_function_formulations.qmd:194-196` explicitly describes random centroid initialization and batch dependence.

Minimum fix: unit-test all five MMD levels, batch sizes 1/2/production, constant signals, identical tensors, AMP, backward finiteness, repeated seeded calls, empty clusters, and permutation behavior.

### MAJOR — Repeated-patient weighting is ignored during batch-distribution losses

Evidence:

- `book/01_dataset_and_preprocessing.qmd:134` establishes repeated PTB-XL records per patient.
- `book/03_loss_function_formulations.qmd:146-156` and `173-196` treat records in a mini-batch as the empirical distribution.
- No patient-balanced sampler or sensitivity analysis is specified.

Patients with more records can receive greater influence in ED/MMD gradients.

Minimum fix: quantify repeat-record frequency and either use patient-balanced sampling or report sensitivity to record-balanced versus patient-balanced batches.

### MINOR — Visualization is too narrow to support scale interpretation

Evidence:

- `book/03_loss_function_formulations.qmd:200-242` shows one arbitrary 10% attenuation/noise perturbation and one eight-record batch.
- The rendered plot is dominated by energy distance, without uncertainty, replicate batches, or all MMD levels.

Minimum fix: show distributions over several real batches and perturbation types, preferably on a log scale, with separate panels for raw values, weighted values, and gradient norms.

### Execution

Python cells execute with current source/data and their hashes match the pinned contract. The rendered mathematical formulation is broken, and the most important IMQ correctness property is untested.

---

## `book/04_regression_to_mean.qmd` — 2/10 — NOT READY

### CRITICAL — Population-risk theorem is promoted to a guaranteed finite-model clinical failure

Evidence:

- `book/04_regression_to_mean.qmd:3` says MSE training “forces” predictions to converge to the conditional expectation and causes variance collapse/amplitude blunting.
- `book/04_regression_to_mean.qmd:50-52` asserts incomplete spatial information necessarily gives strict conditional variance.
- `book/04_regression_to_mean.qmd:54` claims all L2 models systematically shrink QRS peaks and flatten P/T waves into a false low-voltage phenotype.
- The overclaim is present in rendered prose at `book/_book/04_regression_to_mean.html:436`.

The theorem applies to an unrestricted population-risk optimum under squared loss. It does not guarantee that a finite trained neural network, any named morphology statistic, or this checkpoint generation exhibits those effects.

Minimum fix: match the cautious language already used in `book/index.qmd:9-11` and `book/03_loss_function_formulations.qmd:93`: state the theorem, list its assumptions, and treat amplitude/delineation effects as prespecified empirical hypotheses.

### CRITICAL — “MMD forces W1 to zero” is false

Evidence:

- `book/04_regression_to_mean.qmd:67` says MMD regularization forces \(W_1(P,Q)\to0\).
- The same claim renders at `book/_book/04_regression_to_mean.html:446`.

A finite-weight, mini-batch MMD regularizer under an adaptively changing kernel gives no such guarantee. Even convergence in characteristic-kernel MMD does not automatically provide \(W_1\) convergence without additional topology/moment assumptions.

Minimum fix: remove the guarantee. State exactly which empirical distribution metric is optimized and evaluate held-out W1 as a separate endpoint with uncertainty.

### MAJOR — The variance proof is imprecise for vector waveforms

Evidence:

- `book/04_regression_to_mean.qmd:12-16` starts with tensor-valued \(Y\) and a squared norm.
- `book/04_regression_to_mean.qmd:25-35` switches to a scalar integral without stating coordinatewise application.
- `book/04_regression_to_mean.qmd:44-52` uses scalar variance notation for a tensor and then infers amplitude attenuation.

Minimum fix: state the result coordinatewise or use conditional covariance and the law of total covariance. Explicitly separate marginal variance shrinkage from peak-amplitude, interval, and morphology endpoints.

### MAJOR — Bland–Altman battery is mischaracterized as nonparametric tests

Evidence:

- `book/04_regression_to_mean.qmd:84` calls four items “non-parametric statistical tests.”
- Mean bias and limits of agreement are estimators, not tests.
- `book/04_regression_to_mean.qmd:88-93` uses parametric \(1.96s_d\) limits while claiming nonparametric behavior.
- No uncertainty is provided for either limit of agreement.

Minimum fix: call them agreement estimands; specify the normal-difference assumption or use quantile limits; provide clustered confidence intervals for bias and both LoA endpoints.

### MAJOR — P-value non-rejection is incorrectly labeled proof of assumptions

Evidence:

- `book/04_regression_to_mean.qmd:132-141` labels `p>0.05` as “Normal” and “Homoscedastic.”
- The rendered output does exactly that at `book/_book/04_regression_to_mean.html:552-563`.
- `book/04_regression_to_mean.qmd:93` says Levene’s test “verifies” whether variance depends on amplitude.

Failure to reject is not evidence that an assumption is true. A median split also discards information and is not a proper proportional-bias model.

Minimum fix: report statistics/p-values without binary truth labels; add difference-versus-mean regression or a heteroscedastic agreement model and visual residual diagnostics.

### MAJOR — Patient clustering and multiplicity are absent

Evidence:

- `book/01_dataset_and_preprocessing.qmd:134` says uncertainty must cluster repeated records by patient.
- `book/04_regression_to_mean.qmd:125-129` bootstraps scalar differences independently.
- No per-patient hierarchy, per-lead/wave multiplicity control, or prespecified primary endpoint exists.

Minimum fix: bootstrap patients, retaining all within-patient records/beats/leads; define primary endpoints and multiplicity handling.

### MAJOR — Percentage bias is numerically unsafe

Evidence:

- `book/04_regression_to_mean.qmd:143-145` divides by the pair mean with no zero/near-zero handling.

ECG signed amplitudes can cross zero, making percentage differences unstable or undefined.

Minimum fix: avoid percentage BA for signed/near-zero quantities or define an explicit guarded scale and exclusion ledger. Prefer absolute bias and amplitude-ratio/log-ratio analyses where physiologically appropriate.

### MAJOR — The figure title promises a percentage BA plot that does not exist

Evidence:

- `book/04_regression_to_mean.qmd:173` says “Absolute & Percentage Bland-Altman Plots.”
- `book/04_regression_to_mean.qmd:175-206` plots only absolute differences.
- `mse_pct_diff` and `comp_pct_diff` are calculated but never visualized.

Minimum fix: rename the section or add a separate valid percentage/ratio panel. Draw both models’ bias/LoA consistently; currently only MSE bias/LoA lines are shown.

### MAJOR — No real checkpoint analysis

Evidence:

- `book/04_regression_to_mean.qmd:99-103` correctly labels the inputs synthetic.
- `book/04_regression_to_mean.qmd:157-168` hard-codes favorable 30% versus 1% attenuation.

The demonstration cannot support the chapter title “Rigorous Agreement” or any method ranking.

Minimum fix: bind the chapter to record-level checkpoint artifacts, prespecified peak definitions, detector-failure denominators, patient-clustered uncertainty, and actual paired model contrasts. Keep the simulation only as a boxed tutorial.

### MINOR — Formal citations are missing

Evidence:

- `book/04_regression_to_mean.qmd:3` names Presacan et al.
- `book/04_regression_to_mean.qmd:74` names Giavarina.
- No bibliography renders.

Minimum fix: add formal citations and avoid calling an association/agreement distinction “proved” by a review/tutorial article.

### Readiness

Reject in current form. The final caveat at `book/04_regression_to_mean.qmd:209` does not repair the categorical claims earlier in the chapter.

---

## `book/05_clinical_metrics_and_delineation.qmd` — 4/10 — NOT READY

### MAJOR — Bipartite matching optimization is incomplete

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:37-44` asks for a “permutation” minimizing total error when the reference and prediction sets may have different sizes.
- It does not specify matching cardinality, dummy assignments/costs, or whether maximizing valid matches precedes minimizing timing error.

With optional matches, the empty matching minimizes cost; with a literal permutation, unequal set sizes are undefined.

Minimum fix: define a bipartite matching set, maximize cardinality within tolerance first, then minimize total absolute timing error, and unit-test duplicates, ties, missing waves, and unequal counts.

### MAJOR — A single 50 ms tolerance is asserted without endpoint justification

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:37` fixes \(\tau=50\) ms for all events.

Acceptable tolerance and annotation uncertainty may differ for R peaks, P/T peaks, and onsets/offsets.

Minimum fix: prespecify endpoint-specific tolerances from authoritative standards or sensitivity analyses; report results across clinically justified tolerances.

### MAJOR — QTc clinical recommendations are uncited and over-broad

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:20` asserts false-positive long-QT behavior at a specific HR threshold.
- `book/05_clinical_metrics_and_delineation.qmd:25` calls Fridericia an FDA-recommended standard.

No bibliography or regulatory reference is provided, and formula choice depends on protocol/population.

Minimum fix: cite the exact guideline/regulatory document and scope the statement; prespecify the primary correction formula and sensitivity formulas.

### MAJOR — CWT prose does not match the executed detector

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:95-99` describes a Mexican-hat wavelet and frequency bands.
- `book/05_clinical_metrics_and_delineation.qmd:118` actually delegates to NeuroKit2 `method="cwt"`.
- Current NeuroKit2 0.2.12 uses PyWavelets `gaus1`, the first derivative of a Gaussian, at fixed scales `[1,2,4,8,16]`, not the displayed Mexican-hat wavelet.

Minimum fix: describe the exact library/version/wavelet/scales and cite the implemented method, or implement the displayed detector directly.

### MAJOR — The synthetic delineation example is nondeterministic and incomplete

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:109` calls `nk.ecg_simulate` without a seed.
- `book/05_clinical_metrics_and_delineation.qmd:117` promises P, Q, S, and T boundaries.
- `book/05_clinical_metrics_and_delineation.qmd:144-149` plots only R peaks and P/T onsets/offsets; Q/S peaks and QRS boundaries are omitted.

Minimum fix: set a random state, record NeuroKit2/PyWavelets versions, show all claimed fiducials, and count detector failures.

### MINOR — Time vectors are off by one sample interval

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:121` uses `np.linspace(0, len/fs, len)`, putting the last point at 5.0 s rather than \((N-1)/f_s\).

Minimum fix: use `np.arange(len(signal)) / sampling_rate`.

### MAJOR — No executable matching or QT extraction exists

The chapter provides formula demonstrations but no code that extracts QT/RR, applies the matching rule, retains failures, or computes sensitivity/specificity/coverage.

Minimum fix: implement the complete measurement pipeline with unit tests and a failure ledger before calling it a measurement interface.

### MINOR — QTc helper lacks validity checks

Evidence:

- `book/05_clinical_metrics_and_delineation.qmd:166-190` accepts nonpositive or implausible QT/RR values and returns results without flags.

Minimum fix: validate units/ranges, reject nonpositive RR, retain an explicit invalid-reason field, and define how irregular rhythms and beat averaging are handled.

### Readiness

Good as an introductory formula tutorial after correction, but not ready as a clinical methods chapter or evaluator specification.

---

## `book/05_zhejiang_delineation_tutorial.qmd` — 4/10 — BLOCKED

### CRITICAL — The displayed “ground-truth ceiling” fails its own sanity condition

Evidence:

- `book/05_zhejiang_delineation_tutorial.qmd:185` says reference-mask self-comparison should yield perfect overlap and zero boundary error.
- The rendered row named `ground_truth_ceiling` instead reports P-onset F1 ≈ 0.576, QRS Dice ≈ 0.355, and many nonzero MAEs: `book/_book/05_zhejiang_delineation_tutorial.html:927-942`.
- The underlying row is `results/zhejiang_benchmark/zhejiang_delineation_evaluation.csv:2`.

Either the evaluator is broken, the row is mislabeled, or this is a raw-signal detector ceiling rather than mask-against-itself.

Minimum fix: stop the render when a true self-comparison is not exactly perfect within numerical tolerance. If this row is a detector-versus-annotation ceiling, rename it and explain the detector error; do not call it ground-truth self-comparison.

### MAJOR — Result status gate checks row counts, not validity

Evidence:

- `book/05_zhejiang_delineation_tutorial.qmd:190-200` marks a contrast estimable whenever there are at least two non-ceiling rows.
- It does not validate ceiling metrics, schema, model identities, paired record coverage, seeds, checkpoints, or whether two rows actually represent MSE versus MMD on the same cohort.

Minimum fix: validate schema and ranges, enforce the ceiling gate, require paired record IDs and generation digests, and define exact eligible contrast cells/seeds before saying “estimable.”

### MAJOR — Pairing integrity is only aggregate, not record-level

Evidence:

- `book/05_zhejiang_delineation_tutorial.qmd:99-139` separately summarizes signal and mask lengths/dimensions.
- It never asserts, for each record and lead, that signal length equals its label-mask length or that every lead aligns to the same sample grid.

Current output happens to show length 20,000 for all files, but the claimed pairing contract should enforce this directly.

Minimum fix: join every `(record_id, lead)` to its mask and assert per-pair length, dimensionality, finite values, and expected lead set; report any failing IDs.

### MAJOR — Untrusted pickle loading conflicts with missing provenance

Evidence:

- `book/05_zhejiang_delineation_tutorial.qmd:46` says provenance is not inferred.
- `book/05_zhejiang_delineation_tutorial.qmd:101-102`, `114-115`, and `163-166` execute `pickle.load` over thousands of files.

Pickle can execute arbitrary code. That is unsafe when origin/trust and licensing are explicitly unresolved.

Minimum fix: verify a trusted content manifest before loading, sandbox one-time conversion to a non-executable format such as NumPy/Parquet, and run the book only on converted artifacts.

### MAJOR — Sampling rate and units remain unknown, preventing clinical timing claims

Evidence:

- `book/05_zhejiang_delineation_tutorial.qmd:156` says the evaluator assumes 2000 Hz without local documentation.
- `book/05_zhejiang_delineation_tutorial.qmd:209-216` correctly lists missing requirements.

Without an authoritative sample rate, the MAE values in the result CSV cannot safely be interpreted as milliseconds.

Minimum fix: obtain and cite the data dictionary/acquisition protocol; otherwise report native-sample errors only and block clinical interval claims.

### MINOR — The rendered result table lacks an explicit failure banner

The non-perfect “ceiling” row is shown as an ordinary table, followed by a status table that only says no contrast is estimable.

Minimum fix: render a red failure callout with observed versus expected ceiling metrics and suppress downstream status language until resolved.

### MINOR — One-record visualization does not assess annotation consistency

Evidence:

- `book/05_zhejiang_delineation_tutorial.qmd:162-180` plots one sorted record.

Minimum fix: add class-transition validity, duration distributions, wave-order checks, per-record annotated-beat counts, and several stratified examples/outliers.

### Execution

All five cells execute against 4,008 signal files and 334 masks. Execution is not the issue; the executed sanity result invalidates the current evaluation artifact.

### Readiness

Blocked until the ceiling row is explained or fixed and sampling/provenance are established.

---

## `book/06_fairness_robustness_sqi.qmd` — 3/10 — NOT READY

### MAJOR — EMG robustness is claimed but not implemented

Evidence:

- `book/06_fairness_robustness_sqi.qmd:6-7` advertises Gaussian, baseline-wander, and EMG stress testing.
- `book/06_fairness_robustness_sqi.qmd:54-55` describes filtered autoregressive EMG noise.
- `book/06_fairness_robustness_sqi.qmd:115-125` implements only `gaussian` and `baseline_wander`; any EMG request raises `ValueError`.

Minimum fix: implement and test the stated EMG model or remove it from the claimed suite. Add electrode motion, mains interference, clipping/dropout, lead swaps, and cross-lead correlated artifacts if “realistic degradation” remains the claim.

### MAJOR — SQI theory and code use different spectral estimators

Evidence:

- `book/06_fairness_robustness_sqi.qmd:32` defines \(P(f)=|X(f)|^2\) via FFT.
- `book/06_fairness_robustness_sqi.qmd:99` uses Welch PSD.

Minimum fix: define the actual Welch estimator, window, segment length, overlap, detrending, and band integration. Use numerical integration or clearly justify equal-bin summation.

### MAJOR — SQIs are over-described as clinical suitability measures

Evidence:

- `book/06_fairness_robustness_sqi.qmd:13` says SQIs evaluate whether a signal is suitable for clinical delineation without clean ground truth.
- Only kurtosis, skewness, and one band-power ratio are implemented.

These features may correlate with quality but do not establish delineation suitability without task-specific validation.

Minimum fix: state they are candidate proxy features and validate them against expert quality labels or downstream detector failure/performance.

### MAJOR — Synthetic robustness does not exercise a reconstruction model

Evidence:

- `book/06_fairness_robustness_sqi.qmd:129-147` corrupts only a hand-built sinusoidal teaching signal.
- `book/06_fairness_robustness_sqi.qmd:178` acknowledges no named checkpoint is evaluated.

Minimum fix: corrupt only the intended observed input leads of named real records, run exact checkpoints, preserve clean targets, calculate achieved corruption levels, and report paired degradation curves with patient-clustered uncertainty.

### MAJOR — Noise protocol lacks important deployment choices

Evidence:

- `book/06_fairness_robustness_sqi.qmd:107-127` does not specify whether noise is shared or independent across leads, how phase is sampled, whether signal power is centered/band-limited, or the achieved finite-sample SNR.
- Baseline wander always has phase zero.

Minimum fix: prespecify lead covariance, random phase, amplitude/frequency distributions, repeated draws, and measured post-injection SNR; store every noise seed.

### MAJOR — Fairness estimands are underspecified

Evidence:

- `book/06_fairness_robustness_sqi.qmd:61-71` reduces groups to Male/Female and age ≤/>65 and leaves `Metric` undefined.
- Directionality differs across metrics, so subtracting generic metrics is not interpretable.
- Sex versus gender terminology, unknown/missing values, diagnosis/rhythm confounding, subgroup support, and intersectional estimands are absent.

Minimum fix: define the protected attribute and provenance, one primary harm metric with direction, minimum subgroup support, missingness policy, adjusted/unadjusted estimands, intersectional checks, and multiplicity control.

### MAJOR — Cluster bootstrap is stated as automatically sufficient

Evidence:

- `book/06_fairness_robustness_sqi.qmd:73` says patient-clustered bootstrapping “ensures” repeated recordings do not artificially shrink intervals.

Cluster bootstrap is appropriate but does not guarantee validity with few clusters, extreme imbalance, or sparse subgroup endpoints.

Minimum fix: report unique-patient counts per subgroup, use stratified patient resampling, and add small-cluster or hierarchical sensitivity analyses.

### MINOR — The rendered output demonstrates pSQI blindness but does not discuss it

Evidence:

- `book/_book/06_fairness_robustness_sqi.html:558-560` shows pSQI = 1.0 for both the clean signal and 6 dB baseline wander.

This is expected because the denominator starts at 5 Hz, excluding the injected 0.3 Hz artifact, but it contradicts any impression that pSQI alone assesses overall quality.

Minimum fix: explicitly discuss the blind spot and add complementary baseline-wander, mains, and beat-consistency SQIs.

### MINOR — Constant/degenerate inputs are not handled

Evidence:

- `book/06_fairness_robustness_sqi.qmd:88-105` can return NaN kurtosis/skewness for a constant trace and silently regularizes zero spectral power with `1e-8`.

Minimum fix: validate finite input and variance, return flagged missing values/reasons, and test flat/clipped/short signals.

### MINOR — Time axes are off by one sample interval

Evidence:

- `book/06_fairness_robustness_sqi.qmd:112` and `131` use inclusive-endpoint `linspace`.

Minimum fix: use `np.arange(N)/fs`.

### Readiness

A useful teaching scaffold, but it currently provides neither a robustness benchmark nor a fairness analysis.

---

# Cross-chapter contradictions and terminology problems

## 1. Conditional-mean hypothesis versus guaranteed clinical failure

- Cautious: `book/index.qmd:9-11` and `book/03_loss_function_formulations.qmd:93` say attenuation must be empirically measured.
- Categorical: `book/04_regression_to_mean.qmd:3,50-54` says incomplete leads force variance collapse and false low-voltage morphology.

Fix: adopt the cautious formulation everywhere and reserve “observed” for artifact-backed estimates.

## 2. MMD has no clinical guarantee versus “forces W1 to zero”

- Cautious: `book/03_loss_function_formulations.qmd:247-250`.
- False guarantee: `book/04_regression_to_mean.qmd:67`.

Fix: remove the W1 guarantee and add a results-to-claims matrix separating objective behavior from held-out clinical endpoints.

## 3. Locked 48-cell evidence versus absent canonical bundle

- `book/index.qmd:37,52` says the locked benchmark is available.
- `book/02_network_architectures.qmd:22` points to a missing registry and silently renders a 588-row live fallback.

Fix: define one canonical artifact location/generation and gate all cross-chapter claims on it. If `results/factorial_v2/` is the intended bundle, migrate references and verify equivalence explicitly.

## 4. Stored/evaluation length versus training-objective length

- `book/01_dataset_and_preprocessing.qmd:7,86` correctly distinguishes stored 5000 samples from padding to 5120.
- `book/02_network_architectures.qmd:103` describes pad/crop behavior.
- `book/03_loss_function_formulations.qmd:83,200-214` omits the padded zero tail and probes 5000-sample raw tensors as though they were training tensors.

Fix: consistently define \(T_{\text{stored}}=5000\), \(T_{\text{train}}=5120\), and \(T_{\text{eval}}=5000\), including which loss terms see the padded tail.

## 5. Ground truth, detector ceiling, and delineation terminology

- `book/05_zhejiang_delineation_tutorial.qmd:185` defines a perfect mask self-comparison.
- The rendered `ground_truth_ceiling` is highly imperfect.
- `book/05_clinical_metrics_and_delineation.qmd:117` uses P/Q/S/T language, while Zhejiang output uses `R_Onset/R_Offset` and region Dice without an explicit mapping.

Fix: distinguish `mask_self_check`, `raw_signal_detector_ceiling`, and `model_reconstruction_result`; standardize QRS/R-onset terminology and units.

## 6. Patient as the statistical unit is inconsistently enforced

- `book/01_dataset_and_preprocessing.qmd:134` correctly requires patient-clustered uncertainty.
- `book/04_regression_to_mean.qmd:125-129` bootstraps scalar peaks independently.
- `book/03_loss_function_formulations.qmd:146-196` optimizes batch-distribution terms without patient-balanced sampling.
- `book/06_fairness_robustness_sqi.qmd:73` mentions clustering but implements no analysis.

Fix: state the hierarchy once—patient → record → lead → beat/fiducial—and use it in every uncertainty and sampling design.

## 7. Four-bit versus seven-character masks are insufficiently separated

- `book/02_network_architectures.qmd:7` describes 16 four-bit masks.
- `book/03_loss_function_formulations.qmd:7-13` describes a mixed-level seven-character mask.
- The chapters do not provide a mapping table showing which terms, implementations, and weights are comparable across studies.

Fix: add a cross-study mask dictionary and explicitly prohibit like-named terms from being treated as identical without implementation/digest parity.

## 8. Dataset-specific timing units are blurred

- PTB-XL chapters establish 500 Hz.
- `book/05_clinical_metrics_and_delineation.qmd` demonstrates 500 Hz.
- Zhejiang currently assumes an undocumented 2000 Hz (`book/05_zhejiang_delineation_tutorial.qmd:156`).

Fix: require sampling rate and annotation-index transformation in every result row; never transfer 50 ms/sample conversions implicitly across datasets.

## 9. Fairness attribute terminology is not clinically/provenance precise

- `book/06_fairness_robustness_sqi.qmd:61-71` uses Male/Female without defining whether the source field is sex, gender, administrative coding, or self-report.

Fix: use the source variable’s exact name/provenance and avoid implying a broader construct.

## 10. The book lacks formal citations for central clinical/methodological claims

Across the scoped chapters, Presacan, Giavarina, FDA recommendations, QTc formula claims, CWT methodology, ECG lead identities, and SQI bands appear without a rendered bibliography.

Fix: add a shared bibliography and source every clinical threshold, regulatory statement, algorithm, and dataset contract.

# Minimum release gates

1. Fail closed on the missing locked model registry; remove the 588-row fallback from the locked-study audit.
2. Resolve or relabel the Zhejiang `ground_truth_ceiling`; enforce an exact ceiling assertion.
3. Rewrite the regression-to-mean chapter to remove guaranteed attenuation, low-voltage, and MMD→W1 claims.
4. Fix the corrupted energy-distance equation.
5. Correct the shared-bandwidth IMQ MMD implementation/formulation and test all MMD branches.
6. Make PTB-XL content-root and patient-split validation executable in the contract chapter.
7. Convert clinical/fairness chapters from simulations to clearly segregated tutorials, or add generation-bound real checkpoint analyses.
8. Add patient-clustered inference, detector-failure denominators, multiplicity gates, formal citations, and machine-derived evidence status.
9. Activate/document the venv for reproducible Quarto execution and provide an offline archival plot mode.
