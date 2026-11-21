# AF-density feasibility implementation

The source-RR pilot has completed, with a historical integrity audit FAIL retained in its report. The first matched raw-ECG train/validation process has launched; its live log and evaluation lock are authoritative for progress/completion. Matched observed-RR and SetOperator classifier runs remain pending. The full user-supplied AF-density research program is not complete.

`metrics.py` keeps three explicitly different implementations:

- `charitos_density`: continuous-time density from a sorted set of disjoint AF intervals and a single uninterrupted observation duration. It analytically integrates the minimum-window envelope, including jumps. It does not remove or bridge excluded time.
- `discrete_pattern_reference`: literal small-array reference for Butkuviene et al., equations 3–5, using binary beat labels. Quadratic cost; verification only.
- `discrete_pattern_from_runs`: computes the same beat-indexed density and aggregation from AF runs. The minimum window containing p AF beats is p plus the minimum intervening gap among consecutive episode blocks containing at least p AF beats. Counting these minimum-window lengths recovers the maximum AF count by window length. This avoids the reference's quadratic full-window enumeration.

No-AF and continuous-AF density return None because normalization is undefined. Zero must not be substituted. Adjacent, overlapping, unsorted or invalid intervals raise errors. Single episodes have density one. Continuous density and finite beat-indexed density have different normalization; results must retain the named metric and clock.

Tests independently enumerate all windows for binary sequences up to length eight and compare the optimized discrete implementation with the literal equations for every binary sequence up to length nine. Additional checks cover single episodes, undefined edge cases, bounds, time reversal and scaling. These establish algorithmic agreement with the stated definitions, not reproduction of unpublished author software or clinical validation.

Run:

```
/home/mithunmanivannan/.venv/bin/python -m unittest discover -s experiments/af_density -v
/home/mithunmanivannan/.venv/bin/python experiments/af_density/audit_timelines.py
```

The timeline audit requires the completed full-file integrity scan. It uses actual HDF5 lengths, validates source ECG and RR boundary indices, and exports raw descriptive metrics. Raw values include calibration and are not final quality-qualified endpoints.

Sources:

- Charitos definition and monitoring-horizon analysis: https://cinc.org/archives/2015/pdf/0265.pdf
- Beat-indexed definitions, equations 3–5: https://doi.org/10.1109/TBME.2023.3293252 ; primary author manuscript at https://zaguan.unizar.es/record/132307/files/texto_completo.pdf
- IRIDIA-AF descriptor: https://doi.org/10.1038/s41597-023-02621-1

Remaining gate: calibration and excluded-time policy; physical column identity and voltage scaling; waveform quality; horizon-qualified patient counts; subject-level manifest; perturbation replication; external-dataset audit; then matched local representations and temporal decoding. The full research program is not complete.

## Waveform representation and episode evaluation

The frozen observed-channel extraction covers 802,491 actual 30-second windows at 200 Hz, with independently verified sample-index targets and patient partitions. Failed extraction remains explicitly recorded. Run status and committed-shard receipts are under `/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/`; a running status is not completion evidence.

`phase_representation.py` reuses the original PCHIP phase normalization and already fitted whitening/Nyström interfaces. It retains 256 phase samples, 16 cells and 128 response features. It accepts only saved local observed peaks with at least two original 300–2000 ms intervals. It neither fits transforms nor invents beats. The author's loader plots unconverted HDF5 columns as mV; this supports the voltage convention but does not establish canonical operator geometry. Continuous/projective modes remain unadmitted without that geometry.

`episode_metrics.py` evaluates half-open intervals on a single fully observed elapsed-time segment. Detection requires positive-duration overlap; touching boundaries do not count. Sensitivity and PPV use all reference and predicted episodes respectively. Split and merge rates retain all corresponding episodes in their denominators. Boundary errors include only isolated one-to-one overlaps, with the eligible count reported. Empty denominators are undefined, never zero. No short-episode deletion, tolerance-based matching or gap bridging is implicit. Exhaustive binary timelines verify the overlap graph and elapsed-time confusion counts; these checks do not establish clinical event validity.

`temporal_baselines.py` implements D0 and D1 without fitting or selecting their parameters. D0 retains NaN scores as unknown. D1 applies an odd-width binary median with replicated edges separately within observed blocks, then merges bounded non-AF gaps using their actual assigned seconds. Unknown windows and terminal non-AF spans cannot be filled by gap merging. The median weights windows equally, including a partial tail; it is not a time-weighted median. Interval export validates contiguous integer sample assignments and preserves the original sample clock.

`hsmm.py` supplies the D2 numerical Viterbi core. It requires explicit normalized duration PMFs and initial state occupancy. Endpoint censoring is explicit: a censored initial run uses the stationary residual-life law, a censored final run uses survival, and a run spanning both boundaries uses residual survival. Durations count window steps, including a partial final assignment as one step; elapsed-time evaluation must still use actual assignments. Unknown emissions are rejected so callers must split observed blocks. Exhaustive path enumeration verifies every censoring combination for sequences of one through seven windows. Development selection, calibration, missing-time policy and measured D2 evaluation remain pending.

`duration_fit.py` and `fit_waveform_durations.py` fit the prospectively frozen training-only censored lognormal duration model. All 181 training files/90 patients contribute 599 runs: 352 non-AF and 247 AF. Only 113 non-AF and 176 AF runs have both endpoints observed; boundary-cut runs enter the appropriate censored likelihood. Every patient's total file-likelihood weight is one; individual episodes do not receive equal weight. Both optimizations converged away from parameter bounds. `verify_duration_model.py` independently reconstructs every run and initial occupancy, verifies source/input/model hashes, and replays the saved PMFs and likelihoods. Outputs and documentation are under `waveform_v1/duration_model_v1/` and `RUN_DURATION_FIT.md`. The lognormal/stationary assumptions and 30-day finite support are declared modeling choices, not established physiological facts. No validation/test annotations enter this fit, and no measured HSMM performance has been evaluated.

The training-only target-resolution diagnostic covers all 181 training files/90 patients. Assigning the locked window targets preserves all 247 reference episodes without misses, false episodes, splits or merges. Patient-equal mean boundary error is 6.3745 seconds; continuous-density absolute error is 0.00017467 across 129 defined files. These are reference-derived resolution diagnostics, not measured model performance or a lower bound on attainable density error. Outputs are in `waveform_v1/training_target_resolution/` on NFS.

`elapsed_evaluation.py` combines the named pattern and episode measures on a single contiguous source-sample segment. It counts confusion exactly in integer samples, keeps all-reference and known-time sensitivity/specificity separate, and preserves unknown AF/non-AF durations. Any unknown time makes complete-timeline predicted episode/pattern metrics indeterminate. No missing-time strategy is selected or applied implicitly.

The evaluator reports the unchanged verified author-binned Charitos port separately from the exact continuous implementation. Author inputs use equal bins covering the complete segment, with bin count ceil(duration/30 seconds), including the fractional tail without padding or dropping time. The original author edge/clamp conventions are retained. Secondary density/aggregation equal-time-bin analogues use ceil(duration/1 second) equal bins and center-sampled states; the AF-duration quantization difference is exported. They are explicitly not the published RR-beat-clock metrics. Final primary endpoint/population and gap-policy selection must still be frozen before waveform classifier evaluation.

Boundary errors additionally exclude censored reference endpoints at segment edges. The earlier 6.3745-second target-resolution figure includes segment-boundary matches and must not be described as an uncensored onset/offset error. An actual first-training-file preflight independently replays every confusion count against full sample arrays and verifies that a synthetic missing window remains unknown; outputs are in `waveform_v1/elapsed_evaluation_training_preflight.json`.

## Subject-level waveform gate

`waveform_cohort_ledger.py` exports all 152 source patients, 167 recordings and 289 post-calibration daily segments, including original-clock AF intervals, non-AF complements, calibration exclusions, burden, longest observed interval, continuous density and explicitly named time-bin aggregation. It never joins distinct recordings or compresses exclusions. Segment means are descriptive patient summaries, not metrics of a concatenated timeline. Clinical quality is unadjudicated; extraction-completion fields are explicitly a pre-completion snapshot.

The informative multi-episode population comprises 57 patients: 33 training patients/42 segments, 11 validation patients/15 segments, and 13 test patients/14 segments. Those counts cannot be replaced by the much larger window count for inference. No post-exclusion segment is an uninterrupted 24 hours. `verify_waveform_cohort_ledger.py` checks every segment against the prior independent timeline audit, exact AF/non-AF coverage and source-patient partitions. Outputs are in `waveform_v1/cohort_ledger_v1/`.

`matched_selection.py` implements the prospective weighting/threshold components for matched models. Assigned-duration weights sum to one per training patient, with no subsequent class reweighting; an entirely unrepresented training patient causes a hard failure. The fixed validation threshold candidates are .01–.99, selecting patient-mean elapsed-time balanced accuracy, then proximity to .5, then the larger threshold. Unknown rows retain their class-time denominator. The caller must supply exactly the frozen validation patient set. These numerical components have not yet been used to fit a waveform classifier or select a waveform threshold.

## Prospective Phase-KME kernel stage

`fit_phase_kernel.py` freezes the unsupervised training-only kernel protocol before fitting. Production fitting is blocked until extraction completes across all 289 files/802,491 windows. It selects 32 distinct phase-valid windows per training patient and stored channel, then 128 atoms per window: 737,280 atoms across all 90 training patients. This sampling applies only to whitening/landmarks, not classifier training. The original whitening, 128-landmark IMQ map and c2=1 are retained, with a training-only approximation audit required. Source, input, shard, selected-window and atom-array hashes are recorded; prior fit attempts cannot silently overwrite results.

The mean-before-inverse-root evaluation uses linearity to reduce feature computation while retaining the same mathematical representation, with float64 operations before float32 caching. Component tests cover nontrivial inverse-root transforms. A fresh documentation witness verified the temporary actual-training fixture: four channel windows, 16x128 responses, maximum cached float32 difference 0.0 from the original implementation. This witness is not a production kernel fit. Frozen protocol and witness: `waveform_v1/phase_kernel_v1/`; runtime instructions: `RUN_PHASE_KERNEL.md`.

## Matched raw-waveform run

`train_raw_waveform.py` uses the existing eight-block ResNet with two masked voltage channels and two explicit availability channels. It cycles every declared training window through channel0/channel1/both over three epochs, retaining all 504,777 training windows. All 166,047 validation windows are evaluated under all three acquisition patterns. The GPU stores source float32 waveforms; normalization uses unique post-calibration training samples only and is applied per batch. Assigned-duration patient weights have equal totals without class reweighting. Training uses BF16 autocast; checkpoint selection and replay use FP32 validation, deterministic CUDA and TF32 disabled. Detached best/final checkpoints are separate. Strict minimum patient-weighted validation BCE selects the checkpoint; pooled three-pattern validation selects the threshold under the documented fixed policy.

The runner has no held-out inference path and loads no test waveforms. Final test endpoint, temporal-policy and comparison-model protocols must be frozen before a separate test evaluation. A fresh documentation witness verified the exact command, unobserved-channel isolation and batch256 with the full29.988GiB cache reservation (31.184GiB peak allocated). These are synthetic resource checks, not model performance. Run artifacts and live telemetry are under `waveform_v1/raw_mask_resnet_v1/`; runtime command is in `RUN_RAW_BASELINE.md`. `monitor_raw_runtime.py` records GPU/process usage every15seconds and terminates when the declared training process ends, without restarting anything.

## Matched observed-channel RR baseline

`train_observed_rr.py` freezes and prepares the primary matched RR baseline from the same actual waveform windows and locally detected observed-channel peaks. It does not use released native RR. Its float32 design has32 RR-statistic columns,2 acquisition masks and2 RR-valid flags. Unobserved channels cannot affect the design; entirely unavailable requested representations remain indeterminate. Every eligible training window is expanded equally over the same channel0/channel1/both acquisition subsets, with no dataset cap or class reweighting. This estimator uses fixed200 HGB iterations rather than the neural model's three-epoch availability cycle; optimization budgets are explicitly different.

Preparation waits for complete extraction and reads training/validation shards only. Input/target/weight hashes are saved before fit; a fresh joblib reload must exactly replay validation scores from float32 inputs. Threshold selection includes unknown rows in original assigned-time denominators. The runner has no test/external inference path. A fresh documented-command witness passed on32 actual training windows/96 acquisition examples with disposable synthetic labels; this is not a production model fit. Frozen protocol and witness: `waveform_v1/observed_rr_hgb_v1/`; command documentation: `RUN_OBSERVED_RR.md`.

## Authentic SetOperator adapter and numerical witnesses

`set_waveform_model.py` wraps the unchanged original `OperatorSetModel` with a binary AF head, admitting categorical and q-ablated modes only. It preserves PhaseCNN, Transformer, learned-query pooling and256-dimensional latent. Both modes receive no auxiliary reconstruction/orientation loss. The original unused decoder stays present and its parameters are reported separately from parameters with gradients. Categorical IDs0/1 denote training-dataset stored-channel identities; unseen external channels must explicitly useID-1 rather than being mislabeled as source channels or numerical clinical operators.

The adapter masks unavailable/failed responses before encoding and rejects all-channel representation failures for explicit caller-level indeterminate handling. Tests cover identity with the original forward path, joint measurement permutation, unobserved/failed-response isolation, q ablation, unknown zero embeddings and active gradient paths. `torch_phase_response.py` evaluates the same fixed feature mean with float64 torch operations before float32 caching. Its CPU reference tests and actual-training GPU witness agree with original NumPy features within1e-7 (observed maximum0.0 for the four-channel fixture).

`verify_set_gpu.py` and a fresh documented-command witness passed both authentic modes on A100 with BF16 gradients and FP32 permutation/isolation checks. This is a temporary training-only kernel/synthetic-loss fixture, not a production fit. Runtime instructions and receipts: `RUN_SET_GPU_WITNESS.md`, `set_adapter_gpu_preflight.json`, `set_gpu_fresh_doc_witness.json` under `waveform_v1/`. Production phase generation and the SetOperator training/selection protocol remain pending.

`verify_raw_development.py` independently confirms all504777 training and166047 validation windows,90/30 source patients with zero overlap, exact assigned-duration weights,3,028,510,919 unique training source samples, and no test waveform sources in the loading receipt. Its optional full normalization replay uses chunked float64 variance merging on source-float32 training values, separate from the trainer's sum/squares implementation. Live verification status and receipts remain authoritative.

Full normalization replay has completed across all181 training files: maximum mean difference4.44e-15mV and population-SD difference2.75e-11mV. No validation/test sample entered that independent calculation. Receipt: `waveform_v1/raw_mask_resnet_v1/normalization_verification.json`.

## Production phase-feature cache

`generate_phase_features.py` prospectively freezes all-window feature generation, requiring complete observed extraction and a completed training-only kernel with a passing approximation audit. It hashes each source HDF5/observed-peak shard and commits per-file float32 responses, original validity, accepted-beat counts and represented fractions. Every declared window remains present; failed feature cells stay NaN. Reuse requires matching source/kernel/protocol/output hashes; orphan or changed output cannot silently overwrite or pass. The first valid response per file/channel must agree with original NumPy computation within1e-7.

Four spawned CPU workers prepare original per-window PCHIP atoms, with at most8 queued chunks. GPU batches contain at most16 channel representations; variable-beat padding contributes zero feature weight and changes no detected/accepted beats or represented time. This amortizes GPU launch/context overhead while preserving the original mathematical feature mean. Tests cover padding, all-channel failures, corrupted commits and orphan outputs. A fresh documented-command witness passed on256 actual training windows: original-feature discrepancy0.0, exact repeated cached responses, atomic roundtrip and committed-file checks. Warm fixture time2.0453seconds includes NumPy audits and concurrent-job effects, not full-cohort I/O or completion time.

Frozen generation protocol/witness: `waveform_v1/phase_features_v1/`; deployment command: `RUN_PHASE_FEATURES.md`. The production cache has not started until its dependency gates pass. Feature preprocessing of test waveforms uses only frozen training-fitted transforms and does not generate diagnostic predictions.

## SetOperator train/validation protocol

`train_set_waveform.py` prospectively freezes categorical, q-ablated and morphology/RR hybrid runs. All use the raw model's acquisition schedule, batch256, AdamW settings,40-epoch cap/patience8 and BF16-training/FP32-validation contract. Expanded acquisition-example weights total1 per declared training patient, preserving the objective over each3-epoch cycle; unavailable scheduled rows receive no gradient and retain explicit counts. Validation missing rows contribute fixed log2 only to checkpoint-selection loss while their prediction scores remain NaN. Fresh-model exact validation replay is required. No test feature shard is read and no test inference path exists.

`hybrid_set_model.py` preserves the original categorical morphology encoder and adds a single linear head on its256-D latent,36 RR inputs,2 modality flags and2 represented-fraction inputs. RR scaling uses training rows only; supported individual missing statistics are mean-imputed, exact constants use declared unit scale. Explicit RR-only/morphology-only/joint counts distinguish recovery from missing morphology from any measurement-encoding effect. Both modalities absent remains indeterminate. The hybrid's RR-only path cannot support a claim of morphology or operator geometry superiority. Q-ablated is also the identical channel-agnostic set control, so no duplicate run is added.

All83 component tests and fresh three-mode A100 gradient/repeated-logit witnesses pass. These checks do not fit the production models. Full phase cache and adequate free GPU remain prerequisites. Frozen protocol/witness: `waveform_v1/set_models_v1/`; invocation: `RUN_SET_TRAINING.md`.

The raw train/validation run completed14 epochs and selected epoch index5 under the frozen minimum-validation-BCE rule; its separate best/final checkpoints, threshold and same-process exact validation replay are sealed in `raw_mask_resnet_v1/evaluation_lock.json`. A separate source-waveform fresh-process validation replay is performed by `replay_raw_validation.py`; its receipt, rather than this progress description, determines that check's outcome. No held-out waveform predictions have been generated by these runners.

The independent fresh-process raw replay has now passed all166047 validation windows/498141 acquisition predictions from60 source HDF5 files and30 patients. Logits and scores are bitwise exact (maximum difference0.0), the selected checkpoint is the strict history minimum (epoch5; final epoch13), and threshold reselection matches with zero decision changes. No test waveform was read. The fresh documented-command witness ran the complete replay once and observed no runtime/doc divergence. Authoritative receipts: `raw_mask_resnet_v1/fresh_process_validation_replay.json` and `fresh_replay_doc_witness.json`.

## Original Charitos R verification

The original author repository is https://github.com/echaritos/AF-density . Its unchanged 22-Nov-2019 v3 AF.density.function.R was executed using R 4.3.3. `verify_author_r.py` compares the Python binned port on 66 fixtures with and without the author clamp: 132 comparisons passed at absolute tolerance 1e-11, maximum difference 5.33e-15. The report pins the author source SHA256.

The binned implementation sums AF amounts per equal observation bin, approximates the integral with a 0.001 step, returns zero for no AF and by default clamps results above one. These conventions must be recorded; they differ from an exact continuous interval calculation with undefined edge cases. The caller must supply the correct timeunits (bin capacity in the units of AF amount).

R was not installed system-wide. Ubuntu r-base-core 4.3.3-2build2 was downloaded and unpacked at /tmp/af_density_r_runtime; package files under usr/lib/R/etc were copied from the unpacked etc/R directory, with Renviron taken from Renviron.ucf. R_HOME and LD_LIBRARY_PATH were set only for the verification subprocess. This temporary runtime may need rebuilding later. Source, fixtures, R script and outputs are preserved on NFS.

## Gap handling and controlled stress pilot

`gaps.py` implements explicitly named observed-only, non-AF and context policies. Observed-only compresses both clocks and is never presented as preserving the full monitoring period. Context requires both observed flanks; terminal gaps remain indeterminate. For odd-length gaps, the midpoint bin belongs to the right half.

`stress_reference_patterns.py` applies eight declared alterations to every multi-interval calibration-excluded IRIDIA daily segment: 71 segments from 57 patients, 568 case-scenarios. Metrics share a one-second clock, explicitly time-bin analogues of the discrete equations. Report scripts retain undefined cases in declared denominators and use patient-weighted summaries with patient bootstrap intervals. No model is involved.

The controlled pilot shows mixed metric sensitivity, not universal aggregation superiority or reproduction of the published detector/SQI ICCs. Source and output details are in controlled_pattern_stress/REPORT.md on NFS.
