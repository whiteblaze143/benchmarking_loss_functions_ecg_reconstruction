# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Determine which loss components preserve clinically relevant missing-lead ECG information when reconstructing 12 leads from reduced observations.
- **Must-solve bottleneck:** Pointwise error can reward conditional-mean smoothing, while global shape metrics can improve without preserving localized morphology, calibration, or downstream diagnostic information.
- **Non-goals:** Claiming reconstructed ECG replaces measured ECG or echocardiography; declaring one universal loss winner; treating frozen classifiers or machine statements as clinical adjudication.
- **Constraints:** Use the existing fixed MCMA three-lead-to-twelve-lead architecture for the expanded grid; compare only checkpoints sharing the same content-pinned source, batch-size, state-schema, preprocessing, and patient-split contract; preserve patient pairing; acknowledge that training is incomplete.
- **Success condition:** A complete, provenance-locked mixed-level factorial analysis identifies endpoint-specific main effects and interactions without unacceptable diagnostic, calibration, subgroup, robustness, or transfer degradation.

## Anchor Check

- **Original bottleneck:** loss terms may improve global reconstruction scores while smoothing or distorting localized, diagnostic, or tail morphology.
- **Why the revised method still addresses it:** the fixed-generator factorial design isolates loss configurations, while the evaluation ladder preserves per-record pairing, detector failures, diagnostic drift, and external/tail behavior.
- **Suggestions rejected as drift:** a new architecture, diffusion generator, learned clinical score, or foundation-model component would change the question from loss-function benchmarking to model invention.

## Simplicity Check

- **Dominant contribution:** a generation-bound, endpoint-aware mixed-level factorial benchmark within one fixed MCMA generator.
- **Components removed or merged:** three overlapping claims were reduced to two; architecture dependence was removed from the expanded-grid estimand; “full mask” was replaced by an explicit all-binary-on kernel family.
- **Suggestions rejected as unnecessary complexity:** no learned evaluator, post hoc composite utility score, new generator, or inference-time search policy.
- **Why this is the smallest adequate route:** the existing generator, frozen classifiers, annotated/external cohorts, and exact checkpoint store already expose the scientific bottleneck; the missing mechanism is rigorous identity-bound comparison, not extra modeling capacity.

## Changes Made

### 1. Corrected executable design identity

- **Finding:** one chapter named seeds 42/123/456, while the queue manifest schedules 42/200/201.
- **Action:** parse and assert 480 unique model IDs, 160 masks per seed, fixed MSE, five binary positions, and five kernel levels directly from the manifest.
- **Impact:** model identity and planned inference now match executable state.

### 2. Corrected estimands and claim scope

- **Finding:** “seven-bit,” “seven main effects,” and architecture-dependent expanded-grid language contradicted the implementation.
- **Action:** define five binary main effects, categorical kernel contrasts, and prespecified interactions within fixed MCMA; retain architecture heterogeneity only as interim 48-cell evidence.
- **Impact:** the proposal no longer claims a factor or comparison the 480-grid cannot estimate.

### 3. Corrected optimization-scale interpretation

- **Finding:** ~100-scale total losses were attributed mainly to derivative loss.
- **Action:** decode factor positions in the generation-bound training artifact and summarize total-loss scale by ED activation.
- **Evidence:** among 20 compatible runs, ED-off median first validation total is 1.7125 and ED-on is 106.9958, while median first validation MSE is 0.1425 and 0.1351 respectively.
- **Impact:** total loss is treated as a within-mask optimization objective, never as a cross-mask quality metric.

### 4. Updated evaluator status

- **Finding:** the refinement report still called the generation-bound temporal evaluator absent.
- **Action:** record the implemented explicit RBF MMD, full-denominator per-record ledger, detector states, patient-cluster summaries, and tolerance grid.
- **Impact:** the limitation is now incomplete coverage (7/20 compatible models), not missing machinery.

### 5. Closed the first controlled five-level kernel family

- **Finding:** optimization, morphology, distribution, and detector-availability metrics can point in different directions even when seed and all five binary factors are held fixed.
- **Action:** add a builder-bound candidate-minus-kernel-0 training artifact and join it to the full `1000000`–`1000004` generation-bound morphology family with equal-patient common-pair MAE and full-denominator coverage.
- **Evidence:** all four MMD kernels have slightly higher final validation MSE than MSE-only, while anatomical Laplacian lowers V6-QT common-pair MAE by about 20.3 ms but loses 6.64 percentage points of pair coverage; no kernel dominates all four morphology directions.
- **Impact:** the book now demonstrates endpoint discordance with a complete controlled block while retaining its one-seed, detector-conditioned, non-confirmatory boundary.

### 6. Replaced informal direction counting with a feature-local partial order

- **Finding:** pooling adaptive-bandwidth MMD values across unlike clinical features or assigning arbitrary endpoint weights would create a misleading winner score.
- **Action:** compute Pareto dominance separately within each lead-feature row using absolute bias, distance from unit variance, distribution MMD², and coverage shortfall; require a complete five-level block and publish a hash-bound dominance ledger.
- **Evidence:** kernel 4 is nondominated in 11/12 rows, kernels 1 and 2 in 10/12, MSE-only in 6/12, and kernel 3 in 5/12, with reciprocal dominance on some endpoints.
- **Impact:** the result is a transparent partial order that demonstrates trade-offs without inventing cross-feature scale comparability or a universal rank.

### 7. Quantified evaluator throughput

- **Finding:** accepted/compatible counts alone do not determine whether the evaluator is falling irreversibly behind.
- **Action:** bind evaluator service durations to the current training-duration summary and report concurrent arrival, service, and net backlog-drain rates.
- **Evidence:** seven accepted models give a median evaluator duration of 17.80 minutes versus 21.59 minutes for training, corresponding to 3.37 versus 2.78 models/hour and an observed-median 13-model backlog catch-up projection of 21.9 hours.
- **Impact:** the current evaluator keeps pace operationally under explicit stability assumptions; the estimate is automatically invalidated by stale training-summary identity and is not a confidence interval.

## Revised Proposal

### Method Thesis

Within a fixed MCMA three-lead-to-twelve-lead generator, loss components must be evaluated with a generation-bound mixed-level factorial design and paired endpoint ladder because waveform, localized morphology, physiologic consistency, diagnostic probability, robustness, and transfer endpoints can disagree.

### Contribution Focus

- **Dominant contribution:** a 480-identity endpoint-aware factorial benchmark: 160 configurations for each of seeds 42, 200, and 201.
- **Supporting contribution:** an auditable evaluation ladder linking each aggregate to exact checkpoints, patient-level records, detector failures, and external/frozen-model endpoints.
- **Explicit non-contributions:** no new architecture, no universal composite winner, no clinical replacement claim, and no learned scalar “clinical score.”

### Complexity Budget

- **Frozen/reused:** MCMA generator, PTB-XL content-pinned split, EchoNext and ECGFounder classifiers, existing external cohorts, and deterministic evaluator interfaces.
- **New trainable components:** none beyond the scheduled loss configurations.
- **New evaluation machinery:** identity/digest gates, compact per-record sufficient-statistic ledgers, and prespecified aggregation.

### Executable Design

The seven-character configuration is

$$m=(1,c,d,v,e,l,k),$$

where MSE is fixed, $c,d,v,e,l\in\{0,1\}$ activate correlation, first-difference L1, Kors-VCG, energy distance, and limb-lead consistency, and $k\in\{0,1,2,3,4\}$ selects the MMD kernel. This yields $2^5\times5=160$ configurations per seed. MMD level is categorical, not an ordinal dose.

Every admissible checkpoint must match the approved source bundle, batch size 1024, float16 state schema, preprocessing, split inventories, and byte-content roots. Evaluation adds checkpoint, evaluator, target-cache, and output digests.

### Training and Inference Path

1. Train each manifest identity with AdamW and deterministic seed handling on the pinned PTB-XL training split.
2. Select the checkpoint using lowest missing-lead validation MSE, breaking ties with highest missing-lead validation Pearson.
3. Archive exact bytes only after upload/download SHA-256, payload, and strict model-load verification.
4. Materialize one requested checkpoint into a bounded cache, recheck the digest, run inference, atomically publish optional outputs, and prune the cache.
5. Reconstruct the same immutable test records and retain model/record/failure identity through every endpoint.

### Evaluation Ladder

1. **Data contract:** patient separation, units, sampling rate, lead order, and byte-content roots.
2. **Waveform:** paired missing-lead MSE, MAE, Pearson, and variance ratio.
3. **Localized morphology:** QRS/ST windows, P/Q/R/S/T amplitudes, QT intervals, detector failure, pairing coverage, and timing error.
4. **Physiology:** Einthoven/Goldberger residuals and VCG consistency.
5. **Diagnostic behavior:** paired frozen-classifier probabilities, calibration, reclassification, and rare-task support.
6. **External/tail behavior:** LUDB, EchoNext, Sunnybrook, device/simulator transfer, corruption, SQI, subgroup risk–coverage, and influence tails.

### Failure Modes and Diagnostics

- **Generation aliasing:** prevent mask-only joins; require full model/checkpoint/source/data/evaluator identity.
- **Conditional-mean smoothing:** report amplitude slopes, variance ratios, and Bland–Altman behavior.
- **Detector conditioning:** retain target-only, reconstruction-only, both, and neither states on the full denominator.
- **Pairing sensitivity:** report 25/50/75/100/150 ms one-to-one matching sensitivity.
- **Objective-scale confusion:** compare validation MSE trajectories across masks, not total composite loss.
- **Seed instability:** block by seed and separate optimization replication from patient uncertainty.
- **Proxy overclaiming:** treat frozen classifiers and machine-derived labels as diagnostic probes, not adjudication.

### Claim-Driven Validation

#### Claim 1: Loss effects and interactions are endpoint dependent within fixed MCMA

- **Minimal experiment:** complete all 480 identities and estimate five binary effects, four kernel contrasts against level 0, and prespecified interactions for the primary paired endpoint family.
- **Baselines/ablations:** `1000000`, all-binary-on configurations `1111110`–`1111114`, component deletions at fixed kernel, and smaller Pareto candidates.
- **Decisive evidence:** patient-cluster intervals plus seed-blocked effect estimates that differ across endpoints and survive multiplicity control.
- **Falsifier:** negligible or seed-unstable effects, incomplete cells, or effects explained by detector/preprocessing failures.

#### Claim 2: Pareto-selected configurations must preserve diagnostic/external behavior, not morphology alone

- **Minimal experiment:** evaluate a prespecified compact Pareto set on frozen-classifier drift and external/tail datasets.
- **Baselines:** measured ECG, MSE-only reconstruction, and all-binary-on kernel family.
- **Decisive evidence:** bounded morphology improvement without unacceptable prespecified diagnostic, calibration, subgroup, noise, or transfer degradation.
- **Falsifier:** benefits disappear under paired uncertainty or fail on external/tail strata.

### Statistical Plan

- Preserve record pairing and cluster PTB-XL resampling by patient.
- Treat seed as a training-replicate block, not a patient sample.
- Control multiplicity within prespecified endpoint families.
- Use non-inferiority only after margins are clinically justified and frozen.
- Keep rare tasks and under-supported subgroups exploratory.
- Do not open final factorial rankings until identity and endpoint completeness gates pass.

### Current Status

The method remains **REVISE / IN PROGRESS**. Twenty current-contract checkpoints are inference-ready; seven have accepted final-generation temporal-morphology artifacts. This proves the pipeline operates and supplies one complete seed-42 kernel block, but it cannot support expanded-grid factorial claims. Independent cross-family review is unavailable and no numerical review score is assigned.
