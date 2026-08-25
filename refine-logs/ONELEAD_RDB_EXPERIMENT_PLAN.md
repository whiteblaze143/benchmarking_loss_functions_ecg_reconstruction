# One-Lead RDB Waveform-Driven External Delineation Plan

**Problem**: Compare the completed one-lead reconstruction models fairly on a genuinely held-out RDB cohort without letting an architecture-specific delineation head influence the endpoint, while avoiding wasteful full-cohort evaluation of clearly futile checkpoints.

**Method thesis**: A useful one-lead reconstructor must preserve clinically meaningful P/QRS/T morphology in its reconstructed missing leads well enough that a frozen, external LUDB-trained delineator can recover RDB boundaries.

**Date**: 2026-08-25

## Frozen population and data contract

- The requested 60-model population is the 60 `completed` entries in `refine-logs/queue_spatial_1lead/queue_state.json`: 30 Lead-I models (`observed_leads=[0]`) and their 30 matched Lead-II controls (`observed_leads=[1]`).
- Lead I is the primary Track-A population. Lead II is a prespecified input-lead sensitivity/control analysis, not pooled into the Lead-I ranking.
- The extra 24 entries in `refine-logs/queue_1lead/queue_state.json` remain `pending` and are excluded from this frozen 60. The checkpoint catalog currently contains 84 verified assets because it includes both populations.
- Evaluation uses only the 360-record `data/rdb_wavelet_delineation_cache/test` split. The split is Chapman-record-disjoint, rhythm-stratified, hash-frozen with seed `20260822`, and was excluded from architecture selection and sweep training.
- Reconstruction receives only the observed lead. RDB labels are inaccessible to both the reconstructor and the frozen LUDB SemiSeg model and are loaded only for scoring.
- Results are aggregate-only SQLite. No reconstructed waveforms, per-record predictions, or CSV files are retained.

## Claim map

| Claim | Minimum convincing evidence | Block |
|---|---|---|
| C1 (primary) | On held-out RDB, reconstructed missing leads retain enough morphology for strong external six-boundary delineation, with paired uncertainty against A0. | B1, B3 |
| C2 (supporting) | Spatial/physiological variants improve the external endpoint beyond waveform loss alone, and any gain is not merely an observed-lead artifact. | B2, B3 |
| Anti-claim | Results do not arise from RDB leakage, an internal task head, derived-limb inflation, or outcome-dependent cherry-picking. | B1-B4 |

## Experimental blocks

### B1 — Frozen original-waveform ceiling and A0 anchors (must run)

- Run the frozen LUDB SemiSeg delineator on original RDB waveforms to quantify its cross-dataset ceiling.
- Fully evaluate prespecified A0 anchors before any futility rule is learned.
- Report missing-lead waveform PCC/RMSE/bias/limits of agreement and six boundary endpoints (`P-on`, `P-off`, `QRS-on`, `QRS-off`, `T-on`, `T-off`).
- Boundary event metrics: sensitivity/recall, positive predictive value, F1 at 20 ms, and timing bias/MAE/p95 among matches within 150 ms.
- Region metrics: one-vs-rest sample-level sensitivity, specificity, PPV, Dice, and IoU for P/QRS/T on valid RDB annotation samples.
- Primary endpoint: mean micro-F1 at 20 ms across six boundaries and all six missing precordial leads. Derived limb leads are a separate control and never enter the primary endpoint.

### B2 — Threshold-calibration pilot (must run before pruning)

- Pilot subset: 48 test records, exactly six per RDB rhythm, chosen by a frozen SHA-256 ordering. This is an evaluation-compute subset, not a model-selection training set.
- Prespecified calibration models: paired Lead-I/Lead-II versions of A0, E1, permuted-geometry control, panorama-author, and exact-theta variants spanning both `1010010` and `1110000` where available (at least six full models per input lead).
- Every calibration model receives both the 48-record pilot and the full 360-record evaluation.
- A model is considered potentially competitive if its full primary boundary endpoint is within 0.02 absolute F1 of its input-lead A0 anchor **or** its full precordial waveform PCC tail is within 0.05 of A0. This disjunctive definition prevents the waveform gate from deleting a model that is useful for delineation, or vice versa.
- Learn pilot cutoffs separately for Lead I and Lead II. Use the 99% upper confidence bound of each pilot endpoint. A checkpoint may be pruned only when **both** its boundary-F1 UCB and waveform-PCC UCB fall below the corresponding zero-false-skip calibration cutoffs, with additional margins of 0.02 F1 and 0.05 PCC.
- Do not activate pruning unless leave-one-out calibration produces zero false skips for every potentially competitive anchor. If that condition fails, all 60 models run fully.

### B3 — Resumable blinded evaluation of the frozen 60 (must run)

- Pilot all 60 models. Promote every model not meeting the conservative two-endpoint futility condition.
- Fully evaluate every fifth candidate that would otherwise be pruned as a blinded sentinel, plus every model within 0.02 of either cutoff.
- If any sentinel is competitive on the full cohort, invalidate the gate and automatically promote all pruned models.
- Store explicit states: `pilot_complete`, `full_complete`, `pruned_futility`, `error`, and `gate_invalidated`. A pruned model is censored, never assigned a score of zero and never included in complete-case rankings.
- Compare variants within the same observed lead, factorial mask, and compute family. Lead-I and Lead-II results remain stratified.

### B4 — Failure analysis and paper outputs (must run)

- Stratify endpoints by RDB rhythm and lead group without using strata to tune the gate.
- Report bootstrap confidence intervals and paired deltas against A0; control multiplicity for the small prespecified mechanistic contrasts.
- Show Bland–Altman summaries on the physical mV scale, per-lead PCC, boundary error distributions, and representative successes/failures selected by frozen quantiles rather than visual preference.
- Interpret the original-waveform score as the external delineator ceiling, not perfect ground truth performance.

## Run order and decision gates

| Milestone | Runs | Decision gate | Estimated cost |
|---|---|---|---|
| M0 | Loader/hash/schema smoke test on 2 records and 1 model | Strict checkpoint load; observed-lead passthrough; no label access before scoring | 5–10 min CPU |
| M1 | Original waveform ceiling on all 360 records | SemiSeg and RDB scoring produce finite aggregate metrics | 10–20 min CPU |
| M2 | Pilot + full evaluation of prespecified calibration anchors | At least six full anchors per input lead | roughly 1–3 h, depending on contention |
| M3 | Fit and audit the futility gate | Zero leave-one-out false skips, otherwise disable pruning | minutes |
| M4 | Pilot/promote the remaining frozen models | Sentinel audit remains clean | roughly 4–10 h total |
| M5 | Final analysis | Exactly 60 terminal model states plus original ceiling; no unreported exclusions | minutes |

## Compute policy

- CPU only (`CUDA_VISIBLE_DEVICES=''`), one evaluator process.
- The host has 8 physical cores, 15 GiB RAM, no swap, and currently has a load average above 7 with about 4.4 GiB available. Therefore the launcher must wait for safe load/memory conditions.
- Start with two PyTorch threads. It may use up to six threads only when the GPU training data loaders have released CPU/RAM, load is below the configured gate, and at least 7 GiB is available.
- One checkpoint is materialized, SHA-256 verified, evaluated, and evicted at a time to keep disk use bounded.

## Risks and mitigations

- **RDB was used for supervised wavelet/delineation training**: only the untouched 360-record test partition is evaluated; its hash is part of the protocol identity.
- **Pilot pruning creates informative missingness**: conservative dual-UCB gate, leave-one-out zero-false-skip requirement, sentinel full runs, and automatic invalidation.
- **Boundary specificity is ill-defined for event matching**: report event sensitivity/PPV/F1 and separately report sample-level region specificity.
- **The 60-model phrase is ambiguous**: freeze the source queue’s exact 60 completed one-lead jobs and report Lead I (primary) and Lead II (control) separately.
- **Host contention**: load/RAM-gated detached supervisor; no six-core launch while current training saturates the machine.

## Final checklist

- [ ] 360 test IDs and manifest hash frozen
- [ ] exact 60 model IDs frozen from source queue
- [ ] original ceiling complete
- [ ] calibration anchors complete
- [ ] futility threshold audited before activation
- [ ] 60 terminal states recorded
- [ ] Lead-I primary and Lead-II control reported separately
- [ ] compact DB contains aggregates only

