# Experiment Plan

**Problem**: Determine whether physically structured lead conditioning improves ECG-AIM reconstruction when exactly one ECG lead is observed, without changing the established loss, data, masking, or evaluation protocol.
**Method Thesis**: A learned lead-ID residual augmented by canonical lead geometry and explicit observed-to-target geometry should improve worst-case reconstruction, especially when the observed lead varies, because it conditions the model on measurement viewpoint rather than only a 12-class lookup.
**Date**: 2026-08-22

## Context Freeze and Provenance Gate

- Active ECG-AIM implementation: `unified_latents/engineering/experimental/alitok_vae_exp.py`, imported by the training and evaluation registries. `ecg-aimbackup.py` is reference-only and must not be edited.
- Current model geometry: one learned `[12, width]` lead embedding is added in both `_decode()` and `_encode_grid()`.
- Historical geometry baseline: Electrocardio-Panorama's `ThetaEncoder` forms `[theta, phi, theta+phi, theta-phi]`, followed by elementwise sine and cosine, producing 12 features. `model_nefnet2.py` applies the encoded query geometry multiplicatively to latent features.
- Lead order must be remapped from Panorama to ECG-AIM with `[0,1,8,9,10,11,2,3,4,5,6,7]`.
- Frozen objective: factorial mask `1010010`, unchanged loss weights, data, split, normalization, masking policy, patch size, optimizer, scheduler, epoch budget, early stopping, and evaluator.
- **One-lead checkpoint inventory**: `refine-logs/queue_1lead/queue_state.json` defines ECG-AIM jobs for Lead I (`--observed_leads 0`) and Lead II (`--observed_leads 1`), and eight corresponding checkpoint files exist. Their embedded provenance confirms `observed_leads=[0]` or `[1]`, seed 201, and masks `1000000`, `1100000`, `1001000`, or `1000100`. The queue statuses are stale (`pending`) and must not be used as evidence that the files are absent.
- **Exact checkpoint location**: the checkpoint catalog contains `factorial_ecg_aim_1010010_s42` as a `remote_verified` release asset (`asset_id=523581550`, 189,666,614 bytes, SHA256 `5ac89f3d2b52b0e188a7be6f456ff8a0fbd543a023bffd2d79a83ec16275ef4f`). Its payload was validated with mask `1010010`, seed 42, 184 tensors, and state-schema SHA256 `f21a8d316057cb08d8bad75d3542707cf42db81f1b403a5d98f234675e16cd0b`. It is not missing; it must be materialized through `scripts/checkpoint_store.py materialize factorial_ecg_aim_1010010_s42` and digest-checked before A0 evaluation.
- **Observed-lead audit**: the catalog metadata for that exact checkpoint records training-time `observed_leads=[0,1,7]` (I, II, V2). This does not invalidate the artifact or its required reference evaluation, but it does mean the fixed single source for the new controlled 1→11 experiment must be recovered from the intended one-lead evaluation/job configuration rather than inferred from the model ID. Existing one-lead checkpoints establish both Lead I and Lead II protocols; select the authoritative source only after the audit.
- Reference-only reproduction target for the named checkpoint: Pearson p05 0.55063, MSE p95 0.01360, event error p95 0.36868 mV, QRS RMSE p95 0.20915 mV, QRS area p95 10.642 mV·ms, T RMSE p95 0.09633 mV, T area p95 12.296 mV·ms, ST-J p95 0.06582 mV.
- Baseline equivalence means identical outputs after refactoring with the same weights, inputs, and evaluation adapter, within `atol=1e-6, rtol=1e-5` in FP32; checkpoint compatibility alone is insufficient.
- No frontier primitive is claimed. The frontier-necessity block is intentionally omitted.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1 (primary) | Explicit source→target physical geometry improves robust 1→12 reconstruction beyond categorical lead IDs. | On a lead-agnostic exactly-one-source benchmark, lead-field hybrid + relative + FiLM beats learned-ID baseline across 3 seeds on paired patient-level Pearson p05, with no clinically material regression in ST-J/QRS/T tails; gains appear across multiple source and target leads. | B1, B4, B5 |
| C2 (supporting) | The gain comes from geometry and relative conditioning, not extra capacity or a generic modulation path. | Replacement/hybrid/relative/FiLM ablations isolate the contribution; parameter/FLOP-matched learned controls fail to explain the improvement; lead field outperforms Panorama if C/D is claimed. | B2, B3 |
| Anti-claim | The result is checkpoint provenance drift, a 3-lead comparison mislabeled as 1-lead, parameter count, or fixed-source memorization. | Provenance gate, true 1-lead A0, matched seeds and paired records, parameter-matched controls, and variable-source evaluation. | B0–B4 |

## Paper Storyline

- **Main paper must prove**: (1) a valid one-lead baseline; (2) geometry gains on a fixed source without clinical-tail regression; (3) stronger, multi-seed gains when the sole source lead varies; (4) the gain survives capacity controls and is attributable to source→target geometry.
- **Appendix can support**: geometry-only replacement, per-seed tables, all per-target/source matrices, field-similarity diagnostics, warm-start results, and a contingent withheld-source test.
- **Experiments intentionally cut**: loss-weight retuning, patient anatomy, field autoencoders, QT/QRS auxiliary heads, patch-size/masking changes, temporal attention bias, broad architecture sweeps, and G1 until F1 passes.

## Experimental Invariants

Use PTB-XL tensors at 500 Hz, 5000 samples, no normalization change, and the existing train/validation/test inventories (17,418/2,183/2,198 files observed locally). Freeze their content hashes before training. Use ECG-AIM width 384, encoder depth 8, decoder depth 4, heads 8, patch size 25, and the exact optimizer/scheduler/epoch settings recovered from the baseline training path. Train headline comparisons from scratch. Screen with seed 42; confirm only A0 and the selected spatial winner with seeds 42, 123, and 2026. Evaluate the same records and intervals with paired statistics.

For A0, first materialize and evaluate the cataloged `factorial_ecg_aim_1010010_s42` checkpoint exactly as instructed, reproducing the quoted metrics and recording the effective observed-lead mask used by the evaluator. Separately, recover the intended fixed single source from checkpoint/evaluation/job provenance. The headline architecture comparison still trains every row—including learned-ID A0—from scratch with seed 42 under that one-source protocol, so the archived checkpoint is a reference and parity anchor rather than a warm start.

### Active Queue Handoff (2026-08-22)

- The initial screen is **seed 42 only**. Seeds 123 and 2026 are not queued; they remain a later confirmation step after selecting one winner.
- The currently running 3-lead job `ecg_aim_f_1100012_s42` completed successfully and produced its validated checkpoint. The next job `ecg_aim_f_1100013_s42` was restored to `pending` before meaningful training began.
- Priority queue: `refine-logs/queue_spatial_1lead/queue_state.json`.
- Worker: `scripts/run_spatial_1lead_queue.py`; handoff wrapper: `scripts/run_spatial_1lead_then_resume_3lead.sh`.
- The priority queue contains A0–E1 for Lead I and Lead II (10 total runs), all using mask `1010010`, seed 42, identical optimizer/data/split settings, and separate checkpoint IDs.
- After all priority jobs reach a terminal status, the wrapper executes `scripts/run_3arch_queue.py`, resuming the remaining 3-lead queue from `ecg_aim_f_1100013_s42`.
- The original `alitok_vae_exp.py` and `train_1lead_factorial_multimodel.py` remain unchanged. Spatial work is isolated in the user-provided `ecg-aimbackup.py` and new `train_1lead_spatial_ecg_aim.py`.
- Lead-field F1/G1 are not queued because the validated canonical asset gate has not passed.

## Experiment Blocks

### Block 0 (B0): Provenance, evaluator, and refactor parity

- **Claim tested**: comparisons share a valid protocol and the refactor does not alter learned-mode behavior.
- **Why this block exists**: the supplied baseline is stored remotely in the catalog, and its recorded training-time three-lead mask must be kept distinct from the new one-source evaluation/training protocol.
- **Dataset / split / task**: existing LUDB oracle panel for checkpoint reproduction; tiny PTB-XL batch for tensor parity; frozen PTB-XL split inventories.
- **Compared systems**: archived checkpoint through current evaluator; pre-refactor reference loaded from archived source; refactored `learned` mode loaded with identical state.
- **Metrics**: exact eight clinical metrics; max/mean absolute output difference; checkpoint missing/unexpected keys; split and protocol SHA256.
- **Setup details**: centralize `_lead_condition(inherited)` and route both `_decode()` and `_encode_grid()` through it. Keep `ecg_aim_v1` behavior unchanged; expose spatial work under `ecg_aim_spatial_v1`.
- **Success criterion**: quoted checkpoint metrics reproduce within existing evaluator precision; metadata is explicitly labeled 3-lead; learned-mode tensors pass `atol=1e-6, rtol=1e-5`; adapter loads without key drift.
- **Failure interpretation**: stop all GPU training and resolve evaluator, split, source-lead, or refactor drift.
- **Table / figure target**: Appendix provenance table and parity test report.
- **Priority**: MUST-RUN.

### Block 1 (B1): Fixed-source main anchor

- **Claim tested**: structured geometry can improve a controlled fixed-source 1→11 reconstruction task.
- **Why this block exists**: establishes non-inferiority and a first efficacy signal under the exact current objective.
- **Dataset / split / task**: PTB-XL, one fixed observed source, remaining 11 reconstructed; exact existing clinical evaluation on the frozen test cohort.
- **Compared systems**: A0 learned ID; C1 learned + Panorama; E1 learned + Panorama + relative + FiLM; F1 learned + validated lead field + relative + FiLM. B1 geometry replacement appears in B2, not the headline table.
- **Metrics**: primary Pearson p05; secondary MSE p95 and all six clinical tail errors; paired patient bootstrap CIs; parameter count, FLOPs, throughput, peak memory.
- **Setup details**: seed 42 screen from scratch; same initialization policy and training budget. F1 is eligible only after asset validation. Report targets, never the observed source, in global metrics.
- **Success criterion**: minimum non-inferiority Pearson p05 ≥0.54 and no material ST-J/QRS/T regression; strong signal is Pearson p05 >0.58 with simultaneous secondary non-inferiority. Final superiority requires a paired 95% bootstrap CI for improvement excluding zero and confirmation in B4.
- **Failure interpretation**: if C1/E1 fail, angular conditioning is not useful under fixed source; if F1 alone wins, evidence favors richer geometry; if all tie, fixed-source memorization may dominate.
- **Table / figure target**: Main Table 1; architecture × primary metric plot; paired per-record scatter and ΔPearson distribution.
- **Priority**: MUST-RUN.

### Block 2 (B2): Novelty isolation and mechanism ablation

- **Claim tested**: gains arise from hybrid physical conditioning and explicit relative geometry.
- **Why this block exists**: separates feature identity, residual correction, and repeated decoder conditioning.
- **Dataset / split / task**: same fixed-source PTB-XL task and seed-42 screen as B1.
- **Compared systems**: A0 learned; B1 Panorama-only replacement; C1 learned+Panorama; D1 C1+relative; E1 D1+FiLM; F1 learned+lead field+relative+FiLM. G1 is excluded here.
- **Metrics**: primary/secondary endpoints, per-target lead metrics, parameter/FLOP deltas, learned spatial/relative gains.
- **Setup details**: Panorama angles are frozen buffers in ECG-AIM order. Relative context pools only actually observed leads. FiLM projections are zero-initialized. Each row changes only the named component.
- **Success criterion**: an ordered, interpretable contribution pattern; the final model must exceed C1 and a parameter-matched learned control, not merely A0.
- **Failure interpretation**: B1<C1 indicates categorical residual correction is necessary; D1≈C1 rejects explicit relative geometry under fixed source; E1≈D1 rejects repeated modulation; F1≤E1 cannot support lead-field novelty.
- **Table / figure target**: Main Table 2 deletion study; gain-by-target-lead heatmap.
- **Priority**: MUST-RUN.

### Block 3 (B3): Simplicity and capacity control

- **Claim tested**: extra parameters or generic conditioning do not explain the result, and attention bias is unnecessary unless justified.
- **Why this block exists**: the geometry encoders and FiLM add capacity.
- **Dataset / split / task**: fixed-source validation/test, seed 42; promote to 3 seeds only if it challenges the interpretation.
- **Compared systems**: selected spatial model; A0 plus a parameter-matched MLP fed learned lead IDs; selected model with spatial inputs permuted across lead labels; optional G1 lead-field attention bias only if F1 passes B1/B2.
- **Metrics**: eight endpoints, parameter/FLOP/latency/memory, paired differences.
- **Setup details**: matched control uses the same encoder/modulation widths; permutation is fixed before training and preserves feature distribution while destroying semantics.
- **Success criterion**: spatial model beats matched and permuted controls; G1 is adopted only if it adds a reproducible benefit beyond F1 sufficient to justify complexity.
- **Failure interpretation**: matched-control parity downgrades the claim to capacity/regularization; permuted parity rejects geometric semantics; G1 parity means cut attention bias.
- **Table / figure target**: Main or appendix simplicity table; G1 appendix-only unless decisive.
- **Priority**: MUST-RUN for matched/permuted controls; NICE-TO-HAVE for G1.

### Block 4 (B4): Lead-agnostic exactly-one-source test

- **Claim tested**: geometry generalizes across measurement viewpoints instead of memorizing one fixed mapping.
- **Why this block exists**: this is the decisive test for C1 and the dominant paper claim.
- **Dataset / split / task**: each sample exposes exactly one uniformly selected source lead during training; reconstruct the other 11. Use deterministic epoch/sample selection keyed by seed. Evaluate all 12 source leads separately on the same test records.
- **Compared systems**: A0 learned-ID and the best spatial model selected before examining B4 test results; both seeds 42, 123, 2026.
- **Metrics**: macro and worst-source Pearson p05; eight global endpoints; per-source/per-target matrix; paired bootstrap CI; seed mean/SD and individual seeds.
- **Setup details**: equal source-lead exposure, identical training steps, no test-driven model selection. Aggregate per patient before quantiles; preserve source lead identity in outputs.
- **Success criterion**: spatial model improves macro and worst-source Pearson p05 with 95% paired CIs excluding zero, has no material clinical-tail regression, and improves multiple source/target combinations across seeds.
- **Failure interpretation**: fixed-source-only gains imply regularization or memorization, limiting the conclusion to A/B; cross-source gains support C/D.
- **Table / figure target**: Main Table 3; source×target heatmap; per-source tail plot.
- **Priority**: MUST-RUN.

### Block 5 (B5): Failure and physical diagnostic analysis

- **Claim tested**: improvements and failures have a coherent spatial pattern and do not hide harmed patients.
- **Why this block exists**: global quantiles can conceal target- or patient-specific regressions.
- **Dataset / split / task**: test records from B1 and B4.
- **Compared systems**: A0 and selected spatial model; F1/G1 only when a validated field exists.
- **Metrics**: target-level Pearson median/p05, MSE p95, event and ST-J p95; per-record ΔPearson; field cosine similarity versus baseline performance and Δperformance using Spearman correlation with uncertainty.
- **Setup details**: emphasize V1, V3–V6, aVR/aVL/aVF but report all target leads. Show full distributions and named failure strata without post-hoc exclusion.
- **Success criterion**: gains are not driven by a few records, do not systematically harm clinically important leads, and are directionally stable across sources.
- **Failure interpretation**: heterogeneous harm narrows the claim and may reject deployment even if the global endpoint improves; similarity correlations are exploratory and cannot establish mechanism.
- **Table / figure target**: Main failure figure plus appendix per-lead tables.
- **Priority**: MUST-RUN.

## Lead-Field Asset Gate

F1 cannot begin until a canonical asset is documented with paper/DOI or arXiv, repository/license, geometry/model ID, heart and torso meshes, electrode positions, units, coordinate system, field orientation, polarity convention, and file hashes. Do not use `external/lead_field.md` or any numeric file as a field merely because it exists; validate provenance first.

If electrode fields are supplied, derive I/II/III/aVR/aVL/aVF and V1–V6 with the conventional electrode-difference equations, verify signs algebraically and with unit tests, then store ECG-AIM order. Normalize orientation, retain log magnitude, use deterministic SVD with `K=min(8,numerical_rank)`, and freeze the resulting features. Fit SVD using only the canonical geometry asset—not ECG outcome data. If validation fails, mark F1/G1 blocked and complete A0–E1 without inventing fields.

## Statistical Analysis Protocol

- Pre-register directions: higher is better only for Pearson; lower is better for every error metric.
- Compute all headline quantiles from per-record values, with identical records across models.
- Use 10,000 patient-level paired bootstrap resamples for model differences and 95% percentile CIs. For 3-seed confirmation, report each seed and mean±SD; summarize paired differences across matched seeds and records.
- Treat Pearson p05 as the single primary endpoint. Secondary metrics control clinical non-inferiority, not a second route to declaring success. Report all eight regardless of direction.
- Define clinical-tail non-inferiority margins before B1 test evaluation using domain review or a baseline-relative bound; do not infer margins after seeing results. Until approved, use “no material regression” descriptively and do not claim formal non-inferiority.
- Use B4 as confirmatory. B1/B2 select the model; do not select variants on B4 test performance.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost / Turnaround | Main Risk and Mitigation |
|---|---|---|---|---|---|
| M0 | Freeze provenance and evaluator | R001–R003 | Stop unless the cataloged exact checkpoint materializes with the registered digest, its reference metrics reproduce, the effective one-source protocol is recovered, hashes freeze, and learned refactor is numerically equivalent | CPU + <1 A100-hour; 0.5–1 day | Remote-only artifact and mixed task labels; use the catalog/materializer, embedded provenance, and explicit effective-mask logging |
| M1 | Validate implementation | R004–R005 | All 12 required unit tests and tiny overfit pass; no NaNs or contract drift | <2 A100-hours; 0.5 day | Geometry/order bugs; explicit mapping, buffer, sign, shape, and adapter tests |
| M2 | Establish baseline and angular screen | R006–R010 | Seed-42 A0–E1 complete for Lead I and Lead II; A0 converges and spatial rows provide a clear diagnostic | 10 full trainings; approximately 8–14 hours on the active GPU | Wasted screen compute; no extra seeds until a winner is selected |
| M3 | Validate and screen lead field | R011–R013 | Asset provenance passes; F1 beats or plausibly improves on E1; otherwise stop lead-field branch | asset work + ~2 trainings; 1–3 days | Invalid asset; block F1 rather than synthesize geometry |
| M4 | Isolate mechanism/capacity | R014–R015 | Winner beats matched and permuted controls | ~2 trainings; 1 day | Capacity confound; match parameter and optimization budgets |
| M5 | Confirm fixed-source winner | R016–R019 | A0 vs winner across 3 seeds shows stable paired improvement/non-inferiority | 4 additional trainings because seed 42 is reused; 1–3 days | Seed noise; matched seeds and no one-seed superiority claim |
| M6 | Confirm variable-source claim | R020–R025 | Macro and worst-source gains hold across 3 seeds with clinical-tail safety | 6 trainings; 2–5 days | Unequal source exposure; deterministic uniform sampling and per-source reporting |
| M7 | Polish and diagnose | R026–R028 | Required tables/plots complete; conclusion A/B/C/D follows predeclared rule | CPU evaluation + optional 1–2 trainings; 1–2 days | Selective reporting; full per-record and per-lead outputs |

## Compute and Data Budget

- **Current queued training count**: 10 seed-42 screen runs (A0–E1 × Lead I/Lead II). Multi-seed confirmation, variable-source training, F1/G1, and withheld-source tests are deferred and not consuming the current queue.
- **Estimated GPU-hours**: benchmark before committing. Local profiling on A100-40GB reports ~133 ECG/s at batch 32–64 and ~8.5/16.7 GiB peak for model steps, but end-to-end epoch time is not recorded. Budget provisionally **30–80 A100 GPU-hours** for must-runs, then replace this estimate with `2 ×` measured one-epoch time × planned epochs × run count (factor 2 allows validation/evaluation and compilation overhead).
- **Data preparation**: freeze train/val/test inventory/content hashes; create deterministic fixed-source and uniformly variable-source masks; preserve record IDs for paired analyses.
- **Human evaluation**: none. Clinical review is needed only to predeclare meaningful non-inferiority margins and inspect failure cases.
- **Biggest bottleneck**: obtaining and validating a licensed canonical lead-field asset. The initial setup gate is materializing/reproducing the exact archived checkpoint and resolving its recorded three-lead training mask against the intended fixed single-source protocol.
- **Stop/go economy**: do not launch B4 multi-seed runs until A0–E1/F1, provenance, and capacity gates select exactly one spatial candidate.

## Risks and Mitigations

- **Task-label leakage / source mismatch**: both one-lead and 3-lead checkpoint families exist, the exact cataloged baseline records `[0,1,7]`, and queue status is stale. Mitigation: materialize by catalog model ID, verify SHA256, log the evaluator's effective mask, recover the intended fixed source from authoritative configuration, and keep archived-reference and from-scratch headline roles separate.
- **Panorama order/sign error**: positional copying is invalid. Mitigation: named mapping unit test and expected angle rows for all 12 leads.
- **Baseline drift after refactor**: duplicated embedding paths may diverge. Mitigation: output/gradient parity tests with archived source and checkpoint adapter test.
- **Geometry is only extra capacity**: mitigation: matched learned-MLP and permuted-geometry controls.
- **Fixed-source memorization**: mitigation: make B4 confirmatory and report worst-source, not only macro average.
- **Lead-field fabrication or incompatible meshes**: mitigation: metadata gate, license check, coordinate/orientation audit, SHA256, and hard block on F1.
- **Tail metric instability**: mitigation: paired patient bootstrap, 3 seeds for finalists, and complete per-record export.
- **Loss/objective confounding**: mitigation: config diff gate that rejects any change outside architecture/conditioning and source sampling for B4.
- **Observed-lead scoring leakage**: mitigation: exclude the sole observed lead from target metrics and verify target masks per source.

## Required Deliverables After Execution

Write `results/spatial_ecg_aim/architecture_summary.csv`, `per_lead_metrics.csv`, `per_record_metrics.csv`, `seed_summary.csv`, immutable configs, checkpoint/asset/protocol hashes, `lead_field_metadata.json` when applicable, plots, and `REPORT.md`. Required plots are architecture vs primary metric, per-target Pearson p05/MSE p95/event/ST-J tails, paired per-record scatter, patient ΔPearson distribution, source×target results, and lead-field similarity diagnostics when applicable.

Classify the result only after B4: A (no architectural value), B (tail/regularization value), C (lead-field geometry value across sources), or D (fixed-source gain plus materially better changing-source generalization). Only C/D justify integrating lead-field conditioning into the main architecture.

## Final Checklist

- [x] Main paper tables are covered
- [x] Novelty is isolated
- [x] Simplicity is defended
- [x] Frontier contribution is explicitly not claimed
- [x] Nice-to-have runs are separated from must-run runs
- [x] Cataloged exact checkpoint, existing one-lead checkpoints, and their distinct protocol roles are explicit
- [x] Lead-field fabrication is prohibited
- [x] Fixed-source and lead-agnostic claims are separated
