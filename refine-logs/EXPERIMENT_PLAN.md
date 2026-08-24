# Experiment Plan

**Problem**: Determine which loss, architecture, spatial-conditioning, wavelet, and SSL mechanisms preserve clinically relevant missing-lead ECG information without mixing incompatible input contracts or incomplete evidence.
**Method Thesis**: Use a coverage-first evidence ladder: complete within-contract factorial effects first, then clinical preservation, then external morphology and mechanistic one-lead studies.
**Date**: 2026-08-24

## Audited starting state

| Evidence source | Exact current coverage | Interpretation |
|---|---:|---|
| Seven-mask U-Net checkpoints, seed 42 | 160/160 | complete checkpoint grid |
| Seven-mask MSVAE checkpoints, seed 42 | 155/160 | five missing cells |
| Seven-mask ECG-AIM checkpoints, seed 42 | 123/160 | 37 missing cells |
| `missing_leads_v2` U-Net clinical models | 160/160 | complete within-U-Net clinical grid |
| `missing_leads_v2` MSVAE clinical models | 37/160 | descriptive partial coverage only |
| `missing_leads_v2` ECG-AIM clinical models | 0/160 | no result; not a negative result |
| One-lead checkpoint catalog | 84 models | development track; multiple ECG-AIM variants |
| One-lead wavelet/SSL queue | 120 planned cells | live screening queue; replication required |

The seven-character design is fixed MSE plus five binary toggles and a five-level MMD selector: $2^5\times5=160$. It is not seven binary factors. The old $3\times2^4=48$ benchmark, the new three-lead mixed-level grid, and one-lead studies remain separate estimands.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: loss effects are endpoint dependent within a fixed architecture/input contract | MSE alone can reward smoothing while morphology and diagnostic utility disagree | complete grid, common test ledger, paired patient inference, five binary effects, four kernel contrasts, prespecified interactions | B1–B3 |
| C2: ECG-AIM spatial and wavelet/SSL mechanisms can add complementary information | novel mechanisms must be isolated from capacity, architecture, and search effects | matched deletion chain, permuted geometry, Morlet-vs-UEG phase, seed/lead replication, boundary-localized external evaluation | B4–B5 |

**Anti-claims to rule out**: the best validation cell generalizes; absent database rows mean zero performance; E1 is inherently the improved base; wavelet gains come from SSL without a wavelet-only control; UEG-phase gains are generic regularization; architecture differences are estimable under unequal clinical coverage.

## Paper Storyline

- Main paper must prove C1 with a complete, generation-bound within-architecture analysis and endpoint discordance.
- The ECG-AIM section must prove C2 using matched mechanistic contrasts, not novelty language.
- Appendix can contain full 160-cell tables, failed-cell ledgers, kernel interactions, exploratory one-lead screens, and per-endpoint clinical atlases.
- Cut generic UMAPs, post-hoc composite clinical scores, unreplicated top-cell narratives, and cross-architecture rankings with unequal coverage.

## Experiment Blocks

### Block 1: Seven-mask reconstruction anchor

- Claim tested: C1.
- Dataset/task: patient-disjoint PTB-XL; I+II+V2 to 12 leads; score nine missing leads.
- Compared systems: all 160 masks within each architecture; architectures are not replicates.
- Metrics: missing-lead MSE/Pearson first; per-lead and tail errors second.
- Setup: identical preprocessing, source generation, selector, record order, seed block, and digest.
- Success: complete-cell paired estimates with failure accounting.
- Failure: incomplete grids restrict inference to a frozen reduced estimand.
- Target: reconstruction table, coverage diagram, Pareto plot.
- Priority: MUST-RUN.

### Block 2: Component and kernel isolation

- Claim tested: C1 and simplicity.
- Compared systems: five binary effects; MMD 1–4 versus 0; prespecified interactions; `1000000` versus `1111110`–`1111114`; deletion around promoted Pareto candidates.
- Metrics: paired test reconstruction and morphology endpoints, never validation rank alone.
- Success: replicated interpretable effects or an honest null.
- Failure: strong interactions/seed instability imply no universal recommendation.
- Target: effect forest and interaction heatmap.
- Priority: MUST-RUN.

### Block 3: `missing_leads_v2` clinical preservation

- Claim tested: C1's clinical relevance.
- Dataset/task: PTB-XL measured ECG versus reconstructed missing leads; external cohorts only where model rows exist.
- Compared systems: 160 U-Nets now; remaining 123 MSVAEs; then 160 ECG-AIMs with identical code.
- Metrics: AUROC/AUPRC/calibration; QRS, ST, voltage, variance retention, Bland–Altman, PreSACan, paired inference.
- Success: full coverage and endpoint-specific paired estimates.
- Failure: discordance yields a Pareto set, not a forced winner.
- Target: coverage funnel, endpoint forest, variance-retention atlas.
- Priority: MUST-RUN.

### Block 4: One-lead spatial mechanism

- Claim tested: C2.
- Compared systems: A0, panorama, hybrid, relative geometry, FiLM; exact-$\Theta$, capacity-matched, and permuted controls; lead I and II separately.
- Metrics: missing-lead reconstruction, per-lead tails, compute, and seed stability.
- Success: benefit survives capacity/permutation controls and both leads, or is narrowly scoped.
- Failure: E1 remains complementary rather than the default base.
- Target: matched-delta plot and spatial-control table.
- Priority: MUST-RUN for ECG-AIM.

### Block 5: Wavelet, SSL, and physiological second view

- Claim tested: C2.
- Compared systems: `A0→A0+wavelet→A0+wavelet+SSL`; A0/E1 complementarity; Morlet magnitude+phase versus magnitude+UEG-repolarization phase.
- Metrics: reconstruction, P/QRS/T IoU, six boundary timing errors, especially T-on/T-off, compute/memory.
- Success: replicated incremental gain; UEG gain localized to repolarization without unacceptable reconstruction regression.
- Failure: generic/nonlocalized lift weakens the physiology interpretation.
- Target: contrast forest and six-boundary localization plot.
- Priority: MUST-RUN for wavelet/SSL.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | reconcile schemas/identities | read-only DB audit | exact counts and keys agree | CPU minutes | generation aliasing |
| M1 | close checkpoint gaps | 5 MSVAE + 37 ECG-AIM | all verified or reduced estimand frozen | GPU days | queue/disk |
| M2 | close clinical gaps | 123 MSVAE + 160 ECG-AIM `missing_leads_v2` | record order and finite metrics pass | inference days | evaluator mismatch |
| M3 | common paired inference | eligible complete architectures | complete endpoint matrix | moderate | multiplicity |
| M4 | decisive one-lead replication | registered spatial/wavelet contrasts | seed/lead consistency | GPU days | screen regression |
| M5 | external morphology | promoted models on blinded LUDB/RDB | six-boundary gates pass | CPU days | domain shift |

## Compute and Data Budget

- Measure architecture wall times from queue logs before estimating total GPU-hours.
- The immediate clinical workload is 123 remaining MSVAE plus 160 ECG-AIM models.
- Store compact aggregates and required per-record ledgers in SQLite; do not duplicate waveform blobs or emit bloated CSVs.
- Preserve checkpoint/evaluator digests, record order, failure state, and endpoint validity.

## Risks and Mitigations

- Unequal coverage: show denominators and block architecture rankings.
- Validation winner's curse: validation selects; test supports inference.
- Mask ambiguity: decode six fields plus categorical MMD explicitly.
- Historical mixing: join by full model ID and digest, never mask alone.
- Multiplicity: prespecify endpoint families and contrasts.
- RDB invention: use P/QRS/T onset/offset only; peaks remain invalid.
- Disk pressure: compact databases and no duplicate CSV ledgers.

## Book expansion

Chapters 18–23 implement design/coverage, reconstruction, clinical, one-lead spatial, wavelet/SSL, and synthesis views. They read live databases at render time and separate measured evidence from missing coverage.

## Final Checklist

- [x] Main tables mapped
- [x] Novelty isolated
- [x] Simplicity defended
- [x] No irrelevant frontier primitive forced
- [x] Must-run separated from nice-to-have
- [ ] MSVAE/ECG-AIM seven-mask gaps closed or estimand frozen
- [ ] `missing_leads_v2` coverage complete across intended architectures
- [ ] One-lead contrasts replicated across seeds and leads

