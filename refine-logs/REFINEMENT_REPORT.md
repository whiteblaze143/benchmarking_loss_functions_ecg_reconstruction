# Refinement Report

## Thesis Evolution

- **V1:** A composite loss is superior to MSE.
- **V2:** Seven loss components eliminate smoothing and preserve clinical metrics.
- **V3:** A full factorial design will identify the best composite.
- **Current:** Loss effects are architecture- and endpoint-dependent; the contribution is a paired factorial benchmark and clinical evidence ladder, not a predetermined winner.

## Why the Current Thesis Is Stronger

The completed 48-cell results already contradict a universal-winner narrative: full composites improve morphology while EchoNext SHD macro AUROC can fall. The refined thesis treats this discordance as the scientific result to explain.

## Scope Decisions

- Keep the running mixed-level training study and correct its informal `$2^7$` label.
- Preserve the 48-cell grid as interim evidence.
- Use LUDB for lead-specific boundaries.
- Hold ISP waveform claims until provenance is linked.
- Use EchoNext for frozen external SHD evaluation with calibration.
- Use Sunnybrook for small fully measured external stress tests.
- Keep device/simulator transfer separate from prospective human validation.
- Do not add a new architecture or trendy primitive.

## Book Changes Driven by Refinement

- Real-cohort EDA and integrity diagnostics.
- Dedicated EchoNext classifier chapter.
- LUDB/ISP/Sunnybrook dataset atlas.
- Real robustness/fairness/SQI protocol.
- Detailed delineation and interval methodology.
- Device-transfer evidence boundary.
- Explicit placeholder/completeness contract for every 160-condition seed block.
- Current-generation identity gates covering source bundle, batch size, checkpoint digest, state schema, and data content roots.
- Withdrawal of synthetic engineering and cath-lab outcomes that had been presented as empirical results.

## Open Risks

Checkpoint incompleteness, seed variance, external preprocessing, delineator bias, ISP provenance, rare labels, proxy outcomes, multiple testing, subgroup support, and independently adjudicated fiducials remain active risks rather than “resolved concerns.” The generation-bound temporal-morphology evaluator is now implemented with explicit distribution MMD, a full-denominator per-record ledger, detector-state accounting, patient-cluster summaries, five-window pairing sensitivity, and within-feature Pareto diagnostics; at the 2026-08-01 15:09 UTC audit, 23/33 currently compatible checkpoints had completed that final evaluator generation, so it remains partial operational evidence rather than a factorial result. All 33 compatible checkpoints independently passed strict inference with finite `[1, 12, 5000]` output and zero retained audit-cache bytes. The requested independent Claude review bridge is unavailable in the current environment; no local review is labeled as cross-family validation.

## Latest Anchor and Simplicity Check

- **Anchor preserved:** the work still asks which loss components preserve missing-lead ECG information; storage and monitoring remain enabling infrastructure, not the paper contribution.
- **Design identity corrected:** the executable manifest contains seeds 42, 200, and 201, 160 masks per seed, five binary factors, and a categorical five-level MMD selector.
- **Claim scope corrected:** the 480-grid holds MCMA architecture fixed, so architecture dependence is limited to interim 48-cell evidence.
- **Complexity rejected:** no new generator, learned evaluator, or foundation-model component was added.
- **Optimization interpretation corrected:** total composite losses cannot be compared across masks; energy-distance activation changes objective scale sharply even when validation MSE is similar.
