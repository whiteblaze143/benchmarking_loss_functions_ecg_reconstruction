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

## Open Risks

Checkpoint incompleteness, seed variance, external preprocessing, delineator bias, ISP provenance, rare labels, proxy outcomes, multiple testing, and subgroup support remain active risks rather than “resolved concerns.”
