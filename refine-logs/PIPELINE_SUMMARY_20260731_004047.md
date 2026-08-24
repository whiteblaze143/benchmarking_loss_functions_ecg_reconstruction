# Pipeline Summary

**Problem:** Loss-function rankings for ECG reconstruction change across amplitude, morphology, diagnostic, calibration, robustness, subgroup, and transfer endpoints.

**Final Method Thesis:** Use the implemented $2^5\times5$ mixed-level benchmark across three seeds with paired clinical diagnostics and explicit evidence levels; do not assume the largest composite loss wins.

**Final Verdict:** REVISE / IN PROGRESS

**Date:** 2026-07-31

## Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Book: `book/`

## Contribution Snapshot

- Dominant: 160-condition mixed-level loss benchmark across seeds 42, 200, and 201.
- Supporting: real-data diagnostic atlas and morphology–classifier discordance analysis.
- Rejected: universal composite winner, omnibus score, new architecture, unjustified frontier primitive.

## Must-Prove Claims

1. Loss effects depend on architecture and endpoint.
2. Morphology gains can diverge from diagnostic/calibration preservation.
3. Selected masks transfer without unacceptable tail, subgroup, or noise failure.

## First Runs After Training

1. Lock each 160-condition seed block and its record-order manifest.
2. Smoke-test one anchor mask through PTB-XL, LUDB, EchoNext, and Sunnybrook.
3. Launch paired per-record primary morphology evaluation.

## Current Risks

- The mixed-level checkpoint matrix is incomplete while training.
- ISP class/rate/raw-waveform provenance is unresolved.
- Frozen classifier and machine proxy endpoints can be overinterpreted.
- Rare endpoints yield unstable estimates.
- External resampling/unit mistakes can imitate scientific effects.

## Next Action

Continue dataset EDA and adapter validation while checkpoints train; do not publish placeholder factorial values as results.
