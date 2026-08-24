# Pipeline Summary: External Delineation Integration

**Problem**: The factorial GPU queue produces checkpoints faster than the old separate LUDB/ISP/Zhejiang scripts can be manually rerun, and those scripts are not generation-bound.
**Final Method Thesis**: Evaluate every exact compatible checkpoint on endpoint-specific external morphology while preserving dataset evidence tiers and never competing materially with training.
**Final Verdict**: REVISE / RUNNING
**Date**: 2026-08-01

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Unified evaluator: `scripts/evaluate_external_delineation_watch.py`
- Results root: `results/factorial_mixed_level/external_delineation_generation_bound/`
- Runtime log: `refine-logs/queue/external_delineation_generation_bound.log`

## Contribution Snapshot

- **Dominant contribution**: generation-bound external delineation evidence for the fixed-MCMA loss factorial.
- **Supporting contribution**: a low-resource watcher that turns idle waiting time into exact-checkpoint LUDB/ISP/Zhejiang coverage.
- **Explicitly rejected complexity**: GPU evaluation, a second training scheduler, pooled cross-dataset scores, and silent unit inference.

## Must-Prove Claims

1. Loss components and MMD kernels have endpoint-dependent effects on missing-lead morphology.
2. Pareto candidates from PTB-XL survive a documented external delineation gate without detector or preprocessing artifacts.

## First Runs Launched

1. Focused matcher/resampler/checkpoint-store tests — complete.
2. One-checkpoint real-data smoke across LUDB, ISP, and Zhejiang — complete.
3. Full source-signal detector ceiling and compatible seed-42 backfill — running in detached `external_delineation_eval` tmux.

## Resource Contract

- CPU only; `CUDA_VISIBLE_DEVICES=''`.
- Cores 4–5 only; two low-priority workers maximum.
- `nice 19`, idle I/O, one Torch/BLAS thread per process.
- Wait when one-minute load exceeds 6.5/8 or available memory falls below 4 GiB.
- Materialize one SHA-256-verified checkpoint at a time and prune the private cache to zero.
- Poll every 1,200 seconds only after catching up.

## Main Risks

- **ISP/Zhejiang provenance**: outputs are exploratory and retain explicit provisional adapters.
- **Weak detector ceiling**: limits interpretation rather than being hidden.
- **Backlog**: accepted because GPU training remains the priority; two-core affinity is the hard ceiling.
- **Evaluator generation changes**: code/data digest changes invalidate skip decisions and create a new result generation.

## Next Action

- Monitor the production ceiling/backfill for first full-dataset artifact, confirm two-core/RAM/GPU isolation, and add its live coverage summary to the book.

