# Pipeline Summary

**Problem**: Predictable phase state versus standardized phase innovation
**Final Method Thesis**: A past-conditioned location-scale model may isolate phase variation not explained by predictable first- and second-order state.
**Final Verdict**: G1/G2 PASS; G3 REAL-DATA ADMISSIBILITY REQUIRED
**Date**: 2026-09-18

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`

## Evidence so far

- The legacy BCE-only GRU does not implement the claimed flow/SCM mechanism.
- Frozen synthetic seeds 45, 46 and 47 pass all six location-scale gates.
- Stagewise gradient tests prevent diagnosis labels from redefining innovations.
- Reverse-time prediction is easier, so reversal is not a valid destroyer.

## Next run

G3 real phase-cell lineage, coverage and residual diagnostics on development data. Do not launch the legacy production grid.
