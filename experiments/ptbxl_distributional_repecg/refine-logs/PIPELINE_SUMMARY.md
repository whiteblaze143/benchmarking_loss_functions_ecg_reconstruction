# Pipeline Summary

**Problem:** Test distribution-valued local ECG processes as eight independent
PTB-XL mathematical objects.

**Final method thesis:** Treat each cardiac phase cell as an empirical RKHS
distribution, then change one downstream mathematical object per paper and
require a matched control plus mechanism-use falsification.

**Final verdict:** READY for implementation  
**Date:** 2026-09-16

## Final Deliverables

- Proposal: `FINAL_PROPOSAL.md`
- Review summary: `REVIEW_SUMMARY.md`
- Mathematical contract: `../docs/MATHEMATICAL_CONTRACT.md`
- Experiment plan: `EXPERIMENT_PLAN.md`
- Experiment tracker: `EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- Dominant contribution: leakage-safe distribution-valued ECG measurement
  layer with independently falsifiable branches.
- Supporting contribution: practical equivalence and dependence-preserving
  inference without non-rejection graphs.
- Rejected complexity: a monolithic model, wave pseudo-labels, clinical
  metadata, test-triggered changes, and unnecessary large pretrained models.

## First Runs to Launch

1. Shared metadata/label/patient-disjointness tests.
2. Preprocessing, R-detection, phase, and QC integration smoke on folds 1--7.
3. Exact-MMD/Nyström fidelity witness, followed by raw/KME GPU throughput smoke.

## Main Risks

- Beat eligibility and class-dependent exclusion.
- Nyström fidelity for branch-specific descriptor spaces.
- Paper 7/8 memory and uncertainty-computation cost.

## Next Action

Implement the shared harness and run M0/M1. Fold 10 is forbidden until the
final freeze manifest exists.

