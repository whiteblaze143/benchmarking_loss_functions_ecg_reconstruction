# Pipeline Summary: Paper 10

**Problem:** acquisition-shortcut robustness under known ECG interventions.

**Final Method Thesis:** Use same-record, raw-waveform acquisition pairs to regularize diagnostic state stability while preserving transform information in a separate auxiliary state.

**Final Verdict:** RETHINK

**Date:** 2026-09-17

## Contribution Snapshot

- Dominant contribution: paired acquisition-intervention stability.
- Supporting contribution: an executable no-collapse/pairing-falsification suite.
- Explicitly rejected complexity: hospital causality, generic mutual-information claims, lead subsets, and unimplemented CausIRL.

## First Runs to Launch

1. Build and audit the versioned paired transform artifact.
2. Run the synthetic identifiability and wrong-pair tests.
3. Implement and test distinct objective paths before a real grid.

## Main Risk

The intervention bank may be too weak, too destructive, or already erased by the frozen representation. That is a falsifiable stop condition, not a reason to expand the claim.
