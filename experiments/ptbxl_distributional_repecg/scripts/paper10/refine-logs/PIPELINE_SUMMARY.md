# Pipeline Summary: Paper 10

**Problem:** distinguish acquisition stability, transform robustness, and shortcut resistance without confounding hospital membership with an intervention.

**Final Method Thesis:** Same-record raw-waveform matching can test acquisition-state stability more directly than unpaired alignment, but shortcut resistance requires an explicit spurious train-only environment correlation.

**Final Verdict:** RETHINK

**Date:** 2026-09-17

## Contribution Snapshot

- Dominant contribution: paired acquisition-intervention stability.
- Supporting contribution: admissibility, no-collapse, norm-leakage, pairing-falsification, and impossibility gates.
- Explicitly rejected complexity: hospital causality, generic mutual-information claims, lead subsets, and unimplemented CausIRL.

## First Runs to Launch

1. Build G0/G1 provenance, admissibility, and preprocessing-survival artifact audits.
2. Run four frozen synthetic worlds, including the broken train-only shortcut world.
3. Implement objective-gradient tests and only then a paired real-data pilot.

## Main Risk

The intervention bank may be too weak, destructive, or erased by preprocessing; paired stability can also hide environment in latent norms. Each is a falsifiable stop condition, not a reason to expand the claim.
