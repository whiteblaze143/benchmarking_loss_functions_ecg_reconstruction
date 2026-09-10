# Pipeline Summary

**Problem:** Learn a broadly useful embedding while retaining strict Lead-I-to-12-lead ECG-AIM reconstruction.  
**Final Method Thesis:** Add an unchanged LVCG beat/RR representation branch to frozen ECG-AIM through a learned single-lead latent field.  
**Final Verdict:** READY FOR PILOT  
**Date:** 2026-09-10

## Contribution Snapshot

- Dominant contribution: controlled structure/dynamics/rhythm decomposition for strict one-lead ECG-AIM.
- Supporting contribution: causal test of whether beat structure repairs morphology-poor generic pooling.
- Explicitly rejected: physical one-lead VCG inversion, decoder replacement, TTT, and joint training in the pilot.

## Must-Prove Claims

1. Morphology improves over dimension-matched current pooling without losing rhythm utility.
2. Component deletions demonstrate distinct structure and rhythm contributions.

## First Runs

1. Verify frozen reconstruction equivalence on the selected checkpoint.
2. Build the dimension-matched current-pooling baseline.
3. Run the one-seed proposed, triplication, and shuffled-boundary pilot.

## Main Risk

The remaining risk is experimental rather than structural: the learned field may add capacity without improving morphology. Triplication and shuffled-boundary controls isolate that possibility.

## Next Action

Run frozen reconstruction equivalence, then the representation pilot.
