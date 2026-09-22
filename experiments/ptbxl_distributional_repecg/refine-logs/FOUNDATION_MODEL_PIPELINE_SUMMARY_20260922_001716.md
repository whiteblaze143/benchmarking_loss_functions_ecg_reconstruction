# Foundation-Model Comparison Pipeline Summary

**Problem**: Determine whether an externally pretrained ECG encoder alters the conclusion about measurement-operator configuration shift.

**Final verdict**: READY FOR ADMISSION ONLY — no foundation-model result has been claimed or launched.

## Contribution snapshot

- **Dominant comparison**: ECGFounder-1L versus GraphECG and SetOperator on the same PTB-XL Fold-8 Lead-I endpoint.
- **Supporting context**: ECGFounder-12L as a full-12 ceiling, explicitly outside the Q8/missing-lead comparison.
- **Rejected complexity**: No HuBERT substitute, no zero-filled 12-lead ECGFounder subsets, no ICM proxy, and no use of the released 150-task head as the five-label endpoint.

## First runs after the P07 repair gate

1. Verify/archive-extract/strict-load the two real ECGFounder checkpoints.
2. Establish preprocessing, split, and pretraining-membership provenance.
3. Run the frozen ECGFounder-1L five-label Lead-I protocol and paired inference.

## Main risk

PTB-XL pretraining membership may be unknown. If it cannot be proven absent, results remain useful as a labelled foundation-model comparator but cannot support a strict no-leakage superiority statement.
