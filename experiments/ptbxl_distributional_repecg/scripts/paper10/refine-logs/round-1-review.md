# Round 1 Review — Implementation and Assumption Audit

**Review status:** external reviewer backend unavailable in this workspace; this is an evidence-based implementation audit, not an independent external verdict.

## Evidence

- `train_paper10_shared_grid.py` loads only representations, labels, ECG IDs, and patient IDs; it contains no environment tensor and its optimization loss is BCE in every variant.
- `InterventionalRepStatModel` exposes `Z_S`, `Z_A`, and a nine-class acquisition head, but the trainer never calls `factorize` or the acquisition head.
- The claimed mutual-information restriction does not exist in code.
- The existing `evaluate_paper10_ood.py` is retired as unsafe, so no existing OOD result can support the proposal.

## Verdict: RETHINK

The old proposal is not an implementable scientific method. It conflates three distinct claims: site invariance, acquisition robustness, and causal feature recovery. Only controlled same-record acquisition transforms identify the intended intervention while holding patient physiology fixed.

## Required Revision

1. Replace hospital/site environments with a versioned, raw-waveform paired intervention bank.
2. Replace the mutual-information claim with the explicitly optimized paired-stability objective.
3. Make `full`, ERM, IRMv1, and CORAL objective-distinct; remove `causirl` until it has a defined, tested loss.
4. Gate all runs on exact pair membership, patient-disjoint folds, transform/label preservation, and no-collapse diagnostics.
5. Do not call any result clinical OOD generalization; the first claim is robustness to the enumerated transforms.
