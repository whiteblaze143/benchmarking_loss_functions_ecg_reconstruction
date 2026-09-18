# Round 1 Refinement: Stricter Predictive-State Gates

## Problem Anchor

Determine whether a compact state computed from prior ECG beats predicts later beats, rather than merely being a diagnostic feature cluster.

## Corrections Accepted

- Fold 1–6 is fit, fold 7 selects all state count and loss settings, and fold 8 is pseudo-test only.
- G0 must retain patient ID, ECG ID, beat index, target index, fold, R-peak sample, and exact prefix indices. Future-token gradients must be exactly zero.
- G1 adds prefix-length, ordered-prefix versus set/shuffle, cardinality \(K=2,4,8\), continuous-predictor efficiency, same-current/different-future, and horizon tests.
- ARI is diagnostic only; future predictive sufficiency is primary.

## Strict Synthetic Result

The pairing, IID, prefix-order, current-only, set-encoder, horizon, and continuous-retention tests passed over three seeds. The cardinality stress test failed: in seed 44, \(K=8\) obtained 0.545 MSE while \(K=4\) obtained 0.757. Therefore the soft-state implementation is not a stable finite four-state compression.

## Decision

**G1 is FAIL.** Do not build or train the real-ECG artifact. The next method decision is whether a genuinely discrete state-assignment mechanism is justified; do not add a state-use regularizer merely to make this synthetic result pass.
