# Final Proposal: Paper 11 — Continuous Predictive Compression of Ordered ECG Beats

## Problem Anchor

Determine whether a compact representation computed only from ordered prior ECG beats retains future-predictive morphology information beyond autocorrelation and unordered-history alternatives. This is not a claim of causal-state recovery, epsilon-machine minimality, biological state, or clinical utility.

## Method Thesis

Given a causal prefix \(X_{t-m+1:t}\), a GRU emits a continuous \(d\)-dimensional code \(Z_t\), and a decoder predicts the held-out next beat. The dominant contribution is the falsifiable compression claim: a low-dimensional code should retain the predictive gain of a weakly constrained predictor while leaving little recoverable gain for a matched decoder given the raw prefix.

The categorical-state path is permanently retained as a negative result: it failed its predeclared finite-cardinality gate. It is not repaired or included in this method.

## Frozen Synthetic Evidence

Continuous G1 v2 passed all gates for fresh seeds 45, 46, and 47 with a four-dimensional rotational state-space world. In every seed, \(d=1,2\) underfit; \(d=4\) retained at least 99.76% of the \(d=32\) gain over beat-copy; it beat copy, current-only, set, shuffled-prefix, and wrong-future controls; its conditional-prefix residual was at most 0.27% of its predictive gain; and IID had no material gain.

This establishes synthetic predictive compression only. Real-ECG work remains at G0.

## Next Gate

Construct a patient/fold-safe ordered-beat artifact with exact `(patient_id, ecg_id, beat_idx, target_idx, fold, R-peak, prefix)` lineage, patient-equal sampling, and zero future-token gradients. Only after G0 may the real-ECG feasibility gate begin.

**Verdict: REVISE — synthetic mechanism qualified; real-data provenance and feasibility untested.**
