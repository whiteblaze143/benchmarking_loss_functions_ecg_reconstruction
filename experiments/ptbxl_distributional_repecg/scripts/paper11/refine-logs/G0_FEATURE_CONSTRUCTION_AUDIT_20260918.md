# Paper 11 G0 Feature-Construction Causality Audit

**Decision:** the standard PTB-XL Phase-KME cache is ineligible for Paper 11's online causal-prefix claim.

## Evidence

1. `bandpass_ecg` uses `scipy.signal.sosfiltfilt` on an entire record. A numerical perturbation of only samples after a cutoff changed filtered samples before that cutoff by a maximum absolute value of 5.57. The standard filter is therefore acausal.
2. `phase_normalize` creates each token from a valid R-R interval `[R_i, R_{i+1})`. The token at `R_i` consequently uses the next R peak and raw samples up to that future boundary.
3. `detect_rpeaks` is called on the entire filtered record. Its standard pipeline does not establish a causal detection/provenance contract.

No model-level mask can repair either feature-level leak.

## Required replacement contract

Paper 11 must use a dedicated token builder with all of the following:

- causal or independently windowed filtering; never full-record `sosfiltfilt`;
- fixed raw sample windows around a causally confirmed R peak, never R-to-next-R phase cells;
- explicit raw intervals for every prefix and target token, with `prefix_raw_max < target_raw_min`;
- detector-confirmation sample and its raw dependency interval recorded for each token;
- a train-only scaler fit on folds 1–6; no within-record future statistics;
- hierarchical patient → record → target sampling and patient-equal evaluation.

## Status

`P11-CONT-G0` is not yet executable. The next implementation task is to specify and test the causal detector plus fixed-window token contract. No real-ECG training or standard phase-cache reuse is authorized.
