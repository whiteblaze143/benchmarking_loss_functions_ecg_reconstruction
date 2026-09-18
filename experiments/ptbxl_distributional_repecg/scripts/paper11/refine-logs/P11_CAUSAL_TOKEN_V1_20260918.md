# P11-CAUSAL-TOKEN-v1

## Frozen construction

- Detection: fixed Lead II; causal second-order 5–20 Hz Butterworth filter; derivative; square; 120 ms trailing integrator; past-only adaptive threshold; 2 s burn-in; 250 ms refractory interval; backward-only 200 ms anchor search.
- Accepted confirmation latency: 0–250 ms.
- Representation: raw physical-mV 12-lead signal standardized with folds-1–6 global per-lead moments. The detector-filtered signal is never used as a model input.
- Token: fixed `[anchor-200 ms, anchor+300 ms)` support, exactly 250 samples per lead at 500 Hz.
- Availability: `max(confirmation_sample, token_stop)`.
- Target: next causally detected fixed-support token. Its anchor, RR interval, and confirmation time are not model inputs.
- Eligibility: `target_start > max(prefix token_stop)` and `target_start > max(prefix availability)`.
- Interpretation: next causally detected QRS-centred morphology conditional on the future event occurring; beat timing is not predicted.

## Required G0 evidence

Suffix interventions must leave all already-available anchor, confirmation, and token bytes unchanged. Target-region perturbations must leave the entire prefix unchanged. Full-cohort output must preserve raw intervals, detector/config hash, scaler hash, patient, record, fold, exclusion reason, and deterministic token/manifest hashes.

## Development smoke

On the first 20 fold-1 PTB-XL records, all 20 produced at least one eligible four-token prefix: 189 tokens and 109 strictly separated examples. Against full-record XQRS as a QA reference only, within the post-burn-in observable interval, causal-anchor event matching at 75 ms tolerance was precision 1.00, recall 1.00, F1 1.00; median timing disagreement was 8.0 ms and the 95th percentile was 12.4 ms. This is smoke evidence, not full-cohort detector qualification.
