# Experiment Plan: Paper 11 — Continuous Predictive Compression

| Gate | Claim tested | Required evidence |
|---|---|---|
| G0 | target provenance | exact beat, target index, patient, fold, R-peak, prefix hashes, and zero future-token gradients |
| G1 | synthetic continuous compression | dimensionality curve, retained gain, conditional-prefix residual |
| G1b | false claims rejected | wrong-future, IID, current, set, and shuffled-prefix controls |
| G2 | objective executes | future loss has nonzero intended gradients, no decoder bypass, and causal-prefix gradients |
| G3 | real-data feasibility | ordered beat survival and patient-equal future-target coverage |
| G4 | clinical utility | only after G0–G3; folds 1–6 fit, fold 7 selects, fold 8 pseudo-tests |

The main comparisons are continuous bottlenecks \(d\in\{1,2,4,8,16,32\}\), beat-copy, current-only, `wrong_future`, `shuffled_prefix`, and a parameter-matched set encoder. Prefix length, horizon, loss weight, and final dimension are selected only on fold 7. Fold 8 remains pseudo-test. The current Paper 11 grid is retired for this question.

**Current status:** categorical G1 remains failed. Continuous G1 v2 passed its complete frozen fresh-seed suite; G0 real-ECG lineage is now the next required gate, not clinical training.
