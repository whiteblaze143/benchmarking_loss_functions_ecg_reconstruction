# Paper 11 Continuous G1 Protocol

**Status:** frozen before confirmatory execution.

## Thesis

Paper 11 tests whether a low-dimensional, causal-prefix continuous code retains future-predictive ECG-beat information beyond copy, one-beat, unordered-history, and wrong-pairing controls. It does not claim categorical state recovery, epsilon-machine recovery, biological causality, or a unique intrinsic dimension.

## Fixed synthetic world

Use the continuous four-dimensional linear state-space world in `make_linear_state_space_sequences`: stable rotational dynamics, noisy four-dimensional predictive observations, and eight independent nuisance dimensions. The model sees four ordered prior observations and predicts the next full observation. Evaluation is held-out by generated sequence.

## Fixed systems

- Continuous causal GRU bottlenecks: \(d\in\{1,2,4,8,16,32\}\); \(d=32\) is the weakly constrained predictive ceiling.
- `beat_copy`, `current_only`, `set_predictor`, `shuffled_prefix`, `wrong_future`, and IID controls.
- Frozen-code probes: a linear decoder from \(Z_4\), versus the same probe family from \((Z_4,X_{\leq t})\).

## Per-seed gates

1. \(L_1>1.10L_4\) and \(L_2>1.05L_4\).
2. \(\eta_4=(L_{copy}-L_4)/(L_{copy}-L_{32})\geq0.90\).
3. \(L_4\) is lower than copy, current-only, set, and shuffled-prefix MSE.
4. \(L_8\geq0.95L_4\); this permits small overcomplete improvements but rejects a materially larger required code.
5. \((L[Z_4]-L[Z_4,X_{\leq t}])/(L_{copy}-L_4)\leq0.05\).
6. \(L_4<0.90L_{wrong}\).
7. On IID, \(L_{IID}\geq0.95L_{IID,mean}\).

## Decision

All gates must pass for seeds 42, 43, and 44. A pass establishes only synthetic continuous predictive compression in this frozen generator and probe family. Real-ECG G0 remains blocked until then.
