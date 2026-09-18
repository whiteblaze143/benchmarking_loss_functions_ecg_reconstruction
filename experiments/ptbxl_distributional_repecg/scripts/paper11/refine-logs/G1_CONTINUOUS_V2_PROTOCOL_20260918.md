# Paper 11 Continuous G1 v2 Protocol

**Status:** frozen before confirmatory execution on fresh seeds 45, 46, and 47.

## Why this is a new version

The original 400-update continuous protocol failed on seed 44. A diagnostic-only 1,000-update run on that seed showed that \(d=4\) reached the unrestricted predictive ceiling and had a negligible prefix residual. The original failure is retained; it is not relabeled. This version changes one implementation-budget parameter and uses new confirmatory seeds.

## Unchanged scientific object

The claim remains synthetic continuous predictive compression: a causal-prefix \(d=4\) code retains future-predictive information in the frozen four-dimensional rotational state-space world. It is not a causal-state, intrinsic-dimension, biological, or real-ECG claim.

## Frozen execution

- Seeds: 45, 46, 47 only.
- Updates per fitted system and probe: 1,000 exactly.
- Data, architectures, dimensions \(d\in\{1,2,4,8,16,32\}\), controls, split construction, and gates are otherwise identical to `G1_CONTINUOUS_PROTOCOL_20260918.md`.
- The script rejects `--frozen-g1` unless the update count is exactly 1,000.

## Per-seed gates

1. \(L_1>1.10L_4\) and \(L_2>1.05L_4\).
2. \(\eta_4\geq0.90\) relative to \(d=32\) and beat-copy.
3. \(L_4\) beats beat-copy, current-only, set, and shuffled-prefix controls.
4. \(L_8\geq0.95L_4\).
5. The frozen-code prefix-residual fraction is at most 0.05.
6. \(L_4<0.90L_{wrong}\).
7. IID receives no material predictive gain: \(L_{IID}\geq0.95L_{IID,mean}\).

All gates must pass for all three fresh seeds. Only then may Paper 11 proceed to the real-ECG G0 lineage audit.
