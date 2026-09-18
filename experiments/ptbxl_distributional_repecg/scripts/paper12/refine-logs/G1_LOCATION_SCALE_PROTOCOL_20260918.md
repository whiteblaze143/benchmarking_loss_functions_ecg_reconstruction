# Paper 12 G1 — Conditional Location-Scale Innovation Protocol

**Status:** frozen before confirmatory seeds 45, 46, and 47.

## Corrected thesis

Paper 12 tests whether phase-ordered conditional location-scale residuals isolate unpredictable variation beyond predictable phase state. It does not claim independent causal mechanisms, identifiable structural equations, normalizing-flow inversion, or causal direction recovery.

The model estimates conditional mean and diagonal scale from past phase cells, trains them with Gaussian negative log likelihood, and defines standardized innovations as `(z_g - mu_g) / sigma_g`.

## Frozen worlds and gates

All gates must pass independently for all three fresh seeds at exactly 500 updates:

1. Additive world: forward MSE is at most 20% of unconditional MSE.
2. Additive world: innovation-recovery MSE is at most 0.002.
3. Additive world: absolute standardized-innovation/past correlation is at most 0.02.
4. Heteroscedastic world: mean-only squared residual/past correlation is at least 0.10.
5. Heteroscedastic world: location-scale standardized squared residual/past correlation is at most 0.03.
6. IID world: forward MSE is at least 95% of unconditional MSE.

Time reversal is reported only as a sensitivity. It is not a destroyer or causal-direction gate because reverse prediction was easier in every development seed.

## Consequence

Passing G1 validates the synthetic location-scale innovation mechanism only. The legacy Paper 12 training grid remains invalid until it includes the NLL objective, matched controls, and objective-gradient tests.
