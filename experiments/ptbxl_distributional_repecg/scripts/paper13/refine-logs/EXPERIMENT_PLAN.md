# Experiment Plan: Paper 13

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **no_propagation**: Evaluates the specific component or serves as a baseline/sham control.
- **wrong_phase_surgery**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Equivariant layers improve robustness to lead displacement.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
