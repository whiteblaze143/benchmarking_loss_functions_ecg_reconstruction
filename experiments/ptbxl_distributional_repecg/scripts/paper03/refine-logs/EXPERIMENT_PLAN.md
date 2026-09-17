# Experiment Plan: Paper 03

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **order_scrambled**: Evaluates the specific component or serves as a baseline/sham control.
- **time_reversed**: Evaluates the specific component or serves as a baseline/sham control.
- **monotone_warp_sham**: Evaluates the specific component or serves as a baseline/sham control.
- **unordered_kme**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Path signatures provide a robust feature space.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
