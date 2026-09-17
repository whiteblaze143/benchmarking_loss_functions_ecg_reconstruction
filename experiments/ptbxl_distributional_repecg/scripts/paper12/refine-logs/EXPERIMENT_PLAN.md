# Experiment Plan: Paper 12

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **unconditional_z**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Disentangled representations improve interpretability and generalizability.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
