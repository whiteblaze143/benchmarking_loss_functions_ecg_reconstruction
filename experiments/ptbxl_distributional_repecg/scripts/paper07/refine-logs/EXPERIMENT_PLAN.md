# Experiment Plan: Paper 07

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **continuous_primary**: Evaluates the specific component or serves as a baseline/sham control.
- **categorical_primary**: Evaluates the specific component or serves as a baseline/sham control.
- **continuous_auxiliary**: Evaluates the specific component or serves as a baseline/sham control.
- **categorical_auxiliary**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Operator responses are more sample-efficient than raw mappings.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
