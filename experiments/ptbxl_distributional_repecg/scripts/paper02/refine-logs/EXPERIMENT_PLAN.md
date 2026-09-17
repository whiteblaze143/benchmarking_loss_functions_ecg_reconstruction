# Experiment Plan: Paper 02

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_kernel**: Evaluates the specific component or serves as a baseline/sham control.
- **no_circular**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- MMD provides stronger OOD generalization.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
