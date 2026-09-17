# Experiment Plan: Paper 10

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **erm**: Evaluates the specific component or serves as a baseline/sham control.
- **irm**: Evaluates the specific component or serves as a baseline/sham control.
- **coral**: Evaluates the specific component or serves as a baseline/sham control.
- **causirl**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- IB reduces overfitting on spurious lead placements.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
