# Experiment Plan: Paper 09

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **diagnosis_only**: Evaluates the specific component or serves as a baseline/sham control.
- **mismatched_q**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Counterfactual constraints reduce spurious correlations.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
