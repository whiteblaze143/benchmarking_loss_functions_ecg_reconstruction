# Experiment Plan: Paper 14

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **mmd_penalty_only**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- MMD penalty enforces feature alignment across domains.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
