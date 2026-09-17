# Experiment Plan: Paper 06

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **full_signal**: Evaluates the specific component or serves as a baseline/sham control.
- **macro_component**: Evaluates the specific component or serves as a baseline/sham control.
- **residual_marginal**: Evaluates the specific component or serves as a baseline/sham control.
- **macro_residual_marginals**: Evaluates the specific component or serves as a baseline/sham control.
- **shuffled_residual**: Evaluates the specific component or serves as a baseline/sham control.
- **joint_pair_sham**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Residual marginals capture unique diagnostic information.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
