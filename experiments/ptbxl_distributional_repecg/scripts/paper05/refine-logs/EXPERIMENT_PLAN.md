# Experiment Plan: Paper 05

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **occupancy_only**: Evaluates the specific component or serves as a baseline/sham control.
- **chronology_shuffled**: Evaluates the specific component or serves as a baseline/sham control.
- **identity_order_sham**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Cross-spectral matrices outperform spatial-only models.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
