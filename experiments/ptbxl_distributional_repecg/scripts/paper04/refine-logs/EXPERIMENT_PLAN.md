# Experiment Plan: Paper 04

## Objective
Validate the core inductive bias by testing specific architectural variants.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **linear_probe**: Evaluates the specific component or serves as a baseline/sham control.
- **time_shuffled**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Topological features from phase space classify arrhythmias effectively.

## Validation Strategy
Compare the `full` or `primary` variant against the baselines (`linear_probe`, `sham`, etc.) to isolate the contribution of the inductive bias.
