# Experiment Plan: Paper 12

## Objective

Test whether a past-conditioned location-scale model yields standardized phase innovations that remove predictable first- and second-order dependence without erasing clinically useful variation.

This is a predictive-innovation claim, not causal mechanism identification.

## Frozen order of work

1. G1 synthetic identifiability: additive, IID and heteroscedastic worlds.
2. G2 objective execution: Gaussian NLL trains only the density path; freeze that path before fitting diagnosis probes.
3. G3 real-data admissibility: establish exact phase-cell lineage, coverage and patient-equal residual diagnostics on development folds.
4. G4 clinical utility: compare frozen standardized innovations with raw state, mean residuals, an unconditional model and wrong-record histories.

## Required controls

- `state`: raw phase features; establishes information available before residualization.
- `mean_residual`: conditional mean subtraction without scale standardization.
- `innovation`: frozen conditional location-scale standardized residual.
- `unconditional`: no phase-history conditioning.
- `wrong_history`: history from a different patient/record under a deterministic matching rule.

Time reversal is descriptive sensitivity only. It is not a causal-direction destroyer because it was easier than forward prediction in all synthetic development and confirmatory runs.

## Claim boundary

G1 and G2 passing permits G3. It does not validate real-ECG admissibility or clinical utility, and the legacy BCE-only grid must not be launched.
