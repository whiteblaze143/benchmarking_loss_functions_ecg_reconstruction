# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Test whether a flexible-lead SetOperator learns a more clinically transferable representation when its auxiliary task predicts the complete 12-lead ECG, rather than one 16-phase queried response.
- **Must-solve bottleneck:** The existing auxiliary checkpoint has no complete 12-lead waveform target.
- **Non-goals:** No changed clinical splits, no test data, no synthetic ECG targets, and no replacement of existing checkpoints.

## Anchor Check

The revised method still isolates the requested auxiliary-objective change. Using full and sparse views of the same real record serves the flexible-lead bottleneck; introducing an external domain-MMD objective would drift into a different domain-generalization claim.

## Simplicity Check

The encoder is reused unchanged. Only a training-only waveform decoder and non-affine normalization before IMQ MMD² are added. Diffusion, adversarial, multi-stage, and target-cohort modules are explicitly excluded.

## Revised Proposal

The final method is the contract in `FINAL_PROPOSAL.md`: BCE plus unit-weight IMQ MMD² between normalized full and sparse-view encoder latents, plus a fixed 0.1 real full-12-lead waveform MSE objective. Selection remains fold-8 macro-AUROC; reconstruction metrics are descriptive and cannot select a checkpoint.
