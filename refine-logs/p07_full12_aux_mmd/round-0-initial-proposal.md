# Research Proposal: Full-12-Lead Auxiliary Reconstruction for SetOperator

**Date:** 2026-09-22

## Problem Anchor

- **Bottom-line problem:** Test whether a flexible-lead SetOperator learns a more clinically transferable representation when its auxiliary task predicts the complete 12-lead ECG, rather than one 16-phase queried response.
- **Must-solve bottleneck:** The current `continuous_auxiliary` checkpoint only reconstructs a single phase-feature response and cannot test whether whole-ECG morphology provides useful supervision.
- **Non-goals:** Do not alter existing SetOperator, Braid, or GraphECG checkpoints; do not use fold 9/10, external cohorts, fabricated targets, or synthetic waveforms; do not make a superiority claim from the new checkpoint alone.
- **Constraints:** PTB-XL folds 1–7 train, fold 8 selection; real filtered 500 Hz 10-second 12-lead targets only; one A100 is currently occupied by Braid training.
- **Success condition:** A provenance-complete checkpoint enters the identical frozen task-native evaluation, with its full-lead reconstruction and BCE+IMQ-MMD objective independently logged and testable.

## Technical Gap

`continuous_auxiliary` predicts one response tensor of shape `16 × 128` for a sampled operator. That is a useful operator-query task, but it neither reconstructs the 12 standard lead waveforms nor regularizes the latent representation across full and sparse observed lead sets with the requested IMQ MMD² objective.

## Method Thesis

Train one additional continuous SetOperator with real full-12-lead waveform reconstruction and view-alignment regularization:

\[
\mathcal L = \underbrace{\mathcal L_{\mathrm{BCE}}(\hat y_{\mathrm{sub}},y)
+ \widehat{\mathrm{MMD}}^2_{\mathrm{IMQ}}\!\left(\operatorname{LN}(h_{\mathrm{sub}}),\operatorname{LN}(h_{\mathrm{full}});c^2=1\right)}_{\mathcal L_{\mathrm{primary}}}
+ 0.1\,\mathcal L_{\mathrm{MSE}}(D(h_{\mathrm{sub}}),x_{12}).
\]

Here `sub` is an already-supported random real observed subset with `m ∼ Uniform{1,…,8}`; `full` is the eight independent-lead context of the same records; `LN` is non-affine LayerNorm; and `x12` is the real filtered and training-standardized 12-lead waveform. The IMQ term has unit weight as requested. The 0.1 reconstruction coefficient is fixed before training and prevents the 60,000-sample waveform loss from replacing the diagnostic objective.

## Contribution Focus

- **Dominant contribution:** Replace query-response auxiliary supervision with an end-to-end full-12-lead ECG target while retaining a precise flexible-view IMQ alignment objective.
- **Supporting contribution:** Make target, MMD environment pair, weights, and selection rule auditable.
- **Rejected complexity:** No diffusion decoder, adversarial loss, external pretraining, pseudo-target, or per-cohort fine-tuning.

## Proposed Method

The existing SetOperator encoder remains unchanged. It creates `h_sub` and `h_full` by applying the same continuous operator encoder to real PTB-XL response sets. A separate training-only decoder maps `h_sub` to 12 canonical leads × 5,000 samples. The deployment checkpoint contains the encoder state only; the decoder cannot affect task-native inference except through training gradients.

The target builder reads each admitted PTB-XL waveform, applies the existing band-pass contract, and standardizes all 12 canonical leads with statistics fit only on folds 1–7. It checks exact ECG ID, patient ID, and label alignment with the Paper 07 representation artifact. Fold 8 is selection-only.

## Claim-Driven Validation Sketch

1. **Mechanism execution:** Gradient tests must show nonzero encoder gradients from both IMQ MMD² and full-12 reconstruction. Real-target alignment and train-only scaler provenance must pass before training.
2. **Representation comparison:** Under identical PTB-XL splits and task-native protocol, compare the new encoder to `set_operator_robust` and `set_operator_aux`. Report outcomes without declaring a win unless the frozen cross-cohort results and paired inference support it.
3. **Deletion check:** A full-12 reconstruction variant without the MMD term is required before attributing any effect specifically to view alignment.
