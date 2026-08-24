# Poster Figure Captions

## Figure 1 — Complete factorial loss landscape

Complete repaired \(2^4\) factorial benchmark on 2,198 held-out PTB-XL ECGs.
Columns encode MSE/correlation/MMD/derivative inclusion. No loss mask dominates
all architectures or endpoints, motivating architecture-specific objective
design.

## Figure 2 — Component marginal effects

Average matched on/off effects for all four components with 95%
patient-cluster BCa intervals across 1,904 patients (2,198 ECGs), conditional
on the seed-42 trained models. Interpret the MSE effect together with
architecture-specific correlation, MMD, and derivative effects.

## Figure 3 — Morphology and diagnostic utility decouple

Full objective minus MSE-only base. QRS and ST correlations improve in all
three architectures at the prespecified within-family threshold
\(\alpha=0.0167\), while ECGFounder AUROC changes are nonsignificant. Improved
morphology does not automatically imply improved frozen-classifier utility.

## Figure 4 — Robustness and external generalization

Validation-selected models under 0 dB NSTDB noise and EchoNext domain shift.
The latent models lead on clean PTB-XL, whereas U-Net degrades less under
severe noise. High EchoNext Pearson with negative R² exposes amplitude
miscalibration that correlation alone would hide.

## Figure 5 — Smartwatch zero-shot domain gap

Lead-II-only reconstruction across four ECG-capable smartwatches. All selected
family-device cells have negative missing-eleven-lead R². ECGFounder agreement
is a Philips-referenced probability-fidelity proxy on calibrated simulator
signals, not disease classification accuracy.
