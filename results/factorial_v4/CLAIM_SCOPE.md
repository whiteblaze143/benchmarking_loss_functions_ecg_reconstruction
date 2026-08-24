# Factorial v4 Claim Scope

- The primary loss analysis is the prespecified MSE-on \(2^3\)
  correlation/MMD/derivative grid for U-Net, MultiScale-VAE, and ECG-AIM.
- MultiScale-VAE's exploratory MSE toggle is invalid and quarantined because
  its historical decoder-loss output was detached. No MultiScale-VAE MSE main
  effect or MSE interaction may be claimed. U-Net and ECG-AIM MSE toggles have
  verified live gradients and remain exploratory.
- BCa intervals resample 1,904 PTB-XL patients while retaining all 2,198 ECGs
  within sampled patient clusters. They are conditional on each seed-42
  trained model. The two extra seeds cover only base, full, and
  validation-selected configurations.
- ECGFounder macro AUROC uses the 31 of 150 fold-10 tasks containing both
  classes; all 150 task outputs and the 119 single-class statuses remain
  reported. Non-inferiority is a per-cell, pointwise comparison against a
  prespecified engineering margin of 0.02, not a clinically validated
  equivalence threshold.
- The 0.0167 correction controls the three prespecified endpoints within each
  architecture, not all nine architecture-endpoint tests. Factorial effect
  intervals are descriptive and unadjusted.
- Sex and age AUROC gaps compare identical common task sets. PTB-XL sex
  encoding is male=0 and female=1.
- EchoNext SHD endpoints use real echocardiography-derived labels, but the
  frozen classifier consumes reconstructed waveform plus unchanged original
  tabular metadata. Stress and morphology use a deterministic 2,198-record
  subset of the 5,442-record clean test set.
- Smartwatch scalar heart-rate, R-wave, and ST endpoints use calibrated
  simulator targets. The 2-Hz square-wave endpoint is watch-versus-Philips
  maximum aligned cross-correlation under a simulated stimulus, not an
  independent simulator-target accuracy metric. ECGFounder smartwatch outputs
  are probability-fidelity proxies, not disease-ground-truth classification.
