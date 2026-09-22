# Experiment Plan: P07 Full-12-Lead Auxiliary + IMQ MMD²

## Claim Map

| Claim | Minimum evidence | Block |
|---|---|---|
| The requested objective executes on real data | Exact target provenance plus gradient and finite-loss checks | P0 |
| Full-12 supervision changes frozen flexible-lead representations | Identical PTB-XL selection and task-native comparisons | P1 |
| Any alignment effect is not merely the decoder | Full-12 reconstruction deletion of the MMD term | P2 |

## Blocks

### P0: Admission and objective gate

- Build an immutable 12-lead target artifact from folds 1–7/8 only and a separate full-12 context artifact using the existing P07 `response_fit.npz` without refitting whitening or Nyström landmarks.
- Verify exact ECG/patient/label alignment, all finite samples, 5,000 samples at 500 Hz, and train-only 12-lead scaler.
- Verify nonzero finite gradients for BCE, IMQ MMD², and held-out-lead-only waveform MSE.
- Randomly condition on 1 through 11 of the canonical twelve leads; retain lead subset membership for every reconstruction summary. Derived limb leads are reported separately from independent/precordial subsets.

### P1: Registered checkpoint

- Train one `continuous_full12_aux_mmd` encoder with seed 42.
- Select only fold-8 macro-AUROC, not reconstruction loss.
- Save encoder state, decoder state, loss terms, target hashes, and epoch history atomically.

### P2: Isolation

- After P1, train `continuous_full12_aux_no_mmd` with every setting identical except `lambda_mmd=0`.
- Run both only after P0 passes; this is required for a specific MMD claim but is not a substitute for P1.

### P3: Downstream comparison

- Add the completed encoder to the same frozen task-native configurations and prediction-artifact contract.
- Use paired DeLong only on aligned binary outputs; use the 2,000-replicate bootstrap contract after all cohorts finish.

## Run Order

| Milestone | Gate | Status |
|---|---|---|
| P0 | Target and gradient contract | TODO |
| P1 | Seed-42 full12+MMD checkpoint | QUEUED_AFTER_BRAID |
| P2 | MMD deletion checkpoint | PENDING P1 |
| P3 | Benchmark-2 evaluation and inference | PENDING P1/P2 |
| P4 | Sunnybrook personalization eligibility | COMPLETE_NOT_ELIGIBLE | Twenty XML records have twenty distinct retained MRNs, so no within-patient future-record holdout exists. |
