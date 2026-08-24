# Round 0 Initial Proposal: Endpoint-Aware Factorial Benchmarking of ECG Reconstruction Losses

## Frozen Problem Anchor

- **Bottom-line problem:** Determine which loss components preserve clinically relevant missing-lead ECG information when reconstructing 12 leads from reduced observations.
- **Must-solve bottleneck:** Pointwise error can reward conditional-mean smoothing, while global shape metrics can improve without preserving localized morphology, calibration, or downstream diagnostic information.
- **Non-goals:** Claiming reconstructed ECG replaces measured ECG or echocardiography; declaring one universal loss winner; treating frozen classifiers or machine statements as clinical adjudication.
- **Constraints:** Use the existing fixed MCMA three-lead-to-twelve-lead architecture for the expanded grid; compare only checkpoints sharing the same content-pinned source, batch-size, state-schema, preprocessing, and patient-split contract; preserve patient pairing; acknowledge that training is incomplete.
- **Success condition:** A complete, provenance-locked mixed-level factorial analysis identifies endpoint-specific main effects and interactions without unacceptable diagnostic, calibration, subgroup, robustness, or transfer degradation.

## Method Thesis

The contribution is a generation-bound, endpoint-aware factorial benchmark—not a predetermined composite-loss victory. The seven-character mask represents fixed MSE, five binary factors, and one five-level categorical MMD-kernel factor, yielding $2^5\times5=160$ conditions per seed and 480 model identities across seeds 42, 200, and 201.

## Integrity Contract

Every admissible model is keyed by full model ID and seed and bound to checkpoint SHA-256, approved source-bundle SHA-256, state schema, batch size 1024, preprocessing, and train/validation/test content roots. Historical checkpoints trained with batch size 256 or another source bundle are preserved for forensics but excluded from the current factorial estimand.

Every evaluation result must additionally record its evaluation-code digest and the exact checkpoint digest. Mask-only CSV rows are insufficient because a mask can recur after retraining. Aggregation occurs only after the complete eligibility gate passes.

## Evaluation Ladder

1. Verify data identity, patient separation, units, sample rate, lead order, and copied/derived-lead exclusions.
2. Report paired missing-lead waveform error and correlation with patient-cluster uncertainty.
3. Test localized QRS, ST, J-point, P/T morphology, intervals, amplitude slopes, and delineation failures.
4. Test cross-lead physiologic consistency and regression-to-the-mean diagnostics.
5. Compare measured versus reconstructed inputs with frozen ECGFounder and EchoNext models, including probability drift, calibration, reclassification, and task support.
6. Evaluate LUDB delineation, ISP interval integrity once provenance clears, Sunnybrook transfer, device/simulator transfer, noise, SQI, and subgroup risk–coverage.
7. Estimate binary main effects, categorical MMD-kernel contrasts, prespecified interactions, and seed variability; report a Pareto set if endpoints disagree.

## Current Verdict

**REVISE / IN PROGRESS.** The anchor and design are stable, but current-generation training and evaluation are incomplete. No current-grid ranking or clinical claim is admissible yet.
