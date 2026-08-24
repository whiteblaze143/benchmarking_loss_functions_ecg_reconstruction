# LUDB Oracle-Label Fidelity Evaluation Plan

**Problem:** Select ECG-AIM loss configurations by fidelity to cardiologist-annotated LUDB morphology without making an automatic delineator the reference.

**Primary claim:** Loss configurations differ in their preservation of voltage and shape at fixed LUDB fiducials on the genuinely missing precordial leads V1/V3–V6.

**Anti-claim:** A model is not a fidelity winner merely because NeuroKit can re-detect its waves.

## Evidence contract

- Dataset: the complete local PhysioNet LUDB 1.0.1 release, 200 records, 12 leads, 5,000 samples, 500 Hz, mV.
- Clock: LUDB per-lead cardiologist annotations. No landmark is re-found on either original or reconstruction.
- Lead roles: observed I/II/V2; algebraically derivable III/aVR/aVL/aVF; primary missing precordials V1/V3–V6.
- Landmark metrics: reconstructed minus original voltage at every mapped P/QRS/T onset, peak, and offset.
- Wave metrics: fixed-window correlation/RMSE/MAE/max error, baseline-corrected peak amplitude, signed area, and absolute area for complete P/QRS/T intervals.
- ST metrics: exact QRS-offset J voltage; J+20/40/60/80 ms when inside the labeled ST interval; ST mean, area, slope, correlation, and RMSE through the subsequent labeled T onset.
- Signal controls: per-record/per-lead Pearson, MSE, MAE, and derivative MSE.
- Primary scope: V1/V3–V6. Observed and derivable leads are controls and cannot improve the primary claim.
- Selection: report a Pareto set. Do not collapse unlike endpoints into a post-hoc weighted score.
- NeuroKit: catastrophe-only secondary evidence; never ground truth and never the primary ranking.
- Dice/timing: not reported in the oracle track because no independent predicted mask or predicted boundary is emitted. Applying the same LUDB mask to both signals would yield a meaningless Dice of 1.

## Execution gates

1. Reproduce the dataset inventory and retain orphan/invalid annotation counts.
2. Unit-test exact-index sampling and invariance of baseline-corrected area to DC offsets.
3. Verify a one-record smoke run against hand-computed values.
4. Verify a full 200-record model, database integrity, checkpoint/data/protocol hashes, row counts, and storage growth.
5. Launch CPU-only in detached tmux with a 5 GiB free-space reserve and resumable per-checkpoint commits.
6. Use paired record-level uncertainty and multi-seed replication before a final loss claim.

