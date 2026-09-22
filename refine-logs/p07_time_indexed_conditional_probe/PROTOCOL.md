# P07 Time-Indexed Conditional Reconstruction Probe

## Immutable problem anchor

Can the existing operator-aware, distributional SetOperator support patient-specific conditional electrical inference without training a high-capacity waveform generator?

`continuous_full12_aux_mmd` remains an immutable negative multitask ablation. This protocol neither overwrites it nor treats its reconstruction loss as a selection metric.

## Registered blocks

1. **A: Temporal-information gate.** Fit a new time-indexed RKHS response map only on PTB-XL folds 1-7, with input `(tau, voltage, derivative)`. Confirm on a synthetic within-cell temporal permutation that the new map changes while the original time-agnostic KME does not. Train/evaluate a matched SetOperator on fold 8 diagnosis; this is a representation gate, not a superiority claim.
2. **B: Frozen PTB-XL conditional probe.** Freeze the strongest available robustness-trained encoder. For Lead-I and Lead-II contexts separately, fit fold-1-7 linear ridge and finite-rank IMQ conditional mean readouts to eight independent 256-sample waveforms. Tune only readout hyperparameters on fold 8, retain dependent limb leads through exact lead algebra, and report waveform error plus patient-specific variance ratio and morphology calibration. No reconstruction gradient reaches an encoder.
3. **C: VitalDB increment test.** Use the public synchronized `SNUADC/ECG_II`, `SNUADC/ECG_V5`, and `SNUADC/PLETH` 500-Hz streams. Split by VitalDB subject before fitting any preprocessing/readout. Compare `II -> V5`, `II + real PPG -> V5`, `II + wrong-subject PPG -> V5`, and `PPG -> V5` using matched ridge/KRR readouts. A benefit of real PPG is required before PPG can enter the native representation.

## Non-negotiable controls

- No fold 9/10 or external records tune PTB-XL encoders/readouts.
- No dependent limb lead is predicted independently.
- Do not interpret retention above 100 percent as a robustness benefit when its full-context ceiling is weak.
- Report patient/subject-equal aggregates and all denominators.
- MIMIC-III-Ext-PPG is credentialed: acquisition may proceed only after PhysioNet confirms the account has accepted the required DUA and training; credentials are never stored in repository artifacts.
