# Factorial v4 Results Interpretation

Status: empirical results locked; publication-level claim verdict pending an
independent postrun audit.

## Study population and valid design

- Primary inference uses the prespecified MSE-on \(2^3\) grid: 24 seed-42
  cells across U-Net, MultiScale-VAE, and ECG-AIM.
- Confirmation training adds seeds 1337 and 2026 for base, full, and the
  validation-selected mask in each family (18 runs).
- PTB-XL clean and each of 17 stress conditions contain 2,198 ECGs. Paired BCa
  inference clusters those ECGs into 1,904 patients.
- EchoNext contains 5,442 clean ECGs; the deterministic first 2,198 are used
  for morphology and stress evaluation.
- The exploratory MultiScale-VAE MSE-off cells are invalid because the
  reconstruction term was detached in that branch. They are quarantined and
  excluded from four-factor effects. The primary MSE-on loss benchmark is not
  affected.

## Central finding: architecture matters more than adding every loss

On clean PTB-XL, the strongest cell was ECG-AIM with correlation plus
derivative (R² 0.8214), closely followed by the validation-selected ECG-AIM
correlation-plus-MMD cell (R² 0.8196) and the selected MultiScale-VAE
correlation-only cell (R² 0.8155). U-Net's best R² was its base objective
(0.6997); its validation-selected correlation-plus-MMD cell reached 0.6624.

The clean-data ranking therefore does not support a universal “more loss terms
is better” story. It supports an architecture-dependent regularization story.

## Loss-component effects

Patient-cluster BCa marginal effects on missing-lead R²:

| Family | Correlation | MMD | Derivative |
|---|---:|---:|---:|
| U-Net | -0.0282 [-0.0371, -0.0210] | -0.00850 [-0.00926, -0.00776] | -0.00041 [-0.00116, 0.00043] |
| MultiScale-VAE | +0.0141 [0.0127, 0.0157] | -0.00380 [-0.00442, -0.00337] | -0.00004 [-0.00014, 0.00005] |
| ECG-AIM | +0.0132 [0.0117, 0.0151] | +0.00065 [-0.00009, 0.00145] | -0.00172 [-0.00325, -0.00103] |

Correlation is the only component with a clear positive reconstruction effect
in both latent-variable families. The same term reduces U-Net R² while
increasing U-Net Pearson/morphology, revealing a scale-fidelity versus
shape-fidelity trade-off. MMD is neutral to harmful at the tested weight.
Derivative loss is largely neutral for U-Net/MultiScale-VAE and mildly harmful
to ECG-AIM R². Consistently, validation selection omitted derivative loss in
all three families.

## Morphology and diagnostic utility are not the same endpoint

The full objective improved QRS and ST correlations over base in every family
at the prespecified within-family alpha of 0.0167:

- U-Net: QRS +0.0113; ST +0.0209.
- MultiScale-VAE: QRS +0.00452; ST +0.00925.
- ECG-AIM: QRS +0.00823; ST +0.01353.

None of the three full-versus-base ECGFounder AUROC differences was
significant at that threshold. Component-level ECGFounder AUROC effects were
also small. Thus morphology preservation can improve without a measurable
gain in frozen-classifier discrimination.

ECGFounder non-inferiority at the engineering margin of 0.02 passed for all
16 ECG-AIM and all 16 MultiScale-VAE cells, but failed for all 16 U-Net cells.
This inference is based on the 31 of 150 ECGFounder tasks that have both
classes in PTB-XL fold 10; it is not a 150-task clinical validation.

## Seed confirmation

The two extra validation seeds support the architecture-specific pattern:

- MultiScale-VAE correlation-only R²: 0.8017 and 0.8044, versus base 0.7881
  and 0.7881.
- ECG-AIM correlation-plus-MMD R²: 0.8052 and 0.7996, versus base 0.7618 and
  0.7900.
- U-Net correlation-plus-MMD R²: 0.5953 and 0.5967, versus base 0.6599 and
  0.6661.

These runs quantify training variability for three configurations per family;
they do not provide seed-level inference for all eight masks.

## Robustness changes the architecture ranking

For the validation-selected models, clean missing-lead MSE was 0.02139
(U-Net), 0.01792 (MultiScale-VAE), and 0.01805 (ECG-AIM). Under 0 dB NSTDB:

- U-Net: 0.0337–0.0349 across BW/EM/MA.
- MultiScale-VAE: 0.0335–0.0382.
- ECG-AIM: 0.0385–0.0412.

The latent models lead on clean PTB-XL, but U-Net degrades less at the most
severe noise level. Poster figures should show both absolute performance and
degradation, because either view alone reverses part of the story.

## External generalization

On EchoNext, the selected models achieved:

| Family | Missing-lead R² | Pearson | SHD macro AUROC |
|---|---:|---:|---:|
| U-Net | -35.243 | 0.878 | 0.724 |
| MultiScale-VAE | -0.966 | 0.868 | 0.767 |
| ECG-AIM | 0.593 | 0.909 | 0.775 |

U-Net and MultiScale-VAE retain waveform shape correlation while failing
amplitude calibration, especially U-Net. ECG-AIM is the only selected model
with positive external R² and also has the strongest SHD result. EchoNext's
classifier consumes the reconstructed waveform together with unchanged
original tabular metadata, so SHD retention cannot be attributed solely to the
waveform reconstruction.

## Smartwatch transfer is a failure analysis, not clinical validation

All selected models have negative wearable missing-eleven-lead R² on all four
devices. U-Net has the strongest waveform correlations (approximately
0.60–0.65), while MultiScale-VAE and ECG-AIM are generally around 0.34–0.44.
ECGFounder threshold agreement remains high (approximately 0.95–0.98), but
this is agreement with the Philips-derived classifier output on simulator
signals. The dataset has no human disease labels; these values are
probability-fidelity proxies, not diagnostic accuracy.

The calibrated heart-rate, R-wave, and ST analyses compare watch measurements
with METRON simulator targets. The 2-Hz square-wave endpoint is instead a
watch-versus-Philips maximum aligned cross-correlation and must remain labeled
as such.

## Poster narrative supported by the empirical tables

1. Loss components are architecture-dependent; correlation helps the two
   latent families but trades U-Net amplitude fidelity for morphology.
2. MMD and derivative penalties do not provide a universal gain at the tested
   weights.
3. ECG-AIM gives the best balance of clean reconstruction, external amplitude
   calibration, morphology, and frozen-classifier retention.
4. Severe noise narrows or reverses the clean-data architecture advantage.
5. Morphology gains do not automatically become diagnostic-utility gains.
6. Smartwatch results expose a large domain gap and should be presented as
   zero-shot failure analysis plus device-protocol characterization.

These are empirical interpretations, not final publication claims. The ARIS
result-to-claim gate is marked `REVIEW_UNAVAILABLE` until an independent
reviewer emits a verdict.
