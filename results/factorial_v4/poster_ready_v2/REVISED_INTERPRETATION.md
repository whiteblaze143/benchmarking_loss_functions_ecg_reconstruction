# Revised Interpretation for the Loss-Function Benchmark Poster

Status: **superseded draft pending the repaired complete \(2^4\) analysis**.
The text below describes the valid MSE-on slice and must not be used as the
final poster interpretation. Eight historical MultiScale-VAE MSE-off
checkpoints are being replaced before MSE main effects and interactions are
reported. Publication wording also remains provisional because the independent
postrun audit is unavailable (final gate 9/10).

## Central result

The new benchmark does **not** support a universal composite-loss winner. It
supports a stronger and more informative conclusion:

> Correlation, MMD, and derivative penalties have architecture-dependent effects.
> Correlation improves amplitude-aware reconstruction in the two latent families
> but trades away U-Net R², while MMD and derivative penalties do not provide
> consistent cross-family gains. Full composite objectives improve QRS and ST
> morphology without a detectable improvement in frozen ECGFounder AUROC.

Architecture choice is therefore at least as important as loss-mask choice, and
the objective should be selected for the intended endpoint rather than treated as
a universally beneficial regularizer bundle.

## Answers to the three benchmark questions

### 1. Does MSE alone lose clinically meaningful morphology?

Partly. Relative to MSE-only, the full objective improves QRS and ST correlation
in all three families at the prespecified within-family threshold
\(\alpha=0.0167\). This supports the narrower claim that morphology-aware terms
can improve waveform shape.

The tradeoff is endpoint- and architecture-specific. In U-Net, correlation has a
negative marginal effect on missing-lead R²
(\(-0.0282\), 95% patient-cluster BCa CI \([-0.0371,-0.0210]\)). Thus improved
shape correlation does not imply better amplitude-calibrated reconstruction.

### 2. Which loss components contribute?

- **Correlation:** the only component with a reproducible positive R² effect in
  both latent families: MultiScale-VAE \(+0.0141\)
  \([0.0127,0.0157]\), ECG-AIM \(+0.0132\) \([0.0117,0.0151]\).
  Its U-Net effect is negative, so it is not architecture-agnostic.
- **MMD:** small negative R² effects in U-Net and MultiScale-VAE and an
  approximately null ECG-AIM effect. It should not be described as a universal
  distributional improvement.
- **Derivative:** approximately null for MultiScale-VAE, small negative for
  ECG-AIM, and no general cross-family benefit. The current results do not
  establish that it restores sharp clinical transitions.

The appropriate poster conclusion is “component contributions depend on the
decoder/latent geometry,” not “all terms should be retained.”

### 3. Do morphology gains preserve or improve diagnostic utility?

The full objective improves QRS and ST correlation in every family, but its
ECGFounder AUROC change is nonsignificant in every family. Morphology and frozen
classifier utility therefore decouple in this experiment.

ECGFounder inference is based on the 31 of 150 tasks with both classes present in
the held-out fold. It must remain separate from the original five-superclass
paper-parity classifier. “No significant change” is evidence of no detected
difference, not proof of equivalence or clinical safety.

## Architecture-level interpretation

Validation-selected masks were U-Net `110`, MultiScale-VAE `100`, and ECG-AIM
`110`, where bits denote correlation/MMD/derivative. Their held-out PTB-XL R²
values were 0.662, 0.816, and 0.820, respectively. ECG-AIM therefore leads the
selected clean models, closely followed by MultiScale-VAE.

Selection and clean-test ranking must not be conflated: the best clean cell can
differ from the validation-selected cell (for example, U-Net MSE-only has higher
clean R² than selected U-Net `110`). Report selected results for fair downstream
testing and use the complete heatmap to show the loss landscape.

## Robustness and external generalization

At severe 0 dB NSTDB noise, U-Net starts from worse clean reconstruction but
degrades less than the latent models. This is a robustness–accuracy tradeoff, not
evidence that U-Net is globally superior.

On EchoNext, selected ECG-AIM is the only family with positive missing-lead R²
(0.593). U-Net has high Pearson correlation (0.878) but extremely negative R²
(-35.243), demonstrating severe amplitude miscalibration that correlation alone
conceals. EchoNext SHD results also use unchanged tabular metadata and must not be
described as waveform-only diagnostic performance.

All 12 selected family-device smartwatch cells have negative missing-eleven-lead
R². These calibrated simulator recordings are valuable as a zero-shot domain-gap
stress test, but they do not establish clinical wearable reconstruction. The
ECGFounder quantity is Philips-referenced probability agreement, not disease
classification accuracy.

## How the original draft should change

| Original framing | Factorial-v4 evidence | Poster-safe replacement |
|---|---|---|
| The complete composite loss is universally best. | No mask dominates all families and endpoints. | Loss design is architecture- and endpoint-dependent. |
| Correlation is the dominant term and all terms should be retained. | Correlation helps latent-family R² but harms U-Net R²; MMD and derivative are inconsistent. | Correlation is the most consequential term, but its direction depends on architecture. |
| MMD and derivative losses recover distributional and sharp-transition fidelity. | No consistent cross-family benefit establishes those mechanisms. | Treat these as tested regularizers with small or inconsistent marginal effects. |
| Morphology gains occur without diagnostic compromise. | Full-vs-base ECGFounder changes are nonsignificant, with wide enough intervals to avoid an equivalence claim. | Morphology improved, while no diagnostic-utility improvement was detected. |
| Deterministic reconstruction guarantees patient physiology. | A reconstruction model cannot guarantee unobserved physiology. | Reconstruction estimates plausible missing leads and requires external validation. |
| Wearable results demonstrate clinical feasibility. | All smartwatch R² values are negative in zero-shot simulator testing. | Wearable transfer exposes a substantial calibration/domain gap. |
| EchoNext confirms generalization broadly. | Only ECG-AIM has positive external R²; classifier inference includes tabular metadata. | ECG-AIM generalizes best, while other families show substantial external calibration failure. |

## Poster narrative and figure order

1. **Design:** complete MSE-on \(2^3\) factorial across three valid architecture
   families and 2,198 held-out ECGs.
2. **Primary result:** show the complete loss landscape, then patient-cluster
   marginal effects. This keeps the poster centered on loss benchmarking.
3. **Clinical interpretation:** show full-minus-base morphology and ECGFounder
   results to demonstrate endpoint decoupling.
4. **Stress tests:** show NSTDB and EchoNext as evidence that clean ranking does
   not fully determine robustness or external calibration.
5. **Limitation/future work:** show smartwatch transfer only if space permits;
   frame it as a domain-gap result.

Figures 1–3 are the poster core. Figure 4 is the strongest generalization panel.
Figure 5 is supplementary unless wearable translation is central to the venue.

## Cohort and reporting corrections

- Use 2,198 held-out ECGs from 1,904 patients for the factorial-v4 evaluation.
- Keep the original U-Net paper-parity anchor distinct from the extended
  stable-weight factorial.
- State that intervals are patient-cluster BCa intervals conditional on the
  trained seed-42 models; confirmation seeds describe training variance but do
  not replace record-level paired inference.
- Do not include the quarantined MultiScale-VAE MSE-off branch.
- Do not restore cNVAE to the main comparison; it was deferred after failing the
  validity gate.
