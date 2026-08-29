# Completed Results Deep-Analysis Refinement Report

## Scope

This refinement covers the database-backed completed ECG-AIM evidence in the
Quarto book: one-lead `1110000` screening, spatial controls, 10/15-epoch
convergence, one-lead external RDB evaluation, three-lead LUDB/RDB morphology,
frozen-representation probes, and LUDB semi-supervised delineation.

The analysis does not add training runs or fabricate missing uncertainty. It
separates internal validation, external test, representation content,
incremental utility, evaluator sensitivity, and clinical inference.

## Authoritative corrections

1. The convergence queue ledger is stale. Thirty-three terminal artifacts exist:
   23 at 10 epochs and 10 at 15 epochs. The old 12 pending labels are not live
   scientific jobs.
2. Two LUDB semi-seg rows differ only in boundary tolerance while reusing
   identical region and interval predictions. They are not independent region
   results.
3. The calibrated `extended_results` probe generation supersedes the older
   uncalibrated `probe_results` generation for inferential reporting.
4. RDB peak channels are not inferred. The external protocol reports six
   genuine onset/offset boundaries.
5. Measured-signal sentinels are protocol references. Near-retention of their
   delineator F1 is not waveform identity.

## Deep findings

### One-lead wavelet screen

- Wavelets improve P/T much more than QRS at three epochs.
- Adding the wavelet branch increases P IoU by 0.171 (lead I) and 0.083
  (lead II), while QRS changes only 0.001–0.003.
- The wavelet branch costs about 4.16 GB peak memory.
- SSL is lead-dependent: essentially neutral/adverse on lead I and favourable
  for lead-II delineation, at another approximately 8.45 GB.
- UEG-phase versus Morlet-phase effects are small and inconsistent.
- The E1 screen is missing lead I and is effectively tied on lead II at higher
  memory.

### Convergence

- Longer training improves P/T and boundaries much more than QRS.
- R5 has the strongest broad lead-II 15-epoch profile.
- A0 wavelet/no-SSL retains several reconstruction leaders.
- The matched 15-epoch A0 raw → A0+wavelet → A0+wavelet+SSL chain remains
  incomplete, so an SSL main effect is not identified.
- C1/E1 is a complementarity control, not an improved base.

### Spatial controls and external transport

- Internally, E1 beats permuted and capacity-matched controls in all six cells.
- E1 is nearly tied with A0 in five of six internal cells.
- External E1-control effects reverse sign across masks, leads, and endpoints.
- B1 panorama leads RDB boundary F1 for mask 1010010 on both observed leads,
  but is not a universal winner.
- Internal validation has only moderate correlation with external RDB results.
- The 48-record pilot preserves boundary ranks better than signal-tail ranks.

### Three-lead external morphology

- Among 31 shared models, LUDB↔RDB boundary-F1 Spearman is 0.736.
- Waveform-tail rank transport is weak: 0.300 for Pearson p05 and approximately
  zero for MSE p95.
- Best RDB reconstructed boundary F1 is 0.7404 versus measured-signal reference
  0.7430. This is fixed-delineator event retention, not waveform recovery.
- Oracle endpoint leaders differ for signal, QRS, T, area, boundary voltage,
  and ST/J metrics; no scalar winner is validated.

### Frozen ECG-AIM representations

- Test prevalence is 120 AF/AFIB-coded of 360 records.
- Calibrated probes select strong shrinkage (`C=0.01`) and at most eight PCA
  dimensions.
- The 1011011 latent AUROC is 0.787 versus waveform-summary AUROC 0.713.
- Direct 1011011-versus-1000000 AF contrasts are non-significant on every
  metric; objective superiority is not supported.
- 1011011 improves several latent-only QRS-onset–T-offset timing metrics, while
  direct waveform summaries remain much better for absolute MAE.
- Local rhythm-neighbour agreement is significant, but global silhouettes are
  negative and non-significant.
- Latent outlier score has OR 2.15 per SD for internal reconstruction failure,
  based on 24 events; external calibration is absent.

### LUDB semi-supervised delineation

- Student-versus-EMA effects are small and endpoint-dependent.
- Padding/evaluation-window effects can dwarf state effects.
- QRS is the strongest wave; P-wave and interval endpoints remain weaker.
- Checkpoint, state, split, padding, window, evaluator digest, and tolerance are
  required to define a result.

## Claim decisions

| Claim | Decision |
|---|---|
| Wavelets improve slow-wave delineation internally | Supported |
| SSL generally improves wavelet ECG-AIM | Unresolved |
| UEG phase is superior to generic phase | Suggestive only |
| E1 is the improved base architecture | Rejected by current evidence |
| ECG-AIM latents contain rhythm information | Supported internally |
| 1011011 improves AF representation over 1000000 | Not supported |
| Latent outliers can prioritize reconstruction QC | Supported internally |
| Reconstructed ECG is clinically equivalent to measured ECG | Not supported |
| UMAP shows discrete rhythm manifolds | Contradicted |

## Remaining decisive experiments

1. Complete matched 15-epoch A0+wavelet+SSL for both leads.
2. Repeat registered wavelet and phase contrasts across seeds.
3. Externally evaluate preregistered one-lead wavelet checkpoints with six
   genuine RDB boundaries.
4. Extract train/validation/test embeddings for preregistered one-lead models.
5. Compare frozen latents with a strong native-waveform classifier.
6. Add patient-paired uncertainty and external calibration for failure scores.
7. Extend missing_leads_v2 clinical evaluation to ECG-AIM using the identical
   record order and evaluator generation.

## Review independence

The requested independent reviewer bridge is unavailable in this environment.
This report is a local evidence audit and is not represented as independent
review.

