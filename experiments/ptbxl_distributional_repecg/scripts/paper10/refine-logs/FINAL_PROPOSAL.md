# Research Proposal: Paper 10 — Paired Acquisition Matching

## Problem Anchor

Acquisition artifacts can create diagnostic shortcuts, but hospital identity is not itself an acquisition intervention. The paper asks whether a diagnostic ECG representation can be stabilized across a known transform of the *same raw record* without losing usable diagnostic information.

## Precise Claims

The paper separates three claims that cannot be inferred from one another:

1. **Stability:** `Z_S(x_i)` and `Z_S(T_e x_i)` are close for an admissible transform.
2. **Robustness:** this closeness retains diagnostic performance under the enumerated transform bank.
3. **Shortcut resistance:** when `E` is spuriously correlated with `Y` in training but independent at test, the model resists the broken correlation.

Balanced real-data interventions establish only Claims 1–2. Claim 3 requires a synthetic or semi-synthetic shortcut-injection world.

## Intervention Contract and Admissibility

Every view begins with the raw physical waveform and passes through transform, filter, scaling, R-peak detection, phase normalization, and the frozen train-only Phase-KME map. Copying a clinical label makes a view *semantically label-preserving*; it does not prove that diagnostic information survives.

G1 therefore audits, by transform and diagnosis: raw and post-filter distortion, raw and post-filter SNR, R-peak displacement/failure, cycle survival, common-support coverage, and representation drift. The primary bank is gain `{0.8,1.2}`, `500->250->500` resampling, and 20-dB noise with three deterministic realizations per record. 10-dB noise is boundary/stress tier until it passes admissibility. An intervention erased by preprocessing is reported as such and is never used to demand environment prediction from a learned representation.

Lead subsets remain measurement interventions for Paper 9. Site/hospital labels, unimplemented bandwidth changes, and the retired OOD evaluator are excluded.

## Method Thesis

For an admissible environment `e`, learn an auxiliary acquisition-sensitive state `Z_A` and a diagnostic state `Z_S` such that same-record views match in a train-only whitened `Z_S` space while diagnosis operates on that same normalized/whitened state. This prevents environment information from hiding only in `||Z_S||`.

`L_full = L_BCE(Y,h(W(Z_S))) + lambda_A L_CE(e,a(Z_A)) + lambda_pair ||W(Z_S(x_i))-W(Z_S(T_e x_i))||² + lambda_norm (log||Z_S(x_i)||-log||Z_S(T_e x_i)||)²`.

`W` is fitted on folds 1–6 only. A patient is the training unit; one transform and, where applicable, one deterministic realization are sampled per record so differential transform survival cannot reweight patients.

## Comparator and Falsification Contract

| Variant | BCE | acquisition CE | paired loss |
|---|---:|---:|---:|
| `erm_clean` | yes | no | no |
| `erm_aug` | yes | no | no |
| `aux_only` | yes | yes | no |
| `pair_only` | yes | no | yes |
| `full` | yes | yes | yes |
| `mismatch_pair` | yes | matched to full | different-patient pairs |
| `coral` | yes | no | unpaired mean/covariance alignment |
| `irmv1` | yes | no | per-environment gradient penalty |

`erm_aug` sees exactly the views, patient weighting, updates, and budget of `full`; it is the primary comparator. `mismatch_pair` uses ten frozen permutations and preserves environment, transform realization, and predeclared label-matching hierarchy while changing only underlying-record identity. `causirl` remains removed.

## Synthetic Identifiability Suite

| World | Mechanism | Required interpretation |
|---|---|---|
| A | `E` independent of `S,Y` | stability without task loss |
| B | `S->Y`, `E->X`, train `E` spuriously associated with `Y`, test independent | shortcut resistance |
| C | `E->Y` | invariance must cost task performance |
| D | transform erases `S` | no method may claim both perfect invariance and Bayes-optimal diagnosis |

## Claim Gates

For the real paired claim, all must hold with patient-clustered bootstrap intervals: lower stability improvement versus `erm_aug` is positive; clean AUROC is noninferior under a frozen margin; and worst-transform AUROC improves if robustness is claimed. Mechanism additionally requires `pair_only > aux_only` and same-pair superiority over the ten mismatch controls. Non-collapse requires effective rank/variance and retained diagnosis.

Environment probes use held-out patients, multinomial logistic and small MLP probes, and frozen equivalence margins. The `Z_S` probe—including norm alone—must have upper 95% accuracy CI below `1/E_active + 0.05`; `Z_A` must have lower CI above `1/E_active + 0.20`. The architecture makes no claim that `Z_A` is a pure acquisition state.

## Scope Boundary

Evaluating transform definitions seen in training is robustness to trained perturbations, not domain generalization. Severity curves and withheld intensities/families are secondary extrapolation tests. The paper stays **RETHINK / preclinical specification** until G0 provenance through G4 pairing-mechanism gates pass.
