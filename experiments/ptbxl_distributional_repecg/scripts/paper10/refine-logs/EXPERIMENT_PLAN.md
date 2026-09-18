# Experiment Plan: Paper 10 — Paired Acquisition Matching

**Problem:** stabilize diagnostic representations across controlled acquisition transforms without conflating stability, robustness, and shortcut removal.

**Method thesis:** same-record alignment supplies identity information unavailable to unpaired population alignment; it is tested against augmentation-matched ERM, IRMv1, and CORAL.

## Claim Map

| Claim | Anti-claim ruled out | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: paired matching improves stability and robustness to admissible transforms | preprocessing alone, augmentation alone, or collapse explains the effect | full improves paired stability and worst-transform performance over `erm_aug`, retains clean task performance | B0, B4, B6, B7 |
| C2: same-record identity, not auxiliary classification or population alignment, is the active mechanism | auxiliary CE, generic alignment, or accidental pairing is sufficient | `pair_only > aux_only`; same pair beats ten surgical mismatch controls | B2, B3, B5 |
| C3: shortcut resistance has a bounded causal interpretation | balanced transform robustness proves shortcut removal | synthetic/semi-synthetic train `E<->Y`, test `E independent Y` result | B2, B7 |

## Paper Storyline

- **Main paper:** G0–G7 only, if all gates pass.
- **Appendix:** continuous severity curves, withheld transform levels/families, and 10-dB boundary results.
- **Cut:** hospital/site causality, a universal domain-generalization claim, lead-subset interventions, and CausIRL.

## Experiment Blocks

### B0: G0 provenance and G1 admissibility/survival — MUST-RUN

- **Claim tested:** each view is a true same-record controlled intervention and remains usable at the model input.
- **Data:** patient-disjoint PTB-XL folds; transform raw physical waveform, then filter, scale, detect R peaks, phase-normalize, and apply the frozen train-only Phase-KME map.
- **Primary nuisance bank:** gain `0.8/1.2`, `500->250->500`, 20-dB noise. For each noise level use three deterministic realizations with seed hash `(patient_id, ecg_id, environment, realization, master_seed)`.
- **Boundary bank:** 10-dB noise; it is not promoted automatically.
- **Metrics:** raw/post-filter distortion and SNR, R-peak changes, cycle survival, KME drift, active-environment detectability, common-support coverage, and diagnosis-stratified coverage.
- **Success:** exact record/label/fold hashes; all stage denominators reported; active transforms survive preprocessing and meet frozen coverage/information thresholds.
- **Failure:** a view is semantically label-preserving but information-degrading; it is boundary-only or the primary bank fails. Copied labels are never evidence of admissibility.

### B1: G2 four-world synthetic identifiability suite — MUST-RUN

| World | Train/test mechanism | Decisive result |
|---|---|---|
| A | `E` independent of `S,Y` | stable `Z_S`, retained task, no false shortcut claim |
| B | `S->Y`, `E->X`; train `P(E|Y) != P(E)`, test `E independent Y` | paired model beats `erm_aug` after the correlation breaks |
| C | `E->Y` | enforcing invariance costs task information |
| D | transform erases disease state `S` | no method jointly claims perfect invariance and Bayes task preservation |

The generator freezes all coefficients, train/test correlation tables, and seeds before fitting. World B is the only primary shortcut-resistance test; World A is an invariance test only.

### B2: G3 objective and gradient execution — MUST-RUN

- **Compared systems:** `erm_clean`, `erm_aug`, `aux_only`, `pair_only`, `full`, CORAL, and IRMv1.
- **Checks:** each objective is finite; its intended gradient norm is nonzero; gradient routing is correct; and frozen-minibatch parameter updates differ across algorithms.
- **Required details:** IRMv1 computes each balanced-environment risk and `||grad_(w=1) R_e(w Phi)||²`. CORAL is explicitly mean-plus-covariance alignment if both terms are used. Acquisition CE cannot update the diagnosis head; its shared-trunk effect is separately measured.
- **Failure:** named algorithms share BCE-only execution or a supposed term has zero/no target gradient.

### B3: G4 same-record pairing mechanism — MUST-RUN

- **Comparison:** `pair_only` versus `aux_only`, full versus `erm_aug`, and full versus ten `mismatch_pair` permutations.
- **Destroyer:** `(x_i,T_e x_i)` becomes `(x_i,T_e x_j)`, `i != j`, preserving environment, realization, folds, and a predeclared diagnosis matching hierarchy: exact multilabel vector, then superdiagnostic vector, then minimum Hamming distance with deterministic ID tie-break.
- **Metrics:** patient-equal `D_same`, `D_diff`, `R_stab=D_same/D_diff`, task AUROC, and prediction drift.
- **Success:** same-pair `R_stab` and task performance dominate the mismatch distribution; full-to-ERM gap is not explained by augmentation alone.

### B4: G5 non-collapse and G6 nuisance-leakage — MUST-RUN

- **No-collapse:** effective rank, total variance, diagnosis AUROC, and common-support denominator on held-out patients.
- **Probe family:** held-out-patient multinomial logistic and small MLP probes for transform identity from `Z_S`, `Z_A`, and `||Z_S||` alone.
- **Frozen equivalence rules:** with `E_active` active transform types, both `Z_S` probes have upper 95% accuracy CI `< 1/E_active + 0.05`; the `Z_A` probe has lower CI `> 1/E_active + 0.20`. These are conditional on transform survival from B0; an erased transform is reported, not hallucinated by `Z_A`.
- **Boundary:** `Z_A` is acquisition-sensitive, not asserted to be a pure acquisition representation.

### B5: G7 frozen real-data robustness comparison — MUST-RUN only after G0–G6

- **Training:** patient-equal units; fold 1–6 fit, fold 7 selects all weights/settings, fold 8 remains unread. Three seeds are required after one objective-verified pilot.
- **Primary comparator:** `erm_aug` sees identical clean/transformed examples, update count, patient weighting, and budget as full.
- **Primary endpoints:** patient-bootstrap paired-stability improvement; clean macro-AUROC noninferiority to `erm_aug` under a frozen ROPE; worst-primary-transform macro-AUROC improvement if robustness is claimed.
- **Secondary:** macro-AUPRC, calibration, per-transform AUROC, representation/prediction severity curves, and coverage.
- **Failure:** report transform-bank robustness only; do not call same-bank testing domain generalization.

### B6: G8 boundary/extrapolation analysis — APPENDIX

- Evaluate noise SNR `{30,25,20,15,10}` and gain `{0.7,...,1.3}` with held-out severity levels.
- Report paired drift and task degradation as continuous curves.
- Withholding levels/families is the only secondary OOD-style result; it does not rescue a failed primary mechanism gate.

## Run Order and Decision Gates

| Milestone | Runs | Stop/go decision | Cost estimate |
|---|---|---|---|
| M0 | B0 artifact/audit | stop on provenance, survival, or coverage failure | CPU/I/O + one GPU representation pass |
| M1 | B1 synthetic suite | stop if Worlds C/D yield the impossible claim or World B has no shortcut test | minutes on A100 |
| M2 | B2–B4 unit/gradient/probe tests | stop if objectives coincide, pairing is unused, collapse, or norm leakage occurs | minutes to low GPU-hours |
| M3 | B5 fold-7 pilot then 3 seeds | advance only under frozen conjunctive gates | bounded by selected architecture |
| M4 | B6 appendix | characterize limits, never alter primary gates | optional |

## Compute and Data Budget

- **Data prerequisite:** a new versioned paired-intervention artifact; no current Paper 2 array contains environment IDs.
- **Main bottleneck:** raw transformation plus full preprocessing, not GPU training.
- **Frontier primitive:** absent; no LLM/VLM/diffusion/RL component is justified.

## Final Checklist

- [ ] Stability, robustness, and shortcut resistance have separate evidence.
- [ ] Semantic label copying is not called information preservation.
- [ ] `erm_aug` is the primary comparator.
- [ ] Norm-only environment leakage is tested.
- [ ] Gradient/update tests prevent BCE-only variant aliases.
- [ ] Patient-equal training and coverage-aware evaluation are implemented.
