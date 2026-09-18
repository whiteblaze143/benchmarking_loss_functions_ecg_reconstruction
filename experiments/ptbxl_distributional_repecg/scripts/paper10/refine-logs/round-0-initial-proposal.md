# Research Proposal: Paper 10 — Paired Acquisition-Intervention Stability

## Problem Anchor

- **Bottom-line problem:** Test whether a representation can retain diagnostic information while becoming stable to specified ECG acquisition changes.
- **Must-solve bottleneck:** A model must not call an observational site or hospital effect an acquisition intervention, and it must not claim invariance when its training objective never receives environment IDs.
- **Non-goals:** This paper does not establish immunity to arbitrary hospital shift, disease causality, or invariance of nonlinear Phase-KME features by assertion.
- **Constraints:** Use patient-disjoint PTB-XL development folds, transforms applied to raw physical waveforms before the frozen preprocessing path, NFS artifacts, and the existing A100. No external dataset labels are fabricated.
- **Success condition:** On held-out patients subjected to prespecified, label-preserving acquisition transforms, a paired-stability model has higher worst-environment diagnostic performance and lower paired prediction/state drift than capacity-matched ERM, without collapsing diagnostic information.

## Technical Gap

The existing proposal says that restricting mutual information produces causal features, but the implementation has no mutual-information objective, no environment tensor, and no IRM/CORAL/CausIRL loss path. Its `full`, `erm`, `irm`, `coral`, and `causirl` variants therefore optimize the same BCE objective. A site label would also be unsafe: site mixes acquisition, population, referral, and prevalence effects. The smallest adequate route is a controlled intervention battery applied to the same raw record, where physiology and labels are literally held fixed.

## Method Thesis

For a fixed record `i` and a prespecified acquisition transform `T_e`, train `Z_S` to agree between `x_i` and `T_e(x_i)` while predicting diagnosis only from `Z_S`; use `Z_A` only to verify that the transform remains observable elsewhere in the factorization.

The intervention family is initially limited to transforms with executable raw-waveform contracts: global gain `{0.8, 1.2}`, antialiased `500 -> 250 -> 500` resampling, and independently realized per-lead Gaussian noise at `{20, 10}` dB. Lead subsets are excluded because they are a measurement-operator problem in Paper 9, and bandwidth changes are excluded until a concrete pre-normalization contract exists.

## Contribution Focus

- **Dominant contribution:** paired within-record acquisition-intervention stability rather than unpaired site/domain alignment.
- **Supporting contribution:** an executable falsification suite that distinguishes invariant `Z_S` from an uninformative collapsed representation.
- **Explicit non-contributions:** a new generic domain-generalization algorithm, a hospital-causality claim, and an unimplemented `CausIRL` variant.

## Proposed Method

The frozen Paper 2 map embeds every clean and transformed waveform using the same train-only whitening/Nystrom artifact. A shared encoder produces `Z_S,Z_A`; a diagnostic head sees only `Z_S` and an environment head sees `Z_A`.

`L = L_BCE(Y, h(Z_S(x_i))) + lambda_A L_CE(e, a(Z_A(T_e x_i))) + lambda_pair d(norm(Z_S(x_i)), norm(Z_S(T_e x_i)))`.

`lambda_pair` and `lambda_A` are selected on fold 7 only. The primary full model uses the paired term. ERM uses BCE only with matched backbone; IRMv1 and CORAL are separately implemented, named baselines only if their per-environment objectives execute and their objective-specific tests pass. A speculative `causirl` label is removed rather than used as a synonym.

## Claim-Driven Validation Sketch

1. **Acquisition stability:** held-out paired records; compare full versus ERM on state and logit drift, plus worst-environment AUROC.
2. **Pairing necessity:** replace same-record clean/transformed partners with a label-matched different-patient partner. The stability advantage must disappear or degrade.
3. **No-collapse safeguard:** `Z_A` predicts the transform, a held-out probe on `Z_S` is near chance for transform identity, and diagnostic AUROC remains within the prespecified task ROPE.

## Highest-Risk Assumptions

- The chosen transforms are sufficiently diverse to induce meaningful but label-preserving shift.
- The frozen representation does not already erase all acquisition signal.
- Paired state agreement cannot be won through constant representations; this requires the diagnostic and variance safeguards.
