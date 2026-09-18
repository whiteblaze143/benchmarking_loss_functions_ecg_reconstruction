# Round 2 Refinement: Separating Stability, Robustness, and Shortcut Resistance

## Problem Anchor

Test whether a diagnostic representation can remain stable under a controlled acquisition transform of the same ECG, without confusing this with hospital causality or generic domain generalization.

## Anchor Check

Balanced same-record transforms make `E` independent of `S,Y`. They can identify representation stability and transform-bank robustness, but they cannot show removal of a diagnostic shortcut because ERM has no `E-Y` shortcut to exploit. Shortcut resistance is therefore assigned only to the frozen synthetic World B (or a later semi-synthetic PTB-XL injection).

## Simplicity Check

The dominant mechanism remains same-record matching. The additional variants are necessary deletions/isolations of the already proposed full loss: augmentation-only ERM, auxiliary-only, pair-only, and mismatch-pair. No new trainable module, site adversary, or foundation-model component is added.

## Changes Made

1. **Admissibility before training.** A copied label is now called semantically preserved, not information preserved. G1 audits raw-to-KME survival, post-filter SNR, peak/cycle survival, and coverage; 10-dB noise is stress tier.
2. **Leakage closure.** Diagnosis uses the train-only whitened/normalized `Z_S`; paired distance includes a norm term and probes test `E <- ||Z_S||`.
3. **Mechanism isolation.** `erm_aug` is the primary comparator. Ten deterministic wrong-patient pairings preserve everything except identity. IRMv1 and CORAL must execute distinct gradient-tested objectives.
4. **Four synthetic worlds.** The suite now contains independent nuisance, spurious shortcut, genuinely causal environment, and destructive-information worlds.
5. **Conjunctive claims.** Paired stability, clean noninferiority, worst-transform robustness, noncollapse, and mechanism controls are all required; no composite score can compensate for a failed safeguard.

## Revised Proposal

The current canonical proposal is `FINAL_PROPOSAL.md`. It defines G0–G8, the admissible primary and stress banks, patient-equal sampling/evaluation, environment-probe equivalence margins, and the bounded claim: robustness to specified acquisition perturbations, not unobserved hospital generalization.
