# Round 1 Refinement

## Problem Anchor

Test whether a representation can retain diagnostic information while becoming stable to specified ECG acquisition changes, without treating hospital membership as a clean intervention.

## Anchor Check

- The revised method still targets acquisition-induced shortcut sensitivity.
- It rejects a site/hospital claim because environment membership can encode population and prevalence shifts.
- It rejects a generic information-bottleneck claim because no such objective exists or is required.

## Simplicity Check

- **Dominant contribution:** exact same-record pairing across raw acquisition transforms.
- **Reused components:** frozen Paper 2 transform, one shared encoder, existing two projections.
- **New trainable components:** none beyond the already-present diagnostic and acquisition heads.
- **Rejected complexity:** adversarial site removal, `CausIRL`, lead subset intervention, and a universal OOD claim.

## Revised Proposal

Use a finite intervention bank `E={clean,gain0.8,gain1.2,resample250,noise20,noise10}`. Every transform begins with the physical waveform and traverses filtering, scaling, beat detection, phase normalization, and the frozen representation map. For each held-out patient record `i`, all valid views share the immutable pair key `(patient_id, ecg_id)` and label vector.

The full model optimizes diagnosis from `Z_S`, transform classification from `Z_A`, and within-record `Z_S` agreement. The primary mechanism endpoint is patient-equal paired stability under each held-out transform, with worst-environment AUROC as the clinical safeguard. A matched-pair destroyer aligns each transformed view to a different label-matched patient. A collapsed representation fails because it cannot preserve diagnosis and is checked by representation variance plus transform-predictability separation.

The proposal remains `RETHINK` until the intervention artifact, objective implementations, and synthetic/mechanism tests exist.
