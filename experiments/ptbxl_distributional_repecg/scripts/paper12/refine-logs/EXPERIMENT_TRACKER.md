# Experiment Tracker: Paper 12

| Run ID | Gate | Purpose | Decisive metrics | Status |
|---|---|---|---|---|
| P12-LEGACY-G0 | implementation audit | compare claimed flow/SCM mechanism with code | likelihood, invertibility, innovation objective, destroyer | FAIL (BCE-only GRU residual; claimed mechanism absent) |
| P12-MEAN-G1 | mean-residual assumptions | additive, IID, heteroscedastic, reversal diagnostics | recovery, whiteness, direction sensitivity | FAIL as full mechanism (heteroscedastic dependence remains; reversal is not a destroyer) |
| P12-LOC-SCALE-G1 | location-scale identifiability | frozen fresh-seed synthetic suite | prediction, recovery, standardized independence, false worlds | PASS (all six frozen gates, seeds 45/46/47; 500 updates; CUDA) |
| P12-LOC-SCALE-G2 | objective execution | stagewise NLL and diagnosis gradient routing | nonzero intended gradients and distinct updates | PASS (density-only and frozen-density probe routes; 7 Paper 12 tests) |
| P12-LOC-SCALE-G3-MAP | representation fidelity | label-free folds 1–6 map, independent fold-7 confirmation | rank and magnitude fidelity plus tail diagnostics | PASS at 1,024 landmarks (confirm rho=0.9655, median RE=0.0757) |
| P12-LOC-SCALE-G3 | real phase-cell admissibility | phase order, preprocessing lineage, residual diagnostics | matched baselines, destroyers, rotations, patient-equal denominators | FAIL: conditional NLL loses to phase-only at all origins; 93–95% lower-clamp saturation; scale removal negligible |
| P12-LOC-SCALE-G4 | clinical utility | innovations versus state and matched controls | patient-equal AUROC plus residual diagnostics | BLOCKED by failed G3 |
| P12-LOC-SCALE-V2-G1 | scale-conditioned synthetic mechanism | affine scale stress, genuine history, IID, destroyers | recovery, NLL contrasts, false-positive control | PASS (all gates, fresh seeds 48/49/50) |
| P12-LOC-SCALE-V2-G2 | V2 execution | train-only standardization and stagewise routes | exact fit scope, gradients, phase-0 exclusion | PASS (12 mechanism/routing tests) |
| P12-LOC-SCALE-V2-G3 | real residual rescue | standardized Phase-KME folds 1–6/7 | patient-equal baseline and destroyer contrasts | PASS after corrected bijective donor rerun; all intervals decisive |
| P12-LOC-SCALE-V2-G4 | diagnostic retention | matched probes across representation hierarchy | retained accessible diagnosis signal | PRE-LABEL CONTRACT PASS (20 tests); fold-7 Z-only selection running |
