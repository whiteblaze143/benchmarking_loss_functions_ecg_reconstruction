# Experiment Tracker: Paper 10

| Run ID | Gate | Purpose | System / Variant | Split | Decisive metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| P10-R001 | G0 | prove same-record view provenance | paired raw transform builder | folds 1–7 | ID/label/fold hashes, seed and view manifest | MUST | PASS | 152,450 paired views; no folds 8–10 read |
| P10-R002 | G1 | establish transform admissibility/survival | primary and stress banks | patient-equal by diagnosis | stage drift, post-filter SNR, peaks, cycles, coverage | MUST | RUNNING | thresholds frozen before fold 7; confirmatory audit `paper10_g1_fold7`; 10 dB stays boundary-only |
| P10-R003 | G2 | test invariance without shortcut | synthetic World A | held-out states | task, `R_stab`, probes | MUST | TODO | `E independent (S,Y)` |
| P10-R004 | G2 | test shortcut resistance | synthetic World B | train correlated, test independent | robust task vs `erm_aug` | MUST | TODO | only shortcut-removal evidence |
| P10-R005 | G2 | test causal and destructive boundaries | Worlds C/D | held-out states | inevitable task loss / impossibility | MUST | TODO | protects against impossible claims |
| P10-R006 | G3 | objective/gradient execution | ERM, IRMv1, CORAL, pair, full | frozen minibatch | loss, gradient, parameter-update differences | MUST | PARTIAL PASS | 10/10 mechanism/provenance tests pass; full parameter-update matrix remains |
| P10-R007 | G4 | isolate same-record identity | pair/full vs 10 mismatch permutations | held-out synthetic then real | `R_stab`, AUROC distribution | MUST | TODO | predeclared label match hierarchy |
| P10-R008 | G5/G6 | reject collapse and norm leakage | full, pair-only, aux-only | held-out patients | rank, variance, `E<-Z_S`, `E<-||Z_S||`, `E<-Z_A` | MUST | TODO | logistic and MLP probes |
| P10-R009 | G7 | frozen robustness pilot | `erm_clean`, `erm_aug`, pair-only, aux-only, full | fit 1–6/select 7 | paired stability, clean/worst AUROC | MUST | TODO | no fold 8 read |
| P10-R010 | G7 | confirm decisive comparison | selected systems, 3 seeds | frozen development protocol | patient-bootstrap conjunctive gates | MUST | TODO | only after R009 passes |
| P10-R011 | G8 | characterize severity boundary | full vs `erm_aug` | withheld levels/families | severity curves and coverage | APPENDIX | TODO | does not revise main gate |
