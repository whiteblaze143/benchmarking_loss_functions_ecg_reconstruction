# NullVCG validation-oracle decision

`NULLVCG_ORACLE_GATE = PASS`

The result uses all 2,183 ECGs / 1,942 patients in PTB-XL validation fold 9.
Neither PTB-XL test nor EchoNext was accessed by this experiment.

## Raw gate results

| Metric | A0 wavelet | Constrained oracle | Delta | Gate |
|---|---:|---:|---:|---|
| Mean missing-lead Pearson r | 0.810115 | 0.875874 | +0.065760 | pass (required +0.02) |
| p05 missing-lead Pearson r | 0.580873 | 0.744922 | +0.164049 | pass (required +0.02) |
| SemiSeg mIoU | 0.663777 | 0.690164 | +0.026386 | pass |
| P-wave IoU | 0.507309 | 0.534843 | +0.027534 | pass |
| T-wave IoU | 0.693698 | 0.717981 | +0.024284 | pass |
| QRS-duration MAE (ms) | 10.084895 | 8.769773 | -1.315122 | pass |
| Sokolow-Lyon MAE (mV) | 0.740914 | 0.480944 | -0.259969 | pass |
| Precordial variance retained | 0.308183 | 0.562712 | +0.254529 | pass; closer to 1 |

Geometry diagnostics: matrix shape 8-by-3, rank 3; missing-lead `y,z`
block rank 2; condition number 1.790915; zero trainable geometry parameters.

## Ceiling and failure-mode interpretation

The unconstrained full-VCG oracle reaches mean missing-lead r = 0.918175 and
p05 = 0.807320. Enforcing exact Lead I lowers these to 0.875874 and 0.744922.
In MSE terms, the fixed-dipole model error is 0.026351 mV² and the observation
constraint adds 0.014508 mV², for 0.040859 mV² total.

This means the geometry is useful enough to continue, but it is not a faithful
complete embedding of the ECG. The constrained per-lead R² is uneven:

| Lead | I | II | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R² | 1.000 | 0.333 | 0.580 | 0.774 | 0.524 | 0.210 | 0.418 | 0.396 |

The particularly weak II/V4/V5/V6 ceilings warn that a strict dipole output
bottleneck may discard local precordial morphology. Also, the constrained
oracle estimates `y,z` using the true missing leads. It proves only that the
fixed geometry can beat A0 when those coordinates are known; it does not prove
that they are learnable from Lead I.

## Frozen next-stage plan

1. **B0:** retain the existing A0 wavelet checkpoint unchanged as context.
2. **B1 matched direct control:** use the same selected temporal backbone and
   optimization protocol as M1, predict only the seven missing independent
   leads `[II,V1..V6]`, copy Lead I exactly, and derive III/aVR/aVL/aVF
   algebraically.
3. **M1 NullVCG:** use that identical backbone but replace only the final head
   with two `y,z` channels. Concatenate measured Lead I as VCG x, apply the
   fixed 8-by-3 projection, copy Lead I exactly, and derive dependent limb
   leads algebraically.
4. Train both systems only against the seven missing independent ECG leads.
   Do not supervise latent coordinates and do not add SSL, delineation,
   wavelets, residual branches, learned geometry, or probabilistic modeling.
5. Run shape/leakage/gradient tests and a two-batch smoke for both systems.
6. Run seed 42 on folds 1--8 and evaluate only fold 9. Compute a paired
   patient-cluster bootstrap for `M1 - B1`.
7. Continue only if mean missing-lead r improves by at least +0.005 and its
   confidence interval is favorable without material clinical-metric harm.
8. Only after that gate may C1 (a learned rank-2 1-by-1 correction) be tested,
   followed by seeds 43/44. PTB-XL test and EchoNext remain frozen until the
   three-seed decision.

No code or artifact from the prior `lvcg_ecgaim` approach is part of this plan.
