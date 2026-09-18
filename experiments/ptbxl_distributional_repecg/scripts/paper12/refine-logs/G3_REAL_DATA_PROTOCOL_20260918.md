# Paper 12 G3 Real-Data Residual Adequacy Protocol

**Frozen before G3 fitting.** Density fit uses folds 1–6, selection uses fold 7, and fold 8 remains unread. Diagnosis labels are unavailable to the density optimizer. Phase 0 is conditioning context only and is excluded from diagnosis representations.

## Scientific object

The model performs phase-ordered conditional location-scale standardization over frozen Phase-KME cells. It is not physical-time forecasting, causal prediction, a full Gaussian density claim, or covariance whitening.

## Required comparisons

- raw state;
- conditional mean residual;
- conditional standardized innovation;
- phase-specific unconditional location-scale residual;
- deterministic wrong-record/different-patient history;
- deterministic label-matched wrong-record history;
- same-record phase-order permutation.

## Structural sensitivities

Repeat residual diagnostics for phase-origin rotations 0, 4, 8 and 12. Each rotation receives a separate density fit with identical frozen architecture, optimizer and budget, while consuming the exact same frozen Phase-KME map hash. Time reversal remains descriptive only.

The phase-order destroyer uses a deterministic record-specific, non-cyclic permutation keyed by patient ID, ECG ID and the frozen seed. It preserves the phase-cell multiset and cannot be learned as one global remapping. Wrong-history controls keep record `i` as the target and replace only its conditioning history with a frozen different-patient record, either unrestricted or diagnosis-pattern matched.

## Required reporting

Report patient-equal conditional and unconditional loss; mean and variance by phase and coordinate; location- and squared-residual lag curves; history dependence; cross-coordinate covariance descriptively; lower/upper scale-clamp occupancy overall, by phase and coordinate; exact coverage; and patient-bootstrap intervals.

Before residual fitting, the fold-1–6 Nyström map must attain Spearman correlation at least 0.95 and median relative error below 0.10 on 1,000 exact-versus-approximate distance pairs. Relative error is `abs(approx-exact)/max(exact, epsilon)`, with `epsilon=max(1e-8, 0.001*median(positive exact distances))`, frozen before confirmation. Relative-error Q90, Q95, Q99 and exact-distance-quartile medians are descriptive. Reservoir sampling, whitening, landmarks and the basis are label-free and use folds 1–6 only.

Map size is the smallest passing development candidate in `{512,1024}`. The selected frozen map must independently pass on a distinct fold-7-only 1,000-pair sample. If 1,024 fails development or the selected map fails confirmation, this representation version fails G3; larger maps or alternate approximations require a new protocol version.

The selected map has 1,024 landmarks. Its independent fold-7 confirmation passed with Spearman 0.965530 and median relative error 0.075683. Tail errors remain descriptive limitations: Q90 0.259122, Q95 0.354546 and Q99 0.517205; the largest exact-distance quartile has median relative error 0.158863.

## Decision rules

G3 can pass only if all of the following hold on fold 7 without revising the protocol:

1. the frozen Nyström fidelity gate passes, then conditional loss improves over the phase-specific unconditional baseline and both wrong-history controls;
2. same-record phase permutation worsens conditional prediction;
3. standardized innovations reduce location dependence relative to state;
4. standardized innovations reduce scale dependence relative to conditional mean residuals;
5. the direction of those conclusions is unchanged for all four phase origins;
6. every denominator and scale-clamp occupancy is reported;
7. the density path is label-free and frozen before any probe.

Calibration to zero mean and unit variance is reported with patient-bootstrap intervals. It is not converted into a post-hoc tolerance gate. Failure of Gaussian shape or cross-coordinate identity covariance limits the wording to marginal location-scale standardization.

## Frozen fitting budget

Every phase origin and the record-specific phase-permutation destroyer uses width 64, AdamW learning rate `3e-4`, weight decay `1e-4`, batch size 128, exactly 1,000 optimizer updates and seed 42. There is no origin-specific hyperparameter selection. The primary feature basis is the frozen 1,024-landmark map with hash recorded by the fidelity certificate.
