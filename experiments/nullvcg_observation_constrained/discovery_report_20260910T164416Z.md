# NullVCG discovery report

Status: frozen before implementation. This report concerns only the new
observation-constrained fixed-geometry approach.

## Assumptions and success criteria

- The published eight-row LVCG direction matrix is treated as a fixed,
  non-trainable hypothesis, not as ground-truth anatomy.
- The first experiment is validation-only. PTB-XL test and EchoNext remain
  untouched until all prespecified gates permit them.
- The existing 15-epoch A0 wavelet checkpoint is the comparator because it is
  the requested serious baseline. No weights or code from the earlier
  `lvcg_ecgaim` approach are reused.
- Phase 1 succeeds only if all deterministic geometry tests pass. Phase 2
  succeeds only if the constrained oracle clears every preregistered kill
  gate. A failure stops neural-model development.

## Required discovery fields

1. **Existing primary baseline model:**
   `conv15e_A0_wave_noSSL_gated_add_s42_l0`, implemented by
   `AliTokECGAIMWaveletMTL` / `build_wavelet_ecg_aim`; 15 epochs, seed 42,
   Lead I observed, Morlet magnitude branch, no SSL.
2. **Lead-I preprocessing:** the canonical tensor is loaded as float32,
   cropped to 5,000 samples, and unobserved leads are zero-masked. Lead I is
   index 0. The baseline receives no per-lead transform.
3. **Signal units:** physical millivolts, 500 Hz, 10 seconds, shape
   `[12, 5000]`.
4. **Normalization:** disabled for the primary baseline
   (`zscore_norm=false`). The optional legacy normalization is one common
   record-wide mean and standard deviation across all leads and times; it is
   not used here.
5. **Inverse normalization:** none is needed for the primary baseline. The
   evaluator would invert optional record-wide normalization as
   `prediction * record_std + record_mean`.
6. **Patient split implementation:** canonical tensor directories correspond
   exactly to PTB-XL `strat_fold`: folds 1--8 train, fold 9 validation, fold
   10 test. Current metadata has 15,023/1,942/1,904 unique patients and zero
   pairwise patient overlap.
7. **Validation fold:** fold 9, 2,183 ECGs from 1,942 patients. This is the
   only cohort authorized for geometry/oracle development.
8. **PTB test:** fold 10, 2,198 ECGs from 1,904 patients. Frozen and prohibited
   during Phases 1--2.
9. **EchoNext eval:** `data/echonext/EchoNext_test_waveforms.npy` plus the
   official normalization adapter and EchoNext MiniModel. Frozen and
   prohibited during Phases 1--2.
10. **Metric evaluator:** `scripts/evaluate_sap_v2.py` is the canonical SAP
    evaluator. The oracle gate uses its definitions for patient-weighted
    correlations, SemiSeg morphology, QRS duration, Sokolow-Lyon error, and
    precordial variance retention, but operates on validation only.
11. **Patient bootstrap:** patient is the independent unit. Primary mean and
    p05 correlation intervals use 10,000 seeded resamples. Neural comparisons,
    if later authorized, must use paired patient-cluster bootstrap deltas.
12. **Files to modify:** none for the geometry/oracle experiment. Existing
    training, evaluator, and prior LVCG files remain unchanged.
13. **Files to add:** this discovery report; an isolated fixed-geometry module;
    deterministic geometry tests; one validation-only oracle runner; versioned
    oracle results and a concise decision report, all under
    `experiments/nullvcg_observation_constrained/`.
14. **Contradictions:** the broad SAP evaluator currently hardcodes PTB-XL
    test and EchoNext, so invoking it directly would violate the validation-only
    gate. The minimal resolution is a separate validation oracle runner that
    reproduces only the preregistered gate definitions. Also, the constrained
    oracle uses true missing leads to solve `y,z`; it is a geometry ceiling,
    not a deployable Lead-I predictor and must never be described as one.

## Frozen execution plan

1. Implement the fixed 8-by-3 matrix, exact dependent-limb derivation, full
   least-squares oracle, and Lead-I-constrained `y,z` oracle.
2. Test shape, ranks, condition number, exact Lead-I consistency, exact limb
   algebra, gradient flow to `y,z`, and absence of trainable geometry.
3. Smoke the validation oracle on a small deterministic prefix.
4. Only after the smoke succeeds, pause the SAP recovery queue, run all 2,183
   validation ECGs for ground truth, both oracles, and A0 wavelet, then resume
   the SAP queue.
5. Emit `NULLVCG_ORACLE_GATE = PASS|FAIL`. Do not implement B1, M1, residuals,
   flows, or any neural NullVCG model unless the result is `PASS`.

## Review status

The `research-refine` skill requests an external Claude review. No Claude
review tool is installed or callable in this environment, so no external score
is claimed. The plan instead remains explicitly frozen and falsifiable.
