# Artifact Manifest

This manifest separates frozen design artifacts, executable source, tests, and
runtime evidence. Runtime outputs are not scientific results unless explicitly
promoted by the claim gates.

## Frozen design

- `configs/common.yaml`: shared folds, labels, preprocessing, kernel, training,
  statistics, and fold-10 firewall.
- `docs/SPEC_RECONCILIATION.md`: resolution of the two supplied specifications.
- `docs/MATHEMATICAL_CONTRACT.md`: typed mathematical objects and estimators.
- `docs/CLAIM_GATE_MATRIX.md`: preregistered mechanism and promotion gates.
- `refine-logs/FINAL_PROPOSAL.md`: refined eight-paper proposal.
- `refine-logs/EXPERIMENT_PLAN.md`: staged execution plan.
- `refine-logs/EXPERIMENT_TRACKER.md`: run ledger.

## Executable source

- `src/repecg/common/`: PTB-XL access, labels, preprocessing, beat/phase logic,
  exact and Nyström kernels, recurrence operators, and shared models.
- `src/repecg/paper01_recurrence/`: recurrence-operator representation.
- `src/repecg/paper02_kernel_mean/`: kernel-mean phase model.
- `src/repecg/paper03_signature/`: ordered path descriptors.
- `src/repecg/paper04_hankel/`: local Hankel/DMD descriptors.
- `src/repecg/paper05_koopman/`: aligned-observable Koopman operators.
- `src/repecg/paper06_conditional/`: conditional residual recurrence.
- `src/repecg/paper07_operator/`: continuous measurement operators.
- `src/repecg/paper08_tokens/`: uncertainty-bounded token equivalence.
- `scripts/audit_cohort.py`: read-only metadata and patient split audit.
- `scripts/build_smoke_cache.py`: non-scientific 128-record invariant cache.
- `scripts/smoke_train_kme.py`: non-scientific GPU execution smoke.

## Verification

- `tests/test_common.py`: real-data firewall, cohort, phase, kernel, recurrence,
  and streaming-normalization contracts.
- `tests/test_branches.py`: shape and mechanism checks for Papers 3--8.
- `.aris/compute/ptbxl-distributional-repecg-env-spec.json`: environment spec.
- `.aris/compute/local.md`: reproducible invocation and independent witness.

## Runtime evidence

- `outputs/m1_cache_128/manifest.json`: 128-record cache provenance and QC.
- `outputs/m1_gpu_smoke.json`: Paper-2 versus moment-control execution smoke.
- `outputs/m1_gpu_smoke.log`: smoke stdout/stderr.

## Explicitly incomplete

- The eight independently runnable paper directories and their complete stage
  commands are not yet present.
- Full development representations, raw baselines, Nyström fidelity audit, and
  branch gates have not run.
- No configuration is frozen for final training; fold 10 remains locked.
- No smoke result is eligible for a scientific claim.
