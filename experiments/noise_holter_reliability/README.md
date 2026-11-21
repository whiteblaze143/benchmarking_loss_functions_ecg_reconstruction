# Noise Holter admission audit

Artifacts: `/data/mithunmanivannan/codex_artifacts/noise_holter_reliability_20261120`.
This is a retrospective existing-label audit. No training or new reader labels.
Phase 2 failed scientific admission; phases 3–5 must not run from this configuration.
Successful TensorFlow execution does not establish correct score semantics.

## Reproduce the semantic witness

Run from `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction`:

```bash
hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 /home/mithunmanivannan/.venv/bin/python experiments/noise_holter_reliability/verify_export_semantics.py --repository /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction --output /data/mithunmanivannan/codex_artifacts/noise_holter_reliability_20261120
/home/mithunmanivannan/.venv/bin/python -m unittest discover -s experiments/noise_holter_reliability -p test_contracts.py -v
/home/mithunmanivannan/.venv/bin/python experiments/noise_holter_reliability/finalize_admission.py --output /data/mithunmanivannan/codex_artifacts/noise_holter_reliability_20261120
```

The first command uses saved independent-development fixtures, not Noise Holter
ECGs, and reproduces sigmoid(native sigmoid probability) for both exports.
It updates the failure proof and contracts; the last command refreshes hashes.
The native-head function is a diagnostic repair candidate, without an admitted
standalone threshold. The original checkpoints are never modified.

The shared Python environment is reused. cuDNN 9.3 is task-local because the
shared cuDNN layer failed real kernels; see `.aris/compute/local.md` and
`.aris/compute/hrs2027-a100-env-spec.json`. The RR extractor's separate local
WFDB library and source hashes are recorded in the artifact `author_extractor/`
and `configs/extractor_environment_spec.json`; no shared package was installed.

## Rebuild cohort artifacts

Use `build_cohorts.py --release-zip PATH --expected-sha256 SHA --output ROOT`,
then `verify_models.py --repository REPO --output ROOT`, the semantic witness
above, `observed_rr_eligibility.py --output ROOT --workers 4`, and
`finalize_admission.py --output ROOT`, in that order. The release SHA is
`0de40887ee182f6e7556e4cc9daebc772637018f9d455020ec43202710a93326`.
The release is retained in the preceding HRS decision artifact as
`official_release.zip`. The RR census uses observed signal peaks, never source
annotation peaks. Its availability counts do not certify publication-equivalent
preprocessing. The temporary standard-lead transport tag needed by the C
utility does not establish the anatomical identity of A/B.

See `reports/feasibility_report.md`, `reports/hrs_decision.md` and
`manifests/model_admission.csv` for scope, exclusions and stopping reasons.
