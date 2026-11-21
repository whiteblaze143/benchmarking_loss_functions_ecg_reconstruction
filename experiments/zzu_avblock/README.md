# ZZU-ZUA AV-block baseline stage

Run from `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction`.
Reuse `.aris/compute/hrs2027-a100-env-spec.json` and `.aris/compute/local.md`.
No package installation is required. The isolated cuDNN layer is required for
actual convolution kernels; shared cuDNN 9.20 failed in prior witnesses.

Exact preflight invocation (does not train a task classifier or touch test data):

```bash
/home/mithunmanivannan/.venv/bin/python -m unittest discover -s experiments/zzu_avblock -p test_protocol.py -v
hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/mithunmanivannan/.venv/bin/python experiments/zzu_avblock/baseline.py --output /data/mithunmanivannan/codex_artifacts/zzu_avblock_20261120 --smoke
```

After `prepare.py --output ROOT`, run `refine_crop_eligibility.py --output ROOT` to verify actual nested crop flatness and inspect the final contract. Then execute
the same baseline invocation without `--smoke` to train one M0 full-12-lead
10-second classifier. It refuses to overwrite an existing held-out prediction
ledger. `evaluate.py --output ROOT` reproduces analyses from saved predictions;
it does not rerun waveform inference, but refits the fixed metadata-only control
on training rows. This stage returns one baseline before the larger run grid.

The primary common population requires >=30 actual seconds, all 12 leads nonflat,
in the actual first10s prefix, and finite data. Native units are uV; cached model inputs are uV/1000 = mV, still
500 Hz. First prefixes define paired duration views; no tail padding. Secondary
configuration eligibility retains explicit selected-lead failures. Groups unite
source normal-ID strings conservatively and near-identical observed waveforms.
They are not verified patients. Near-duplicate retrieval is approximate and cannot
exclude every offset or repeated patient. Complete parameters are in the frozen
`clinical_cohort_contract.json`, written before any training or held-out scores.

M1 must reuse the existing OperatorSetModel/Phase-KME mathematics with a training-only
ZZU fit and observed-channel-only peak detection. The old EchoNext convenience
extractor invents intervals on detection failure and must not be used unchanged.
M2/M3, external-endpoint testing and physiological perturbations remain gated until
the first baseline and original M1 are reported. No QT reconstruction is planned.
