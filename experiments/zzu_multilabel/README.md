# ZZU-ZUA 15-label first comparison

From `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction`, reuse `.aris/compute/local.md` environment `hrs2027-a100-cudnn93`; no installation.

Exact preflight, synthetic tensors only:

```bash
hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/study.py smoke
```

Expect four `MULTILABEL_GPU_WITNESS` lines, raw/set x full12/II, finite losses and exit0. No cohort or prediction archives are touched by smoke.

Use the same environment prefix for `prepare`, `represent`, then four `train --family raw|set --config full12|II` commands. Execute sequentially on the shared A100. Outputs are in `/data/mithunmanivannan/codex_artifacts/zzu_multilabel_20261120`. Source and prior binary artifacts are read-only. Preparation freezes all source hashes; training refuses changed sources or overwritten predictions. Targets are 15 independent labels, Normal allzero. Source diagnoses are record-level. The old record-group partitions persist; patient independence and an untouched test are not claimed. Future loss/lead/temporal ablations remain gated.
