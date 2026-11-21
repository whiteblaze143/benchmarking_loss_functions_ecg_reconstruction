# Locked Lead-II comparison

Run from `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction`.
Reuse `.aris/compute/local.md` environment `hrs2027-a100-cudnn93`; no installation.

Preflight (synthetic tensors only; verifies both actual GPU models and gradients):

```bash
hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/mithunmanivannan/.venv/bin/python experiments/zzu_avblock/leadii.py smoke
```

Expect `LEADII_KERNEL_GRADIENT_WITNESS M0` and `LEADII_KERNEL_GRADIENT_WITNESS M1`, finite losses, exit0. Source cohort and prior full12 artifacts are read-only. New outputs go to `/data/mithunmanivannan/codex_artifacts/zzu_leadii_20261120`.
Use the same environment prefix and replace `smoke` by `prepare`, then `M0`, then `M1` for execution. Preparation freezes source hashes; do not edit sources after freezing. Predictions cannot be overwritten. Kernel fitting uses training only. Failed M1 extraction remains indeterminate; completed-only ranking metrics must be labeled. No M3 training is authorized by this run document.
