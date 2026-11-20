# Local Compute Environment Ledger

### env: hrs2027-a100-cudnn93

- how: existing `/home/mithunmanivannan/.venv`; isolated `nvidia-cudnn-cu12==9.3.0.75` under `/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93`; shared packages unchanged
- spec: `.aris/compute/hrs2027-a100-env-spec.json`
- tier: `{cpus: 8 available, mem_gib: 15, gpus: 1 x A100-PCIE-40GB}`
- gotcha: shared cuDNN 9.20 raises `ptrDesc->finalize()` for both frameworks; prefer isolated cuDNN 9.3 through the loader path. pip temporary files must use `/data` because the root filesystem has limited space.
- validated: seeded PyTorch Conv1d forward/backward and TensorFlow Conv1D on A100 passed; fresh agent ran documented invocation verbatim, exit 0, including official RawECGNet real ECG inference witness (64 predictions, 2.65 seconds); no divergence
- documented invocation: `hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:'); LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/mithunmanivannan/.venv/bin/python scripts/verify_hrs2027_a100.py`

### env: ptbxl-distributional-repecg@fc49e470

- how: warm reuse of `/home/mithunmanivannan/.venv`; added `iisignature==0.24` and editable project package with `--no-deps`
- spec: `.aris/compute/ptbxl-distributional-repecg-env-spec.json`
- tier: `{cpus: affinity-visible, mem_gib: host-available, gpus: 1 x A100-PCIE-40GB}`
- versions: Python 3.12.3; torch 2.6.0+cu124; CUDA 12.4; numpy 2.3.5; iisignature 0.24
- validated: 2026-09-16 tier-1 imports and seeded A100 kernel witness `WITNESS_REPECG (8, 8) NVIDIA A100-PCIE-40GB`; fresh agent-follows-doc passed (11 tests, exact documented invocation, no divergence)
- gotcha: `iisignature` must build after NumPy with `--no-build-isolation`; editable install must use `--no-deps` or pip attempts to re-resolve the full CUDA wheel stack.
- documented invocation: `cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction && /home/mithunmanivannan/.venv/bin/python -m pytest -q experiments/ptbxl_distributional_repecg/tests && /home/mithunmanivannan/.venv/bin/python -c 'import torch;torch.manual_seed(0);x=torch.randn(8,8,device="cuda");print("WITNESS_REPECG",tuple((x@x).shape),torch.cuda.get_device_name())'`

### env: native-nef-a0@49c6ae5b

- how: warm reuse of `/home/mithunmanivannan/.venv`; no environment rebuild
- spec: `.aris/compute/native-nef-a0-env-spec.json`
- tier: `{cpus: affinity-visible, mem_gib: host-available, gpus: 1 x A100-PCIE-40GB}`
- versions: Python 3.12.3; torch 2.6.0+cu124; CUDA 12.4; numpy 2.3.5
- weights: strict A0 SHA-256 `fdf6e997...a5acf`; FULL-e13 SHA-256 `d5d9197e...ef054`
- validated: 2026-09-15 real fold-9 batch-32 BF16 forward/backward; identity, shared initialization, gradients, output shape, and hidden-target gates passed
- gotcha: cuDNN 9 Conv1d raises `ptrDesc->finalize()`; production trainer disables cuDNN while retaining CUDA ATen execution. Exact batch-32 preflight requires BF16 autocast and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### env: ecgaim-rdb-oracle@d4f03f36

- how: warm reuse of `/home/mithunmanivannan/.venv`; no environment rebuild performed
- spec: `.aris/compute/rdb-oracle-env-spec.json`
- tier: `{cpus: 8 affinity-visible, mem_gib: 15, gpus: 0 used}`
- versions: Python 3.12.3; torch 2.6.0+cu124; numpy 2.3.5; openpyxl 3.1.5; tmux 3.4; util-linux taskset 2.39.3
- weights: project checkpoint-store resolver and `checkpoints/`; smoke restored `factorial_ecg_aim_1000000_s42`
- validated: 2026-08-22 tier 1 imports, seeded CPU witness, and fresh agent-follows-doc smoke passed
- gotcha: the process affinity is currently 0-7; hard-coding 8-11 fails. Production launcher now inherits the caller's allowed CPU set.

### env: ecgaim-rdb-oracle@4016318b

- how: warm reuse of `/home/mithunmanivannan/.venv`; no environment rebuild performed
- spec: `.aris/compute/rdb-oracle-env-spec.json`
- tier: `{cpus: 7 selected from caller affinity, mem_gib: 15, gpus: 0 used}`
- versions: Python 3.12.3; torch 2.6.0+cu124; numpy 2.3.5; openpyxl 3.1.5; tmux 3.4; util-linux taskset 2.39.3
- weights: project checkpoint-store resolver and `checkpoints/`
- validated: 2026-08-22 tier 1–2 warm reuse plus fresh agent-follows-doc handoff preflight; exact seven-core witness `0,1,2,3,4,5,6`
- gotcha: select the first seven CPUs from `os.sched_getaffinity(0)`; do not assume a particular CPU numbering scheme.

### env: semiseg-ludb-cpu@e394dc37

- how: warm reuse of `/home/mithunmanivannan/.venv` with 65 MiB of no-dependency packages layered through `external/semiseg/runtime_deps`; shared environment unchanged
- spec: `.aris/compute/semiseg-ludb-cpu-env-spec.json`
- tier: `{cpus: 7 selected from caller affinity, mem_gib: 15, gpus: 0 used}`
- versions: Python 3.12.3; torch 2.6.0+cu124; numpy 2.3.5; torchmetrics 1.5.2; tensorboard 2.21.0
- validated: 2026-08-22 tier 1 imports, seeded seven-core CPU witness `WITNESS (8, 8) 7 False`, and fresh agent-follows-doc `WITNESS_SEMISEG_MT_CPU` passed with no functional divergence
- gotcha: the vendor Mean Teacher loop calls `torch.cuda.synchronize()` unconditionally; `SEMISEG_CPU_ONLY=1` activates a scoped sitecustomize no-op before vendor imports.

### env: wavelet-rdb-ssl@2fccd6d0

- how: warm reuse of `/home/mithunmanivannan/.venv`; no environment rebuild performed
- spec: `.aris/compute/wavelet-rdb-ssl-env-spec.json`
- tier: `{cpus: affinity-visible, mem_gib: 15, gpus: 1 x A100-PCIE-40GB after dependency gate}`
- versions: Python 3.12.3; torch 2.6.0+cu124; numpy 2.3.5; tmux 3.4
- data: immutable RDB cache contract `81d69552522ae2b86116e33f578cbdf5965214f284c5092523d19660acefda83`; frozen train/validation/test = 1678/360/360
- validated: 2026-08-22 tier 1 CPU self-tests and queue/cache tests; tier 2 GPU validation is an enforced 10-job preflight after the spatial queue releases the GPU
- gotcha: the existing spatial wrapper would otherwise resume the 3-architecture queue; the persistent priority barrier plus held 3-architecture lock enforce spatial -> wavelet -> 3-architecture ordering.

### env: ludb-semiseg-blinded@5ad939e6

- how: warm reuse of `/home/mithunmanivannan/.venv` plus `external/semiseg/runtime_deps`; no environment rebuild performed
- spec: `.aris/compute/ludb-semiseg-blinded-env-spec.json`
- tier: `{cpus: 0-5 pinned, mem_gib: 15, gpus: 0}`
- checkpoint: validation-selected `best-MeanIoU.pth:model_ema`, SHA-256 `b404e0dbe198b7f5d4961b7a5fe04a2bd16d281a9a030a3db2ba4996ce696fb4`
- validated: 2026-08-22 13 tests, test-only two-record/one-model execution smoke, compact SQLite integrity, and fresh agent-follows-doc launch audit passed
- gotcha: CPU affinity is not exclusive; the active spatial trainer and its workers retain affinity to all eight cores.
