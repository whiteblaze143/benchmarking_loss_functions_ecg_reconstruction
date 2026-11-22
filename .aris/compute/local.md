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

### env: af-density-rr-cpu@e1907214

- how: unchanged /home/mithunmanivannan/.venv; no installations or GPU use
- spec: .aris/compute/af-density-rr-cpu-env-spec.json
- tier: CPU 2 threads, GPU 0
- validated: seeded HGB deterministic fit/predict and actual eight-window source-RR feature witness passed; fresh documented-invocation witness pending
- documented invocation: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/mithunmanivannan/.venv/bin/python experiments/af_density/verify_rr_cpu.py`

### env: af-waveform-prep@49e86c3f

- how: unchanged /home/mithunmanivannan/.venv; no package installation
- spec: .aris/compute/af-waveform-prep-env-spec.json
- tier: six CPU processes, one numerical thread each, no GPU
- witness: actual eight-window/sixteen-channel training-only XQRS preflight executed; fresh documented-invocation validation recorded separately
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_PREPROCESSING.md
- documented invocation: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/mithunmanivannan/.venv/bin/python experiments/af_density/extract_observed_waveforms.py --smoke`

### env: af-duration@e41cac93

- how: unchanged /home/mithunmanivannan/.venv, no installs
- spec: .aris/compute/af-duration-env-spec.json
- tier: one CPU process, one numerical thread, GPU 0
- witness: seeded synthetic discrete-lognormal fit passed; fresh documented-command witness pending
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_DURATION_FIT.md
- validated: seeded synthetic witness and fresh /root/af_duration_doc_witness command passed without divergence; real annotation fit and independent replay passed, outputs duration_model_v1/verification.json

### env: af-phase-kernel@acfa0a54

- how: unchanged /home/mithunmanivannan/.venv; no installations
- spec: .aris/compute/af-phase-kernel-env-spec.json
- tier: one CPU process/thread, GPU 0
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_PHASE_KERNEL.md
- validated: seeded actual-training feature-equivalence witness passed; fresh documented-command witness pending
- fresh documented-command validation: /root/af_phase_kernel_doc_witness passed, four training-channel examples,16x128, maximum float32 discrepancy0.0; production fit awaits complete extraction

### env: af-raw-gpu@3d18ce1a

- how: unchanged venv with existing isolated cuDNN9.3 override; no installs
- spec: .aris/compute/af-raw-gpu-env-spec.json
- tier: CPU1, A10040GB,29.988GiB development cache
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_RAW_BASELINE.md
- seeded witness: finite BF16 gradients, FP32 unobserved-channel isolation, batch256 with full-cache reservation, peak31.18GiB; fresh documented-command witness pending
- fresh documented-command witness: /root/af_raw_doc_witness passed, batch256/full-cache reservation,31.184GiB peak; raw train/validation process launched, no test inference

### env: af-observed-rr@c9def370

- how: unchanged venv; no installs
- spec: .aris/compute/af-observed-rr-env-spec.json
- tier: CPU1/thread1, GPU0
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_OBSERVED_RR.md
- seeded witness:32 actual training windows/96 examples,36 float32 features, synthetic labels only; fresh documented-command witness pending
- fresh documented-command witness: /root/af_observed_rr_doc_witness passed,32 training windows/96 acquisition examples,36 float32 columns; production preparation waits for complete extraction

### env: af-phase-cache@584e3f04

- how: unchanged venv and existing isolated cuDNN9.3; no installs
- spec: .aris/compute/af-phase-cache-env-spec.json
- tier:4 spawned CPU prep workers plus1 controller, each numerical thread1; one CUDA float64 feature evaluator,16 channel representations per batch,8 queued chunks
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_PHASE_FEATURES.md
- seeded witness:256 actual training windows, original NumPy agreement0.0, exact commit roundtrip; final fresh documented-command witness pending
- fresh documented-command witness: /root/af_phase_cache_doc_witness passed256 actual training windows, NumPy discrepancy0.0, exact cache replay/commit roundtrip; warm2.0453s under concurrent jobs; production run awaits dependency gates

### env: af-set-gpu@a6046ce7

- how: unchanged venv with existing cuDNN9.3 override, no installs
- spec: .aris/compute/af-set-gpu-env-spec.json
- tier:CPU1/A10040GB; production requires14GiB free GPU
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/RUN_SET_TRAINING.md
- seeded witness:categorical/q-ablated/hybrid BF16 finite gradients and bitwise repeated FP32 logits; fresh documentation witness pending
- fresh documented-command witness: /root/af_set_training_doc_witness passed categorical/q-ablated/hybrid BF16 gradients and repeated FP32 logits; production models await complete phase cache
- raw independent replay: /root/af_raw_replay_doc_witness exact documented full run passed166047 validation windows/498141 predictions, bitwise logits/scores, strict best-checkpoint history minimum and exact threshold reselection; no test waveform

### env: af-set-math-sdp-v2@e3ec8eae

- how: warm reuse unchanged venv and cuDNN9.3; no installs; new explicit MATH attention-dispatch policy
- spec: .aris/compute/af-set-math-sdp-gpu-env-spec.json
- tier:CPU1/A10040GB,14GiB free GPU required for full development training
- documentation: /data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/set_models_math_sdp_v2/RUN_MATH_SDP_V2.md
- gotcha: saved actual q-ablated batch has nonfinite BF16 default-SDP gradients; MATH BF16 and FP32 alternatives finite. All3 corrected variants share MATH.
- validation: parent saved-batch witness executed; fresh agent-follows-doc pending. No test authorization.
- fresh agent-follows-doc: /root/af_math_sdp_doc_witness passed exact documented command;4 finite-gradient cases,0 optimizer updates; parent verified command equality and protocol/witness hashes. No test authorization.
