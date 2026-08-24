# Wavelet SSL throughput profile (2026-08-23)

## Hardware and measured headroom

| Dimension | Observation |
|---|---|
| GPU | NVIDIA A100 PCIe, 40 GiB, 250 W limit |
| CPU | 8 Cascadelake vCPUs, one hardware thread each |
| RAM | 15 GiB total; about 9 GiB available while profiling |
| Disk | 96 GiB volume; about 16 GiB free |
| Batch 16 peak | 14.88 GiB for the heaviest measured BYOL/cross-attention preflight |
| Batch 32 peak | 28.41 GiB for the heaviest measured BYOL/cross-attention preflight |
| Batch 32 active sample | 89% GPU utilization, about 200 W |

## Applied throughput changes

1. Increased reconstruction and delineation batch sizes from 16 to 32.
2. Increased DataLoader workers from 2 to 4, leaving capacity for the trainer and OS.
3. Reduced the serial queue's idle-GPU stability interval from 30 seconds to about 4 seconds between jobs; two independent samples are still required.
4. Retained one job per GPU. Concurrent training would compete for the same 40 GiB device and is not supported by the measured memory peaks.

Batch 48 was rejected: linear extrapolation from the measured 28.41 GiB batch-32 peak would leave inadequate safety margin on a 40 GiB device.

## Instrumentation changelog

| File | Change type | What was added/modified | Lines |
|---|---|---|---|
| `profile_output/wavelet_ssl_1110000/PROFILE.md` | created | Hardware observations, measured peaks, and applied decisions | all |

No runtime instrumentation was inserted into the trainer. Measurements came from completion summaries, `nvidia-smi`, `lscpu`, `free`, and live queue state.
