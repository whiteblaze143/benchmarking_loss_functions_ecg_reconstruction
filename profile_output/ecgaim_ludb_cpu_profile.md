# ECG-AIM LUDB CPU profile

Profiled 2026-08-21 while the ECG-AIM factorial training queue remained active.

## Hardware and live load

| Dimension | Observation |
|---|---:|
| Logical CPUs | 8 (single NUMA node) |
| RAM | 15 GiB, approximately 7.1 GiB available during sampling |
| Evaluator CPU | 533.5% average (about 5.34 cores) |
| Whole-host CPU | 69.0% user, 13.6% system, 17.3% idle |
| Evaluator RSS | approximately 1.7--1.9 GiB at batch 4 |
| Evaluator disk I/O | 0 KiB/s during steady-state inference |
| Disk utilization | 0.1--2.2%, zero material I/O wait |
| GPU use | none; `CUDA_VISIBLE_DEVICES` is empty |
| Minor page faults | approximately 326,000/s during inference |

The workload is CPU/allocation and memory-bandwidth bound, not storage bound.
Running multiple model evaluators would oversubscribe the host and compete with
the active training data loaders.

## Batch-size benchmark

One ECG-AIM checkpoint, 48 fixed LUDB records, six CPU threads:

| Batch | Records/s | Relative to batch 4 |
|---:|---:|---:|
| 2 | 1.089 | -6.2% |
| 4, run 1 | 1.166 | +0.4% |
| 4, run 2 | 1.156 | -0.4% |
| 8 | 1.150 | -0.9% |
| 12 | 1.089 | -6.2% |

Batch 4 is the stable optimum. Larger batches increase resident memory without
increasing throughput.

## Thread-count benchmark

Batch 4, 48 fixed LUDB records:

| Threads | Records/s | Relative to six threads |
|---:|---:|---:|
| 4 | 0.915 | -21.2% |
| 6 | 1.161 | baseline |
| 8 | 0.799 | -31.2% |

Six threads is optimal under concurrent ECG-AIM training. Eight threads loses
throughput because training and evaluation contend for the same eight CPUs.

## End-to-end production timing

The first two full 199-record models completed in 157.2 and 177.5 seconds.
At the observed mean, 115 models require approximately 5.3 hours of inference,
plus checkpoint materialization overhead. The focused daemon is substantially
narrower than the previous pipeline: LUDB only, ECG-AIM only, no downstream
datasets, no linear probes, no detector multiprocessing pool, and one model in
memory at a time. A direct old-versus-new throughput ratio is unavailable
because the old daemon did not record equivalent LUDB per-model timings.

## Recommendation

Keep the production configuration at CPU-only, six threads, batch size four,
CPUs 0--7, and one model at a time. Do not add parallel model workers while
training is active. The next material speedup would require using the GPU after
training finishes or changing/model-compiling the inference implementation,
both of which have higher operational risk than the modest remaining CPU gain.

## Instrumentation changelog

| File | Change type | What was added/modified | Lines |
|---|---|---|---|
| `scripts/profile_ecgaim_ludb_cpu.py` | created | Reusable same-checkpoint CPU batch/thread benchmark | all |
| `profile_output/ecgaim_ludb_cpu_batch_benchmark.json` | created | Raw batch benchmark results | all |
| `profile_output/ecgaim_ludb_cpu_threads4.json` | created | Raw four-thread benchmark result | all |
| `profile_output/ecgaim_ludb_cpu_threads8.json` | created | Raw eight-thread benchmark result | all |
| `profile_output/ecgaim_ludb_cpu_profile.md` | created | This profile report | all |

