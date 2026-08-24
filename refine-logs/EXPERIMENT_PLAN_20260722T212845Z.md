# Experiment Plan: Complete Factorial ECG Loss Benchmark v2

Version: 2026-07-22T21:28:45Z

- Four families by eight correlation/MMD/derivative masks at seed 42.
- Base/full/best-nontrivial at seeds 1337 and 2026 for each family.
- Neutral missing-lead validation MSE/Pearson selection and a cNVAE validity gate.
- Full PTB-XL, ECGFounder, morphology, 17 paired stresses, paired BCa inference, and provenance-gated EchoNext.
- Authoritative controls: `experiment_queue/factorial_v2/manifest.json` and `model_registry.json`.
