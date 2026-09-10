# Geometry-specific latent completion experiment tracker

Created: 2026-09-10 17:30:54 UTC

## State

- Phase: B0/B1 implementation and preflight.
- Neural training: blocked by the preregistered geometry-specificity oracle gate.
- PTB-XL fold 10: sealed.
- EchoNext waveform evaluation: sealed.

## Frozen decisions

- Plan: `EXPERIMENT_PLAN_20260910_173054.md`.
- Trunk: `refine-logs/lean_abl2/runs/lean2_T_patch10_s42_l0/best.pt`.
- Trunk checkpoint SHA-256: `25fef94eda49c5ce853f226fba08e95824a09123dbcad3d12f0e2499010fc602`.
- PCA fit split: PTB-XL folds 1–8 only.
- Selection/evaluation split: PTB-XL fold 9 only.
- Random bases/bootstrap seed: 20260910.
- Random bases: 100; bootstrap replicates: 10,000.

## Evidence log

- 2026-09-10: Existing oracle audited. It compares constrained VCG to the older A0 model, uses all 11 missing channels as primary metrics, and lacks PCA/random-subspace controls; it is retained as historical evidence and cannot authorize neural training.
- 2026-09-10: Local A100 preflight passed; `.venv` reports PyTorch 2.6.0+cu124 and CUDA available. No environment rebuild is required.
- 2026-09-10: No geometry-specificity oracle or latent neural job was running at audit time.
- 2026-09-10: End-to-end two-batch smoke completed. It found source limb-identity residuals of 0.0010–0.0015 mV, consistent with 0.001 mV storage quantization. The provisional 0.0001 mV audit threshold was invalid because it was below source resolution; before any full run it was transparently amended to 0.002 mV. Scientific gates were not changed. Five geometry unit tests pass.

## Next milestone

Implement and smoke-test the fold-1–8 streaming PCA fit, fold-9 limb audit, G0/P0/R0/T0 evaluator, independent patient metrics, clinical panel, and deterministic artifact writer. Then run B1 and record the gate before changing neural code.
