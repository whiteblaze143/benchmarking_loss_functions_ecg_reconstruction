# Pipeline Summary

**Problem**: Determine which frozen GRAIL latent should become the reference clinical state.
**Final Method Thesis**: Select the reference using leakage-safe linear accessibility, clinical dimensionality, and complementary-subspace evidence.
**Final Verdict**: READY
**Date**: 2026-09-11

## Contribution snapshot

- Dominant contribution: a frozen, patient-aware representation audit.
- Supporting contribution: residual and cross-representation geometry.
- Explicitly rejected complexity: more encoder training and premature 4,095-subset evaluation.

## First runs

1. Export six frozen representations for folds 1–8.
2. Run deterministic per-concept, PCA, and repeated low-shot probes.
3. Complete geometry, residual-subspace, retrieval, and paired-bootstrap analyses.

## Next action

- Monitor `tmux:grail_r3_audit` and `tmux:grail_r3_probe_queue`.

