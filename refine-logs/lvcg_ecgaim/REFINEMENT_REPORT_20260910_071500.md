# Refinement Report

## Starting Idea

Use LVCG to improve the latent space of ECG-AIM while retaining as much upstream code as possible.

## Refinements Applied

1. Rejected literal one-lead VCG inversion because the measurement matrix has rank at most one.
2. Narrowed the contribution to an auxiliary representation branch so reconstruction is held fixed.
3. Reused only independently importable upstream components after verifying the released LVCG package is incomplete.
4. Replaced LVCG's first-beat structural readout with masked mean pooling to reduce early-beat brittleness.
5. Made beat boundaries explicit required inputs and prohibited silent segmentation fallbacks.
6. Limited new trainable machinery to a 1x1 field adapter and one-layer GRU.
7. Rejected downstream-supervised encoder training after identifying probe leakage; added self-supervised beat reconstruction and next-state prediction using LVCG's unchanged BeatDecoder.

## Planning Gate

- Final thesis: explicit beat structure/dynamics/rhythm decomposition can repair ECG-AIM's morphology-poor generic pooling.
- Dominant contribution: controlled one-lead representation decomposition.
- Rejected complexity: physical pseudo-inversion, TTT, decoder replacement, joint training in the pilot.
- Validation concern: distinguish useful structure from added capacity or RR leakage.
- Frontier primitive: absent and unnecessary.

## Implementation

- `unified_latents/engineering/models/ecg_aim_lvcg_variant.py`
- `tests/test_ecg_aim_lvcg_variant.py` (including self-supervised gradient flow and real ECG-AIM bitwise equivalence)
- Upstream `external/LVCG/lvcg/models/blocks/beat_modules.py` remains unchanged.
