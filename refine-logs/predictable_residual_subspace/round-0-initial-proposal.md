# Research Proposal: Predictable Residual Subspace Completion

## Problem Anchor

- Bottom-line problem: Determine the smallest low-dimensional residual subspace that preserves clinically relevant missing-lead ECG information and how much of that subspace is predictable from Lead I.
- Must-solve bottleneck: A PCA oracle measures representational capacity using true missing leads; it does not establish that the corresponding latent coefficients can be inferred from Lead I.
- Non-goals: No VCG neural model, probabilistic generator, new loss, architecture sweep, 3-epoch screen, latent supervision, or external-test access.
- Constraints: PTB-XL folds 1–8 fit/train, fold 9 select/evaluate, fold 10 and 4,097 pristine EchoNext ECGs sealed; existing T_patch10 trunk; one A100; seed 42 and 15 epochs first.
- Success condition: Freeze rank from a closed-form gate, then show with paired patient evidence whether fixed or learnable low-rank output geometry materially improves over a matched direct decoder without clinical harm.

## Technical Gap

The rank-2 PCA oracle reaches 0.9427 independent correlation, but its latent coefficients are computed from the targets. Fixed VCG is meaningful population structure (98th percentile of random rank-2 bases) yet is 0.0756 below PCA. The missing mechanism is therefore neither a fixed physiological projection nor an oracle PCA bottleneck: it is a compact output subspace whose orientation may need to reflect conditional observability from Lead I.

## Method Thesis

The smallest adequate intervention is to replace only the direct seven-channel output head with a rank-`K_STAR` residual decoder and test whether a QR-constrained learned subspace improves upon both direct regression and a frozen PCA subspace.

## Contribution Focus

- Dominant contribution: predictive residual subspace learning—jointly orienting a tiny orthonormal output subspace toward residual structure learnable from Lead I.
- Supporting contribution: separating representational rank from conditional predictability with a closed-form oracle gate.
- Explicit non-contributions: no physiological interpretation of individual coordinates and no claim that PCA-oracle quality implies reconstructability.

## Proposed Method

### Complexity budget

- Frozen/reused: T_patch10 architecture, preprocessing, balanced waveform/clinical objective, optimizer, scheduler, data order, augmentation, and validation aggregation.
- New: one `7 × K_STAR` matrix in C2; thin QR supplies an exactly orthonormal basis.
- Excluded: orthogonality penalty, manifold framework, VCG decoder, diffusion, uncertainty, extra losses.

### System and training

`Lead I → shared trunk → output head → seven independent missing leads`.

- B1 emits seven signals directly.
- C1 emits `K_STAR` coefficients and reconstructs with frozen folds-1–8 `c` and PCA basis.
- C2 emits the same number of coefficients and reconstructs with frozen `c` plus the Q factor of a learned `7 × K_STAR` parameter.

All systems optimize the same objective in reconstructed signal space on `[II,V1,…,V6]`. C1 receives no coefficient supervision. C2 is identified and interpreted only through its column space.

## Claim-driven validation sketch

1. Rank gate: ranks 1–6; choose the smallest rank satisfying the frozen energy/saturation rule.
2. Seed-42 main comparison: B1/C1/C2, identical initialization and training protocol; paired patient bootstrap and frozen clinical-harm screen.
3. Conditional replication: seeds 43/44 only if C1 or C2 materially beats B1.

