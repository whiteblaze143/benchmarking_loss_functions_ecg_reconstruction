# Research Proposal: Theta-repSpat

## Problem Anchor

- Bottom-line problem: detect temporally separated recurrence of cardiac electrical states without making recurrence depend on the incidental recorded lead subset.
- Must-solve bottleneck: the attribute geometry supplied to repSpat must be stable to view availability while retaining physiologically distinct waveform states.
- Non-goals: no claim of SO(3) equivariance, no interpretation of an internal neural vector as cardiac state, and no changes to CAHC, MMD, permutation, BH, or graph construction.
- Constraints: train the exact released Nef-Net v2 GeoVT architecture from scratch on the downloaded PanoBench training split; freeze it before repSpat; never tune it on recurrence outcomes.
- Success condition: cross-view geometry becomes more stable than raw eight-view geometry while positive-control phase separability is not degraded, followed by improved or non-inferior patient-level recurrence behavior.

## Method

Train `nefnet_plus.layer` with Any-Pairs reconstruction on PanoBench. I and II are fixed anchors; a third torso view and a distinct query target are deterministically sampled from the 42 torso views. After the final epoch, freeze the operator.

For downstream ECG input `{I, II, V3}`, query the frozen operator at the five missing standard chest-lead angles and form

`z_t = [I, II, V1_hat, V2_hat, V3, V4_hat, V5_hat, V6_hat]_t`.

Observed coordinates are copied, never reconstructed. `z_t` is the eight-dimensional continuous repSpat attribute and `s_t=(t_seconds,0)` is the temporal domain. Euclidean distance in `z` is a finite canonical-view approximation to distance between angular query-response functions.

## Claims

1. The frozen query operator induces lead-subset-stable temporal geometry without erasing cardiac-state distinctions.
2. That geometry supports patient-level recurrence inference at least as well as Kors VCG while leaving repSpat unchanged.

## Explicit exclusions

No learned query grid, purity cutoff, effect threshold, dense sphere, spherical harmonics, new kernel, Gate-3 fine-tuning, or result-dependent hyperparameter selection is permitted in v1.
