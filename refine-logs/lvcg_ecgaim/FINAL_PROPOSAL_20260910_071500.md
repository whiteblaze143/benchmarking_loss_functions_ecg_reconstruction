# Research Proposal: Beat-Structured ECG-AIM

**Status:** READY FOR PILOT, NOT READY FOR A PAPER CLAIM  
**Date:** 2026-09-10

## Problem Anchor

- **Bottom-line problem:** Reconstruct the eleven missing clinical ECG leads from wearable Lead I while learning a compact record representation useful beyond waveform reconstruction.
- **Must-solve bottleneck:** The present ECG-AIM encoder is rhythm-informative but morphology-duration-poor, while target theta codes act mainly as output-slot identifiers rather than a physical cardiac latent.
- **Non-goals:** Do not claim identifiable VCG recovery from one lead; do not replace the proven ECG-AIM reconstructor; do not modify LVCG source; do not add TTT, diffusion, or a new peak detector in the first experiment.
- **Constraints:** PTB-XL, 500 Hz, 5000 samples, strict Lead I input, existing patient-disjoint splits and evaluation, no LVCG checkpoint, and an incomplete upstream package release.
- **Success condition:** On the same held-out records, the new embedding improves at least one prespecified morphology endpoint without losing rhythm utility, while reconstruction remains unchanged by construction in the representation-only pilot.

## Method Thesis

Attach LVCG's unchanged beat encoder and global RR encoder to ECG-AIM through a learned one-to-three-channel latent-field adapter, then concatenate masked-mean beat morphology, GRU dynamics, and RR rhythm into an explicit record embedding.

The dominant contribution is the **controlled separation of structure, dynamics, and rhythm for strict one-lead ECG-AIM**, not physical VCG inversion. The supporting contribution is a clean test of whether beat structure repairs the morphology weakness of generic encoder pooling.

## Complexity Budget

- Reused unchanged: current ECG-AIM reconstruction model; LVCG `BeatEncoder`, `BeatDecoder`, and `GlobalRREmbedding`.
- New trainable components: one 1x1 Conv1d mapping Lead I to a three-channel learned field; one single-layer GRU.
- New non-trainable interface: explicit beat boundaries and left-aligned validity mask supplied by preprocessing.
- Intentionally excluded: LVCG pseudo-inverse, fixed VCG projection, beat decoder, refinement decoder, TTT, wavelets, contrastive projector, and feedback from the new embedding into reconstruction.

## System

```text
Lead I ───────────────► existing ECG-AIM ───────────────► 12-lead reconstruction
   │
   └─► 1x1 learned field ─► boundary/resample ─► original LVCG BeatEncoder
                                                ├─► masked mean ─► structure
                                                └─► 1-layer GRU ─► dynamics
         beat durations ─► original LVCG GlobalRREmbedding ─────► rhythm

                 concat(structure, dynamics, rhythm) ─► explicit embedding
```

Beat boundaries are required. The model fails on missing, malformed, or non-left-aligned beats; it does not silently substitute uniform windows.

## Representation and Training

The default embedding is 640 dimensions: 256 structure, 256 dynamics, and 128 rhythm. Structure uses a masked mean across valid beats rather than LVCG's brittle first-complete-beat choice. Dynamics uses the last valid hidden state of a one-layer GRU. Rhythm reuses LVCG's global RR statistics, sinusoidal sequence encoding, differences, and attention pooling.

Pilot pretraining freezes ECG-AIM and trains the auxiliary branch without diagnostic labels. The original LVCG beat decoder reconstructs each resampled Lead-I beat through a fixed mean over its three decoded channels. A one-step GRU objective predicts the next detached beat state. The embedding loss is beat L1 plus 0.1 times temporal Smooth-L1. Downstream labels are used only by frozen linear probes. This prevents a reconstruction change or task-specific encoder supervision from being mistaken for a general embedding improvement.

## Novelty and Scope

This is not a new VCG method and not a reproduction of LVCG. It is a strict one-lead decomposition study that preserves the existing reconstructor and tests whether LVCG's beat-structured representation transfers when physical inversion is impossible. No foundation-model primitive is necessary.

## Risks

1. **Beat-boundary availability:** PTB-XL training lacks guaranteed R peaks. The pilot must materialize and audit boundaries before training; no model-side fallback is allowed.
2. **Pseudo-field semantics:** the learned three channels may be arbitrary. They must be called a learned latent cardiac field, never VCG.
3. **Rhythm leakage:** duration-derived rhythm may dominate. Component-wise probes and deletion ablations are mandatory.
4. **Upstream completeness:** LVCG's package import fails because tracked code imports omitted `lvcg.data` modules. Only its self-contained beat module is reused.

## Claims Allowed After a Passing Pilot

- C1: Beat-structured decomposition improves morphology information relative to dimension-matched generic ECG-AIM pooling.
- C2: The structure/dynamics/rhythm components contain measurably distinct information under component deletion probes.

No physical-field, view-invariance, cross-site, or reconstruction-improvement claim is permitted from this pilot.
