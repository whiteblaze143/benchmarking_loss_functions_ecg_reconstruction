# Final Research Proposal: SetOperator Continuous Measurement Operator Architecture

## 1. Problem Statement & Anchor
Deep learning models for electrocardiograms typically assume a fixed measurement coordinate frame (such as the standard 12-lead ECG). However, clinical monitoring, ambulatory patches, handheld wearables, and spatial vectorcardiograms measure the cardiac electrical field from nonstandard, reduced, or arbitrarily rotated lead geometries.

SetOperator models ECG acquisition as a variable set of operator-response pairs $\{(q_i, \mu_{q_i})\}_{i=1}^m$, where each continuous measurement operator $q_i \in \mathbb{R}^8$ maps the underlying 8-dimensional cardiac source dipole to a 1D lead signal, and $\mu_{q_i}$ represents the phase-conditioned Kernel Mean Embedding (Phase-KME) in an RKHS.

Our central hypothesis is:
> Explicitly representing the continuous measurement-operator geometry and relative cardiac phase morphology enables the neural network to learn robust, generalizable representations that transfer to unseen, out-of-distribution measurement subspaces.

## 2. Dominant Contribution & The 5 Scientific Pillars
We structure the architectural investigation into five targeted mechanistic comparisons:

1. **Measurement Operator Geometry (Group A)**:
   - Does explicit second-order projective geometry $\gamma(q) = [q, \operatorname{vech}_{\sqrt{2}}(qq^\top)] \in \mathbb{R}^{44}$ improve generalization over a capacity-matched raw-$q$ encoder ($\text{Linear}(8,96) \to \text{GELU} \to \text{Linear}(96,64)$)?
   - How does sign-insensitive projective geometry ($\mathbb{R}^{36}$) compare to signed geometry ($\mathbb{R}^{44}$)?
   - Does squared-Gram attention bias $\beta (q_i^\top q_j)^2$ provide additional benefit on nonorthogonal or redundant measurement configurations?

2. **Phase-KME Distributional Morphology (Group B)**:
   - Our numerical analysis established that the original circular PhaseCNN is strictly invariant to cyclic shifts of cardiac phase ($\Delta < 6 \times 10^{-8}$). We evaluate whether breaking this cyclic invariance via a learned softmax phase weighting $\sum_{p=1}^{16} \alpha_p \cdot \text{conv\_feature}_p$ improves diagnostic discrimination.
   - We ablate phase conditioning (Global-KME), local phase convolutions, circular padding, and phase order to isolate the contribution of cardiac phase.

3. **Cross-View Set Reasoning (Group C)**:
   - We test whether cross-token Transformer self-attention is necessary by comparing against an independent per-measurement token encoder with permutation-invariant pooling.
   - We evaluate learned-query attention pooling versus masked mean pooling and test depth (1 vs 2 vs 4 layers) and attention head count (1 vs 4 heads).

4. **Clinical Modality Attribution (Group D)**:
   - We establish the diagnostic baseline of the 7 clinical tabular covariates alone (Tabular-Only MLP) and isolate pure ECG waveform diagnosis (ECG-Only SetOperator).

5. **Constructive Interactions (Group E)**:
   - We test whether validated operator geometry and phase-position-aware readout combine constructively.

## 3. Rejected Complexity
- No speculative 5-seed sweeps prior to single-seed screening.
- No heavy generative decoders (diffusion, VAE, flow matching).
- No unverified foundation models or external representation shifts.
- All neural components trained jointly from random initialization on EchoNext SHD binary detection.
