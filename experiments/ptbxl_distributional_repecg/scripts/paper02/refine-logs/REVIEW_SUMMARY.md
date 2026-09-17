# Multi-Round Adversarial Review Summary: Paper 02
# Phase-KME: Phase-Conditioned Kernel Mean Embeddings for Distribution-Valued ECG Representation

## Review Round 1: Critical Evaluation

### Summary of Proposal
The authors propose **Phase-KME**, representing 12-lead ECGs as a sequence of 16 phase-conditioned Kernel Mean Embeddings (KMEs) using an Inverse Multiquadric (IMQ) kernel approximated via $M=256$ Nyström landmarks, processed by a 3-block 1D CNN with circular padding ($S^1$ topology).

### Primary Criticisms & Collision Audit

#### 1. Collision with KME Prior Art in ECG
**Reviewer Objection**: The authors frame KME for ECG as a novel contribution. This is factually incorrect. A 2025 Scientific Reports paper (*Robust screening of atrial fibrillation with distribution classification*, PMC12283955) already represents patient ECG beat ensembles using KME into an RKHS and classifies them with an SVM. What is the actual technical novelty beyond this prior work?
- **Author Response**: The 2025 Scientific Reports paper constructs a single **global** KME: $\{x_1, \dots, x_N\} \to P_{\text{global}} \to \mu_{P_{\text{global}}}$, completely discarding cardiac phase chronology and waveform shape. Our contribution is the **phase-conditioned distribution field**: $\theta \mapsto \mu_{P(X \mid \theta)}$ across $C_{16}$. To directly isolate this, we introduce `global_kme` as a mandatory baseline. If `full` outperforms `global_kme`, the diagnostic value of localized phase conditioning over prior art is conclusively established.

#### 2. Collision with Cardiac Equivariance Prior Art (Winder, Aug 2026)
**Reviewer Objection**: The claim of "exploiting circular cardiac topology" collides directly with the August 2026 Winder preprint (arXiv:2608.21147), which derives phase-equivariant representations on PTB-XL using harmonic transport operators. 
- **Author Response**: We acknowledge Winder (2026). Our contribution is NOT "the cardiac cycle is circular." It is the **distribution-valued** cardiac phase field $\mu: C_{16} \to \mathcal{H}_k$. Furthermore, we downgrade the mathematical terminology from continuous $S^1$ to discrete cyclic group $C_{16} \cong \mathbb{Z}_{16}$, which accurately describes 16-bin circular convolution.

#### 3. Confounded Linear Probe
**Reviewer Objection**: The original linear probe averaged the 16 KME vectors into $\frac{1}{16}\sum_g \mu_g \in \mathbb{R}^{256}$. This removes both nonlinear architecture AND phase organization simultaneously, confounding the ablation.
- **Author Response**: We bifurcate the probe into:
  1. `phase_kme_linear_probe`: Linear classifier on the flattened sequence $z = \operatorname{vec}([\mu_0, \dots, \mu_{15}]) \in \mathbb{R}^{4096}$ with ridge regularization, strictly testing representation linearity while preserving all 16 phase bins.
  2. `global_kme_linear_probe`: Linear classifier on $\bar{\mu} \in \mathbb{R}^{256}$, isolating the phase-free KME prior-art baseline.

#### 4. The "Exact Gaussian" Flaw
**Reviewer Objection**: Calling the finite recoloured sample an "exact Gaussian distribution" is mathematically invalid. Once samples are conditioned on empirical sample moments matching target sample moments, they are no longer i.i.d. draws from a Gaussian.
- **Author Response**: We rephrase and lock the control as a **moment-matched Gaussian surrogate**. The algorithm:
  $$Z_i \sim \mathcal{N}(0, I) \implies \tilde{Z}_i = S_Z^{-1/2}(Z_i - \bar{Z}) \implies Y_i = \bar{X} + S_X^{1/2}\tilde{Z}_i$$
  achieves $\bar{Y} = \bar{X}$ and $S_Y = S_X$ to $< 10^{-8}$ numerical precision. It serves as an unyielding destroyer control: if $\mu_P$ beats $\mu_Y$, the performance gain cannot be attributed to mean or covariance.

#### 5. Sample-Size Identifiability on 10-Second ECGs
**Reviewer Objection**: A 10-second PTB-XL record has only $B \sim 8-12$ beats. KME estimation error scales as $\mathcal{O}(B^{-1/2})$. Can a 16-landmark IMQ KME reliably distinguish non-Gaussian distributions at $B=8$?
- **Author Response**: We introduce **Synthetic World E (Sample-Size Identifiability Power Curve)** evaluating $B \in \{2, 3, 4, 5, 6, 8, 10, 12, 16, 32\}$ and measuring empirical discrimination power $\Pr[D(P, Q) > D(P, P')]$. On real data, we stratify test evaluation by beat count ($2-4, 5-7, 8-10, >10$) and report AUROC$(B)$ and split-half stability $S_B$.

---

## Review Round 2: Re-evaluation & Final Verdict

### Assessment of Revisions
1. **Mathematical Terminology**: Renaming to **Phase-KME** and framing the domain as discrete $C_{16}$ is mathematically clean, rigorous, and truthful.
2. **Claim Matrix**: The 12-variant claim-identification matrix cleanly decouples phase conditioning (`global_kme`), higher-order shape (`gaussian_surrogate`), linear signal averaging (`linear_kernel`), circular boundaries (`no_circular`), and architecture (`phase_kme_linear_probe`).
3. **Synthetic Proofs**: The 5 synthetic worlds (A: matched mean/cov non-Gaussian separation, B: Gaussian surrogate precision, C: $C_{16}$ discrete equivariance/invariance, D: Nyström fidelity gate, E: sample-size power curve) provide verifiable mathematical grounding.

### Verdict
**ACCEPT WITH LOCKED CLAIM MATRIX**.
Proceed to synthetic test implementation, variant registry update, and smoke verification.
