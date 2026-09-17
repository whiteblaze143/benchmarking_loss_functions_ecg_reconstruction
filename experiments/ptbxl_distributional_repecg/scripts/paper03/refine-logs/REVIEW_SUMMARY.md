# Multi-Round Adversarial Review Summary: Paper 03
# Phase-Path-Signature: Distributional Path Signatures on Cardiac Phase Trajectories

## Review Round 1: Critical Evaluation

### Primary Criticisms & Audit Points

#### 1. Collision with PathFusion-Net Prior Art
**Reviewer Objection**: Path signatures for ECG classification already exist. *PathFusion-Net* (PubMed 41196785) combines rough path theory, path signatures, CNNs, and LSTMs for ECG arrhythmia classification. What is the actual technical novelty?
- **Author Response**: PathFusion-Net applies signatures to unaligned, global, or sliding-window 1D ECG signals into generic deep models: $\gamma \to \operatorname{Sig}(\gamma) \to \text{CNN/LSTM}$. Our contribution is the **distribution-valued cardiac phase field**:
  $$\theta \longmapsto \mu_{P(\operatorname{LogSig}(\gamma_{b, \theta}))} \in \mathcal{H}_{k_{\text{sig}}}$$
  We introduce mandatory baselines `global_signature` (PathFusion-style) and `whole_beat_signature` to isolate the exact hierarchy:
  $$\text{global path} \longrightarrow \text{beat path} \longrightarrow \text{phase-local path} \longrightarrow \text{distribution of phase-local paths}$$

#### 2. The Reparameterization Invariance Bug ($ar{\gamma}$)
**Reviewer Objection**: The plan claims the 228-D descriptor is invariant to monotone time reparameterization. This is mathematically false. The descriptor appends $\bar{\gamma} = \int_0^1 \gamma(t) dt$. Under non-linear monotonic warping $\phi(t)$, $\int_0^1 \gamma(\phi(t)) dt \neq \int_0^1 \gamma(t) dt$. Therefore the descriptor is NOT invariant.
- **Author Response**: Crucial catch. We decouple the descriptor:
  $$d^{\text{sig}} = [\gamma(0), \gamma(1), \operatorname{LogSig}_3(\gamma)] \in \mathbb{R}^{220}$$
  and
  $$d^{\text{aug}} = [d^{\text{sig}}, \bar{\gamma}] \in \mathbb{R}^{228}$$
  Formal invariance applies strictly to $d^{\text{sig}}$ (`signature_pure`). We run `signature_pure` vs `signature_plus_mean` to test whether $\bar{\gamma}$ provides any empirical benefit.

#### 3. Start Point & Translation Invariance
**Reviewer Objection**: Path signatures depend purely on increments $d\gamma$ and are translation invariant: $S(\gamma + c) = S(\gamma)$. Appending $\gamma(0)$ breaks translation invariance to restore baseline voltage.
- **Author Response**: We explicitly formalize the representation as a dual-component vector:
  $$\boxed{\text{absolute morphology } (\gamma(0), \gamma(1)) + \text{translation-invariant path geometry } (\operatorname{LogSig}_3(\gamma))}$$
  and include ablations isolating pure $\operatorname{LogSig}$ vs $[\gamma(0), \operatorname{LogSig}]$ vs $[\gamma(0), \gamma(1), \operatorname{LogSig}]$.

#### 4. Imprecise Terminology (VCG & Conduction Chirality)
**Reviewer Objection**: Calling $(V_1, V_5)$ trajectories "vectorcardiograms" is inaccurate, as they are scalar projections, not orthogonal physical dipoles. Calling Lévy area "conduction chirality" claims invasive tissue-level conduction direction that is unproven.
- **Author Response**: Corrected across all documentation:
  - "Vectorcardiographic loop" $\to$ **"multilead lead-space trajectory"** or **"planar ECG trajectory in $(V_1, V_5)$ coordinates"**.
  - "Conduction chirality" $\to$ **"trajectory orientation/chirality"**.

#### 5. Missing Ablation: `phase_mean_signature`
**Reviewer Objection**: Without comparing the KME embedding against a simple beat-averaged signature $\bar{s}_g = \frac{1}{B}\sum_b s_{b,g}$, the necessity of the Kernel Mean Embedding is completely unproven.
- **Author Response**: Added `phase_mean_signature` as a mandatory core ablation under an identical PhaseCNN architecture. Comparing `full` vs `phase_mean_signature` directly tests: *Does beat-to-beat distributional dispersion matter, or is the mean path signature sufficient?*

#### 6. Discretization, Sampling Rate, and Tree-Like Ambiguity
**Reviewer Objection**: High-order iterated integrals are sensitive to discretization. Path signatures also fail to distinguish tree-like excursions ($\gamma * \eta * \eta^{-1}$).
- **Author Response**:
  1. We implement a sampling rate stability test ($500 \to 250 \to 100$ Hz) evaluating degradation across levels $L_1, L_2, L_3$.
  2. We add Synthetic World F demonstrating that tree-like backtracking excursions cancel in the signature, and formally state that the signature characterizes paths up to tree-like equivalence.

---

## Review Round 2: Final Assessment & Verdict

### Assessment of Revisions
1. **Mathematical Precision**: The descriptor decoupling into $d^{\text{sig}}$ (pure invariant) and $d^{\text{aug}}$ resolves the reparameterization bug.
2. **Prior Art Anchoring**: Structural hierarchy vs PathFusion-Net and Paper 02 is now airtight.
3. **Claim Matrix**: The 15-variant suite thoroughly isolates order (`order_scrambled`), time arrow (`time_reversed`), reparameterization (`monotone_warp_sham`), depth ($m=1, 2, 3$), beat distribution (`phase_mean_signature`), and architecture (`phase_signature_linear_probe`).
4. **Verification Gates**: 6 synthetic worlds with null-relative tests replace arbitrary thresholds.

### Verdict
**ACCEPT WITH HIGHEST DISTINCTION**.
