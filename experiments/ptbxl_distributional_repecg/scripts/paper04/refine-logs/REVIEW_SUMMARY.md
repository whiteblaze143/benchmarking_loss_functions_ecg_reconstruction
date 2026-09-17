# Adversarial Review Summary: Paper 04
## Differentiable Cyclic Delay-State Operators on Phase-Conditioned ECG Distributions

### Executive Assessment: ACCEPT WITH PRODUCTION REVISIONS (Grade: 9.4/10)

### Critical Findings & Blocking Issues Resolved

#### Round 1: Mathematical & Scope Audit
1. **Prior Art & Novelty Boundary (User Point 1)**:
   - *Critique*: Raw waveform Hankel-DMD has substantial direct prior art (HAVOK by Brunton et al., 2017; Liang et al., 2025; SPTDMD-WST, 2024). Overclaiming "first Hankel/DMD on ECG" would invite immediate rejection.
   - *Fix*: Narrowed claim strictly to differentiable cyclic delay-state operators over phase-conditioned distribution embeddings ($U_g = \mu_{P(X \mid \theta_g)}$). Renamed "Nyström ineligibility paradox" to "historical spectral-descriptor ineligibility result".
2. **Takens Attractor Claim (User Point 2)**:
   - *Critique*: 16 phase-indexed distribution summaries cannot mathematically substantiate "reconstructing the unobserved phase-space attractor of the heart".
   - *Fix*: Removed all attractor reconstruction claims. Framed as delay-coordinate augmentation motivated by delay-embedding methods, and designated $A$ as a phase-progression operator.
3. **Matrix Solve Transpose Orientation (User Point 3)**:
   - *Critique*: `torch.linalg.solve(C11_reg, C21)` computes $C_{11}^{-1} C_{21}$, not $C_{21} C_{11}^{-1}$. Due to non-commutativity, this produced relative errors exceeding 500.
   - *Fix*: Formulated as $A = \operatorname{solve}(C_{11}^{\text{reg}}, C_{21}^\top)^\top$. Verified in float64 against direct matrix inversion with tolerance $< 10^{-12}$.
4. **Zero-Trace Ridge Degeneracy (User Point 4)**:
   - *Critique*: Relative ridge $\lambda = 10^{-4} \operatorname{Tr}(C_{11})/D$ evaluates to 0 on all-zero inputs ($x_t=0$), causing $C_{11}^{\text{reg}}=0$ to be singular.
   - *Fix*: Enforced prespecified absolute floor $\lambda = \max(\alpha \frac{\operatorname{Tr}(C_{11})}{D}, 10^{-6})$.
5. **Operator Underidentification (User Point 5)**:
   - *Critique*: A $64 \times 64$ operator (4096 entries) estimated from 12 open-chain transitions is wildly underdetermined ($	ext{rank} \le 12$).
   - *Fix*: Set $d=4, L=4 \implies D_H = 16$. With $T=16$ cyclic transitions, $X, Y \in \mathbb{R}^{16 \times 16}$, yielding a fully identified 256-parameter operator.
6. **Cyclic Topology & Wrap-Around (User Point 7)**:
   - *Critique*: Open-chain construction discarded the essential physiological wrap-around $16 \to 1$.
   - *Fix*: Implemented $C_{16}$ cyclic delay state $h_g = [x_{(g+\ell)\%16}]_{\ell=0}^{L-1}$. Retained `open_chain` as an ablation.
7. **Primary Comparator (User Point 9)**:
   - *Critique*: Comparing only against linear probe does not prove dynamics adds value beyond Phase-KME.
   - *Fix*: Established Paper 02's `PhaseCNN` on identical frozen $U$ as the primary baseline comparator (`full_cyclic_hankel` vs `paper02_phasecnn`).

#### Round 2: Verification of Proof Worlds & Reproducibility
- Synthetic proof worlds expanded to 9, including exact linear recovery on sufficiently observed systems, AR(2) memory order tracking, out-of-sample temporal ordering, cyclic closure advantage, regularization path tradeoff, and PyDMD external oracle validation.
- Added operator reproducibility metric $S_A$ under clinical perturbations (beat subsets, 250 vs 500 Hz, R-peak jitter) with kill criterion.
