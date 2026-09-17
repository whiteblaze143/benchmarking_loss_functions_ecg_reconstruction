# Final Research Proposal: Paper 04
## Differentiable Cyclic Delay-State Operators on Phase-Conditioned ECG Distributions

### 1. Abstract & Defensible Novelty Boundary
Electrocardiographic interpretation fundamentally depends on the spatiotemporal progression of electrical activation and repolarization across the cardiac cycle. While conventional Dynamic Mode Decomposition (DMD) and Hankel-DMD methods have been applied directly to raw 1D or multilead ECG voltage waveforms (Brunton et al., 2017; Liang et al., 2025; SPTDMD-WST, 2024), Paper 04 establishes a fundamentally distinct object: **a differentiable cyclic delay-state operator over phase-conditioned ECG distribution embeddings**.

$$\boxed{ U_g = \mu_{P(X \mid \theta_g)} \quad \longrightarrow \quad h_g = [x_g, x_{(g+1)\%T}, \dots, x_{(g+L-1)\%T}]^\top \quad \longrightarrow \quad h_{(g+1)\%T} \approx A_i h_g }$$

Rather than assuming raw continuous voltages follow a global linear operator, Paper 04 consumes validated non-parametric Kernel Mean Embeddings (KME) $U \in \mathbb{R}^{T \times D_{\text{in}}}$ across $T=16$ cardiac phase sectors, projects them into a low-dimensional manifold ($d=4$), constructs a delay-augmented state on the topologically cyclic group $C_{16}$, and differentiably estimates the patient-specific phase-progression transition operator $A_i \in \mathbb{R}^{16 \times 16}$. This directly tests whether diagnostic information resides in the dynamical progression between cardiac phase distributions beyond static distribution features alone.

### 2. Resolution of Historical Spectral-Descriptor Ineligibility
In preliminary investigations, local cell-level spectral descriptors (eigenvalues and singular values of short 32-sample lead segments) were evaluated. That formulation strictly failed its prespecified Nyström approximation-fidelity criterion:
- $M=128$: Spearman $\rho = 0.649 < 0.90$, Median relative error $= 0.861 > 0.15$ (FAILED)
- $M=256$: Spearman $\rho = 0.671 < 0.90$, Median relative error $= 0.816 > 0.15$ (FAILED)

The mathematical failure stems from discontinuous branch cuts in eigenvalue angles $\operatorname{Arg}(\lambda) \in (-\pi, \pi]$, sorting permutation jumps, and short window discretization. Crucially, this negative finding is preserved fail-closed in `manifest.json` (`status: ineligible_nystrom_fidelity`), demonstrating scientific integrity. Modern Paper 04 completely bypasses heuristic spectral decomposition by differentiably learning the transition operator $A$ directly on validated Phase-KME distribution states.

### 3. Mathematical Formulation

#### 3.1 Latent Projection & Cyclic Delay-State Embedding
Let $U = [U_0, \dots, U_{T-1}]^\top \in \mathbb{R}^{T \times D_{\text{in}}}$ ($T=16, D_{\text{in}}=128$) denote the phase-conditioned distribution summaries of a patient record.
1. Each phase distribution is projected to latent space:
   $$x_g = W_{\text{proj}} U_g + b_{\text{proj}} \in \mathbb{R}^d, \quad d=4$$
2. The delay-augmented state vector incorporates $L=4$ consecutive phases under $C_{16}$ cyclic topology:
   $$h_g = \begin{bmatrix} x_g \\ x_{(g+1)\%T} \\ \vdots \\ x_{(g+L-1)\%T} \end{bmatrix} \in \mathbb{R}^{D_H}, \quad D_H = d \cdot L = 16$$
3. Assembling across all $T=16$ cyclic transitions yields snapshot matrices:
   $$X = [h_0, h_1, \dots, h_{T-1}] \in \mathbb{R}^{16 \times 16}, \quad Y = [h_1, h_2, \dots, h_{T-1}, h_0] \in \mathbb{R}^{16 \times 16}$$
   Because $X, Y \in \mathbb{R}^{16 \times 16}$, the resulting operator $A \in \mathbb{R}^{16 \times 16}$ has exactly 256 coefficients, eliminating the gross underidentification of the former $64 \times 64$ operator (4096 parameters from 12 snapshots).

#### 3.2 Regularized Operator Solve & Orientation
We solve for the optimal transition operator minimizing regularized prediction error:
$$\min_A \|Y - A X\|_F^2 + \lambda \|A\|_F^2$$
Analytical solution:
$$A = C_{21} (C_{11} + \lambda I_{D_H})^{-1}, \quad C_{11} = X X^\top, \quad C_{21} = Y X^\top$$

To ensure numerical stability on degenerate and all-zero signals, the ridge parameter is scale-relative with a prespecified absolute floor:
$$\lambda = \max\left( \alpha \frac{\operatorname{Tr}(C_{11})}{D_H}, \lambda_{\min} \right), \quad \alpha = 10^{-4}, \quad \lambda_{\min} = 10^{-6}$$

**Correct Solve Orientation**:
Because matrix multiplication is non-commutative, $C_{21} (C_{11}^{\text{reg}})^{-1} \neq (C_{11}^{\text{reg}})^{-1} C_{21}$. Using the symmetry of $C_{11}^{\text{reg}}$, PyTorch computes:
$$A = \operatorname{solve}(C_{11}^{\text{reg}}, C_{21}^\top)^\top$$

#### 3.3 Dual Ridge Formulation for High-Lag Regimes
When $D_H > T$ (e.g. $D_H=64, T=16$), the $D_H \times D_H$ auto-covariance is rank-deficient. By the dual ridge identity:
$$A = Y (X^\top X + \lambda I_T)^{-1} X^\top$$
The solve operates on the $T \times T$ Gram matrix, guaranteeing the minimum-ridge extension from the observed transition subspace.

#### 3.4 Classification Heads & Invariant Summary Probe
- **Flattened Operator Head**: The coefficients $\operatorname{vec}(A) \in \mathbb{R}^{256}$ represent the best regularized linear predictor of delay-state progression and are decoded via a 2-layer GELU MLP:
  $$\hat{y} = W_2 \operatorname{GELU}(W_1 \operatorname{vec}(A) + b_1) + b_2 \in \mathbb{R}^5$$
- **Operator Summary Probe**: To directly test whether performance stems from simple coordinate-free dynamical summaries rather than overparameterized memorization, we extract 8 invariant scalars:
  $$\phi(A) = \left[ \|A\|_F, \|A\|_2, \operatorname{Tr}(A), \operatorname{Tr}(A^2), \operatorname{Tr}(A^3), \frac{\|Y - AX\|_F}{\|Y\|_F}, \|A^\top A - AA^\top\|_F, \ln \kappa(C_{11}^{\text{reg}}) \right] \in \mathbb{R}^8$$
  and classify via MLP.
