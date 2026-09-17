# Refinement Report: Paper 02
# Phase-KME: Phase-Conditioned Kernel Mean Embeddings for Distribution-Valued ECG Representation

## 1. Mathematical Grounding & Formal Derivations

### 1.1 Characteristic Property & RKHS Injectivity
Let $\mathcal{P}(\mathbb{R}^D)$ be the space of Borel probability measures on $\mathbb{R}^D$ ($D=8$ electrical leads). Let $k_c(x, y) = (\|x - y\|_2^2 + c^2)^{-1/2}$ be the Inverse Multiquadric (IMQ) kernel with scale parameter $c > 0$.
- **Theorem (Sriperumbudur et al., 2011)**: The IMQ kernel is strictly positive definite, bounded, and $c_0$-universal. Consequently, $k_c$ is **characteristic**: the kernel mean embedding map:
  $$\mu: \mathcal{P}(\mathbb{R}^D) \longrightarrow \mathcal{H}_k, \quad P \longmapsto \mu_P = \int_{\mathbb{R}^D} k_c(x, \cdot) \, dP(x)$$
  is **injective**. That is, $\operatorname{MMD}(P, Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}_k} = 0 \iff P = Q$.
- **Implication**: Any two non-identical distributions $P \neq Q$, even those with identical mean vectors $\mathbb{E}_P[X] = \mathbb{E}_Q[X]$ and identical covariance matrices $\operatorname{Cov}_P(X) = \operatorname{Cov}_Q(X)$, satisfy $\operatorname{MMD}^2(P, Q) > 0$.

### 1.2 Nyström Feature Map & Operating Fidelity
Let $Z = \{z_1, \dots, z_M\} \subset \mathbb{R}^D$ be $M=256$ landmark points. The Nyström kernel approximation is:
$$k(x, y) \approx \psi(x)^\top \psi(y), \quad \psi(x) = k(x, Z) (K_{ZZ} + \lambda I)^{-1/2} \in \mathbb{R}^M$$
Chatalic et al. (ICML 2022) establish that with $M = \mathcal{O}(\sqrt{N})$, Nyström KME estimators retain the optimal $\mathcal{O}(N^{-1/2})$ statistical convergence rate.
At our operating point ($M=256, c^2=1.0$), we enforce an empirical acceptance gate across 2,000 exact cell pairs:
- Spearman rank correlation: $\rho_S(D_{\text{exact}}, D_M) \ge 0.95$.
- Median relative error: $\operatorname{median} \frac{|D_M - D_{\text{exact}}|}{D_{\text{exact}}} \le 0.15$.

### 1.3 Moment-Matched Gaussian Surrogate Construction
To isolate non-Gaussian information, we construct a surrogate $Y$ matching target $X \in \mathbb{R}^{N \times D}$:
1. Draw standard normal noise: $Z_i \sim \mathcal{N}(0, I_D)$ for $i=1,\dots,N$.
2. Compute sample mean $\bar{Z} = \frac{1}{N}\sum Z_i$ and sample covariance $S_Z = \frac{1}{N-1}\sum (Z_i - \bar{Z})(Z_i - \bar{Z})^\top$.
3. Exact whitening: $\tilde{Z}_i = S_Z^{-1/2}(Z_i - \bar{Z})$ so that $\bar{\tilde{Z}} = 0$ and $S_{\tilde{Z}} = I_D$.
4. Exact recolouring to target sample mean $\bar{X}$ and sample covariance $S_X$:
   $$Y_i = \bar{X} + S_X^{1/2} \tilde{Z}_i$$
5. Numerical validation:
   $$\max_{j} |\bar{Y}_j - \bar{X}_j| < 10^{-8}, \quad \max_{j,k} |(S_Y)_{jk} - (S_X)_{jk}| < 10^{-8}$$

### 1.4 Discrete $C_{16}$ Cyclic Convolution & Shift Equivariance
The 16 phase cells index elements of the cyclic group $C_{16} \cong \mathbb{Z} / 16\mathbb{Z}$.
The group action of $g \in C_{16}$ on a discrete phase sequence $x \in \mathbb{R}^{16 \times D}$ is the circular shift operator:
$$(T_k x)[g] = x[(g - k) \bmod 16]$$
A 1D convolutional layer with kernel size $K=3$, weights $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 3}$, and circular padding:
$$(F(x))[g] = \sum_{m=-1}^1 W_{m+1} x[(g + m) \bmod 16]$$
satisfies exact equivariance:
$$F(T_k x) = T_k F(x), \quad \forall k \in \mathbb{Z}_{16}$$
In contrast, standard zero-padding produces boundary truncation errors at $g=0$ and $g=15$:
$$(F_{\text{zero}}(T_k x))[g] \neq (T_k F_{\text{zero}}(x))[g]$$

### 1.5 Mask-Aware Kernel for Sparse-Lead Deployment
To evaluate reduced-lead configurations ($8 \to 6 \to 3 \to 2 \to 1$) without altering the RKHS dimensionality, we define the mask-augmented representation:
$$\tilde{x} = [m \odot x, \; \gamma m] \in \mathbb{R}^{16}$$
where $m \in \{0, 1\}^8$ indicates lead availability and $\gamma = 1.0$. The fixed kernel:
$$k(\tilde{x}, \tilde{y}) = \frac{1}{\sqrt{\|\tilde{x} - \tilde{y}\|_2^2 + c^2}}$$
distinguishes a true zero voltage from an unobserved lead, preserving a single fixed RKHS across all lead configurations.

### 1.6 Split-Half Stability (Odd-Even Beats)
Within each recording containing $B \ge 4$ cycles, we partition beats into odd and even subsets: $\mathcal{B}_{\text{odd}} = \{1, 3, 5, \dots\}$, $\mathcal{B}_{\text{even}} = \{2, 4, 6, \dots\}$.
We evaluate Normalized Frobenius Similarity (NFS):
$$\operatorname{NFS}(U_{\text{odd}}, U_{\text{even}}) = \frac{2 \langle U_{\text{odd}}, U_{\text{even}} \rangle_F}{\|U_{\text{odd}}\|_F^2 + \|U_{\text{even}}\|_F^2} \in [0, 1]$$
We test whether $\Delta_{\text{NFS}} = \operatorname{NFS}_{\text{kernel}} - \operatorname{NFS}_{\text{moments}} > 0$ via patient-clustered bootstrap ($2,000$ replicates).
