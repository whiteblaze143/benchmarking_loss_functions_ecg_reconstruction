# Refinement Report: Paper 01 — Distributional Recurrence Operator

## 1. Technical Audit of the Proposed Mathematical Object

### Prior-Art Point Recurrence vs Proposed Distributional Recurrence
- **Existing Prior Art (Mathunjwa et al., 2022; Classical RP-CNN)**:
  $$x_g, x_h \in \mathbb{R}^D \implies R_{gh} = \|x_g - x_h\|_2^2 \quad \text{or} \quad \Theta(\epsilon - \|x_g - x_h\|)$$
  *Limitation*: Operates on single points. Cannot capture beat-to-beat variability, morphological dispersion, or multi-beat distributional characteristics.
- **Proposed Distributional Recurrence (repStat)**:
  $$C_g \to P_g \to \mu_{P_g} \to D_{gh}$$
  For phase region $g$, define an empirical distribution across beats:
  $$P_g = \mathcal{L}\left\{ X_b(s) : s \in C_g, \; b = 1, \dots, B \right\}$$
  Kernel mean embedding in RKHS $\mathcal{H}$:
  $$\mu_{P_g} = \mathbb{E}_{X \sim P_g}[\phi(X)] \in \mathcal{H}$$
  Pairwise divergence via squared Maximum Mean Discrepancy (MMD):
  $$D_{gh} = \operatorname{MMD}^2(P_g, P_h) = \|\mu_{P_g} - \mu_{P_h}\|_{\mathcal{H}}^2$$
  Normalized Recurrence Operator:
  $$\boxed{ R_{gh} = \exp\left( -\frac{\operatorname{MMD}^2(P_g, P_h)}{\tau} \right) }$$

---

## 2. Derivation of the Unbiased MMD Estimator

For phase regions $P_g$ and $P_h$ sampled with $m$ and $n$ observations across beats:
$$\widehat{\operatorname{MMD}}_u^2(P_g, P_h) = \frac{1}{m(m-1)} \sum_{i=1}^m \sum_{j \ne i}^m k(x_i, x_j) + \frac{1}{n(n-1)} \sum_{i=1}^n \sum_{j \ne i}^n k(y_i, y_j) - \frac{2}{mn} \sum_{i=1}^m \sum_{j=1}^n k(x_i, y_j)$$
where $k(x, y)$ is:
1. **Gaussian RBF Kernel**: $k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$ (captures all higher-order statistical moments).
2. **Inverse Multiquadric (IMQ) Kernel**: $k(x, y) = \left(c^2 + \|x - y\|^2\right)^{-\frac{1}{2}}$ (heavy-tailed, robust to outlier ectopic spikes).

---

## 3. Four Pre-Production Synthetic Recovery Worlds

Before deploying to the expensive PTB-XL grid, the mathematical object must satisfy four synthetic sanity proofs:

### Synthetic World A: Same Means, Distinct Distributions
- **Setup**: Two phase distributions $P_1 = \mathcal{N}(0, 1)$ and $P_2 = 0.5\mathcal{N}(-2, 0.2) + 0.5\mathcal{N}(2, 0.2)$.
- **Property**: $\mathbb{E}[P_1] = \mathbb{E}[P_2] = 0$.
- **Test**: Euclidean recurrence fails ($\| \mathbb{E}[P_1] - \mathbb{E}[P_2] \| = 0$). MMD recurrence successfully separates them ($D_{12} > 0$).
- **Success Criteria**: $\operatorname{MMD}^2(P_1, P_2) \ge 0.45$.

### Synthetic World B: Finite Sample Convergence
- **Setup**: $X \sim P$ and $Y \sim P$ drawn from identical distributions with sample sizes $N \in \{10, 50, 100, 500\}$.
- **Test**: $\widehat{\operatorname{MMD}}^2(X, Y) \to 0$ as $N \to \infty$ with rate $\mathcal{O}(1/\sqrt{N})$.
- **Success Criteria**: Monotonic convergence without negative variance artifacts.

### Synthetic World C: Cyclic Shift Equivariance
- **Setup**: Apply cyclic phase shift $g \to (g + k) \pmod{16}$.
- **Test**: The circular recurrence CNN satisfies $p(P_k R P_k^\top) \approx p(R)$ with $\Delta_{\text{cyclic}} \le 0.01$.

### Synthetic World D: Amplitude Scale Disentanglement
- **Setup**: Waveform scaled by arbitrary factor $X' = a X$ with $a \in [0.5, 2.0]$.
- **Test**: The shape recurrence $R_{\text{shape}}$ computed on $\tilde{x} = x / a$ remains exactly invariant ($\|R_{\text{shape}}(X') - R_{\text{shape}}(X)\|_F = 0$), while scalar energy $a$ is preserved in the composite head $[R_{\text{shape}}, a]$.

---

## 4. Representation Stability & Reproducibility Metrics

To ensure that the representation captures stable underlying cardiac physiology rather than individual beat noise:

### Split-Half Beat Stability
For each 10-second ECG recording, partition all valid cardiac cycles into odd beats $B_{\text{odd}}$ and even beats $B_{\text{even}}$. Construct independent recurrence operators $R^{\text{odd}}$ and $R^{\text{even}}$.
Evaluate:
1. **Normalized Frobenius Similarity (NFS)**:
   $$\operatorname{NFS} = 1 - \frac{\|R^{\text{odd}} - R^{\text{even}}\|_F}{\|R^{\text{odd}}\|_F + \|R^{\text{even}}\|_F} \in [0, 1]$$
2. **Spectral Correlation**:
   $$\rho_{\text{spectral}} = \operatorname{corr}\left(\lambda(R^{\text{odd}}), \lambda(R^{\text{even}})\right)$$
   where $\lambda(R)$ is the sorted eigenvalue spectrum of $R$.
- **Hypothesis**: MMD distributional recurrence achieves higher NFS and $\rho_{\text{spectral}}$ than Euclidean single-beat recurrence because it averages out respiratory and positioning noise across cycles.

---

## 5. Perturbation & Test-Retest Robustness Protocol
Evaluate sensitivity of $R$ and downstream AUROC under realistic acquisition degradations:
- **Resampling**: $500\text{ Hz} \to 250\text{ Hz}$, $500\text{ Hz} \to 100\text{ Hz}$.
- **Fiducial Jitter**: Add Gaussian R-peak jitter $\sigma_t \in \{\pm 10\text{ ms}, \pm 20\text{ ms}\}$.
- **Additive Noise**: Add pink/baseline noise at $\text{SNR} \in \{20\text{ dB}, 10\text{ dB}\}$.
- **Gain Shift**: Multiply lead gains by factors $\{0.8, 1.2\}$.
- **Subsampling**: Randomly drop $25\%$ and $50\%$ of valid beats before distribution estimation.

Measure relative degradation:
$$\Delta_R = \frac{\|R - R'\|_F}{\|R\|_F}, \quad \Delta_{\operatorname{AUROC}} = \operatorname{AUROC}_{\text{clean}} - \operatorname{AUROC}_{\text{perturbed}}$$
