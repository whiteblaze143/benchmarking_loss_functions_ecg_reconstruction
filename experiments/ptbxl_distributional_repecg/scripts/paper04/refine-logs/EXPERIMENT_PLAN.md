# Experiment Plan: Paper 04
## Differentiable Cyclic Delay-State Operators on Phase-Conditioned ECG Distributions

### 1. Variant Hierarchy & Scientific Questions

| Variant | Delay ($L$) | Order | Topology | Operator ($A$) | Primary Scientific Question |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `full_cyclic_hankel` | 4 | True | Cyclic | Yes | **Proposed Method** |
| `paper02_phasecnn` | None | True | Cyclic | No | **Primary Comparator**: Does dynamics add value over static Phase-KME? |
| `flat_phase_mlp` | None | True | None | No | Is an unconstrained nonlinear readout of $U$ sufficient? |
| `phase_aware_linear_probe` | None | True | None | No | Linear information in $U$ ($16 \times 128 \to 5$) |
| `linear_probe` | None | True | None | No | Mean-pooled linear baseline ($128 \to 5$) |
| `lag1` | 1 | True | Cyclic | Yes | Markovian memory ($L=1$): does history help? |
| `lag2` | 2 | True | Cyclic | Yes | Minimum short-memory model ($L=2$) |
| `open_chain` | 4 | True | Open | Yes | Topological ablation: does $16 \to 1$ wrap-around matter? |
| `time_shuffled` | 4 | Destroyed | Cyclic | Yes | Order ablation: does transition sequence matter? |
| `time_reversed` | 4 | Reversed | Cyclic | Yes | Directionality ablation: forward vs backward activation |
| `operator_summary_probe` | 4 | True | Cyclic | Scalar | Can simple coordinate-free scalar dynamics explain performance? |
| `ridge_strength_sensitivity` | 4 | True | Cyclic | Yes | Regularization sensitivity ($\alpha = 10^{-3}$ vs $10^{-4}$) |

### 2. Core Falsifiable Hypotheses
- **$H_1$ (Dynamical Gain)**: $\text{AUROC}(\texttt{full\_cyclic\_hankel}) > \text{AUROC}(\texttt{paper02\_phasecnn})$ on identical frozen $U$.
- **$H_2$ (Memory Depth)**: $\text{AUROC}(L=4) \ge \text{AUROC}(L=2) > \text{AUROC}(L=1)$, demonstrating multi-phase memory gain over 1st-order Markovian dynamics.
- **$H_3$ (Temporal Order)**: $\text{AUROC}(\text{ordered}) > \text{AUROC}(\text{time\_shuffled})$.
- **$H_4$ (Cyclic Closure)**: $\text{AUROC}(\text{cyclic}) > \text{AUROC}(\text{open\_chain})$.

### 3. Suite-Wide Evaluation Battery
- **Lead Degradation**: 12, 8, 6, 3, 2, 1I, 1II, V3-V2 lead configurations.
- **Perturbation Battery**: Additive Gaussian noise (SNR 30 to 10 dB), sampling rate decimation (500 to 100 Hz), R-peak temporal jitter ($\pm 20$ ms), and random beat dropping.
- **Mechanistic Endpoints**:
  - Condition number $\kappa(C_{11}^{\text{reg}})$
  - Effective rank $r_{\text{eff}} = \frac{\operatorname{rank}_\epsilon(X)}{D_H}$
  - Operator norm $\|A\|_2$
  - Relative one-step transition residual $\frac{\|Y - AX\|_F}{\|Y\|_F}$
  - Operator reproducibility score $S_A = 1 - \frac{\|A^{(1)} - A^{(2)}\|_F}{\|A^{(1)}\|_F + \|A^{(2)}\|_F}$
