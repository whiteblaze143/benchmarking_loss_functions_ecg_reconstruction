# Paired Patient-Level Bootstrap Uncertainty Report (3D Theta Spatial Benchmark)

Generated at: 2026-09-03T02:23:29.232704+00:00 UTC  
Evaluation Cohort: PTB-XL held-out validation cohort ($N = 2183$ patients)  
Bootstrap Resamples: $B = 10,000$ iterations  

## 1. Causal Mechanism Contrasts & 95% Confidence Intervals

| Causal Hypothesis Contrast | Missing Lead $\mathbb{E}[\Delta_i]$ | 95% Bootstrap CI | Emp. $p$-val | Precordial $V_1$–$V_6$ $\Delta$ | 95% CI (Chest) | Precordial Transition $\Delta V_k$ | 95% CI (Transition) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Physical Geometry Effect (D3 - D5)** | **+0.0022** | `[+0.0005, +0.0038]` | 0.0066 | **+0.0005** | `[-0.0010, +0.0019]` | **-0.0001** | `[-0.0018, +0.0017]` |
| **Multiplicative Fusion on Physical Theta (D3 - D6)** | **+0.0012** | `[-0.0007, +0.0031]` | 0.2236 | **+0.0006** | `[-0.0011, +0.0023]` | **+0.0002** | `[-0.0016, +0.0020]` |
| **Multiplicative Fusion on Learned Code (D4 - D7)** | **-0.0005** | `[-0.0023, +0.0015]` | 0.6094 | **-0.0003** | `[-0.0022, +0.0016]` | **+0.0020** | `[-0.0004, +0.0044]` |
| **Trigonometric Structure vs Fixed Random (D3 - D8)** | **-0.0000** | `[-0.0013, +0.0012]` | 0.9486 | **+0.0006** | `[-0.0006, +0.0018]` | **-0.0017** | `[-0.0031, -0.0004]` |
| **Physical Theta vs Learned Matrix (D3 - D4)** | **+0.0071** | `[+0.0052, +0.0090]` | 0.0000 | **+0.0048** | `[+0.0029, +0.0066]` | **+0.0072** | `[+0.0051, +0.0094]` |

## 2. Epistemological Interpretation Guidelines
- **Physical Geometry ($D_3 - D_5$)**: If the 95% confidence interval for $\Delta_i$ overlaps zero or is tightly concentrated below $+0.003$, the hypothesis that explicit 3D coordinate assignment provides meaningful anatomical advantage is rejected in favor of the structured target code hypothesis.
- **Multiplicative Modulation ($D_3 - D_6$ and $D_4 - D_7$)**: Isolates whether multiplicative scaling ($H \odot g_l$) provides an authentic inductive bias over additive broadcast ($H + g_l$) when holding spatial codes strictly constant.
- **Trigonometric Representation ($D_3 - D_8$)**: Distinguishes whether trigonometric coordinates $\Theta(\theta, \phi)$ provide an inductive smoothing bias over arbitrary fixed continuous codes.
