# Stage A & A.1: 3DRECON-QT 3D Theta Spatial Reconstruction Benchmark

Generated at: 2026-09-03T02:23:30.029506+00:00 UTC

## 1. Cell Performance Table

| Cell | Run Name | Code Mode | Fusion | Loss | Missing $r$ | Tail $p_{05}$ | Chest (V1-V6) $r$ | Precordial $\Delta r$ | L1 (mV) | MSE | Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | `D0_current_id_currentloss_s42_l0` | `-` | `add` | `-` | **0.6936** | 0.3388 | 0.7002 | 0.5729 | 0.0916 | 0.0421 | Completed |
| **D1** | `D1_theta_mul_currentloss_s42_l0` | `-` | `mul` | `-` | **0.7056** | 0.3573 | 0.7115 | 0.5822 | 0.0900 | 0.0403 | Completed |
| **D2** | `D2_current_id_l1_s42_l0` | `-` | `add` | `-` | **0.7035** | 0.3495 | 0.7083 | 0.5777 | 0.0888 | 0.0409 | Completed |
| **D3** | `D3_theta_mul_l1_s42_l0` | `-` | `mul` | `-` | **0.7134** | 0.3623 | 0.7183 | 0.5859 | 0.0876 | 0.0398 | Completed |
| **D4** | `D4_learned12_mul_l1_s42_l0` | `-` | `mul` | `-` | **0.7071** | 0.3563 | 0.7126 | 0.5808 | 0.0888 | 0.0403 | Completed |
| **D5** | `D5_permuted_theta_mul_l1_s42_l0` | `-` | `mul` | `-` | **0.7118** | 0.3513 | 0.7169 | 0.5845 | 0.0884 | 0.0400 | Completed |
| **D6** | `D6_theta_add_l1_s42_l0` | `-` | `add` | `-` | **0.7130** | 0.3558 | 0.7174 | 0.5836 | 0.0879 | 0.0398 | Completed |
| **D7** | `D7_learned12_add_l1_s42_l0` | `-` | `add` | `-` | **0.7037** | 0.3552 | 0.7089 | 0.5737 | 0.0894 | 0.0407 | Completed |
| **D8** | `D8_random12_mul_l1_s42_l0` | `-` | `mul` | `-` | **0.7138** | 0.3639 | 0.7181 | 0.5870 | 0.0875 | 0.0397 | Completed |

## 2. Hypothesis Contrast Matrix & Anatomical Decomposition

| Contrast | Cell A | Cell B | $\Delta r$ (A - B) | Causal Mechanism Tested | 95% Bootstrap CI | $p$-value |
|:---|:---:|:---:|:---:|:---|:---:|:---:|
| **D2 - D0** | 0.7035 | 0.6936 | **+0.0098** | Missing-lead L1 optimization relative to prior objective on learned ID anchor | $[+0.0078, +0.0119]$ | $< 0.0001$ |
| **D3 - D5** (All Missing) | 0.7134 | 0.7118 | **+0.0022** | Physical geometry test: true vs permuted spherical coordinate assignment | $[+0.0005, +0.0038]$ | $0.0066$ |
| **D3 - D5** (Frontal Only) | 0.5312 | 0.5270 | **+0.0042** | Frontal limb leads (II, III, aVR, aVL, aVF): algebraically redundant subset | $[+0.0015, +0.0068]$ | $0.0032$ |
| **D3 - D5** (Chest Only) | 0.7164 | 0.7159 | **+0.0005** | Precordial chest leads (V1–V6): spatial chest-wall projection sequence | $[-0.0010, +0.0019]$ | $0.5288$ |
| **D3 - D5** (Precordial $\Delta$) | 0.5859 | 0.5860 | **-0.0001** | Precordial transition progression ($V_{k+1} - V_k$) | $[-0.0018, +0.0017]$ | $0.9238$ |
| **D3 - D6** | 0.7134 | 0.7130 | **+0.0004** | Multiplication effect holding theta constant: ($H \odot g_l$) vs ($H + g_l$) | $[-0.0007, +0.0031]$ | $0.2236$ |
| **D4 - D7** | 0.7071 | 0.7037 | **+0.0034** | Multiplication effect holding learned 12-D code constant: ($H \odot g_l$) vs ($H + g_l$) | $[-0.0023, +0.0015]$ | $0.6094$ |
| **D3 - D4** | 0.7134 | 0.7071 | **+0.0063** | Trigonometric basis representation vs learned 12-D continuous code | $[+0.0041, +0.0084]$ | $< 0.0001$ |
| **D3 - D8** | 0.7134 | 0.7138 | **-0.0000** | Trigonometric code structure vs fixed random continuous code | $[-0.0013, +0.0012]$ | $0.9486$ |

## 3. Scientific Falsification Verdict
> **Epistemological Summary**: 
> 1. **Frontal vs. Chest Decomposition**: The small overall benefit of correct theta assignment ($\Delta \bar{r} = +0.0022$) was concentrated entirely in the frontal limb leads ($\Delta_{\text{frontal}} = +0.0042, p = 0.0032$) and did not extend to precordial reconstruction ($\Delta_{\text{chest}} = +0.00048, p = 0.5288$) or V1–V6 spatial progression ($p = 0.9238$). Because the frontal leads are algebraically redundant under Kirchhoff's laws, this true-theta advantage occurs in the algebraically constrained subspace, providing no empirical evidence that anatomical coordinates resolve missing precordial spatial geometry.
> 2. **Code Structure Realization**: In this seed-42 realization, the tested fixed-random codebook was statistically indistinguishable from the true-theta codebook ($\Delta = -0.00004, 95\%\ \text{CI: } [-0.00128, +0.00122], p = 0.9486$).
> 3. **Fusion Invariance**: Multiplicative gating provides no statistically detectable advantage over additive conditioning when holding spatial codes constant ($p > 0.20$).

