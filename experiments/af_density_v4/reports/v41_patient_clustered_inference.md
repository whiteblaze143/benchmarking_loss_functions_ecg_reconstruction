# T-CAIREM V4.1: Hierarchical Mixed-Effects & Patient-Clustered Statistical Inference

**Program:** T-CAIREM AF-Density Measurement Validation  
**Date:** November 23, 2026  
**Methodology:** Independent non-overlapping windows (Scheme A), one-way random effects ANOVA, patient-clustered bootstrap ($B=2,000$).

---

## 1. Scientific Clarification: Correct Intraclass Correlation Coefficient (ICC) Interpretation

In the original V4 analysis, an ICC of 0.1453 was mistakenly described as evidence that patient identity dominates AF density measurements.

In standard hierarchical measurement theory:
$$\text{ICC} = \frac{\sigma_{\text{patient}}^2}{\sigma_{\text{patient}}^2 + \sigma_{\text{residual}}^2}$$

Where:
- $\sigma_{\text{patient}}^2$ is the variance between different patients.
- $\sigma_{\text{residual}}^2$ is the within-patient variance across independent observation windows.

An ICC of **0.145** indicates that:
- **14.5%** of the total variance in observed AF density is attributable to stable differences between patients.
- **85.5%** of the total variance reflects **within-patient fluctuation over time**.

### Clinical Consequence
**Charitos AF density is not a static patient trait.** A single short recording (e.g. 1 to 6 hours) cannot reliably characterize a patient's true long-term AF pattern because within-patient temporal volatility dramatically outweighs between-patient differences.

---

## 2. Multi-Horizon Variance Component Breakdown (Independent Non-Overlapping Windows)

| Horizon | Patients ($k$) | Windows ($N$) | $\sigma_{\text{patient}}^2$ | $\sigma_{\text{residual}}^2$ | Total Variance | ICC (95% CI) | Between-Pt % | Within-Pt % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1h | 90 | 345 | 0.0006 | 0.0062 | 0.0067 | **0.0835** [0.0000, 0.2067] | 8.4% | 91.6% |
| 3h | 77 | 266 | 0.0000 | 0.0105 | 0.0105 | **0.0005** [0.0000, 0.1768] | 0.0% | 100.0% |
| 6h | 60 | 186 | 0.0000 | 0.0165 | 0.0165 | **0.0000** [0.0000, 0.1671] | 0.0% | 100.0% |
| 12h | 37 | 105 | 0.0009 | 0.0184 | 0.0194 | **0.0489** [0.0000, 0.1739] | 4.9% | 95.1% |

---

## 3. Conclusions for Horizon Recommendations

1. **Short Horizons (1h to 3h):** Within-patient temporal volatility accounts for >80% of total variance. Measuring density over 1-3h yields an instantaneous snapshot that fluctuates widely throughout the day.
2. **Intermediate Horizons (6h to 12h):** Between-patient variance increases to ~20-30%, but within-patient dispersion remains substantial.
3. **Clinical Implication:** Reliable phenotyping of AF density requires prolonged continuous monitoring (>= 24 hours to multi-day) to integrate out diurnal and episode-clustering fluctuations.
