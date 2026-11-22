# Statistical Analysis Plan v2 (SAP v2)
## T-CAIREM Continuous SetOperator AF-Density Research Program

**Document Version**: 2.0 (Post-Stage 1 Critical Review Corrective Protocol)  
**Date**: 2026-11-22  
**Author**: Statistical Governance & Refinement Planner (@planner)  
**Target Repository**: `experiments/af_density/`  
**Storage Destination**: `statistical_analysis_plan_v2.md`, `refine-logs/statistical_analysis_plan_v2.md`, and `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/statistical_analysis_plan_v2.md`  

---

## 1. Executive Summary & Governance Mandates

This revised Statistical Analysis Plan (SAP v2) incorporates all corrective governance mandates from the Stage 1 Critical Review. It establishes the binding inferential, metric, and missing-data standards for evaluating the Continuous SetOperator architecture against matched baseline models.

### Five Inviolable Governance Principles:
1. **Patient-Clustered Bootstrap Resampling ($B=2,000$)**: All statistical inference, hypothesis testing, and 95% confidence intervals must cluster strictly at the unique patient level. Resampling 30-second windows as independent observations is fundamentally invalid and strictly forbidden.
2. **Four Distinct Estimand Rules**: Analyses must clearly segregate: (1) Continuous AF Burden, (2) Episode-level Delineation, (3) Charitos Continuous Temporal Density, and (4) Wearable Completeness Sensitivity. No conflation of estimands is permitted.
3. **Exact Metric Disambiguation**: Charitos continuous-time density, author 30s-binned density (R v3 verified), and discrete RR-beat-clock aggregation are mathematically distinct estimands that must be reported under separate, explicit headings.
4. **Manski Partial-Identification Missing-Time Bounds**: Unobserved time (artifacts, lead disconnects, device removals) must be bounded by sharp worst-case intervals $[Burden_{\min}, Burden_{\max}]$. Missing time must never be silently zero-filled or imputed without bound verification.
5. **Strict Noninferiority Prohibition**: In the absence of an established, clinically calibrated margin $\Delta_{\text{NI}}$, no claims of clinical noninferiority, equivalence, or therapeutic parity may be asserted. All findings must be reported as descriptive benchmark comparisons.

---

## 2. Four Distinct Estimand Specifications

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             FOUR DISTINCT ESTIMANDS                              │
├──────────────────────────┬───────────────────────────────────────────────────────┤
│ Estimand 1: Burden       │ Cumulative AF duration fraction over monitoring span. │
│ Estimand 2: Episodes     │ Episode detection, sensitivity, PPV, boundary error.  │
│ Estimand 3: Density      │ Temporal clustering/dispersion over continuous time.  │
│ Estimand 4: Completeness │ Robustness under partial wear and missing data bounds.│
└──────────────────────────┴───────────────────────────────────────────────────────┘
```

### 2.1 Estimand 1: Continuous AF Burden over Total Monitoring Horizon
- **Target Parameter**: The cumulative fraction of observed time occupied by atrial fibrillation:
$$\text{Burden} = \frac{\sum_{k=1}^K (v_k - u_k)}{T_{\text{obs}}} \in [0, 1]$$
- **Primary Population**: All eligible recordings with valid observed time $T_{\text{obs}} \ge 6$ hours.
- **Evaluation Metric**: Patient-weighted Mean Absolute Error:
$$\text{MAE}_{\text{burden}} = \sum_{i=1}^{N_{\text{patients}}} w_i |\widehat{\text{Burden}}_i - \text{Burden}_{i, \text{ref}}|$$
- **Primary Contrast**: Model Burden vs Reference Burden across D0, D1, and D2 decoders.

---

### 2.2 Estimand 2: Episode-Level Delineation & Temporal Boundary Precision
- **Target Parameter**: Delineation of discrete AF episodes $\{[u_k, v_k]\}_{k=1}^K$ on the continuous elapsed-time axis.
- **Definitions**:
  - Overlap requires strictly positive Lebesgue measure: $\mu(P_j \cap R_i) > 0$. Boundary contact ($u_R = v_P$) does not count.
  - **Episode Sensitivity ($\text{Se}_{\text{ep}}$)**: $\frac{|\{R_i : \exists P_j, \mu(R_i \cap P_j) > 0\}|}{|\mathcal{R}|}$.
  - **Episode Positive Predictive Value ($\text{PPV}_{\text{ep}}$)**: $\frac{|\{P_j : \exists R_i, \mu(P_j \cap R_i) > 0\}|}{|\mathcal{P}|}$.
  - **Episode Split Rate**: Proportion of detected reference episodes overlapping with $\ge 2$ predicted episodes.
  - **Episode Merge Rate**: Proportion of predicted episodes overlapping with $\ge 2$ reference episodes.
  - **Isolated Boundary Error**: Mean absolute onset and offset error evaluated exclusively on isolated 1-to-1 matched episode pairs without splits or merges. Boundaries coincident with segment start/end ($t=0$ or $t=T_{\text{obs}}$) are censored and excluded.

---

### 2.3 Estimand 3: Charitos Continuous Temporal Density
- **Target Parameter**: The temporal dispersion and clustering of AF episodes over an uninterrupted physical timeline $T$:
$$m(w) = \inf \left\{ \Delta \in [w, T] : \exists t \in [0, T - \Delta] \text{ s.t. } \int_{t}^{t+\Delta} \mathbf{1}_{\{\text{AF}\}}(s) \, ds \ge w \right\}$$
$$D(w) = \frac{w}{m(w)}, \quad \bar{D} = \frac{1}{T_{\text{AF}}} \int_0^{T_{\text{AF}}} D(w) \, dw \in [0, 1]$$
- **Mandatory Subgroup Restriction**:
  - Continuous density $\bar{D}$ is evaluated **strictly on the Informative Multi-Interval Cohort ($K \ge 2$)**.
  - **Undefined Cases**:
    - Zero AF ($T_{\text{AF}} = 0$, Burden = 0.0): Returns `None`. Substituting 0.0 is an integrity violation.
    - Continuous AF ($T_{\text{AF}} = T_{\text{obs}}$, Burden = 1.0): Returns `None`.
    - Single Episode ($K = 1$): Trivially $m(w) = w$ for all $w$, hence $\bar{D} = 1.0$.
  - Informative Cohorts: 11 patients (15 segments) in IRIDIA validation; 49 subjects in SHDB; 66 records in LTAF.

---

### 2.4 Estimand 4: Wearable Completeness & Monitoring Horizon Sensitivity
- **Target Parameter**: Model robustness and uncertainty expansion under partial wear time ($W_{\text{wear}} \in [0.10, 0.90]$) and variable daily horizons (6h, 12h, 24h, full-prefix).
- **Evaluation Standard**: Width of Manski partial-identification bounds $[Burden_{\min}, Burden_{\max}]$ and percentage of reference episodes missed under intermittent spot-checking (e.g., 2x30-second checks per day).

---

## 3. Mathematical Disambiguation of Density Estimands

| Estimand Heading | Underlying Clock | Normalization / Domain | Zero-AF Convention | Implementation Reference |
|---|---|---|---|---|
| **Charitos Continuous Density** | Continuous physical wall-clock (seconds) | $\frac{1}{T_{\text{AF}}} \int_0^{T_{\text{AF}}} \frac{w}{m(w)} dw$ | **Undefined (`None`)** (Never 0.0) | `metrics.py: charitos_density` (analytic envelope integration) |
| **Charitos Binned Density** | Uniform 30s observation bins | Riemann sum with 0.001 step; capacity 30s | Returns 0.0 (by author convention) | `metrics.py: echaritos/AF-density v3` (R 4.3.3 verified) |
| **Shared-RR-Clock Aggregation** | Discrete R-peak heartbeat clock | Total beat count $N_{\text{beats}}$ (Butkuviene eq 3–5) | Returns 0.0 or `None` | `metrics.py: discrete_pattern_from_runs` |

**Governance Rule**: A table or figure must never present "AF Density" without qualifying the clock (Continuous physical-time vs Binned 30s vs Beat-clock).

---

## 4. Patient-Clustered Resampling Methodology ($B=2,000$)

### 4.1 Cluster Bootstrap Algorithm
To obtain valid inferential coverage under arbitrary intra-patient correlation and temporal dependence:

```
Algorithm: Patient-Clustered Bootstrap Resampling
Input: Patient-indexed cohort D = { (Patient_i, Timeline_i, Predictions_i) }_{i=1}^N
Parameter: Number of bootstrap replicates B = 2,000
Output: Empirical bootstrap distribution { θ*(b) }_{b=1}^B and 95% Confidence Intervals

For b = 1 to B:
    1. Draw N patient IDs with replacement: S_b = { i_1, i_2, ..., i_N } from { 1, ..., N }
    2. Construct bootstrap sample D*(b) by pooling all daily files, windows, and 
       timeline evaluations belonging to the drawn patients.
    3. For each unique patient i in S_b, compute patient-level metrics θ_i*(b).
    4. Compute cohort aggregate statistic using assigned duration weighting:
       θ*(b) = ∑_{i ∈ S_b} w_i · θ_i*(b)
    5. For paired model comparison (Model A vs Model B):
       Δ*(b) = θ_A*(b) - θ_B*(b)

Construct Two-Sided 95% Confidence Interval:
    Percentile Method: CI_95 = [ θ*(0.025·B),  θ*(0.975·B) ]
    BCa Method: Applied for skewed episode-rate and boundary error distributions.
```

### 4.2 Paired Model Hypothesis Testing
For any comparison between the proposed Continuous SetOperator ($\mathcal{M}_{\text{prop}}$) and a baseline ($\mathcal{M}_{\text{base}}$):
- The null hypothesis is $H_0: \mathbb{E}[\Delta] = 0$ versus $H_1: \mathbb{E}[\Delta] \ne 0$.
- Two-sided bootstrap $p$-value:
$$p = 2 \cdot \min \left( \frac{1}{B} \sum_{b=1}^B \mathbf{1}_{\{\Delta^*(b) \le 0\}}, \, \frac{1}{B} \sum_{b=1}^B \mathbf{1}_{\{\Delta^*(b) \ge 0\}} \right)$$
- Significance threshold: $\alpha = 0.05$ after multiplicity adjustment.

---

## 5. Missing Time Governance & Manski Bounds

```
                ┌──────────────────────────────────────────────┐
                │        Total Recording Period: T_total       │
                ├──────────────────────┬───────────────────────┤
                │   Observed: T_obs    │   Unobserved: T_unobs │
                │  (Confirmed AF/Non)  │  (Artifacts/Dropout)  │
                └──────────────────────┴───────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        Worst-Case Lower Bound                Worst-Case Upper Bound
        (Assume Unobserved = Non-AF)          (Assume Unobserved = AF)
        Burden_min = T_AF,obs / T_total       Burden_max = (T_AF,obs + T_unobs) / T_total
```

### 5.1 Sharp Partial-Identification Bounds
For any patient recording with unobserved time $T_{\text{unobs}} > 0$:
$$Burden_{\min} = \frac{T_{\text{AF, obs}}}{T_{\text{obs}} + T_{\text{unobs}}}, \quad Burden_{\max} = \frac{T_{\text{AF, obs}} + T_{\text{unobs}}}{T_{\text{obs}} + T_{\text{unobs}}}$$
- **Coverage Criterion**: A model's timeline reconstruction is mathematically admissible under missing time if and only if:
$$\widehat{\text{Burden}} \in [Burden_{\min}, Burden_{\max}]$$
Across all evaluation cohorts, the empirical coverage rate of reference burden within $[Burden_{\min}, Burden_{\max}]$ must be 100%.

### 5.2 Conditional Context Filling Rules
- Gap duration $\le 300$ seconds (5 minutes) flanked by identical observed states:
  - If preceding window is AF and succeeding window is AF $\to$ Gap is imputed as **AF**.
  - If preceding window is Non-AF and succeeding window is Non-AF $\to$ Gap is imputed as **Non-AF**.
- If flanking states disagree (transition across gap) $\to$ State is **Indeterminate**.
- Terminal gaps (start or end of file) $\to$ State is **Indeterminate** (never bridged).

---

## 6. Multiplicity Control

### 6.1 Primary Confirmatory Family ($\mathcal{F}_{\text{primary}}$, $M=6$)
1. Charitos Continuous Density MAE
2. Charitos Binned Density MAE
3. Continuous AF Burden MAE
4. Episode Sensitivity ($\text{Se}_{\text{ep}}$)
5. Episode Positive Predictive Value ($\text{PPV}_{\text{ep}}$)
6. Isolated Boundary Onset/Offset Error (seconds)

**Adjustment**: Controlled at FWER $\alpha = 0.05$ via **Holm-Bonferroni step-down**:
$$p_{(k)} \le \frac{0.05}{6 - k + 1}$$

### 6.2 Secondary Exploratory Family ($\mathcal{F}_{\text{secondary}}$)
- Episode Split Rate, Episode Merge Rate, Window Balanced Accuracy, Brier Score, ECE, Manski Bound Width.
- **Adjustment**: Controlled at FDR $q = 0.05$ via **Benjamini-Hochberg (BH)** procedure.

---

## 7. Noninferiority Caveats & Reporting Charter

### 7.1 Strict Prohibition
- In the absence of an FDA-accepted or guideline-endorsed clinical noninferiority margin for continuous AF density ($\Delta_{\text{NI}}$), **no claim of noninferiority or equivalence is permitted**.
- Phrases such as "noninferior to 12-lead ECG", "equivalent to raw ResNet", or "therapeutically interchangeable" are prohibited.

### 7.2 Permitted Reporting Language
- "The Continuous SetOperator achieved an observed continuous density MAE of 0.2575 [95% CI: 0.2104, 0.3082] compared to 0.3210 [95% CI: 0.2645, 0.3812] for raw ResNet, representing a descriptive reduction of 19.8%."
- "Lead-dropout sensitivity resulted in a burden shift of $< 0.15\%$ points."
