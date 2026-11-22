# Statistical Analysis Plan (SAP)
## T-CAIREM Continuous SetOperator AF-Density Research Program

**Document Version**: 1.0 (Final Frozen Protocol)  
**Date**: 2026-11-22  
**Author**: Statistical Governance & Refinement Planner  
**Target Repository**: `experiments/af_density/`  
**Storage Destination**: `refine-logs/statistical_analysis_plan.md` and `/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/statistical_analysis_plan.md`  

---

## 1. Executive Summary & Core Principles
This Statistical Analysis Plan (SAP) governs the evaluation of machine learning models for Atrial Fibrillation (AF) episode reconstruction and continuous temporal density estimation.

The core principles of this governance framework are:
1. **Patient-Clustered Resampling**: All inferential confidence intervals and hypothesis tests must cluster at the patient level. Window-level resampling is strictly prohibited due to severe intra-subject autocorrelation.
2. **Exact Metric Disambiguation**: Charitos continuous-time density, Charitos binned density, and beat-clock AF aggregation must be mathematically distinguished and reported under separate, explicit headings.
3. **Formal Missing-Time Accounting**: Unobserved intervals (artifacts, lead disconnects, device removals) must not be silently imputed. Sharp Manski partial-identification bounds $[Burden_{\min}, Burden_{\max}]$ serve as the primary truth standard.
4. **Multiplicity Control**: Family-Wise Error Rate (FWER) is controlled via the Holm-Bonferroni procedure across primary confirmatory endpoints.
5. **No Speculative Noninferiority Claims**: In the absence of an accepted, clinically grounded noninferiority margin, all comparative conclusions must be framed descriptively.

---

## 2. Endpoints & Mathematical Definitions

### 2.1 Continuous AF Burden
Let a recording have total observed duration $T_{\text{obs}}$ and a set of disjoint AF episodes $\{[u_k, v_k]\}_{k=1}^K$, where $0 \le u_k < v_k \le T_{\text{obs}}$. The total AF duration is:
$$T_{\text{AF}} = \sum_{k=1}^K (v_k - u_k)$$
The continuous AF burden is defined as:
$$\text{Burden} = \frac{T_{\text{AF}}}{T_{\text{obs}}} \in [0, 1]$$
- **Burden Error**: $|\widehat{\text{Burden}} - \text{Burden}_{\text{ref}}|$.

---

### 2.2 Charitos Continuous-Time AF Density
As defined by Charitos et al. (Computers in Cardiology, 2015), continuous temporal density quantifies the temporal dispersion and clustering of AF episodes over a single uninterrupted observation duration $T$:

For any evaluation window duration $w \in (0, T)$, let $m(w)$ be the minimum elapsed observation time required to encompass an accumulated AF duration of $w$:
$$m(w) = \inf \left\{ \Delta \in [w, T] : \exists t \in [0, T - \Delta] \text{ s.t. } \int_{t}^{t+\Delta} \mathbf{1}_{\{\text{AF}\}}(s) \, ds \ge w \right\}$$
The continuous density function $D(w)$ is:
$$D(w) = \frac{w}{m(w)} \in \left[\frac{w}{T}, 1\right]$$
The integrated scalar continuous density $\bar{D}$ is the normalized integral across all admissible AF durations $w \in (0, T_{\text{AF}}]$:
$$\bar{D} = \frac{1}{T_{\text{AF}}} \int_0^{T_{\text{AF}}} D(w) \, dw$$

#### Mathematical Conventions & Undefined Cases:
- **Zero AF Episodes ($T_{\text{AF}} = 0$, Burden = 0.0)**: $\bar{D}$ is mathematically **undefined** ($\frac{0}{0}$). The function must return `None`. Substituting 0.0 is strictly prohibited, as zero density would imply infinitely dispersed AF episodes.
- **Continuous AF ($T_{\text{AF}} = T_{\text{obs}}$, Burden = 1.0)**: $\bar{D}$ is mathematically **undefined** (normalization denominator degenerates). The function must return `None`.
- **Single Isolated Episode ($K = 1$)**: For any single uninterrupted episode, $m(w) = w$ for all $w \le T_{\text{AF}}$, hence $D(w) = 1.0$ and $\bar{D} = 1.0$.
- **Disjoint Multi-Interval Episodes ($K \ge 2$)**: This is the strictly **informative cohort** for density benchmarking. $\bar{D} \in (0, 1)$ strictly decreases as intervening sinus intervals lengthen.
- **Continuous Density Absolute Error**: $|\widehat{\bar{D}} - \bar{D}_{\text{ref}}|$ computed only on defined, multi-interval recordings.

---

### 2.3 Charitos Binned AF Density (Author R Implementation)
Following the verified author implementation (echaritos/AF-density v3):
- The recording segment is partitioned into equal contiguous observation bins of duration $\Delta t_{\text{bin}} = 30$ seconds: $N_{\text{bins}} = \left\lceil \frac{T_{\text{obs}}}{\Delta t_{\text{bin}}} \right\rceil$.
- Each bin $j \in \{1, \dots, N_{\text{bins}}\}$ is assigned an AF amount $a_j \in [0, 30]$ seconds based on reference/predicted overlap.
- For an AF duration threshold $p$, the algorithm evaluates the shortest consecutive sequence of bins containing accumulated AF time $\ge p$.
- The binned density is computed via numerical Riemann integration with step size $0.001$.
- Clamping: Results exceeding 1.0 due to edge bin quantization are clamped to 1.0. If $T_{\text{AF}} = 0$, the binned R port returns 0.0.
- **Binned Density MAE**: $|\widehat{D}_{\text{binned}} - D_{\text{binned, ref}}|$.

---

### 2.4 Shared-RR-Clock Aggregation (Beat-Indexed Density)
Following Butkuviene et al. (IEEE TBME 2023, equations 3–5):
- Evaluated on binary beat-level classifications $y_b \in \{0, 1\}$ indexed by detected R-peaks $b \in \{1, \dots, N_{\text{beats}}\}$.
- For an integer count of AF beats $p \in [1, N_{\text{AF}}]$, the minimum window containing $p$ AF beats is:
$$W_{\min}(p) = p + \min_{i} \left\{ \text{gap between episode blocks containing } \ge p \text{ AF beats} \right\}$$
- Beat aggregation quantifies the concentration of AF beats on the discrete heart-rate clock.
- **Mandatory Reporting Separation**: Because R-peak counts fluctuate with instantaneous heart rate, beat-indexed aggregation cannot be equated with physical wall-clock density. Results must explicitly state whether the underlying clock is the continuous physical time clock (seconds) or the discrete beat clock (R-peaks).

---

### 2.5 Episode Delineation & Boundary Endpoints
Let reference episodes be $\mathcal{R} = \{R_1, \dots, R_I\}$ and predicted episodes be $\mathcal{P} = \{P_1, \dots, P_J\}$ on a contiguous observation interval $[0, T_{\text{obs}}]$.
- **Episode Overlap Condition**: A predicted episode $P_j$ detects a reference episode $R_i$ if and only if their intersection has positive Lebesgue measure: $\mu(P_j \cap R_i) > 0$. Point contact at boundaries does not constitute detection.
- **Episode Sensitivity**:
$$\text{Se}_{\text{ep}} = \frac{|\{R_i \in \mathcal{R} : \exists P_j \in \mathcal{P}, \, \mu(R_i \cap P_j) > 0\}|}{|\mathcal{R}|}$$
- **Episode Positive Predictive Value (PPV)**:
$$\text{PPV}_{\text{ep}} = \frac{|\{P_j \in \mathcal{P} : \exists R_i \in \mathcal{R}, \, \mu(P_j \cap R_i) > 0\}|}{|\mathcal{P}|}$$
- **Episode Split Rate**: Proportion of detected reference episodes that overlap with two or more predicted episodes ($\ge 2$).
- **Episode Merge Rate**: Proportion of true positive predicted episodes that overlap with two or more reference episodes ($\ge 2$).
- **Boundary Onset/Offset Errors**:
  - Evaluated exclusively on isolated, one-to-one matched episode pairs ($|R_i \cap P_j| > 0$ with no splits or merges).
  - Onset Error: $|\text{onset}(P_j) - \text{onset}(R_i)|$ in seconds.
  - Offset Error: $|\text{offset}(P_j) - \text{offset}(R_i)|$ in seconds.
  - Segment-Boundary Censoring: If $R_i$ starts at $t=0$ or ends at $t=T_{\text{obs}}$, the corresponding boundary is censored and excluded from the uncensored boundary error calculation.

---

### 2.6 Window-Level Classification & Calibration Endpoints
- **Balanced Accuracy**: $\frac{1}{2} (\text{Sensitivity} + \text{Specificity})$ on 30s windows.
- **Expected Calibration Error (ECE)**: Evaluated over 10 equal probability bins:
$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Brier Score**: $\frac{1}{N} \sum_{n=1}^N (p_n - y_n)^2$, evaluated with patient-equal duration weighting.
- **AUROC & AUPRC**: Area under ROC and precision-recall curves.

---

## 3. Patient-Clustered Resampling Methodology

### 3.1 Failure of Window-Level Resampling
In continuous physiological monitoring, consecutive 30-second windows within the same subject exhibit extreme temporal autocorrelation and share patient-specific baseline morphology. Treating windows as independent observations ($N \approx 500,000$) violates the fundamental i.i.d. assumption, artificially shrinking standard errors by a factor of $\sqrt{M_{\text{windows/patient}}} \approx \sqrt{2880} \approx 53.6$. This produces catastrophic Type I error inflation ($p$-values of $10^{-30}$ on clinically negligible differences).

### 3.2 Patient-Clustered Bootstrap Protocol
All statistical inference (standard errors, confidence intervals, hypothesis tests) must utilize **non-parametric patient-clustered bootstrap**:

1. **Cluster Definition**: The sampling unit is the unique patient identifier $i \in \{1, \dots, N_{\text{patients}}\}$.
2. **Resampling with Replacement**:
   - For bootstrap iteration $b \in \{1, \dots, B\}$ (where $B = 2,000$):
   - Sample $N_{\text{patients}}$ patient IDs with replacement from the cohort.
   - For each sampled patient, include all daily recording segments, 30s windows, and timeline predictions belonging to that patient.
3. **Endpoint Recomputation**:
   - For each resample $b$, compute the patient-level summary metrics:
     $$\theta_{i}^{(b)} = f(\text{patient } i)$$
   - Compute the cohort parameter $\hat{\theta}^{(b)}$ using assigned duration weighting:
     $$\hat{\theta}^{(b)} = \sum_{i=1}^{N_{\text{patients}}} w_i \theta_{i}^{(b)}$$
4. **Confidence Interval Construction**:
   - **Percentile Bootstrap**: The 95% two-sided confidence interval is given by the empirical 2.5th and 97.5th percentiles of $\{\hat{\theta}^{(b)}\}_{b=1}^B$:
     $$\text{CI}_{95\%} = \left[ \hat{\theta}^*_{(0.025)}, \, \hat{\theta}^*_{(0.975)} \right]$$
   - **Bias-Corrected and Accelerated (BCa) Bootstrap**: Used for skewed endpoints (e.g., split rates, boundary errors) to correct for median bias and variance acceleration.
5. **Paired Model Comparisons**:
   - When comparing Model A (e.g., Continuous SetOperator) vs Model B (e.g., Raw Mask-ResNet):
   - For each resample $b$, evaluate both Model A and Model B on the exact same resampled patient cohort:
     $$\Delta^{(b)} = \hat{\theta}_{A}^{(b)} - \hat{\theta}_{B}^{(b)}$$
   - The 95% CI for the difference is constructed directly from $\{\Delta^{(b)}\}_{b=1}^B$. Model A is statistically significantly superior to Model B at $\alpha = 0.05$ if and only if the 95% CI of $\Delta$ excludes 0.0.

---

## 4. Missing Time Governance & Gaps Policy

In long-term ambulatory Holter monitoring, recordings invariably contain missing data arising from:
- Device calibration cycles (first 30s of first recording, first 60s of subsequent daily files).
- Lead disconnection or electrode detachment.
- Muscle tremor / motion saturation triggering hardware squelch.
- Patient removal of the monitor.

The research program establishes three explicit gap-handling policies:

```
Timeline: [── Observed AF ──] [???? Missing Time ????] [── Observed Non-AF ──]
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   1. Manski Bounds           2. Context Filling        3. Time Compression
   [Min: Non-AF, Max: AF]    [Bridged if same state]   [Discard & Concatenate]
   (Primary Truth Standard)   (Conditional Policy)      (Sensitivity Check Only)
```

### 4.1 Policy 1: Manski Partial-Identification Bounds (Primary Truth Standard)
To avoid introducing unverifiable distributional assumptions, missing time must be evaluated using sharp worst-case partial-identification bounds (Manski, 1990):
- Let $T_{\text{obs}}$ be the total known duration, $T_{\text{unobs}}$ be the unobserved duration, and $T_{\text{total}} = T_{\text{obs}} + T_{\text{unobs}}$.
- Let $T_{\text{AF, obs}}$ be the confirmed AF duration during observed periods.
- **Lower Bound ($Burden_{\min}$)**: Imputes all unobserved time as non-AF:
$$Burden_{\min} = \frac{T_{\text{AF, obs}}}{T_{\text{total}}}$$
- **Upper Bound ($Burden_{\max}$)**: Imputes all unobserved time as AF:
$$Burden_{\max} = \frac{T_{\text{AF, obs}} + T_{\text{unobs}}}{T_{\text{total}}}$$
- **Validity Gate**: Any predicted burden $\widehat{\text{Burden}}$ is admissible if and only if $\widehat{\text{Burden}} \in [Burden_{\min}, Burden_{\max}]$. The model's nominal prediction is bounded by this interval.

### 4.2 Policy 2: Context Filling (Conditional Evaluation)
- Missing segments of duration $\Delta t_{\text{gap}} \le 300$ seconds (5 minutes) are bridged if and only if both the immediate preceding observed window and the immediate succeeding observed window exhibit the **identical rhythm state** (both AF or both non-AF).
- If the flanking states differ (e.g., transition from AF to non-AF across the gap), the gap is assigned state **Indeterminate**.
- Terminal gaps (unobserved time at the start or end of a daily file) lack bidirectional flanks and must **never** be filled; they remain strictly indeterminate.

### 4.3 Policy 3: Time Compression / Observed-Only Clock (Sensitivity Analysis Only)
- Discards all unobserved windows and concatenates observed segments into a shortened virtual timeline $T_{\text{comp}} = T_{\text{obs}}$.
- **Mandatory Caveat**: Compressing time distorts inter-episode duration distributions, artificially shortens sinus recovery intervals, and invalidates Charitos continuous density (which assumes uninterrupted real elapsed time). This policy may only be reported as a sensitivity check and must never serve as the primary endpoint.

---

## 5. Multiplicity Control & Hypothesis Testing

### 5.1 Confirmatory vs Exploratory Endpoints
To prevent false discovery across multiple reported endpoints, we establish a formal hierarchy:
- **Primary Confirmatory Family ($\mathcal{F}_{\text{primary}}$)**:
  1. Charitos Continuous Density MAE
  2. Charitos Binned Density MAE
  3. Continuous AF Burden MAE
  4. Episode Sensitivity ($\text{Se}_{\text{ep}}$)
  5. Episode PPV ($\text{PPV}_{\text{ep}}$)
  6. Mean Boundary Onset/Offset Error
- **Secondary Exploratory Family ($\mathcal{F}_{\text{secondary}}$)**:
  - Episode Split Rate, Episode Merge Rate, Window Balanced Accuracy, ECE, Brier Score, AUROC.

### 5.2 Family-Wise Error Rate (FWER) Control
For the primary confirmatory family ($\mathcal{F}_{\text{primary}}$, $M=6$ hypotheses):
- The family-wise error rate is controlled at $\alpha = 0.05$ using the **Holm-Bonferroni step-down procedure**:
  1. Sort two-sided bootstrap $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$.
  2. For $k = 1, \dots, M$, compare $p_{(k)}$ to the adjusted threshold $\alpha_k = \frac{\alpha}{M - k + 1}$.
  3. If $p_{(k)} \le \alpha_k$, reject $H_{0,(k)}$ and proceed to $k+1$.
  4. At the first index $j$ where $p_{(j)} > \alpha_j$, stop and fail to reject all remaining hypotheses $H_{0,(j)}, \dots, H_{0,(M)}$.

### 5.3 False Discovery Rate (FDR) Control
For exploratory secondary endpoints, the False Discovery Rate is controlled at $q = 0.05$ using the **Benjamini-Hochberg (BH)** procedure.

---

## 6. Noninferiority & Superiority Governance Caveats

### 6.1 Prohibition of Speculative Noninferiority
In medical device regulation and clinical trials (FDA / EMA guidance), asserting that a single-lead or reduced-lead model is "noninferior" or "equivalent" to a 12-lead reference or complex baseline requires:
1. A prospectively registered noninferiority margin $\Delta_{\text{NI}} > 0$.
2. Formal clinical and statistical justification that any degradation smaller than $\Delta_{\text{NI}}$ is clinically inconsequential.
3. A two-sided $95\%$ CI (or one-sided $97.5\%$ CI) lying entirely within $[-\Delta_{\text{NI}}, +\infty)$.

**Current Research Status**:
- No international electrophysiological consensus exists for an allowable noninferiority margin in continuous AF density estimation ($\Delta_{\text{density}}$).
- Therefore, **NO claims of noninferiority or therapeutic equivalence are permitted in the manuscript or reports.**
- All model comparisons must be framed strictly as **descriptive benchmark evaluations**, reporting the point estimate and the 95% patient-clustered bootstrap CI without asserting noninferiority.

### 6.2 Pre-Specified Population Subgroups
All primary tables must stratify results across three mutually exclusive subject cohorts:
1. **Informative Multi-Interval Cohort ($K \ge 2$)**: The only cohort where continuous density and interval clustering are defined. Primary density claims are evaluated exclusively here.
2. **Single-Episode Persistent Cohort ($K = 1$)**: Evaluated for burden error and episode boundary accuracy; density is trivially 1.0.
3. **Zero-AF Negative Control Cohort ($K = 0$)**: Evaluated strictly for patient specificity, false episode count, and false burden generation.

---

## 7. Quality Assurance & Independent Audit Checklist
- [x] Resampling script explicitly groups windows by Patient ID, not window index.
- [x] Zero-AF and Continuous-AF files return `None` for Charitos continuous density, never 0.0.
- [x] Shared-RR-clock beat aggregation is reported under a distinct heading from continuous time density.
- [x] Manski bounds $[Burden_{\min}, Burden_{\max}]$ are computed for all files with missing data.
- [x] Holm-Bonferroni correction is applied to all primary hypothesis tests.
- [x] Manuscript contains zero ungrounded noninferiority assertions.
