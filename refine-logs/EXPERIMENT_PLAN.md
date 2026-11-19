# Rigorous 12+ Hour GPU Discovery & Validation Plan: Heart Rhythm 2027

## 1. Clinical Motivation: Post-Ablation Recurrence Ascertainment vs. Risk Prediction

A critical appraisal of clinical electrophysiology literature (2025–2026) demonstrates why our study MUST focus on **Recurrence Ascertainment** rather than recurrence prediction or marginal AUROC tuning:
1. **The Fallacy of Pre-Ablation Prediction:** Predicting recurrence from baseline resting ECGs has hit a ceiling of modest discrimination (e.g. 2026 Mayo Clinic pulsed-field ablation AI-ECG study: C-statistic 0.63–0.67; 2026 PMC multi-center AI-ECG: modest prognostic discrimination beyond clinical factors). Pre-ablation prediction is crowded and offers limited incremental clinical value.
2. **The Recurrence Ascertainment Problem:** Post-catheter ablation, clinicians must definitively answer: *Has atrial fibrillation recurred, when did it recur, and what is the arrhythmia burden?* This guides antiarrhythmic drug cessation, repeat ablation, and stroke-prevention anticoagulation.
3. **Absence of Detected AF $\neq$ Evidence of Absence:** Real-world wearable ECG monitoring (garments, patches, smartwatches) suffers from incomplete wear time (charging, sleep displacement, bathing) and uninterpretable ambulatory motion artifacts (Noise Holter N2 tier: 20.3% of ambulatory segments).
4. **The Inconclusive-Data Crisis (2025 Heart Rhythm):** Prior studies that discard inconclusive or noisy recordings artificially exaggerate diagnostic accuracy. In real-world surveillance, abstained or missing intervals directly degrade recurrence ascertainment power.
5. **The 2026 HRS-Endorsed AFA-ARC Consensus Standards:**
   - Evaluates both consensus $\ge 30$-second episodes (for historical trial comparability) and $\ge 6$-minute episodes (clinically actionable stroke-risk threshold).
   - Mandates reporting of **Monitoring Completeness** ($\frac{T_{\text{interpretable}}}{T_{\text{scheduled}}} \times 100\%$).
   - Mandates reporting of AF burden with explicit uncertainty intervals reflecting missing and uninterpretable monitoring time.
6. **2026 Wearable Precedents (CIRCA-DOSE & JACC 2026):**
   - 2026 CIRCA-DOSE simulation demonstrated that wearable intermittent surveillance achieved only 64.4%–82.2% sensitivity for post-ablation recurrence vs continuous ICM.
   - 2026 randomized patient-led smartwatch study (*JACC* 2026) showed wearable surveillance accelerates recurrence detection, but leaves open how to handle noise, missing time, and true burden quantification.

---

## 2. Frozen Paper Claims & Minimum Convincing Evidence

### Primary Claim
> Fusing continuous Phase-KME morphology and sequential RR dynamics with a Chow's rule selective abstention gate achieves $\ge 95\%$ sensitivity on consensus $\ge 30$s episodes and 100% sensitivity on stroke-actionable $\ge 6$min episodes, while eliminating false alarms ($0.0\%$ false alert rate on accepted windows) under severe ambulatory noise.

### Supporting Claim 1 (Burden Agreement via State-Space Decoding)
> Incorporating temporal state-space Markov smoothing (enforcing physiological episode transition priors) reduces cumulative AF burden Mean Absolute Error by $\ge 40\%$ relative to unsmoothed window averaging across 24-hour continuous recordings, slashing burden MAE to $<15\%$.

### Supporting Claim 2 (Wearable Completeness & Burden Uncertainty Bounds)
> By formulating selective abstention and wear-time dropouts as missing-time uncertainty bounds $[Burden_{\min}, Burden_{\max}]$, the framework maintains $\ge 90\%$ post-ablation recurrence ascertainment sensitivity at $\ge 60\%$ monitoring completeness and bounds true continuous AF burden within a 95% coverage interval, overcoming the inconclusive-data bias of prior wearable benchmarks.

### Anti-Claim to Rule Out
> *"The model's reported accuracy is an artifact of high AF prevalence; when deployed on true negative controls or long continuous recordings with sinus arrhythmia, the model produces frequent false alerts."*
- **Minimum Convincing Evidence:** Evaluating across all 33 non-AF control Holters (792 hours of pure sinus rhythm), demonstrating $<0.5$ false alerts per 24 hours and $>95\%$ patient-level specificity.

---

## 3. The 6 Experimental Blocks (12+ Hour GPU Campaign)

### Block 1: Full-Corpus Calibration on Complete Noise Holter Database
- **Scale:** All 6,255 segments (34.9 hours) across Clean (2,665), N1 (2,202), and N2 (1,263) tiers.
- **Protocol:** Subject-disjoint 5-fold cross-validation across all 316 unique subjects.
- **Ablation Matrix:**
  - $M_1$: ArNet2 (Biton et al., 2023)
  - $M_2$: RawECGNet (Ben-Moshe et al., 2024)
  - $M_3$: SetOperator Phase-KME (Morphology only)
  - $M_4$: SetOperator + RR (Multimodal Fusion)
  - $M_5$: Quality-Aware SetOperator (Selective Abstention with calibrated Pareto thresholds)
- **Target:** Full noise-stratified ROC/PR curves and false alarm suppression.

### Block 2: Full SHDB-AF Population Benchmark (All 128 Holters, 3,050+ Hours)
- **Scale:** All 128 continuous 24-hour recordings (>368,000 30-second windows).
- **Subgroups:**
  - Post-Ablation Surveillance ($N=41$, 34 post-blanking $>90\text{d}$)
  - Non-AF Negative Controls ($N=33$, 792 hours of sinus rhythm)
  - Pre-Ablation / Non-Ablated PAF ($N=54$, 1,296 hours)
- **Target:** Continuous 24h false alert rate ($\text{FA}/24\text{h}$) and global patient-level classification.

### Block 3: True Negative Control Specificity Validation (The 33 Non-AF Holters)
- **Objective:** Definitively refute the anti-claim.
- **Endpoint:** False positive episode rate per 24 hours, patient-level specificity (target $>95\%$), and cumulative false burden (target $<1.0\%$).

### Block 4: Temporal HMM & State-Space Markov Smoothing Ablation
- **Objective:** Crush the 30.0% MAE burden error.
- **Mechanism:** Two-state Hidden Markov Model:
  $$P(Y_t = \text{AF} \mid Y_{t-1}) = \begin{bmatrix} 1 - p_{01} & p_{01} \\ p_{10} & 1 - p_{10} \end{bmatrix}$$
  Parametrized by minimum episode duration priors (1 window for 30s; 12 windows for 6min).
- **Comparison:** Raw Window Averaging vs. Moving Average (3, 5, 11 windows) vs. HMM Viterbi Decoding.
- **Success Criterion:** Burden MAE drops from 30.0% to $<15.0\%$, and Pearson $r$ increases to $>0.60$.

### Block 5: Multi-Lead & Sensor Invariance Sweep
- **Objective:** Demonstrate continuous operator representation across lead configurations.
- **Evaluations:** Lead 1 alone (CC5) vs. Lead 2 alone (NASA) vs. Joint Dual-Lead vs. Random Dropout (simulating lead detachment).
- **Success Criterion:** Model maintains $\ge 90\%$ performance when a lead drops out, proving true set-invariance.

### Block 6: Monitoring Completeness, Missing-Time Uncertainty, and Burden Reliability Quantification under Imperfect Wearable Coverage
- **Clinical Anchor & Motivation:**
  The 2026 HRS-endorsed Atrial Fibrillation Ablation Academic Research Consortium (AFA-ARC) consensus mandates reporting **Monitoring Completeness** alongside AF burden. In real-world wearable monitoring, devices face nonwear (charging, showering, sleep displacement) and high-noise uninterpretable intervals (Noise Holter N2 tier: 20.3% of ambulatory records). Discarding inconclusive strips without accounting for unobserved time biases recurrence ascertainment (2025 *Heart Rhythm* inconclusive smartwatch analysis). Clinically, **absence of detected AF is not evidence of absence of AF**.
- **Experimental Design & Protocol:**
  1. **Wear-Time Coverage Degradation Sweep:** Subsample the continuous 24-hour SHDB-AF post-ablation cohort to simulate realistic wearable adherence schedules:
     - 100% (24-hour continuous reference)
     - 66.7% (16-hour daytime wear)
     - 50.0% (12-hour workday wear)
     - 33.3% (8-hour sleep/morning wear)
     - 16.7% (4-hour casual wear)
     - Intermittent spot checks (2 $\times$ 30-second strips per day, matching typical patient-led smartwatch checks)
  2. **Noise Abstention & Inconclusive Data Modeling:**
     - Quantify analyzable vs. abstained time under Chow's rule selective abstention across authentic noise tiers (Clean, N1, N2).
  3. **Formal Burden Uncertainty Bounds:**
     - Compute rigorous burden bounds:
       $$\text{Burden}_{\min} = \frac{T_{\text{AF, confirmed}}}{T_{\text{scheduled}}}$$
       $$\text{Burden}_{\text{nominal}} = \frac{T_{\text{AF, confirmed}}}{T_{\text{analyzable}}}$$
       $$\text{Burden}_{\max} = \frac{T_{\text{AF, confirmed}} + T_{\text{abstained}} + T_{\text{nonwear}}}{T_{\text{scheduled}}}$$
     - Evaluate the **Ascertainment Uncertainty Gap**: $\Delta_{\text{burden}} = \text{Burden}_{\max} - \text{Burden}_{\min}$.
  4. **Quantitative Endpoints:**
     - **Recurrence Ascertainment Sensitivity vs. Completeness:** Ascertainment sensitivity for $\ge 30\text{s}$ (consensus) and $\ge 6\text{min}$ (actionable) episodes as completeness degrades. Benchmark against CIRCA-DOSE 2026 smartwatch sensitivities (64.4%–82.2%).
     - **Ascertainment Delay (Time-to-First-Detected Recurrence):** Median and IQR delay in hours as coverage decreases.
     - **Indeterminacy Rate:** Fraction of patients whose uncertainty gap $\Delta_{\text{burden}}$ spans across clinical decision cutoffs ($0.1\%$ or $1.0\%$ burden), defining when wearable surveillance is diagnostic vs. inconclusive.
     - **Algorithmic Comparison:** Compare fixed-output baselines (RawECGNet, ArNet2 without quality-awareness) against Quality-Aware SetOperator + Markov state-space smoother.
- **Success Criteria:**
  - Identify the critical completeness threshold (e.g. $\ge 60\%$) required to maintain $\ge 90\%$ recurrence ascertainment sensitivity.
  - Demonstrate that Quality-Aware SetOperator bounds the true continuous 24-hour burden within $[Burden_{\min}, Burden_{\max}]$ in $\ge 95\%$ of patient days, establishing a mathematically sound framework for inconclusive wearable recordings at Heart Rhythm 2027.

---

## 4. Compute Budget & Execution Architecture
- **Hardware:** NVIDIA A100-PCIE-40GB GPU (100% available).
- **Execution Harness:** Persistent detached `tmux` session: `hrs2027_gpu_discovery_12h`.
- **Target Execution Duration:** 12 to 16 hours continuous GPU computation.
- **Checkpointing:** State and intermediate CSV/JSON summaries written after each completed block.
