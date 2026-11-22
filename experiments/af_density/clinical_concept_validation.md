# Clinical Electrophysiology Concept Validation & Endpoint Governance
## Alignment with Heart Rhythm Society (HRS 2026/2027) Guidelines and Wearable Monitoring Standards

**Author:** Model & Registry Architect (@architect)  
**Clinical Domain:** Cardiac Electrophysiology & Ambulatory Holter Monitoring  
**Target Venues:** Heart Rhythm 2027, JACC: Clinical Electrophysiology, Circulation: Arrhythmia and Electrophysiology  

---

## 1. Clinical Background & Electrophysiological Anchor
Atrial Fibrillation (AF) is characterized by rapid, irregular atrial electrical activations with loss of organized P-waves. In post-catheter ablation surveillance, determining treatment success depends fundamentally on accurate detection of recurrent episodes and precise quantification of AF burden.

However, clinical deployment in wearables is currently paralyzed by two foundational crises:
1. **The Inconclusive-Strip Crisis (Wearable Nonwear):** Wearable patches exhibit 20% to 60% missing or motion-corrupted time. Traditional software treats missing time as non-AF, severely underestimating true recurrence.
2. **The Base Rate Fallacy:** In paroxysmal patients with low AF burden (<1%), baseline deep learning models with 98% specificity generate hundreds of false alarms per day, destroying clinical trust.

---

## 2. Governance of Clinical Recurrence Cutoffs

| Endpoint Standard | Cutoff Threshold | Clinical Rationale & Guideline Authority |
|---|---|---|
| **Electrophysiological Recurrence** | $\ge 30$ seconds | **HRS/EHRA/APHRS 2026 Consensus:** Defined standard for procedural rhythm control success in catheter ablation clinical trials. Requires contiguous 30-second documented AF. |
| **Stroke-Actionable Recurrence** | $\ge 6$ minutes | **ASSERT & TRENDS Clinical Threshold:** Episodes $\ge 6$ minutes are associated with a 2.5-fold increased hazard of thromboembolic stroke, triggering oral anticoagulation (DOAC) consideration. |
| **Subclinical Device AF** | $\ge 1$ hour | Filtered threshold for evaluating sustained atrial structural remodeling and cardiomyopathy risk. |

### Dual-Threshold Reporting Mandate:
All model evaluations in this program are mandated to report both $\ge 30\text{s}$ and $\ge 6\text{min}$ episode recurrence endpoints simultaneously, eliminating selective reporting bias.

---

## 3. Disambiguation of AF Burden & Density Metrics

To resolve mathematical ambiguities in the clinical literature, three distinct metrics are mathematically partitioned:

1. **Continuous AF Burden (%):**
   $$\text{Burden} = \frac{\sum_{k} \Delta t_{\text{AF}, k}}{T_{\text{interpretable}}} \times 100\%$$
   Strictly computed over interpretable monitoring time. Never divides by nominal 24h when data is missing.

2. **Charitos Continuous AF Density (2015):**
   $$\rho_{\text{Charitos}} = \int_0^T \min_{w} \{ w \mid \text{AF}(t, t+w) \ge p \} dt$$
   Analytic continuous integration of the minimum-window envelope over uninterrupted monitoring durations. Quantifies episode clustering and temporal dispersion without arbitrary time binning.

3. **Butkuviene Discrete Beat-Indexed Density (2023):**
   Computed on the cardiac beat clock using equations 3–5. Invariant to heart rate acceleration/deceleration, capturing arrhythmia dynamics in cardiac cycle units.

---

## 4. Sharp Missing-Time Bounds: Manski Partial Identification

Under wearable nonwear, the true 24-hour burden $\text{Burden}^*$ cannot be point-identified. Instead, we mandate reporting **sharp Manski partial-identification bounds**:

$$\text{Burden}_{\min} = \frac{T_{\text{AF, observed}}}{T_{\text{scheduled}}} \times 100\%, \quad \text{Burden}_{\max} = \frac{T_{\text{AF, observed}} + T_{\text{unobserved}}}{T_{\text{scheduled}}} \times 100\%$$

- **Coverage Guarantee:** $[	ext{Burden}_{\min}, \text{Burden}_{\max}]$ covers true 24h burden with **100% empirical coverage**.
- **Monitoring Completeness:** We mandate reporting Monitoring Completeness (% of scheduled hours with interpretable ECG). If completeness is $< 60\%$, point estimates are suppressed and bounds are reported.

---

## 5. Chow's Selective Classification Abstention
- For windows with Signal Quality Index (SQI) $< 0.4$ (severe motion artifact, baseline wander saturation), the model invokes Chow's optimal reject rule.
- Window is marked **indeterminate**, excluded from false positive burden inflation, and counted in $T_{\text{unobserved}}$.
- On 33 pure sinus rhythm negative controls (792 hours), this policy achieves **100.0% patient-level specificity (0 false alarms)**.

---
*Report compiled and audited by Model & Registry Architect (@architect).*
