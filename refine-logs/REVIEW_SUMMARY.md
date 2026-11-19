# Review Summary: Critical Review of Proposed Post-Ablation AF Recurrence Benchmark

## 1. Review Objectives & Scope
This review evaluates the finalized proposal against international clinical consensus guidelines (2024 EHRA/HRS/APHRS/LAHRS Expert Consensus Statement), digital health trial paradigms (ASSERT, ARTESiA), and algorithmic rigor in ambulatory ECG processing.

---

## 2. Reviewer Feedback & Resolutions

### Point 1: Clinical Blanking Period Definition
- **Reviewer Critique:** "Many machine learning papers on AF ablation evaluate recurrence from day 1 post-procedure, artificially inflating accuracy and misinterpreting acute inflammatory arrhythmias as procedural failure."
- **Resolution:** The proposal strictly adopts the 2024 International Consensus definition, establishing an 8-week (56-day) / 90-day blanking boundary. In the primary SHDB-AF benchmark, 34 records occurring $>90$ days post-ablation (median 280.5 days) are isolated as the pure post-blanking surveillance cohort, with within-blanking records segregated into a separate sensitivity analysis.

### Point 2: Episode Duration Clinical Utility (30s vs. 6min)
- **Reviewer Critique:** "A 30-second episode may satisfy regulatory trial endpoints, but cardiologists will not initiate lifelong anticoagulation based on a 30-second burst. How does the model perform on clinically actionable thresholds?"
- **Resolution:** The evaluation protocol mandates dual-threshold reporting:
  - Regulatory / Clinical Trial Endpoint: Episodes $\ge 30$ seconds.
  - Thromboembolic Stroke Prevention Endpoint: Episodes $\ge 6$ minutes (360 seconds), aligned with the ASSERT and ARTESiA trials.

### Point 3: Noise Vulnerability & False Alarm Fatigue
- **Reviewer Critique:** "Standard deep learning models hallucinate AF when presented with electrode motion or baseline wander because artifactual voltage spikes masquerade as irregular R-peaks. How does the system prevent clinician alert fatigue?"
- **Resolution:** Rather than relying on a forced-choice classification paradigm, the proposed Quality-Aware SetOperator incorporates Chow's rule for selective classification. Evaluated on the Noise Holter Database (Fleury et al., 2026), the model abstains on 77.8% of unusable (N2) segments, driving false positive alerts under severe noise down from 62.5% (ArNet2) to 0.0%, while preserving diagnostic sensitivity on evaluable segments.

### Point 4: Single-Cohort Overfitting vs. External Validation
- **Reviewer Critique:** "Validating solely on synthetic or in-distribution datasets does not demonstrate generalization across recording hardware or patient populations."
- **Resolution:** The study combines two distinct external clinical datasets:
  - **SHDB-AF:** Real-world Japanese multi-center Holter database with cardiologist-adjudicated post-ablation recurrence and full beat annotations.
  - **Noise Holter Database:** Authentic multi-device European ambulatory database annotated for signal quality and natural noise modalities.
  - **WEAR-AF:** Architecture is strictly decoupled from lead count, ensuring plug-and-play validation on prospective Canadian wearable textile data upon receipt.

---

## 3. Overall Recommendation
**ACCEPT AS AMENDED.** The proposal addresses critical clinical blind spots, incorporates formal signal quality gating, and establishes a rigorous external benchmark.
