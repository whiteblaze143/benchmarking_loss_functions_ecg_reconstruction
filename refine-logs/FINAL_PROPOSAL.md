# Final Research Proposal: Quality-Aware Multimodal SetOperator for Post-Ablation AF Recurrence Ascertainment Under Real-World Ambulatory Noise

## 1. Problem Statement & Clinical Anchor
Continuous cardiac monitoring following catheter ablation of atrial fibrillation (AF) is critical for determining procedural success, guiding antiarrhythmic drug cessation, and managing oral anticoagulation (OAC). However, clinical deployment of deep learning models on continuous wearable and ambulatory ECG streams faces three major challenges:
1. **The Post-Ablation Blanking Dilemma:** Arrhythmias occurring within the initial post-procedural period (defined as 8 weeks / 56 days under the 2024 EHRA/HRS/APHRS/LAHRS Consensus, historically 90 days) represent benign, transient inflammation rather than procedural failure. Algorithms must evaluate recurrence strictly beyond this blanking threshold.
2. **Clinical Actionability vs. Trial Cutoffs:** Consensus efficacy trials require detecting any atrial arrhythmia lasting $\ge 30$ seconds. In contrast, landmark clinical stroke trials (ASSERT, ARTESiA, NOAH-AFNET 6) demonstrate that thromboembolic risk and net clinical benefit of anticoagulation emerge at episodes lasting $\ge 6$ minutes (360 seconds). A clinically deployable algorithm must provide dual-threshold verification.
3. **The Ambulatory False Alarm Crisis:** Fixed-output deep learning models (e.g., ArNet2, RawECGNet) are forced to classify every segment, regardless of quality. Under authentic ambulatory motion artifacts and EMG noise (Noise Holter N2 tier), these models suffer from massive false alarm spikes (up to 62.5% false positives in sinus rhythm), leading to alarm fatigue and clinical abandonment.

Our central hypothesis is:
> Jointly modeling invariant cardiac phase morphology (Phase-KME) and sequential RR interval dynamics via cross-attention, coupled with a quality-aware selective abstention gate (Chow's rule on SQI + predictive entropy), maximizes post-blanking AF recurrence sensitivity while eliminating false alarms under uninterpretable ambulatory noise.

---

## 2. Dominant Methodological Pillars

### Pillar 1: Phase-Conditioned Morphological Representation (Phase-KME)
Rather than passing raw high-frequency waveforms that are distorted by baseline wander, cardiac cycles are normalized into canonical phase $\phi \in [0, 2\pi)$. Morphological features are embedded into a reproducing kernel Hilbert space across 16 canonical phase bins:
$$\mu_q(\phi) = \frac{1}{|I_\phi|} \sum_{t \in I_\phi} \psi(x(t))$$
This extracts stable P-wave absence, fibrillatory f-wave patterns, and QRS shapes invariant to heart rate variation.

### Pillar 2: Sequential RR Interval Temporal Dynamics
Because morphological phase normalization strips beat-to-beat irregularity, a parallel lightweight temporal network processes the consecutive sequence of RR intervals:
$$\Delta t_i = t_{i+1} - t_i, \quad x_{\text{RR}} = [\Delta t_1, \dots, \Delta t_M]$$
Capturing the pathognomonic irregularly irregular ventricular response of atrial fibrillation.

### Pillar 3: Cross-Modal Set Attention Fusion
The morphological token set $\{\mu_{q, k}\}$ and temporal RR tokens $\{h_{\text{RR}, j}\}$ are fused via cross-attention:
$$z = \text{CrossAttention}(Q=\mu_q, K=h_{\text{RR}}, V=h_{\text{RR}})$$
Allowing morphological evidence (absence of P-waves) to modulate rhythm evidence (irregular RR intervals).

### Pillar 4: Quality-Aware Selective Abstention (Chow's Rule)
We formulate AF detection as a selective classification problem with a rejection option:
$$\text{Decision}(x) = \begin{cases} \hat{y}(x) & \text{if } \text{bSQI}(x) \ge \tau_{\text{sqi}} \text{ and } \mathcal{H}(\hat{P}) \le \tau_{\text{ent}} \\ \bot \text{ (Abstain)} & \text{otherwise} \end{cases}$$
When electrode displacement, EMG interference, or severe motion artifact renders the segment uninterpretable (matching Noise Holter tier N2), the model abstains, deferring to re-acquisition rather than emitting a hallucinated false alert.

---

## 3. Rejected Complexity
- **No heavy diffusion or generative models:** Generative imputation of corrupted ECGs risks hallucinating non-existent P-waves or f-waves; explicit selective abstention is clinically safer and computationally lighter.
- **No speculative 12-lead reconstruction:** Projecting 2-lead Holters or 3-lead textile wearables into synthetic 12-lead spaces introduces ill-conditioned dipole inversion artifacts; we operate directly on the observed lead set.
- **No ungrounded hyperparameter sweeps:** Thresholds for abstention ($\tau_{\text{sqi}} = 0.5, \tau_{\text{ent}} = 0.5$) are derived deterministically from the calibration tier of the Noise Holter Database.

---

## 4. Empirical Evaluation Protocol
1. **Primary Clinical Validation (SHDB-AF Post-Ablation Cohort):**
   - Full evaluation on all 41 post-ablation Holters (34 post-blanking >90d recordings, 25 annotated recordings with expert cardiologist beat-by-beat annotations).
   - Evaluation at both $\ge 30$ seconds and $\ge 6$ minutes episode thresholds.
   - Metrics: Episode-level AUROC, F1, Sensitivity, Specificity, PPV, NPV; Patient-level binary recurrence ascertainment; Cumulative AF burden agreement (Pearson $r$, Spearman $\rho$, MAE); False alarms per 24 hours.
2. **Noise-Tier Stress Benchmark (Noise Holter Database v1.0.0):**
   - 6,255 segments across Clean (43.7%), N1 (marginal, 36.0%), and N2 (unusable, 20.3%).
   - Abstention rate and false positive suppression across noise tiers.
3. **Monitoring Completeness & Missing-Time Uncertainty Bounds (2026 AFA-ARC Consensus):**
   - Formalize $[Burden_{\min}, Burden_{\max}]$ bounds under wearable nonwear and Chow's selective abstention.
   - Quantify the relationship between monitoring completeness ($\ge 60\%$ vs $<40\%$) and post-ablation recurrence ascertainment sensitivity under $\ge 30\text{s}$ and $\ge 6\text{min}$ episode criteria, addressing the 2025 *Heart Rhythm* inconclusive recording crisis.
4. **Upcoming Prospective Cohort (WEAR-AF):**
   - Designed to seamlessly ingest prospective Skiin 320 Hz 3-lead continuous monitoring data upon release without algorithmic re-engineering.
