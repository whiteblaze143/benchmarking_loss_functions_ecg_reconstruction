# Slide Outline: Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial

**Meeting**: Research Strategy Alignment with Dr. Christopher Cheung & Dr. Alex Mariakakis  
**Investigators**: Christopher Cheung, MD; Alex Mariakakis, PhD; Mithun Manivannan, BSc; Robert Wu, MD; Alice Tu  
**Institutions**: Division of Cardiology, Sunnybrook Research Institute & Department of Computer Science, University of Toronto  
**Base Protocol**: *Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation* (Protocol Version 1.1, Oct 21, 2025)  
**Total Slides**: 19 slides (~18-22 min talk + discussion)  
**Aspect Ratio**: 16:9 widescreen  

---

### Slide-by-Slide Outline

| Slide | Title | Core Message & Conceptual Focus | Visual / Figure | Time |
|:-----:|-------|---------------------------------|:---------------:|:----:|
| **1** | **Title Slide** | Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial | Sunnybrook & UofT logos / Title block | 0:30 |
| **2** | **Purpose of Today's Meeting** | Established plan as foundation: cleanly integrating 12L reconstruction on the side without altering primary trial endpoints | 3-box alignment layout | 1:00 |
| **3** | **The Established Clinical Protocol (v1.1)** | Post-ablation persistent AF: daily smartwatch at home vs. 14-day CardioSTAT patch at 3 & 6 months; primary outcome = time to AF recurrence | Study timeline & cohort diagram | 1:30 |
| **4** | **Why Reconstruction Cannot Be the Primary Endpoint** | 1-lead reconstruction is mathematically ill-posed; staking primary trial gate on it is fatal; keeping AF recurrence as primary protects trial | Contrast table: Clinical Trial vs. Inverse Problem | 1:15 |
| **5** | **The Two-Track Architecture: Efficacy + Discovery** | Track 1: Sovereign clinical efficacy (AF recurrence, $N=96$); Track 2: Secondary translational AI (12L reconstruction on the side) | Two-track architectural blocks | 1:15 |
| **6** | **The 3DRECON Methodology (Ansari et al., Circulation 2026)** | Multitask spatial reconstruction forces the single-lead encoder to learn 3D vector loops, overcoming single-lead f-wave blindness for AF detection | 3DRECON multitask concept diagram | 1:30 |
| **7** | **Figure 1: The ECG-AIM Model Architecture** | Champion model `conv15e_A0_wave_noSSL_gated_add`: Continuous Morlet wavelet bank + TimeSformer + Gated-Add fusion + Axial Decoder | `conv15e_a0_wave_architecture.png` | 1:45 |
| **8** | **Figure 2: ECG-AIM Multi-Task Training Pipeline** | Synthetic single-lead masking on public 12-lead archives + multi-objective loss supervision (combinatorial, limb consistency, delineation) exposes the hardware gap motivating our trial | `ecg_aim_training_pipeline.png` | 1:45 |
| **9** | **Empirical Validation of ECG-AIM: Benchmark Performance** | Champion model evaluated on 175,890 patient observations: Precordial $r=0.776$, V3 $r=0.739$, median QRS error 8.0 ms; overcomes Ansari et al.'s EchoNext structural collapse (71.6% vs 46.5%) | Comprehensive benchmark table + 2 callout cards | 1:45 |
| **10** | **Zero-Shot Multi-Cohort Generalization \& Flexible Lead Inputs** | Dynamic masking ($M_{\text{obs}}$) accepts any lead set: Smartwatch Lead I (AF AUROC 0.952) $\to$ In-Clinic 3-Lead Triad (AF AUROC 0.981, VT 0.989); zero-shot transfer on EchoNext and Sunnybrook | Dynamic lead invariance table + translation callout | 1:30 |
| **11** | **Single-Lead Wearable Gallery: Smartwatch (Lead I) vs. Patch (Lead II)** | Unseen PTB-XL Record #10283: Transverse Lead I yields superior precordial tracking ($r=0.988-0.995$) vs. inferior Lead II axis ($r=0.919$) | `best_ecg_aim_lead1_reconstruction.png` & `best_ecg_aim_1lead_reconstruction.png` | 1:30 |
| **12** | **In-Clinic Escalation Gallery: 1-Lead Smartwatch vs. 3-Lead Diagnostic Triad** | Escalating from wrist Lead I ($r=0.941$) to in-clinic Triad (I, II, $V_2$) locks horizontal/frontal planes, achieving near-lossless diagnostic reconstruction ($r=0.980-1.000$) | `best_ecg_aim_lead1_reconstruction.png` & `best_ecg_aim_12lead_reconstruction.png` | 1:30 |
| **13** | **External Validation: Sunnybrook Physical 10-Wire XMLs** | Zero-shot evaluation on Dr. Cheung's own institution hardware (ECG004): standard diagnostic voltages ($>2.1$ mV in $V_2$), limb completion $r=1.000$, authentic $18\,\mu$V noise preserved | `sunnybrook_best_reconstruction_1lead.png` & `sunnybrook_best_reconstruction_3lead.png` | 1:45 |
| **14** | **External Validation: Stanford EchoNext Structural Cohort** | Zero-shot on 100k echo-paired patients (Record #10): true diagnostic voltages ($0.9-1.4$ mV), 3-Lead $r=0.964$, 81.4% LVEF concordance, preventing Stanford 3DRECON's collapse | `echonext_best_reconstruction_1lead.png` & `echonext_best_reconstruction_3lead.png` | 1:45 |
| **15** | **Data Collection Methodology: Dual-Stream Protocol** | Stream A (Ambulatory Home): Daily 30s watch ECG + 14-day CardioSTAT; Stream B (In-Clinic Paired): Simultaneous Watch + 12L ECG at routine hospital visits | Clinical workflow timeline diagram | 1:30 |
| **16** | **The Paired In-Clinic Evaluation Set: The Gold Standard** | Why this is the project's most critical dataset: prospective zero-time-delta ($\Delta t = 0$) pairing across baseline AF, acute ablation, and 3/6-mo healed states (vs Ansari et al.'s $\pm 7$-day limit) | Evaluation benchmark contrast table | 1:45 |
| **17** | **Figure 3: Integrated Clinical-AI Trial Pipeline** | Tri-modal engine: Public foundation pre-training (MIMIC/PTB-XL/Icentia) $\to$ In-clinic paired calibration $\to$ Prospective longitudinal testing | `paired_trial_model_integration.png` | 1:30 |
| **18** | **Sample Size, Statistical Power \& Clustered Modeling** | Primary trial power ($N=96$, 20 vanguard) + secondary AI evaluation with MMRM/GEE to account for within-patient repeated measures | Power & statistical modeling schema | 1:00 |
| **19** | **Action Items \& Alignment for Today's Meeting** | 4 concrete decisions: approve two-track protocol amendment, in-clinic paired acquisition SOP, pre-specified secondary biomarker endpoints, vanguard rollout | Decision checklist | 1:00 |

---

### Key Methodological Pillars
1. **The Primary Trial Remains Sovereign**: Chris's primary clinical trial (time-to-recurrent-AF vs. 14-day CardioSTAT patch) is untouched and guaranteed.
2. **Reconstruction is the Auxiliary Regularizer**: Following Ansari et al. (*Circulation* 2026), 12-lead reconstruction acts as a multitask regularizer that forces the single-lead encoder to learn 3D atrial/ventricular dynamics, boosting at-home AF detection.
3. **The Training Pipeline Motivates the Prospective Trial**:
   - Figure 2 details how the AI model was pre-trained using single-lead masking on retrospective 12-lead hospital databases (PTB-XL, MIMIC-IV).
   - Because retrospective public databases use wet electrodes and supine patients, this synthetic single-lead proxy lacks true wearable physics (dry electrode impedance, skin motion artifacts, smartwatch DSP filters).
   - This training gap proves that synthetic math is insufficient and directly justifies the clinical necessity of our Sunnybrook prospective paired in-clinic dataset.
4. **The Paired In-Clinic Evaluation Set is the Crown Jewel**:
   - Ansari et al. relied on derived $V3-V2$ for training and only 26 retrospective CareLink pairs within $\pm 7$ days for device validation.
   - Our prospective protocol captures **exact $\Delta t = 0$ synchronous paired Smartwatch + 12-Lead ECGs** during routine baseline, procedure, 3-month, and 6-month hospital visits ($\approx 384$ paired recordings across 96 patients).
   - This provides the world's first prospective, multi-state (baseline AF, acute ablation, healed myocardium) real-hardware wearable-to-12-lead evaluation benchmark!
