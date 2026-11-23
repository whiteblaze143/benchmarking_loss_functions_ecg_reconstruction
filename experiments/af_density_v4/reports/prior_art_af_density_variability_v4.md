# Prior Art and Literature Foundations: AF Density, Multiscale Dynamics, Identifiability & Robust Reconstruction (V4)

**Program:** T-CAIREM Continuous SetOperator / AF-Density Research  
**Date:** November 23, 2026  
**Document:** `prior_art_af_density_variability_v4.md`  

---

## 1. Executive Scientific Appraisal

Our research program focuses on the **measurement science, multiscale stability, identifiability, and robust reconstruction of Atrial Fibrillation (AF) density and episode dynamics**, rather than conventional binary classification.

A critical review of the literature demonstrates that while the conceptual definition of AF temporal aggregation was introduced over a decade ago, subsequent work has revealed fundamental measurement vulnerabilities:
1. **Detector Sensitivity & Error Vulnerability:** As Butkuvienė et al. (IEEE TBME 2024) demonstrated, AF density is markedly more fragile than AF burden to false detections, missed short episodes, and signal outages. A single false episode or fragmented detection can alter the measured density by $>50\%$.
2. **Ambiguity Across Observation Horizons:** Charitos et al. (2012, 2014) and Rosman et al. (EHJ 2026) defined AF density across fixed monitoring horizons (e.g. 7-day or 30-day monitoring). However, no prior study has established a formal mathematical framework for **multiscale density stabilization**, nor established the minimum horizon required for density to attain metric stability.
3. **Confusion Between Burden, Density, and Variability:** Prior studies frequently conflate distinct temporal estimands. For example, the 2025 *JMIR Cardio* wearable study introduces metrics M3 (density) and M4 (hourly variability), but M3 represents a binned density, while M4 reflects circadian variability, neither of which is equivalent to the continuous lower-envelope Charitos density $D_{\text{AF}}$.
4. **The Identifiability Void under Incomplete Monitoring:** When monitoring contains outages (missing waveforms or artifact rejection), traditional pipelines either impute non-AF or drop incomplete windows. No prior literature has formulated **provable identifiability bounds (Density Identifiability Width, DIW)** to represent what can mathematically be bounded about continuous density without fabricating unobserved rhythm states.

---

## 2. Detailed Literature Synthesis by Thematic Pillar

### Pillar A: AF Density, Temporal Clustering & Monitoring Horizons
- **Charitos et al. (Heart Rhythm 2012; JACC 2014):**
  - *Contribution:* Introduced continuous AF density $D_{\text{AF}} = \frac{2}{1-b}\int_0^1 [m - L(m)]\, dm$, where $L(m)$ is the minimum fraction of monitoring time containing fraction $m$ of total AF burden $b$. Proved that patients with identical burden can have dispersed (low density) vs clustered (high density) temporal profiles.
  - *Limitations:* Evaluated strictly on continuous implantable cardiac monitor (ICM) data without noise; assumed 100% signal availability; did not evaluate detector error propagation or multiscale horizon variation.
- **Butkuvienė et al. (IEEE TBME 2024):**
  - *Contribution:* First quantitative stress-test of AF episode-pattern reconstruction under simulated detector errors and monitoring shutdowns. Showed that AF density has significantly lower reliability coefficients ($ICC < 0.60$) than AF burden ($ICC > 0.90$) under realistic false alarm rates.
  - *Limitations:* Restricted to synthetic Bernoulli error injection and random dropout; did not evaluate modern deep neural detectors (e.g. ArNet2, RawECGNet, SetOperators), nor did it provide constructive algorithms for identifiability bounding or quality-aware fallback.
- **Rosman et al. (European Heart Journal 2026):**
  - *Contribution:* Evaluated AF density over 30-day continuous patch monitoring in 3,200 post-stroke patients, showing temporal clustering correlates with recurrent thromboembolism independent of burden.
  - *Limitations:* Used coarse 1-hour binned discretization; did not address single-lead dropout, signal quality uncertainty, or short ambulatory horizons (6h, 12h, 24h).
- **2025 ESC/EHRA Consensus on AF Burden & 2026 Editorial:**
  - *Contribution:* Recognized AF burden as a continuous prognostic spectrum, but warned that temporal clustering requires standardized definitions and quality control before clinical adoption.
- **2025 JMIR Cardio PPG AF Study:**
  - *Contribution:* Defined four wearable metrics: M1 (burden), M2 (episode count), M3 (binned density), and M4 (hourly variability).
  - *Distinction:* Proved mathematically distinct: M3 is a discrete bin-count ratio, whereas Charitos density is a continuous envelope. M4 measures circadian rhythmicity, not spatial clustering within episodes.

### Pillar B: AF Detection, Episode Modeling & Neural Architectures
- **Gilon et al. (IEEE TBME 2020 / IRIDIA-AF Release):**
  - *Contribution:* Released 100+ patient continuous 2-lead ambulatory recordings with expert annotations and established baseline RR-XGBoost (AUROC ~0.96) and 1D CNNs.
  - *Findings:* High window AUROC is readily achieved via R-R interval chaos, but episode boundary preservation requires specialized temporal postprocessing.
- **ArNet2 & RawECGNet (Modern Benchmark Architectures):**
  - *ArNet2:* Autoregressive neural sequence model specialized for ventricular rhythm dynamics over 60-beat R-R intervals.
  - *RawECGNet:* Deep convolutional residual network operating directly on raw voltage waveforms to capture subtle P-wave absence and f-wave fibrillatory baseline.
- **T-CAIREM Stage 1D Findings (November 2026):**
  - *Continuous SetOperator:* When evaluated on unaligned whole-window Phase-KME features, Continuous SetOperator `A001` exhibits negative Brier Skill Scores ($-0.0560$), failing to beat constant prevalence.
  - *Multimodal Hybrid `G061`:* Relies almost entirely on its R-R branch (shuffling morphology leaves AUROC at 0.8686; shuffling R-R drops it to 0.5057). Under genuine lead dropout, `G061` collapses to AUROC **0.1885** on Lead 1, whereas non-neural RR-GBDT achieves **0.9706** (Lead 0) and **0.9622** (Lead 1) symmetrically.

### Pillar C: Benchmark Datasets & Noise Environments
- **IRIDIA-AF:** Continuous 2-channel ambulatory monitoring (90 train, 30 val, 32 test). Train and val authorized; test strictly quarantined.
- **SHDB-AF (PhysioNet v1.0.1, 2025):** 1,000+ Holter recordings with expert beat-by-beat and rhythm annotations for external validation.
- **LTAFDB (PhysioNet):** Long-term AF database with 84 continuous 24-hour recordings, ideal for external transfer of multiscale density profiles.
- **MIT-BIH Noise Stress Test Database (NSTDB):** Standardized baseline wander (bw), muscle artifact (ma), and electrode motion (em) recordings for calibrated SNR corruption sweeps (24 dB down to -6 dB).
- **PhysioNet Noise Holter Database (NHD v1.0.0, November 2026):** Real-world clinical noise benchmark providing segment-level Clean, N1 (QRS identifiable), and N2 (QRS unidentifiable) expert labels alongside competing arrhythmias (atrial flutter, SVT, PVCs, PACs).

---

## 3. Explicit Conceptual Distinctions

To prevent methodological confusion, the following distinctions are strictly enforced across V4:

1. **AF Density $\neq$ AF Burden:**
   - Burden ($b = \frac{1}{T}\int_0^T I_{\text{AF}}(t)\, dt$) is a 0D scalar measuring total accumulated duration.
   - Density ($D_{\text{AF}}$) is a 1D topological metric measuring spatial aggregation of episodes along the observation timeline.
2. **AF Density $\neq$ Episode Frequency:**
   - A patient with 10 episodes clustered within a 2-hour window has identical episode frequency to a patient with 10 episodes uniformly spaced across 24 hours, but radically different density.
3. **AF Density $\neq$ 2025 PPG M3 or M4:**
   - M3 measures the fraction of AF time in the top 20% most active hours. Charitos density is continuous and scale-free.
   - M4 measures hourly standard deviation (circadian rhythmicity), not episode coalescence.
4. **High Window AUROC $\neq$ Accurate AF Density Reconstruction:**
   - A detector with 0.98 window AUROC that produces 2 false 30s episodes per hour fragments long sinus intervals, artificially multiplying episode count and collapsing density estimates.
5. **Within-Patient Horizon Variation $\neq$ Measurement Error:**
   - A patient's true AF density varies over a 6h vs 24h vs 72h window due to multiscale episode dynamics; this is an intrinsic property of the observation window, distinct from sensor noise or detector classification error.
6. **Classifier Softmax Confidence $\neq$ Calibrated Patient Timeline Uncertainty:**
   - Window-level posterior probability cannot be assumed independent; temporal correlation and semi-Markov duration constraints dictate true timeline uncertainty.

---

## 4. Prior Art Matrix Summary Table

The complete formal matrix is codified in [`prior_art_matrix_v4.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v4/reports/prior_art_matrix_v4.csv), indexing 15 landmark studies across 18 audited methodological variables.
