# AF-DENSITY V6 — Unverified & Disputed Evidence Archive

This archive documents hypotheses, preliminary findings, and historical claims that failed rigorous forensic verification, were found to be statistically confounded, or lack empirical support.

---

## 1. Disputed / Invalidated Claims

### INVAL-001: Early Post-AF PMV as an Independent Recurrence Predictor
- **Prior Claim:** Early post-termination P-wave morphological variability strongly predicts subsequent AF recurrence (unadjusted HR = 0.7782, p < 0.0001).
- **Forensic Re-examination:** Re-evaluated using a landmark Cox proportional hazards model adjusting for cumulative prior AF burden, episode count, and observation horizon in LTAFDB.
- **Finding:** The adjusted hazard ratio is $HR = 0.9983$ (95% CI 0.8467 to 1.1771, $p = 0.9842$). The recurrence effect completely vanishes.
- **Status:** **INVALIDATED**. The association was an artifact of immortal time bias and unadjusted underlying burden.

### INVAL-002: Inverse AF Density Association with Left Atrial Diameter
- **Prior Claim:** Multi-episode AF density exhibits an inverse association with echocardiographic left atrial diameter ($\beta_D = -13.60$ mm, $p = 0.0089$).
- **Forensic Re-examination:** Re-evaluated across echo-to-Holter time latency strata and leave-one-patient-out Cook's distance influence diagnostics.
- **Finding:** Restricting to contemporaneous echocardiograms within 90 days produces a complete sign reversal to $\beta_D = +7.77$ mm ($p = 0.3086$). Furthermore, excluding a single high-leverage patient (`shdb_2148042`, Cook's D = 0.318) attenuates the effect to $-3.64$ mm ($p = 0.4750$).
- **Status:** **INVALIDATED**. Association is biologically unstable and heavily driven by observation lag and outlier leverage.

### INVAL-003: Robustness of Point-Process & Recurrent Decoders
- **Prior Claim:** Point-process neural architectures (A001) and recurrent networks (ArNet2) provide superior temporal tracking of AF episode dynamics.
- **Forensic Re-examination:** Evaluated against matched benchmark test sets using Brier Skill Score and calibration error.
- **Finding:** Both architectures yield severe negative Brier Skill Scores (A001: -1.0097, ArNet2: -0.5760), indicating predictions that perform worse than predicting cohort prevalence.
- **Status:** **DISPUTED / REJECTED**.

---

## 2. Unresolved Scientific Hypotheses Requiring Future Primary Data

1. **AFR Harmonic Ventricular Contamination:** While spatiotemporal QRST cancellation achieves >20 dB QRS attenuation, residual ventricular repolarization harmonics contaminate the 3-9 Hz fibrillatory band in tachycardic AF (>110 bpm). Direct intra-atrial electrogram validation is required.
2. **Cross-Modality Density Translation:** Ambulatory 24-48h Holter density exhibits 36.2% category migration compared to continuous multi-month CIED monitoring. Holter-derived density cannot be directly substituted for CIED density without an observation-horizon adjustment factor.
3. **Within-Patient PMV Baseline Elevation:** Patient-paired $\Delta PMV$ against remote baseline shows a positive trend (+0.003172), but the 95% confidence interval spans zero ([-0.001471, +0.006725], $p = 0.0614$). A larger prospective panel is required to achieve statistical power.
