# AF-DENSITY V6 — Clinical & Physiological Verification Report

**Audit Standard:** Independent statistical re-estimation of episode entrenchment, PMV recovery, recurrence landmarking, and structural substrate models.
**Status:** RECONCILED & SCIENTIFICALLY ADJUDICATED

---

## 1. Episode Entrenchment & Frailty Modeling

1. **Termination Hazard Collapse:**
   - Empirical hazard at 15s: **0.5300** per minute.
   - Empirical hazard at 1h: **0.0072** per minute.
   - Reduction ratio: **73.7x** reduction ($p < 0.0001$).
2. **Shared Gamma Frailty Weibull Model:**
   - Within-patient shape $\alpha_{within}$: **0.5854** ($< 1.0$ indicates negative duration dependence).
   - Frailty variance $\sigma^2$: **2.3398**.
   - Model improvement: $\Delta AIC = -6,855.5$ vs homogeneous Weibull ($p < 10^{-100}$).
3. **ARC Clinical Trial Discordance:**
   - Exactly **26 patients** have episodes between 30s and 6m, creating regulatory classification discordance under ARC trial definitions.

---

## 2. P-Wave Morphological Variability (PMV) Dynamics

1. **Post-AF Recovery Trajectory:**
   - Fixed panel median PMV drops by **16.58%** from early (0-2m) to late (30-60m) across 207 complete episodes.
   - Patient-paired comparison against remote personal baseline yields median $\Delta PMV = +0.003172$ (95% CI [-0.001471, +0.006725], $p = 0.0614$). Inconclusive.
2. **Difference-in-Differences vs Negative Controls:**
   - Matched DiD effect: $\Delta PMV_{DiD} = +0.007393$ ($p = 0.0157$).
   - Mock transition negative controls show flat trajectory ($p = 0.3501$).
3. **Subsequent AF Recurrence Landmark Analysis:**
   - Unadjusted Cox PH in LTAFDB: $HR = 0.7782$ ($p < 0.0001$).
   - **Rigorous Adjusted Landmark Model:** $HR = 0.9983$ (95% CI 0.8467 to 1.1771, **$p = 0.9842$**).
   - *Adjudication:* Prior claim of recurrence prediction is **INVALIDATED**. The association was driven by immortal time bias.

---

## 3. SHDB Structural Associations & Echocardiographic Substrate

1. **Multi-Episode Subcohort (N=49):**
   - $\beta_D = -13.60$ mm ($p = 0.0089$). Apparent inverse association.
2. **Contemporaneous Echocardiogram Subcohort (N=44, $\le 90$ days):**
   - $\beta_D = +7.77$ mm ($p = 0.3086$). **Complete sign reversal**.
3. **Leverage & Outlier Analysis:**
   - Max Cook's distance: **0.318** on `shdb_2148042`.
   - Excluding top outlier shifts overall model effect from $-7.66$ mm to **-3.64 mm ($p = 0.4750$)**.
   - *Adjudication:* Claim of stable structural remodeling is **INVALIDATED**.

---

## 4. Rosman 2026 Translation & Horizon Migration

1. **Category Replication:**
   - High Density tier (>0.75): **61.82%** in ambulatory Holter vs 54.9% in CIED.
2. **Horizon Migration (12h to Full):**
   - **36.20%** of recordings migrate across Rosman burden-density tiers when changing observation duration from 12h to full recording.
3. **Test-Retest Stability (ICC):**
   - AF Burden ICC: **0.9048** (High rank stability).
   - AF Density ICC: **0.5300** (Moderate/low rank stability).
