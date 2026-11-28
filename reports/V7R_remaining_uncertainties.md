# AF-DENSITY V7-R: Remaining Uncertainties & Methodological Constraints

1. **Annotation Harmonization Gap Across Cohorts:**
   - `IRIDIA-AF` contains only `2` runs `<30s` out of `388` runs (`0.5%`), vs `310 / 794` (`39.0%`) in `SHDB-AF` and `4,704 / 7,358` (`63.9%`) in `LTAFDB`. Retrospective files do not document whether IRIDIA-AF annotation rules filtered sub-30s events; therefore raw cross-cohort comparisons of brief-run incidence remain confounded by annotation convention.
2. **Physical-Time vs Non-AF Risk-Time Distortion (E01):**
   - Under pooled `risk_time_h`, `Pre vs Remote RR = 6.236` ($p=1.59\times 10^{-7}$), but under pooled `physical_time_h`, `Pre vs Remote RR = 1.856` ($p=0.102$, NS). In LTAFDB alone, physical-time RRs are $<1.0$ (`0.831` Pre, `0.544` Post) because peri-prolonged windows contain dense adjacent AF, compressing the non-AF denominator and inflating conditional rates (`27.32/h`) while conditional Null A (`Obs/Exp = 0.905` Pre, `0.674` Post) shows no excess above boundary-constrained random placement. Only SHDB-AF exhibits genuine clustering in both physical time (`~11.1x`) and Null A (`5.7x–6.0x`).
3. **Between-Patient Stratification vs Within-Patient Episode Prediction (P02):**
   - While P02 `S2_Architecture` achieves strong within-cohort (`0.8664` LTAFDB, `0.7173` IRIDIA) and LOCO transfer (`0.8548` LTAFDB, `0.7664` IRIDIA, `0.7283` SHDB) AUROCs, **patient-level equal-weight AUROC across the `n=91` patients who exhibit both brief and prolonged runs is `0.4789` (Logistic) / `0.5245` (LightGBM)**. Thus 1–6h episode architecture predominantly captures patient/regime state rather than episode-to-episode triggers within mixed-phenotype patients.
4. **Underpowered Structural Remodeling Cohort (E03):**
   - Only `13–28` patient-deduplicated multi-run PAF patients in SHDB-AF have defined 12h density and matched echocardiography within `30d–Unrestricted` windows. Null findings (`+0.188 mm`, 95% CI `[-2.298, +2.270]`) reflect severe underpowering and absence of LA volume index, strain, or LGE-MRI fibrosis data — not proof of zero biological relationship.
5. **Lack of Longitudinal Clinical Outcomes:**
   - Neither IRIDIA-AF, SHDB-AF, nor LTAFDB includes prospective ischemic stroke, heart failure hospitalization, or AF progression follow-up endpoints; questions on clinical thromboembolic risk remain unanswerable with these cohorts.
6. **SHDB-AF Repeated Subjects & 30 Unannotated Recordings:**
   - Raw `AdditionalData.csv` has `128` recordings for `122` unique subjects (`6` repeat Holters) and `30` completely unannotated recordings (`719.88h` of unknown rhythm). Any analysis must enforce both annotation filtering and patient deduplication / clustering.
7. **Foundation Model Input/Domain Mismatches & Lack of Pretraining Advantage (Excluding ECGFounder/EchoNext):**
   - Zero-padding 1-lead Holters into 12-lead architectures (`ECGFM_KED`, `EchoNext`, `DINO-SR`, `JEPA`) or assuming Lead I/II alignment for uncalibrated Holter leads (`CSFM`, `F0/F2`, `G1`) introduces channel mismatch; `F2` (`+0.0073` / `-0.0150`) and `CSFM` (`-0.0041` / `-0.0259`) fail to outperform untrained random-init weights on P04.
