# Prespecified Clinical Review Analysis Plan (Investigator-Only)

**Study:** Heart Rhythm 2027 — Reference Annotation Ambiguity and Temporal AF Density Reliability  
**Freeze Status:** Case selection frozen (`2026-11-25T05:30:00Z`) in `case_selection_manifest.csv` and `case_mapping.csv` prior to any clinical verdicts.  
**Design:** Single-reader (`Dr. Christopher Cheung`), blinded expert electrocardiographic assessment of purposively selected high-impact interruptions in high-burden LTAFDB 12-hour recordings plus disjoint clear-AF and organized-sinus control segments.

---

## 1. Analytical Scope & Methodological Constraints

1. **Non-Prevalence Constraint:** Because the `58` target interruptions (`19` Batch A primary + `39` Batch B additional) were purposively sampled by high single-gap Charitos density impact across `19` high-burden ($B \ge 0.80$) `MIXED` LTAFDB patients, **no population-level prevalence of WFDB annotation error will be estimated** from this sample.
2. **Single-Reader Design:** Because only one clinical electrophysiologist is currently reviewing the strips, **no inter-reader agreement ($\kappa$ / ICC between readers) will be computed**. All human outputs are reported strictly as **single-reader, blinded expert assessment of selected influential rhythm intervals**.
3. **Unreviewed Intervals:** Any interval in the `19` patients not submitted for blinded review—or skipped if only Batch A is completed—must **never** be treated as clinically adjudicated.
4. **Control Calibration Check (Descriptive):** Compare the `8` blinded control assessments (`4` continuous `AFIB`, `4` continuous `N`) descriptively as an internal quality check on atrial rhythm classification without pooling controls into target interruption totals.

---

## 2. Prespecified Tabulation After Form Receipt (`05_EP_Review_Form.xlsx`)

Once `05_EP_Review_Form.xlsx` is returned with completed assessments, run a read-only decoding script joining `Case_ID` strictly to `investigator_only/case_mapping.csv` and report:

1. **Review Counts & Completeness:**
   - Number and fraction of reviewed intervals (`Batch A / 19`, `Batch B / 39`, `Controls / 8`) and unique patients covered (`<= 19`).
2. **Distribution of Primary Atrial Rhythm Verdicts (Overall & by Batch):**
   - `1. Continued AF throughout the target interval`
   - `2. Organized sinus rhythm`
   - `3. Other identifiable organized atrial rhythm`
   - `4. AF termination or reinitiation within the interval`
   - `5. Atrial rhythm indeterminate`
3. **Concurrent Ventricular Rhythm & Quality Metrics:**
   - Cross-tabulation of Primary Atrial Verdict $\times$ Secondary Concurrent Ventricular Rhythm (`Normally conducted`, `Ventricular ectopy`, `VT`, `IVR`, `Bradycardia/pause`, `Other`, `Indeterminate`).
   - Diagnostic confidence (`Low`, `Moderate`, `High`) and ECG interpretability (`Good`, `Fair`, `Poor`, `Uninterpretable`).
4. **Stratification by Original Native WFDB Annotation Category (Post-Unblinding Only):**
   - Tabulate expert atrial verdicts separately across native WFDB groups (`NATIVE_VT`, `NATIVE_IVR`, `NATIVE_B`, `NATIVE_T`, `NATIVE_SBR`, `NATIVE_N`, `NATIVE_OTHER`) and across automated screening strata (`r1_class`) as secondary concordance.

---

## 3. Prespecified Parallel Reference-Timeline Recalculation (`R0`, `R_EP`, `R_EP_Unknown`)

For each reviewed patient with fixed `12.0-hour` (`43,200 s`) known recording horizon:

1. **`R0` (Original Native Timeline):** Exact native WFDB `(AFIB`-based interval timeline.
2. **`R_EP` (Confirmed Definitive Expert Re-Annotation Timeline):**
   - Re-classify **only** reviewed intervals assigned a definitive atrial verdict with moderate/high diagnostic confidence (`Continued AF` $\to$ AF; `Organized sinus` or `Other organized atrial` $\to$ non-AF; `Within-interval termination/reinitiation` $\to$ segmented using reviewer's `Estimated_Transition_Timestamps_s` where provided cleanly).
   - Unreviewed intervals and `Indeterminate` / low-confidence intervals retain their native status under **`R_EP` only as operational defaults**—explicitly labeled as mixed confirmed + unreviewed baseline (`R_EP_Partial`), distinguishing confirmed changes from unreviewed segments.
3. **`R_EP_Unknown` (Conservative Confirmed + Masked-Unknown Timeline):**
   - Map all intervals rated `5. Atrial rhythm indeterminate` (and optionally low-confidence intervals in sensitivity check) to **`UNKNOWN`**, excluding them cleanly from known denominator $T_{\text{known}}$ while preserving confirmed `Continued AF` and confirmed `Organized non-AF` determinations. Clearly mark any patient whose remaining known non-AF intervals vanish as **`ALL_KNOWN_AF`** (never `CONFIRMED_ALL_AF`).

---

## 4. Target Patient-Level Metrics & Partial Identification

Across `R0`, `R_EP`, and `R_EP_Unknown` for each reviewed patient (`12 h` clock), recompute exact continuous-time metrics using `experiments/af_density_v8/src/v8_exp_a_reference_layers.py::metrics_from_intervals`:
- **AF Burden ($B$)**
- **Charitos AF Density ($D = 2A / (1 - B)$)** & **Density Measurability Status** (`MIXED` [defined] vs `ALL_KNOWN_AF` / `NATIVE_ALL_AF` [undefined])
- **Butkuvienė/Petrėnas AF Aggregation ($G = 2A = (1 - B)D$)** (`af_aggregation` and `af_aggregation_zeroext`)
- **Episode Count (`n_runs`)** & **Longest-Episode Share (`longest_share = L_max / M`)**
- **Partial Identification Under Unresolved Intervals:** Over any remaining `UNKNOWN` (indeterminate + unreviewed) gaps inside `[0, 43,200 s]`, compute exact **`[certified_burden_min, certified_burden_max]`** and feasible density intervals (`EXACT_EXHAUSTIVE_WHOLE_INTERVAL` when $M \le 12$, `SAMPLED_ATTAINABLE_EXTREMA_ONLY` when $M > 12$), separating `MIXED` vs `ALL_AF` outcomes without interpolating unreviewed segments.
