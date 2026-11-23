# Annotation Mapping and Temporal Continuity Protocol (V4)

**Program:** T-CAIREM Continuous SetOperator / AF-Density Research  
**Document:** `annotation_mapping_and_continuity_v4.md`  
**Date:** November 23, 2026  

---

## 1. Source-to-Target Rhythm State Mapping

To ensure interoperability across datasets without clinical conflation, the following explicit label mapping is enforced:

| Source Dataset | Source Annotation Tag | Canonical RhythmState | QualityClass | ObservationStatus | Clinical Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IRIDIA-AF** | `AF` / `1` | `RhythmState.AF` | `CLEAN` | `OBSERVED` | Primary target rhythm |
| **IRIDIA-AF** | `Non-AF` / `0` | `RhythmState.NON_AF` | `CLEAN` | `OBSERVED` | Normal sinus rhythm or non-AF |
| **IRIDIA-AF** | `noisy_window` | `RhythmState.UNKNOWN` | `N2_QRS_LOST` | `CORRUPTED` | Unanalyzable ECG |
| **SHDB-AF** | `(AFIB` | `RhythmState.AF` | `CLEAN` | `OBSERVED` | Atrial fibrillation episode |
| **SHDB-AF** | `(AFL` | `RhythmState.OTHER_ARRHYTHMIA` | `CLEAN` | `OBSERVED` | Atrial flutter (separately audited) |
| **SHDB-AF** | `(SVT` / `(AT` | `RhythmState.OTHER_ARRHYTHMIA` | `CLEAN` | `OBSERVED` | Supraventricular tachycardia |
| **SHDB-AF** | `(N` | `RhythmState.NON_AF` | `CLEAN` | `OBSERVED` | Normal sinus rhythm |
| **LTAFDB** | `(AFIB` | `RhythmState.AF` | `CLEAN` | `OBSERVED` | Atrial fibrillation episode |
| **LTAFDB** | `(AFL` | `RhythmState.OTHER_ARRHYTHMIA` | `CLEAN` | `OBSERVED` | Atrial flutter |
| **LTAFDB** | `(N` | `RhythmState.NON_AF` | `CLEAN` | `OBSERVED` | Normal sinus rhythm |
| **PhysioNet NHD** | `Clean` | `RhythmState.AF` / `NON_AF` | `CLEAN` | `OBSERVED` | Benchmark noise stratum |
| **PhysioNet NHD** | `N1` | `RhythmState.AF` / `NON_AF` | `N1_QRS_OK` | `CORRUPTED` | Noise present; QRS identifiable |
| **PhysioNet NHD** | `N2` | `RhythmState.UNKNOWN` | `N2_QRS_LOST` | `UNOBSERVABLE` | Severe artifact; QRS lost |

---

## 2. Temporal Continuity Rules

1. **Half-Open Intervals:** All annotations are represented as $[start\_s, end\_s)$ with $end\_s > start\_s$.
2. **No Silent Day Stitching:** Disconnected daily recordings (e.g. Day 1 file vs Day 2 file) must **never be assumed continuous**. Unless an explicit continuous recording header or telemetry timestamp confirms continuity, the boundary between daily files is marked `MISSING_DROPOUT`.
3. **No Bridging Gaps:** Annotation gaps exceeding 30 seconds are treated as unobserved time (`ObservationStatus.MISSING_DROPOUT`), not imputed as sinus rhythm.
4. **Preservation of Competing Arrhythmias:** Atrial flutter (`AFL`) and supraventricular tachycardia (`SVT`) must remain annotated as `OTHER_ARRHYTHMIA` in the source layer. While binary AF detection treats them as non-AF negatives, false AF alarms during AFL/SVT must be explicitly audited in Family Q.
