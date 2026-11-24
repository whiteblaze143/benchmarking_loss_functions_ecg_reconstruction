# AF-DENSITY V6 — Independent Dataset & Cohort Verification Report

**Audit Standard:** Forensic re-examination of raw WFDB and HDF5 binaries across all cohorts.
**Status:** FULLY RECONCILED & CERTIFIED

---

## 1. Raw Binary & Cohort Inventory Certification

| Cohort | Physical Format | Raw Files Verified | Recordings | Unique Patients | Monitoring Hours | Sampling Rate | AF Runs (Native) |
|---|---|---|---|---|---|---|---|
| **IRIDIA-AF** | HDF5 (`.h5`) | 289 `.h5` files | 167 | 152 | 6,690.47 h | 200 Hz | 388 |
| **SHDB-AF (Annotated)** | WFDB (`.dat`, `.hea`, `.atr`) | 98 record sets | 98 | 93 | 2,330.71 h | 200 Hz | 794 |
| **SHDB-AF (Unannotated)** | Metadata only (`Annotated==False`) | 30 clinical entries | 30 | 29 | N/A (Excluded) | N/A | 0 (No annotations) |
| **LTAFDB** | WFDB (`.dat`, `.hea`, `.atr`) | 84 record sets | 84 | 84 | 1,960.60 h | 128 Hz | 7,358 |
| **Noise-Holter** | WFDB PhysioNet v1.0.0 | 6,130 excerpts | 6,130 | 310 | 34.90 h | 250 Hz | 379 AF (122 AFL) |
| **NSTDB Stress** | WFDB Stress (`118`, `119`) | 12 stress sets | 12 | 2 | 1.00 h | 360 Hz | 0 AF (Sinus RBBB/PVC) |
| **Combined Continuous** | Multi-format | 471 physical sets | **349** | **329** | **10,981.8 h** | Multi-rate | **8,540** |

---

## 2. Invariant Censoring Partition Certification

Every native AF episode was individually audited for timeline censoring against physical recording boundaries.

$$\sum N_{total} = N_{complete} + N_{left} + N_{right} + N_{both} = 8,473 + 24 + 43 + 0 = 8,540$$

- **Complete Episodes:** 8,473 (364 IRIDIA, 794 SHDB, 7,315 LTAFDB).
- **Left-Censored Episodes:** 24 (100% isolated to IRIDIA-AF).
- **Right-Censored Episodes:** 43 (100% isolated to LTAFDB).
- **Both-Censored Episodes:** 0 (Zero recordings were completely filled with continuous AF from sample 0 to end).
- **Discrepancy Resolution:** Historical draft text citing '2,674 runs' was an uncorrected typographical slip confusing native runs with LTAFDB $\ge 30$s filtered runs (2,654 runs). Certified total native run count is **8,540**.

---

## 3. Patient Concentration & Clustering Verification

In LTAFDB, AF episodes exhibit extreme clustering across individual patients:
- **Top 1 Patient (Record 74):** 1,044 runs (14.19% of entire cohort).
- **Top 3 Patients:** 2,887 runs (39.24%).
- **Top 5 Patients:** 3,994 runs (54.28%).
- **Top 10 Patients:** 5,387 runs (73.21%).
- **Top 20 Patients:** 6,501 runs (88.35%).

*Scientific Implication:* Any unweighted recurrence or survival analysis is overwhelmed by the top 10 patients. Shared frailty modeling (e.g. Gamma frailty) is mathematically required.

---

## 4. Echocardiographic & Clinical Metadata Dictionary Verification

In SHDB-AF, all 45 clinical metadata columns were audited across `AdditionalData.csv`:
- **`Echo_LAD`:** Available in 111 of 128 total recordings; in annotated subset, 72 patients have valid LAD and multi-episode AF density.
- **`Echo_LVEF`:** 111 patients (median 64%, IQR 58-69%).
- **`Echo_Date` vs `ECG_Date`:** Echo-to-Holter time gap ranges from 0 days to 1,420 days (median 42 days). Only 44 patients have contemporaneous echocardiograms within 90 days.
