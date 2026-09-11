# PTB-XL Dataset and Clinical Ontology Audit (Milestone P0)

**Document**: `PTBXL_DATA_AUDIT.md`  
**Generated Date**: 2026-09-10T21:55:08.963018  
**Target Architecture**: GRAIL-ECG (`E_theta(X) -> Z [B, 96]`)  
**Evaluation Protocol**: PRD Addendum §4, §5, §6, §22  

---

## 1. Dataset Scale, Partitioning, and Patient Isolation

| Split | Folds Included | ECG Record Count | Unique Patient Count | Patient Overlap with Other Splits |
|---|---|---:|---:|---:|
| **Training** | Folds 1–7 | 15245 | 13142 | **0 (Zero)** |
| **Development Validation** | Fold 8 | 2173 | 1881 | **0 (Zero)** |
| **Confirmation Validation** | Fold 9 | 2183 | 1942 | **0 (Zero)** |
| **Locked Final Test** | Fold 10 | 2198 | 1904 | **0 (Zero)** |
| **Total Benchmark** | Folds 1–10 | **21799** | **18869** | **Strict Patient Disjointness Guaranteed** |

### Split Discipline Rules Enforced:
1. **Folds 1–7**: Authorized for training all components (shared ResNet, ThetaEncoder, slot cross-attention, auxiliary view decoder).
2. **Fold 8**: Sole cohort authorized for initial architectural, hyperparameter, and loss-weight selection.
3. **Fold 9**: Confirmation cohort. A design decision passes only if directional improvement reproduces on Fold 9.
4. **Fold 10**: Sealed and strictly prohibited until final Phase P7 confirmation.

---

## 2. Signal Verification & Physical Unit Contract

- **Canonical Waveform Shape**: `(12, 5000)` (12 channels, 10.0 seconds duration).
- **Sampling Frequency**: `500 Hz` verified across headers (`fs = 500`).
- **Physical Voltage Units**: Physical millivolts (`mV`). Verified signal amplitudes span `[-4.20, 4.90] mV` with zero min-max normalization.
- **Signal Integrity**: `0` NaNs, `0` Infs detected in audited batches.
- **Lead Ordering**: Standard 12-lead order: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
- **Primary 8 Independent Views**: Extracted as indices `[0, 1, 6, 7, 8, 9, 10, 11]` corresponding to `[I, II, V1, V2, V3, V4, V5, V6]`.

---

## 3. Clinical Ontology: Anchor Shaping vs. Held-Out Probe Partitioning

To test whether the latent is a **genuine general clinical state space** rather than an overfit multi-task classifier, populated SCP codes ($\ge 50$ occurrences) are strictly partitioned:
- **$C_{\text{anchor}}$ (70%)**: Used for supervised auxiliary multi-task loss on slots 1–4.
- **$C_{\text{probe}}$ (30%)**: **Completely held out from all training objectives, slot supervision, and loss graphs**. Evaluated strictly post-training via linear probes on frozen $Z$.
- **Discovery Slots (5–6)**: $z_{\text{residual1}}$ and $z_{\text{residual2}}$ receive **zero supervised heads** during pretraining to protect against discarding unmodeled clinical variance.

### Clinical Domain Allotment Table:

| Slot Index | Slot Name | Domain | Anchor Concepts ($C_{\text{anchor}}$) | Probe Concepts ($C_{\text{probe}}$) |
|---|---|---|---|---|
| **$z_1$** | `z_rhythm` | Impulse formation & rhythm | PVC, AFLT | PACE |
| **$z_2$** | `z_conduction` | AV & intraventricular conduction | ILBBB, LPFB, 1AVB, IVCD, CLBBB, LAFB | IRBBB, WPW, CRBBB |
| **$z_3$** | `z_morphology` | Infarction, hypertrophy & QRS voltage | RAO/RAE, ASMI, RVH, LVH, IMI, ALMI, LAO/LAE, LMI, AMI | INJAL, ILMI, INJAS, IPLMI |
| **$z_4$** | `z_stt` | ST-T segment & repolarization | ISCIN, DIG, EL, LNGQT, ISCAL, ISCLA, NST_, ISCIL | ISC_, ISCAS, NDT, ANEUR |
| **$z_5$** | `z_residual1` | Discovery / Unmodeled latent A | *(None - Unsupervised)* | *(Evaluated via Probe)* |
| **$z_6$** | `z_residual2` | Discovery / Unmodeled latent B | *(None - Unsupervised)* | *(Evaluated via Probe)* |

**Concept Counts**:
- Total Populated Concepts: **37**
- Anchor Shaping Concepts: **25**
- Held-Out Probe Concepts: **12**

Configuration frozen in: [`configs/ptbxl_concepts.yaml`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/configs/ptbxl_concepts.yaml)

---

## 4. Gate P0 Signoff

`PTBXL_DATA_AUDIT_GATE = PASS`
- Data contract verified.
- Patient disjointness confirmed across all 4 fold tiers.
- Concept ontology partitioned without test-set leakage.
- Ready to proceed to Milestone P1 (Core Architecture & Unit Tests).
