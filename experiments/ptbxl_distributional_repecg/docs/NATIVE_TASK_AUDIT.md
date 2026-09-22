# Native Dataset Task Audit

**Date:** 2026-09-17  
**Policy:** use each dataset's native labels and split; never replace missing labels with zeros and never force every dataset into PTB-XL's five superclasses.

**Audit conclusion:** every dataset in scope has native supervision. The current
production blocker is label ingestion: the legacy OOD representation builder
discards those labels and writes a fabricated five-zero PTB-XL target for every
record. That output is invalid and must not be interpreted as evidence that the
source datasets are unlabeled.

The machine-readable contract is `configs/dataset_tasks.json`.

| Dataset | Native task(s) | Audited label evidence | Split contract | Production note |
|---|---|---|---|---|
| PTB-XL | Five-superclass multilabel diagnosis | `NORM, MI, STTC, CD, HYP` | Patient-disjoint official folds | Fold 10 stays locked until final promotion. |
| EchoNext | 12 structural-heart-disease endpoints | 100,000 metadata rows: 72,475 train, 4,626 validation, 5,442 test, 17,457 `no_split`; all 12 binary flags present | Official `split` column | Local labels are complete. Local waveform archive currently contains only the 5,442 test waveforms; do not fit a repECG head on test. Locate/restore train/validation waveforms first. |
| Kingston-ICU | SINUS versus AFIB/AFLT | 596 records: 497 SINUS, 99 AFIB/AFLT; 298 Train and 298 Test | Official `TrainOrTest` | Zero-padding negative control. Inputs retain I/II only; six precordial slots are zero after frozen PTB-XL standardization. |
| LUDB | Rhythm, broad diagnostic categories, P-QRS-T delineation | 200 labeled patients. The source field is `Rhythms` (plural), and all rows are populated; nonempty counts: axis 190, conduction 66, extrasystoles 14, hypertrophy 142, pacing 10, ischemia 51, repolarization 49, other 9; physician delineations are present | Frozen patient-level stratified five-fold CV | Parse newline-separated native terms; report broad-category multilabel results and rhythm separately. Rare labels need explicit denominators. |
| RDB | Eight-class canonical rhythm and P-QRS-T delineation | Frozen cache has 1,678 train, 360 validation, 360 test. Rhythm vocabulary: `ST, AF, SA, AFIB, SR, SB, AT, SVT`; delineation masks stored per record | Existing train/val/test cache directories | Preserve the cache split and patient IDs; do not infer labels from filenames when payload fields exist. |
| ISP | P-QRS-T delineation, sex classification, age regression | 403 train and 72 test rows; every row has interval targets, binary sex, and age in years | Official train/test CSV files | No diagnosis field is released. Keep age continuous rather than inventing bins. |
| Zhejiang | RVOT versus LVOT origin, PVC versus VT type, P-QRS-T delineation | Official `Diagnosis.xlsx` joins exactly to all 334 mask IDs: 257 RVOT/77 LVOT and 329 PVC/5 VT; audited pixel counts: ISO 5,546,062, P 176,499, QRS 354,438, T 603,001 | Frozen record-hash five-fold CV, stratified on RVOT/LVOT | RVOT/LVOT is the primary ablation-validated classification task. PVC/VT is secondary and extremely imbalanced. Source is 2,000 Hz and model coordinates are 500 Hz after exact 4:1 mapping. Voltage gain remains undeclared, so amplitude-dependent production evaluation is blocked. |
| Emory-MUSE | 12SL diagnosis-code multilabel classification | 974,172 unique `FileName` rows join to 998,844 metadata rows; all 968,680 diagnosis-eligible rows have exact WFDB `.hea`/`.dat` pairs. The frozen admission contains 941,679 ECGs from 343,424 patients and all 180 training-derived codes; exclusions are 5,492 empty-code rows and 27,001 coded rows with blank source `BDSPPatientID` | Frozen SHA-256 patient split: 751,980 train / 95,774 validation / 93,925 test records, with zero patients crossing splits | Use `benchmark2/emory_muse_admission_v1`; never infer patient IDs for the explicit blank-ID exclusions. Keep raw codes, exclusions, and dictionary text in provenance. |
| Sunnybrook | Philips statement-code multilabel evaluation | All 20 XML records contain interpretation statement codes; 29 distinct codes observed | External-only, no fitting (`n=20`) | Report per-record predictions and aggregate only labels with valid positive and negative denominators. This is a small external case series, not a training cohort. |

## EchoNext test denominators

Positive counts in the 5,442-record official test set are: LVEF <=45 962; LV wall thickness >=13 1,061; moderate-or-greater aortic stenosis 286; aortic regurgitation 66; mitral regurgitation 337; tricuspid regurgitation 353; pulmonary regurgitation 20; RV systolic dysfunction 419; moderate/large pericardial effusion 69; PASP >=45 699; TR max >=3.2 375; any moderate-or-greater SHD 2,318.

## Required production consequences

1. Replace the current OOD builder's fabricated five-zero label vector with native label loaders.
2. Give every result row a `task_id`; metrics from different tasks are not pooled as though they share a target.
3. Use task-specific heads over frozen representations. Classification heads are fit only on native training data and selected only on native validation data.
4. Use sequence heads and sequence metrics for ISP, Zhejiang, and delineation tracks in LUDB/RDB.
5. Store complete and ineligible metric rows with explicit positive/negative denominators.
6. Kingston remains tagged `zero_padding_negative_control` even though its native AF rhythm labels are valid.
7. Treat `tit_ecg/src/dataset_adapters.py` as waveform-only legacy code until
   each adapter returns the native label payload declared above. In particular,
   absence of a label key in an adapter return value is an integration defect,
   not absence of labels in the dataset.

## Kingston zero-padding control results

The complete matched PTB-XL fold-8 control contains 2,173 ECGs from 1,881
patients. The full and I/II-only conditions use identical records, labels,
beats, checkpoints, and Gaussian-control random seeds. With inverse-record-count
patient weighting, masking six standardized channels reduced macro AUROC by
0.1347 for the kernel representation (95% patient-bootstrap CI -0.1461 to
-0.1233) and by 0.2408 for moments (CI -0.2578 to -0.2251). Because fold 8 was
also used for checkpoint selection, these are descriptive robustness-control
results rather than independent test evidence.

On Kingston's separate native rhythm task, heads selected only within the
official Train split achieved Test AUROC 0.8786 for kernel and 0.8843 for
moments (48 AFIB/AFLT and 245 SINUS eligible Test records). This shows that a
task-specific head can still learn useful rhythm information from I/II despite
zero padding. It does not contradict the matched PTB-XL result: the latter
measures degradation when a model trained on the full eight-lead interface is
given six zero-filled channels. Kingston provides no patient identifier, so its
confidence intervals are record-bootstrap and must not be called patient-level.
