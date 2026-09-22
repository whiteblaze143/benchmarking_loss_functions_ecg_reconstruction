# Foundation-model PTB-XL leakage admission

## Rule

No public foundation-model checkpoint may contribute a PTB-XL fold-8 metric if
PTB-XL was used at any pretraining or finetuning stage.  The sole exception is
a released, checkpoint-specific record manifest that proves every official
fold-8 record was withheld before training.  A paper-level dataset list, a
downstream split, or a claim of a "test set" is insufficient.

This is independent of external-cohort evaluations.  A model trained on PTB-XL
may be evaluated on EchoNext or another independently held-out cohort, but that
result must not be presented as a PTB-XL fold-8 comparison.  The identical rule
applies to Emory-MUSE: a checkpoint whose training corpus includes HEEDB/Emory
cannot contribute an Emory score without a record-level proof that the released
Emory patient-test split was withheld.

## Initial source audit

| Checkpoint family | Public checkpoint/source state | PTB-XL fold-8 admission | Evidence / next required proof |
| --- | --- | --- | --- |
| CLEF small / medium / large | Official source and Zenodo weights; pretraining guide specifies MIMIC-IV-ECG | **Provisional pass** | Verify the released checkpoint configuration and hash against its MIMIC-only pretraining claim before scoring. |
| ECG-JEPA random / multi-block | Official source; released checkpoint links; documented pretraining command uses Shaoxing and CODE-15 | **Provisional pass** | Verify the downloaded checkpoint metadata/config maps to that command before scoring. |
| ECG-FM | Official source states MIMIC-IV-ECG plus PhysioNet Challenge 2021 pretraining | **Blocked** | Challenge-2021 composition alone cannot establish fold-8 exclusion; require record-level manifest. |
| 2026 pretraining-strategies checkpoints | Five packaged checkpoints; repository explicitly lists PTB-XL and contains HEEDB-Emory data configurations | **Blocked on PTB-XL and Emory** | Require checkpoint-specific official PTB-XL fold-8 and Emory test-patient exclusion manifests. |
| ECGFounder | Weights archived locally; README does not identify the released checkpoint's pretraining record manifest | **Blocked for PTB-XL** | Obtain checkpoint-specific pretraining provenance and fold-8 exclusion proof. EchoNext-only cell remains separately admissible. |
| HuBERT-ECG | Public large model; source describes 9.1M ECGs but no record-level PTB-XL exclusion manifest in the repository | **Blocked** | Obtain dataset membership / official fold-8 exclusion proof. |
| ST-MEM | Official source and public encoder link, but source checkout does not establish training-record provenance | **Blocked** | Obtain exact pretraining corpus and fold-8 exclusion proof. |
| ECGFM-KED | Public MIMIC-IV-trained checkpoint stated in official README | **Provisional pass** | Verify checkpoint payload and author preprocessing before scoring. |
| anyECG-chat encoder | Official README states MIMIC-trained ECG encoder; model is multimodal | **Not a core comparator yet** | First demonstrate a frozen ECG-only embedding interface and equal task protocol. |
| ExChanGeAI/CardX | Source code available; public checkpoint provenance not established | **Blocked** | Find a public, checksumable checkpoint with training-corpus provenance. |

## Enforcement

Every PTB-XL foundation-model evaluation command must read a model-specific
admission JSON that includes: source URL/commit, checkpoint hash, all training
datasets, an explicit `ptbxl_fold8_status`, and a pointer to the supporting
release artifact.  The evaluator must fail closed unless status is `pass`.
No zero-padding or lead replication is admissible as a reduced-lead input.
