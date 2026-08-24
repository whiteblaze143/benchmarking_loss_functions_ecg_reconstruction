# Experiment Tracker: RDB Checkpoint Representation Study

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| E000 | M0 | smoke + resume + STOP integrity | enhanced ECG-AIM encoder | 8 RDB test | finite shape, exact resume | MUST | TODO | CPU core 7, batch 1 |
| E001 | M1 | simple learned baseline | ECG-AIM A0 `1000000_s42` | train/val/test, 2,398 | 2,304-D activation contract | MUST | TODO | exact checkpoint SHA |
| E002 | M1 | enhanced representation | ECG-AIM `1011011_s42` | train/val/test, 2,398 | 2,304-D activation contract | MUST | TODO | frozen before analysis |
| E003 | M2 | primary code-level probe | waveform vs A0 vs enhanced | train→val→test | AUROC/AUPRC/Brier/log loss/calibration | MUST | TODO | AF/AFIB-coded membership |
| E004 | M2 | secondary multinomial probe | waveform vs A0 vs enhanced | train→val→test | macro AUROC/F1/balanced accuracy | MUST | TODO | rare codes descriptive |
| E005 | M2 | repolarization content | waveform vs A0 vs enhanced | train→val→test | QRSon–Toff MAE/correlation/agreement | MUST | TODO | not clinical QT/QTc |
| E006 | M2 | projection robustness | A0 + enhanced | all splits | trustworthiness/Jaccard/Procrustes/null | MUST | TODO | 5 seeds × 3 neighbors |
| E007 | M2 | quality-control association | enhanced outlier score | validation→test | OR/CI/risk contrast | MUST | GATED | requires nonleaky oracle linkage |
| E101–E109 | M3 | checkpoint sensitivity panel | nine frozen blinded-complete ECG-AIM IDs | test, 360 each | neighbor overlap/CKA/probe deltas | NICE | TODO | no raw-axis comparison |
| E200 | M4 | live book integration | compact SQLite reader | complete + partial jobs | tables/plots/provenance | MUST | TODO | no CSV export |
