# Experiment Tracker: RDB Checkpoint Representation Study

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| E000 | M0 | smoke + resume integrity | enhanced ECG-AIM encoder | 8 RDB records | finite 2,304-D shape, exact resume | MUST | COMPLETE | first pass committed 8 rows; second pass emitted `job_skip_complete`; small-split inference correctly gated |
| E001 | M1 | simple learned baseline | ECG-AIM A0 `1000000_s42` | train/val/test, 2,398 | activation contract | MUST | RUNNING | tmux `checkpoint_embedding_rdb`, CPU 7 only; 324/2,398 at 2026-08-24 00:55 EDT |
| E002 | M1 | enhanced representation | ECG-AIM `1011011_s42` | train/val/test, 2,398 | activation contract | MUST | QUEUED | frozen before analysis |
| E003 | M2 | primary code-level probe | waveform vs A0 vs enhanced | train→val→test | AUROC/AUPRC/Brier/log loss/calibration | MUST | QUEUED | AF/AFIB-coded membership |
| E004 | M2 | secondary multinomial probe | waveform vs A0 vs enhanced | train→val→test | macro AUROC/F1/balanced accuracy | MUST | QUEUED | rare codes descriptive |
| E005 | M2 | repolarization content | waveform vs A0 vs enhanced | train→val→test | QRSon–Toff MAE/correlation/agreement | MUST | QUEUED | not clinical QT/QTc |
| E006 | M2 | projection robustness | A0 + enhanced | all splits | trustworthiness/Jaccard/Procrustes/null | MUST | QUEUED | 5 seeds × 3 neighbors |
| E007 | M2 | quality-control association | enhanced outlier score | validation→test | OR/CI/risk contrast | MUST | ARMED | SHA-matched oracle evaluation 10 verified: 2,398/2,398 `primary_missing_precordial` rows; validation p10 Pearson/p90 MSE failure rule; train PCA-8 + Ledoit–Wolf score |
| E008 | M2 | rigorous resumable post-analysis | all completed feature jobs | matched test records | calibration, paired 2,000× bootstrap, 1,000-label nulls, Holm, QRSon–Toff agreement/Bland–Altman, CKA/RSA/kNN overlap, UMAP Procrustes/continuity | MUST | ARMED | tmux `checkpoint_embedding_postanalysis`; database gate requires all 11 exact counts and released extractor lock, so PID changes/resume are safe; CPU 7 only; synthetic probe/regression/projection tests passed |
| E101–E109 | M3 | checkpoint sensitivity panel | nine frozen blinded-complete ECG-AIM IDs | test, 360 each | neighbor overlap/CKA/probe deltas | NICE | QUEUED | no raw-axis comparison |
| E200 | M4 | live book integration | compact SQLite reader | complete + partial jobs | status, estimands, calibration, paired effects, multiplicity, geometry, UMAP | MUST | COMPLETE-LIVE | all 19 cells render; all 38 local resources resolve; all 18 sidebar HTML targets restored; stale legacy chapters rendered without executing absent archived inputs |
