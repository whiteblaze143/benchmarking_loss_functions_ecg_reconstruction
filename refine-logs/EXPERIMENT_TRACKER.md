# Braid Clinical Extension Tracker

| Run ID | Milestone | System / task | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|
| B01-B05 | M0 | Five Braid PTB-XL variants | PTB-XL development train/select | MUST | RUNNING | Three trainers active; remaining variants start after capacity frees. |
| C01 | M1 | Existing four-cohort task-native comparison | Native frozen splits | MUST | PENDING | Depends on all five PTB-XL checkpoints. |
| C02 | M2 | RDB canonical rhythm | Frozen cache train/val/test | MUST | QUEUED | Full 2,398-record loader validated; Braid cells run in catch-up evaluation, then the remaining model roster runs from `rdb_remaining_models_after_braid`. |
| D01 | M3 | ISP delineation | Official train/test | MUST | PENDING | P/QRS/T intervals are present. |
| D02 | M3 | LUDB delineation | Frozen patient folds | MUST | PENDING | Per-lead physician annotations. |
| D03 | M3 | RDB delineation | Frozen cache train/val/test | MUST | PENDING | Native masks and fiducials available. |
| D04 | M3 | Zhejiang delineation | Frozen record-hash folds | MUST | PENDING | Preserve 2,000 Hz to 500 Hz mapping. |
| H01 | M4 | HEEDB-derived Emory-MUSE 12SL diagnoses | Patient-disjoint cohort | MUST | COMPLETE | Frozen admission: 941,679 records / 343,424 patients; all 968,680 coded rows have waveform pairs; 27,001 blank-patient and 5,492 empty-code rows explicitly excluded; 180-code training vocabulary. |
| E01 | M5 | EchoNext SHD | Official split | MUST | BLOCKED | Only test waveforms are local; train/validation restoration required. |
| S01 | M6 | Paired bootstrap/DeLong | Frozen prediction tables | MUST | PENDING | 2,000 paired patient bootstraps; DeLong only matched binary endpoints. |
