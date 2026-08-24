# Experiment Tracker

| Run ID | Block | Purpose | Dataset | Status | Gate / Evidence |
|---|---|---|---|---|---|
| DATA-PTB-01 | B | metadata/split audit | PTB-XL | PASS | 21,799 rows; patient-disjoint folds; 2,198 test |
| DATA-ECHO-01 | F | schema/normalization audit | EchoNext | PASS | 100k metadata; 5,442 test waveforms; provenance read |
| DATA-LUDB-01 | D | WFDB parser/inventory | LUDB | PASS | 200/200 records, 12×5,000, 500 Hz |
| DATA-LUDB-02 | D | annotation-count audit | LUDB | PASS | all 2,400 lead streams parsed |
| DATA-ISP-01 | E | interval integrity | ISP | PASS-PARTIAL | 403/72 CSV rows parsed; source rate/class provenance pending |
| DATA-SUN-01 | F | file/provenance inventory | Sunnybrook | PASS | 20 XML files; 500 Hz; 5 µV/bit |
| GRID-2X7-01 | A | enumerate checkpoint matrix | PTB-XL | RUNNING | checkpoints training; authoritative manifest required |
| GRID-2X7-02 | A | completeness/hash gate | all | BLOCKED-BY-TRAINING | run after checkpoints finish |
| EVAL-PTB-01 | B | primary morphology | PTB-XL | TODO | requires locked 160-condition-per-seed manifest |
| EVAL-CLS-01 | C | classifier/calibration | PTB-XL | TODO | per-record paired probabilities |
| EVAL-ECHO-01 | C/F | SHD + transfer | EchoNext | TODO | smoke-test preprocessing first |
| EVAL-LUDB-01 | D | source delineator baseline | LUDB | TODO | establish algorithm ceiling/bias |
| EVAL-LUDB-02 | D | selected-mask delineation | LUDB | TODO | beat-aware matching |
| EVAL-ISP-01 | E | interval endpoint | ISP | HOLD | sampling rate/raw waveform/class mapping |
| EVAL-SUN-01 | F | external waveform/morphometry | Sunnybrook | TODO | retain transient sensitivity |
| EVAL-DEV-01 | F | four-device transfer | smartwatch | TODO | resolve Samsung n=179 |
| EVAL-NOISE-01 | G | 17-condition degradation | PTB-XL | TODO | deterministic paired corruptions |
| EVAL-FAIR-01 | G | subgroup degradation | PTB-XL/EchoNext | TODO | invalid-age rule and support gates |
| EVAL-ABSTAIN-01 | G | risk–coverage | all applicable | TODO | uncertainty score validation |
