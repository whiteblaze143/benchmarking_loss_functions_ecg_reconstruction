# Experiment Tracker: Temporal repSpat (`temporal_rep_stat_ecg`)

| Run ID | Experiment Target | Command / Script | Status | Key Results / Artifacts |
|---|---|---|---|---|
| **EXP-TEST** | Mathematical Unit Tests | `pytest temporal_rep_stat_ecg/tests/` | **PASSED** | 12/12 unit tests passed (1.70s) |
| **EXP-SIM** | Autoregressive Simulation Benchmark | `scripts/run_ar_simulation_benchmark.py` | **SCHEDULED** | Table 4 replication across $(\eta, p, n)$ grid |
| **EXP-CLIN** | Clinical Multi-Lead ECG Benchmark | `scripts/run_clinical_ecg_benchmark.py` | **SCHEDULED** | PTB-XL records, Euclidean vs Jaccard |
| **EXP-FULL** | Master Comprehensive Suite | `scripts/run_comprehensive_evaluation.py` | **LAUNCHING (tmux)** | Full evaluation, metrics CSV, publication figures |
