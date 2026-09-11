# Experiment Tracker: Q-VCG Quotient Representation Learning

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| **R_M0_GEOM** | M0 | Unit verification of Table 7 direction matrix and SVD inverse | Geometry Unit Tests | Synthetic | Max error $< 10^{-5}$, rank=3 | MUST | TODO | `tests/test_vcg_geometry.py` |
| **R_M0_DYN** | M0 | Unit verification of Savitzky-Golay and intrinsic descriptor $\xi$ | Dynamics Unit Tests | Synthetic | Circle curvature $= 1.0$, rotation invariance | MUST | TODO | `tests/test_vcg_dynamics.py` |
| **R_M0_REPSPAT**| M0 | Synthetic repSpat unit tests (Tests A–F) | RepSpat Unit Tests | Synthetic | Recurrence merge, different separation, null control | MUST | TODO | `tests/test_motif_discovery.py` |
| **R_M0_REPR** | M0 | Representation extraction contracts | Repr Unit Tests | Synthetic | Occupancy sum=1, dwell ms, run compression | MUST | TODO | `tests/test_qvcg_representation.py` |
| **R_M1_CENSUS**| M1 | Patient-balanced training microstate extraction | Microstate Builder | PTB-XL Folds 1–7 | $N_{\text{patient}} \approx 2,500$, zero NaNs | MUST | TODO | `scripts/build_vcg_microstates.py` |
| **R_M2_DOMAINS**| M2 | Spatially contiguous domain discovery via CAHC | 3D Lattice + CAHC | PTB-XL Folds 1–7 | $M=64$ contiguous spatial domains | MUST | TODO | `scripts/discover_vcg_domains.py` |
| **R_M3_QUOTIENT**| M3 | Self-calibrated MMD matrix & patient block permutation | RepSpat Quotient | PTB-XL Folds 1–7 | $\epsilon_{\text{MMD}}$ threshold, $K < M$ motifs | MUST | TODO | `scripts/discover_repeated_motifs.py` |
| **R_M4_FEAT** | M4 | Feature serialization ($R_0, R_1, R_2, R_3$) | Feature Extractor | PTB-XL Folds 1–9 | Parquet serialization, OOD rate logged | MUST | TODO | `scripts/build_qvcg_features.py` |
| **R_M5_PROBES**| M5 | Frozen linear probing across 4 diagnostic tasks | Logistic Regression | Train: Folds 1–7, Val: Fold 8 | Macro AUROC / AUPRC across 4 tasks | MUST | TODO | `scripts/evaluate_qvcg.py` |
| **R_M5_FACTOR**| M5 | Function vs. location factorization ablation | $Z_{\text{motif}}$ vs $Z_{\text{where}}$ vs $R_3$ | PTB-XL Fold 8 | Delta AUROC (Outcome A check) | MUST | TODO | Superclass & Subclass |
| **R_M5_LOWSHOT**| M5 | Sample efficiency evaluation (1%, 5%, 10%, 25%, 50%, 100%)| Low-shot Probes | PTB-XL Fold 8 | AUROC curve vs label percentage | MUST | TODO | Compare R0, R1, R2, R3 |
| **R_M6_STABIL**| M6 | Patient split-half stability ($H_1$ vs $H_2$) | RepSpat Split-Half | PTB-XL Folds 1–7 | AMI, Variation of Information | MUST | TODO | Reproducibility report |
| **R_M6_RETRIEVE**| M6 | Clinical multi-label nearest-neighbor retrieval | Vector Retrieval | PTB-XL Fold 8 | Precision@k, Recall@k, nDCG@k | MUST | TODO | Evaluation on Fold 8 |
| **R_M6_GATE** | M6 | Transparent Gate qualification verdict | Kill Gate Evaluator | PTB-XL Fold 8 | 7 criteria PASS/FAIL | MUST | TODO | `QVCG_TRANSPARENT_GATE.json` |
| **R_M7_CONFIRM**| M7 | Fold 9 confirmation evaluation | Frozen Pipeline | PTB-XL Fold 9 | Confirmation AUROC/AUPRC | MUST | TODO | Sealed until Gate PASS |
