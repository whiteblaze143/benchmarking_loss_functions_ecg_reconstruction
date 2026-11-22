# Experiment Tracker: T-CAIREM Continuous SetOperator AF-Density Research Program

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Output Artifacts / Notes |
|---|---|---|---|---|---|---|---|---|
| **R001** | M0 | Integrity Sanity: Tensor shapes & BF16 gradient validity | A001 (Continuous SetOperator) | Synthetic 32-window batch | Gradient norm, shape match `(B, 2, 16, 128)` | MUST | READY | `experiments/af_density/test_stage0_integrity.py` |
| **R002** | M0 | Lead Permutation Invariance Preflight | A001 vs Swapped Leads | Synthetic batch | Logit diff $|\hat{s}(C_0, C_1) - \hat{s}(C_1, C_0)|$ | MUST | READY | Expected diff $< 1 \times 10^{-5}$ |
| **R003** | M0 | All-Absent Channel Handling | A001 with mask `[0, 0]` | Synthetic batch | Indeterminate flag, zero fabricated beats | MUST | READY | Must raise indeterminate or return NaN |
| **R004** | M1 | Balanced Overfit Sanity: 128 AF+ / 128 AF- | A001 vs A002 | 256 training windows | BCE loss, AUROC, Epochs to converge | MUST | READY | Target: BCE $< 0.05$, AUROC $> 0.98$ |
| **R005** | M1 | Layer-wise Activation Dispersion Probe | A001 vs A002 vs A003 | 256 validation windows | Active token activation SD, Score SD | MUST | READY | Proves non-collapse ($SD > 0.10$) |
| **R006** | M1 | Continuous vs Categorical Symmetry Test | A001 vs A002 | Validation subset | Cross-channel cosine similarity | MUST | READY | Proves $q$ breaks token symmetry |
| **R007** | M1 | Linear Head Baseline: No Transformer | F053 (Masked Mean Pooling) | 256 validation windows | Score SD, Overfit convergence | MUST | READY | Ablation anchor for set-transformer need |
| **R008** | M2 | Master Model Training: Continuous D1 SetOperator | A001 (EchoNext 1.027M params) | IRIDIA Train (504,777 w) | Patient-weighted validation BCE | MUST | TODO | `best_continuous_setoperator_a001.pt` |
| **R009** | M2 | Matched Categorical Control Training | A002 (Categorical 1.027M params) | IRIDIA Train (504,777 w) | Patient-weighted validation BCE | MUST | TODO | `best_categorical_setoperator_a002.pt` |
| **R010** | M2 | Distributional Morphology Training | D032 (Mean + Var Pooling) | IRIDIA Train (504,777 w) | Patient-weighted validation BCE | MUST | TODO | `best_distributional_setoperator_d032.pt` |
| **R011** | M2 | Ordered-Beat Dynamics Training | D037 (BiGRU Beat Dynamics) | IRIDIA Train (504,777 w) | Patient-weighted validation BCE | MUST | TODO | `best_ordered_dynamics_setoperator_d037.pt` |
| **R012** | M2 | Full Residual ResNet Fusion Training | I082 (SetOperator + ResNet Bypass) | IRIDIA Train (504,777 w) | Patient-weighted validation BCE | MUST | TODO | `best_residual_fusion_setoperator_i082.pt` |
| **R013** | M3 | Raw Validation Score Replay & Calibration | A001, A002, D032, D037, I082 | IRIDIA Val (166,047 w) | ECE, Brier Score, AUROC, AUPRC | MUST | TODO | Temperature calibration parameter $T_{\text{cal}}$ |
| **R014** | M3 | D0 Window Thresholding Benchmarking | All models vs Raw & RR Baselines | IRIDIA Val (60 recordings) | Window Se, Sp, F1, Balanced Acc | MUST | TODO | Evaluated under fixed policy $\tau \in [0.01, 0.99]$ |
| **R015** | M3 | D1 Median Filter Episode Delineation | All models | IRIDIA Val (11 informative pts) | Episode Se, PPV, Split/Merge Rate | MUST | TODO | Gap merge $\le 60$s within observed blocks |
| **R016** | M3 | D2 HSMM Episode & Density Benchmarking | All models | IRIDIA Val (11 informative pts) | Charitos Continuous MAE, Binned MAE | MUST | TODO | Primary Table 1 with Bootstrap CIs ($B=2,000$) |
| **R017** | M3 | Missing Time Sensitivity: Manski vs Context | I082 vs Raw Mask-ResNet | IRIDIA Val (30 patients) | $[Burden_{\min}, Burden_{\max}]$, MAE shift | MUST | TODO | Evaluates Manski worst-case bounds |
| **R018** | M4 | Zero-Tuning External Benchmark: SHDB | I082 vs Baselines | SHDB (98 records, 93 subjects) | Patient Specificity, False Burden, MAE | MUST | TODO | 49 multi-interval informative subjects |
| **R019** | M4 | Zero-Tuning External Benchmark: LTAF | I082 vs Baselines | LTAF (84 records, 128 Hz) | Episode Se, PPV, Burden Pearson $r$ | MUST | TODO | 66 multi-interval informative records |
| **R020** | M4 | Multi-Lead Dropout & Permutation Audit | I082 | Full Validation & SHDB | Logit max diff, Burden spread (%) | MUST | TODO | Tests invariance under lead swap |
