# Detailed interpretation of ECG reconstruction results so far

**Snapshot:** 2026-08-12T17:36:52-04:00

This report separates *what the data currently demonstrates* from what is still incomplete or invalid. The corrected evaluator is authoritative for clinical claims. Legacy rows are retained only for exploratory signal-level analysis restricted to the nine reconstructed leads.

## Executive verdict

1. **No architecture-level conclusion is currently permitted.** The registry contains 160/160 U-Nets, 61/160 MSVAEs, and 0/160 ECG-AIM models. The formal architecture gate is therefore closed.
2. **The strongest repeatable U-Net loss finding is that correlation loss helps and energy distance hurts.** Across all 160 U-Net masks, correlation improved mean missing-lead Pearson on PTB-XL, EchoNext, and Sunnybrook; energy distance reduced it for every one of 80 matched pairs on each cohort.
3. **The best fully logged MSVAE training run so far is `msvae_f_1100000_s42`** with best validation missing-lead Pearson 0.8736 at epoch 10. This is a training/validation result, not yet a corrected clinical or external-cohort result.
4. **The one corrected PTB-XL model shows clinically meaningful degradation.** For `f_1000000_s42`, reconstruction lowers paired ECGFounder macro AUROC by 0.0349 and macro AUPRC by 0.0774, while worsening Brier score and calibration error. All four patient-cluster bootstrap intervals exclude zero.
5. **The corrected evaluator is not advancing.** It crashes on NaN LVH scores after completing the first PTB-XL model, then the watcher restarts it. The CSV contains 1,287 rows but only 33 unique rows (1,254 duplicates). SQLite correctly retains only the latest unique rows, and is the source used below.

## Evidence and completeness audit

| Layer | Current coverage | Usable for | Not usable for |
| --- | --- | --- | --- |
| MSVAE training logs | 60 fully parsed; 1 running; 3 failed | Within-MSVAE validation ranking and convergence | Architecture or clinical superiority |
| Corrected V2 evaluator | 1 model, 1 dataset, 33 metric rows, 8 paired rows | Paired PTB-XL ECGFounder, QRS, and LVH agreement for one U-Net | Model ranking, EchoNext, external generalization, architecture claims |
| Legacy evaluator | 179 models across 4 datasets | Exploratory nine-missing-lead signal and ST metrics | QRS/delineation endpoints and formal clinical claims |
| ECG-AIM | 0/160 complete | Nothing yet | Any performance statement |

SQLite integrity check: **`ok`**.

Queue state at report generation: 61 completed, 1 running, 3 failed, and 255 pending out of 320 jobs.

## Experimental design and mask decoding

A mask is `MSE-Corr-Deriv-VCG-ED-Lead-MMD`. MSE is always enabled. Correlation, first-derivative L1, Kors vectorcardiogram, energy distance, and Goldberger lead-consistency losses are binary. The final digit selects no MMD (0), global adaptive RBF (1), anatomical Laplacian (2), anatomical multiscale IMQ (3), or temporal K-means multiscale IMQ (4).

| Position | Symbol | Meaning |
| --- | --- | --- |
| 1 | MSE | Anchor reconstruction loss; always 1 |
| 2 | Corr | Pearson correlation loss |
| 3 | Deriv | First-difference L1 loss |
| 4 | VCG | Kors VCG angle and magnitude loss |
| 5 | ED | Empirical energy distance |
| 6 | Lead | Goldberger limb-lead consistency |
| 7 | MMD | Kernel variant 0–4 |

All current MSVAE runs use seed 42. Consequently, differences among masks do not yet include seed-to-seed uncertainty. Final composite losses are not comparable between masks because enabling an additional term changes the scale and meaning of the optimized objective; validation Pearson is the comparable training endpoint.

## MSVAE training results

Across 60 fully logged runs, best validation missing-lead Pearson is mean 0.7672, median 0.8131, SD 0.1646, and range 0.1095–0.8736. The median runtime is 48.6 minutes. 43/60 runs achieve their best Pearson at epoch 10, so many configurations are still improving at the fixed training horizon.

Checkpoint selection matters: 10 runs lose more than 0.01 Pearson from their best epoch to epoch 10. The most unstable run drops 0.3036. Rankings below use the best validation checkpoint rather than the last epoch.

### Best and worst fully logged MSVAE configurations

| Model | Best r | Best epoch | Epoch-1 r | Final r | Best→final drop | Minutes |
| --- | --- | --- | --- | --- | --- | --- |
| msvae_f_1100000_s42 | 0.8736 | 10 | 0.7877 | 0.8736 | 0.0000 | 47.3 |
| msvae_f_1100003_s42 | 0.8701 | 10 | 0.7903 | 0.8701 | 0.0000 | 49.0 |
| msvae_f_1101000_s42 | 0.8677 | 10 | 0.7575 | 0.8677 | 0.0000 | 47.5 |
| msvae_f_1011002_s42 | 0.8675 | 9 | 0.7860 | 0.8593 | 0.0082 | 48.3 |
| msvae_f_1010003_s42 | 0.8660 | 8 | 0.7203 | 0.7773 | 0.0887 | 49.0 |
| msvae_f_1011000_s42 | 0.8656 | 9 | 0.7356 | 0.8599 | 0.0057 | 47.5 |
| msvae_f_1011004_s42 | 0.8656 | 10 | 0.7478 | 0.8656 | 0.0000 | 49.2 |
| msvae_f_1011003_s42 | 0.8642 | 10 | 0.7479 | 0.8642 | 0.0000 | 49.0 |
| msvae_f_1010000_s42 | 0.8634 | 9 | 0.7301 | 0.8341 | 0.0293 | 47.3 |
| msvae_f_1001003_s42 | 0.8615 | 10 | 0.7860 | 0.8615 | 0.0000 | 49.3 |
| msvae_f_1001004_s42 | 0.8615 | 10 | 0.7541 | 0.8615 | 0.0000 | 49.1 |
| msvae_f_1100002_s42 | 0.8603 | 8 | 0.7600 | 0.8548 | 0.0055 | 48.2 |
| msvae_f_1000101_s42 | 0.7216 | 10 | -0.0069 | 0.7216 | 0.0000 | 48.5 |
| msvae_f_1000113_s42 | 0.7175 | 10 | 0.0600 | 0.7175 | 0.0000 | 50.1 |
| msvae_f_1010101_s42 | 0.6714 | 9 | 0.0007 | 0.6687 | 0.0027 | 48.2 |
| msvae_f_1010104_s42 | 0.6681 | 10 | 0.0434 | 0.6681 | 0.0000 | 49.5 |
| msvae_f_1000112_s42 | 0.5598 | 10 | 0.0188 | 0.5598 | 0.0000 | 49.4 |
| msvae_f_1100102_s42 | 0.4517 | 10 | 0.0000 | 0.4517 | 0.0000 | 48.8 |
| msvae_f_1000114_s42 | 0.3947 | 10 | -0.0150 | 0.3947 | 0.0000 | 50.1 |
| msvae_f_1001100_s42 | 0.3046 | 4 | 0.0081 | 0.0010 | 0.3036 | 48.1 |
| msvae_f_1010103_s42 | 0.1279 | 4 | 0.0405 | 0.1111 | 0.0168 | 49.4 |
| msvae_f_1000110_s42 | 0.1095 | 2 | 0.0108 | 0.0600 | 0.0495 | 48.0 |

The top run, `1100000`, is MSE plus correlation with no other auxiliary term. The next-best runs mostly add either MMD-3, VCG, or derivative loss without energy distance. This pattern is consistent with the U-Net external signal results, where the same `110000x` family dominates.

The severe failures cluster around energy distance: `1000110`, `1010103`, and `1001100` reach only 0.1095, 0.1279, and 0.3046 best Pearson. Some MMD variants rescue those combinations, producing very large positive pairwise effects, but that is recovery from an unstable baseline—not evidence that MMD is generally beneficial.

### Matched MSVAE loss-component effects

| Change | Matched pairs | Mean Δr | Median Δr | Beneficial | Range |
| --- | --- | --- | --- | --- | --- |
| Correlation | 9 | -0.00404 | 0.02350 | 8/9 | -0.3105 to 0.0882 |
| Derivative | 18 | -0.00512 | 0.00805 | 13/18 | -0.6398 to 0.4795 |
| VCG | 21 | 0.06340 | 0.02510 | 17/21 | -0.0466 to 0.6953 |
| Energy distance | 27 | -0.15900 | -0.08150 | 2/27 | -0.7492 to 0.0120 |
| Lead consistency | 9 | -0.05849 | -0.00120 | 4/9 | -0.3627 to 0.0933 |
| MMD-1 vs none | 12 | 0.08112 | -0.00910 | 3/12 | -0.0844 to 0.7054 |
| MMD-2 vs none | 10 | 0.05484 | -0.00475 | 4/10 | -0.3593 to 0.5020 |
| MMD-3 vs none | 11 | 0.04189 | 0.00150 | 6/11 | -0.6279 to 0.6080 |
| MMD-4 vs none | 10 | 0.04807 | -0.00015 | 4/10 | -0.1025 to 0.5033 |

These MSVAE effects are interim because the completed subset is not the full factorial and contains only one seed. The robust directional finding is energy distance: mean matched Δr is strongly negative and only 2/27 pairs improve. VCG is usually protective, especially in energy-distance failures. Correlation and derivative losses have positive medians but their means are pulled negative by single catastrophic interactions.

### Failed and provenance-limited MSVAE jobs

| Job | Recorded reason | Interpretation |
| --- | --- | --- |
| msvae_f_1000004_s42 | training exited with status 1 | CUDA/Triton hardware-access failure; not evidence that the loss mask is invalid |
| msvae_f_1000012_s42 | training exited with status 1 | CUDA/Triton hardware-access failure; not evidence that the loss mask is invalid |
| msvae_f_1000100_s42 | training exited with status 1 | CUDA/Triton hardware-access failure; not evidence that the loss mask is invalid |

The failed logs report `Invalid access of peer GPU memory over nvlink or a hardware error`. They should be retried after GPU health is established. The administratively completed `msvae_f_1111112_s42` checkpoint has no matching per-job log in the current log directory and is excluded from the quantitative training table; its compatibility audit also lacks the current source/data contract.

## Corrected V2 PTB-XL evaluation: `f_1000000_s42`

This is the only model/dataset pair currently evaluated with independent delineation over the nine missing leads. It contains 2,198 ECG records from 1,904 patients for ECGFounder and 2,197 records from 1,903 patients for QRS. All paired intervals below use 500 patient-cluster bootstrap replicates.

### Paired original-versus-reconstructed clinical inference

| Endpoint | Metric | Original/reference | Reconstruction | Δ | 95% CI | p | Patients |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ECGFounder_Macro | auroc | 0.88406 | 0.84918 | -0.03488 | [-0.04394, -0.02611] | 0.0040 | 1904 |
| ECGFounder_Macro | auprc | 0.47694 | 0.39955 | -0.07739 | [-0.09959, -0.05225] | 0.0040 | 1904 |
| ECGFounder_Macro | brier | 0.02604 | 0.02653 | 0.00048 | [0.00022, 0.00076] | 0.0040 | 1904 |
| ECGFounder_Macro | ece | 0.04466 | 0.04967 | 0.00500 | [0.00453, 0.00542] | 0.0040 | 1904 |
| LVH_SokolowLyon | mae | 0.00000 | 0.55923 | 0.55923 | [0.53558, 0.58520] | — | 1607 |
| LVH_SokolowLyon | bias | 0.00000 | -0.39718 | -0.39718 | [-0.42979, -0.36521] | 1.93e-113 | 1607 |
| QRS_MissingLeads | mae | 0.00000 | 16.83535 | 16.83535 | [16.37918, 17.35223] | — | 1903 |
| QRS_MissingLeads | bias | 0.00000 | -14.31317 | -14.31317 | [-14.94425, -13.72028] | 1.18e-223 | 1903 |

Reconstruction reduces ECGFounder macro AUROC from 0.8841 to 0.8492 (Δ −0.0349, 95% CI −0.0439 to −0.0261) and macro AUPRC from 0.4769 to 0.3995 (Δ −0.0774, 95% CI −0.0996 to −0.0523). Brier score rises by 0.00048 and expected calibration error rises by 0.00500; because lower is better for both, calibration also worsens. These are paired degradation estimates, not merely differences between two independent summaries.

The QRS estimate is systematically short by 14.31 ms, with MAE 16.84 ms and patient-bootstrap MAE interval 16.38–17.35 ms. LVH voltage is systematically attenuated by 0.397 mV, with MAE 0.559 mV. Both biases exclude zero by a wide margin, indicating directional amplitude/timing distortion rather than only random reconstruction noise.

### Corrected missing-lead QRS endpoint

| MAE ms | Pearson | R² | Bias ms | Limits of agreement ms | AUROC | AUPRC | Sensitivity | Specificity | Adjusted OR (95% CI), p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.8353 | 0.5322 | 0.2833 | -14.3132 | -43.5067 to 14.8804 | 0.9069 | 0.9993 | 0.9931 | 0.3333 | 0.6458 (0.2454–1.6997), p=0.3759 |

The continuous QRS agreement is only moderate (r=0.532, R²=0.283), with wide limits of agreement from −43.51 to +14.88 ms. The near-perfect AUPRC and 99.3% sensitivity should not be read as overall clinical fidelity: specificity is only 33.3%, and the adjusted association with conduction disease is non-significant (OR 0.646, 95% CI 0.245–1.700, p=0.376). The threshold result is therefore prevalence-sensitive and poorly discriminates the negative class despite a high ranking metric.

### ECGFounder task-level performance on reconstructed ECGs

Among 31 reported tasks, 17 have AUROC ≥0.90, but 4 have AUPRC <0.10, 7 have F1=0, and 16 have sensitivity <0.20 at the fixed 0.5 threshold. Median AUROC is high, but median sensitivity is only 0.150. This is classic rare-label behavior: ranking can look strong while thresholded detection remains clinically weak.

| Task | AUROC (CI) | AUPRC (CI) | F1 | Sensitivity | Specificity | PPV | NPV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SUPRAVENTRICULAR_TACHYCARDIA | 0.9977 (0.9946–0.9997) | 0.4001 (0.1476–0.9108) | 0.5000 | 0.6000 | 0.9982 | 0.4286 | 0.9991 |
| SINUS_TACHYCARDIA | 0.9902 (0.9807–0.9950) | 0.7921 (0.7126–0.9261) | 0.7900 | 0.9634 | 0.9816 | 0.6695 | 0.9986 |
| VENTRICULAR_TACHYCARDIA | 0.9899 (0.9744–0.9991) | 0.2720 (0.0798–0.5615) | 0.4444 | 0.8000 | 0.9959 | 0.3077 | 0.9995 |
| LEFT_BUNDLE_BRANCH_BLOCK | 0.9821 (0.9522–0.9990) | 0.9176 (0.8398–0.9637) | 0.8467 | 0.9355 | 0.9920 | 0.7733 | 0.9981 |
| ATRIAL_FIBRILLATION | 0.9776 (0.9642–0.9889) | 0.8983 (0.8446–0.9443) | 0.8252 | 0.9474 | 0.9741 | 0.7310 | 0.9960 |
| PREMATURE_VENTRICULAR_COMPLEXES | 0.9767 (0.9664–0.9874) | 0.7515 (0.6378–0.8353) | 0.6820 | 0.9123 | 0.9583 | 0.5445 | 0.9950 |
| ATRIAL_FLUTTER | 0.9677 (0.9253–0.9945) | 0.4224 (0.0496–0.7951) | 0.2424 | 0.5714 | 0.9900 | 0.1538 | 0.9986 |
| SEPTAL_INFARCT | 0.9528 (0.9378–0.9640) | 0.7421 (0.6779–0.7978) | 0.7315 | 0.7393 | 0.9664 | 0.7238 | 0.9689 |
| QT_HAS_LENGTHENED | 0.9452 (0.9069–0.9788) | 0.1419 (0.0402–0.3815) | 0.1818 | 0.2727 | 0.9913 | 0.1364 | 0.9963 |
| SINUS_BRADYCARDIA | 0.9400 (0.9186–0.9612) | 0.3031 (0.2321–0.4419) | 0.2731 | 0.9688 | 0.8463 | 0.1590 | 0.9989 |
| RIGHT_ATRIAL_ENLARGEMENT | 0.9340 (0.8993–0.9880) | 0.2324 (0.0550–0.4832) | 0.2667 | 0.2000 | 0.9986 | 0.4000 | 0.9964 |
| ANTEROSEPTAL_INFARCT | 0.9211 (0.8936–0.9409) | 0.6904 (0.6290–0.7405) | 0.2444 | 0.1410 | 0.9985 | 0.9167 | 0.9070 |
| LEFT_ANTERIOR_FASCICULAR_BLOCK | 0.9197 (0.9003–0.9352) | 0.6076 (0.5505–0.6715) | 0.1878 | 0.1049 | 0.9990 | 0.8947 | 0.9335 |
| RIGHT_BUNDLE_BRANCH_BLOCK | 0.9141 (0.8893–0.9307) | 0.6212 (0.5565–0.6832) | 0.5421 | 0.5241 | 0.9665 | 0.5613 | 0.9613 |
| WOLFF_PARKINSON_WHITE | 0.9061 (0.8223–0.9948) | 0.5189 (0.2783–0.9630) | 0.2222 | 0.1250 | 1.0000 | 1.0000 | 0.9968 |
| WITH_1ST_DEGREE_AV_BLOCK | 0.9046 (0.8809–0.9296) | 0.3665 (0.2787–0.4983) | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9641 |
| RIGHT_VENTRICULAR_HYPERTROPHY | 0.9026 (0.8315–0.9668) | 0.1371 (0.0188–0.3048) | 0.1429 | 0.0833 | 0.9995 | 0.5000 | 0.9950 |
| NORMAL_ECG | 0.8171 (0.8038–0.8349) | 0.7556 (0.7241–0.7862) | 0.6860 | 0.6864 | 0.7547 | 0.6857 | 0.7553 |
| SINUS_RHYTHM | 0.8067 (0.7897–0.8224) | 0.9216 (0.9090–0.9325) | 0.8440 | 0.8793 | 0.3473 | 0.8115 | 0.4740 |
| ANTEROLATERAL_LEADS | 0.8060 (0.7653–0.8676) | 0.1127 (0.0860–0.1805) | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9700 |
| LEFT_POSTERIOR_FASCICULAR_BLOCK | 0.8060 (0.6775–0.9079) | 0.1669 (0.0419–0.3834) | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9918 |
| ANTEROLATERAL_INFARCT | 0.8013 (0.7210–0.8993) | 0.1507 (0.0833–0.2899) | 0.0667 | 0.0370 | 0.9991 | 0.3333 | 0.9882 |
| LATERAL_INFARCT | 0.7949 (0.7570–0.8296) | 0.2152 (0.1712–0.2902) | 0.2000 | 0.1500 | 0.9833 | 0.3000 | 0.9604 |
| ELECTRONIC_ATRIAL_PACEMAKER | 0.7826 (0.7116–0.8373) | 0.0912 (0.0433–0.1975) | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9873 |
| LEFT_VENTRICULAR_HYPERTROPHY | 0.7607 (0.7406–0.7917) | 0.3271 (0.2798–0.3836) | 0.0968 | 0.0513 | 0.9990 | 0.8571 | 0.8984 |
| INFERIOR_INFARCT | 0.7394 (0.7027–0.7729) | 0.2841 (0.2507–0.3507) | 0.0075 | 0.0037 | 1.0000 | 1.0000 | 0.8789 |
| NONSPECIFIC_INTRAVENTRICULAR_BLOCK | 0.6809 (0.6248–0.7635) | 0.0830 (0.0637–0.1349) | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9641 |
| LOW_VOLTAGE_QRS | 0.6541 (0.6353–0.6708) | 0.2518 (0.2271–0.2833) | 0.2713 | 0.2422 | 0.9067 | 0.3083 | 0.8746 |
| LEFT_ATRIAL_ENLARGEMENT | 0.6395 (0.5779–0.6944) | 0.0343 (0.0223–0.0643) | 0.0484 | 0.1429 | 0.9072 | 0.0291 | 0.9819 |
| ANTERIOR_INFARCT | 0.5814 (0.5233–0.6558) | 0.0196 (0.0143–0.0308) | 0.0000 | 0.0000 | 0.9769 | 0.0000 | 0.9837 |
| WITH_QRS_WIDENING | 0.5319 (0.5003–0.5584) | 0.1566 (0.1388–0.1815) | 0.0000 | 0.0000 | 0.9979 | 0.0000 | 0.8532 |

Task-level intervals above use the evaluator's 50-replicate, record-level bootstrap rather than the 500-replicate patient-cluster bootstrap used for macro paired deltas. They are useful descriptively but do not yet satisfy the requested patient-level inference standard for every task.

## Legacy signal-level results: exploratory only

The legacy evaluator directly compares reconstructed and target waveforms per lead. Those rows remain informative after removing the copied observed leads I, II, and V2. Its QRS, boundary, and morphology rows are invalid: zero QRS and boundary error and Dice ≈1.0 for every U-Net reveal that the old code measured the copied V2 rather than reconstructed leads. No claim below uses those rows.

| Dataset | Architecture | Models | Mean r | Median r | r range | Mean MAE | Median MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| echonext | MSVAE | 19 | 0.6720 | 0.7576 | 0.0781–0.8144 | 0.3316 | 0.3276 |
| echonext | U-Net | 160 | 0.4898 | 0.4324 | 0.3057–0.7767 | 0.3707 | 0.3734 |
| ludb | U-Net | 7 | 0.6418 | 0.6477 | 0.5739–0.7035 | 0.0987 | 0.0972 |
| ptb_xl | MSVAE | 19 | 0.7010 | 0.8032 | -0.0215–0.8834 | 0.0850 | 0.0735 |
| ptb_xl | U-Net | 160 | 0.5483 | 0.5022 | 0.3543–0.8598 | 0.1284 | 0.1295 |
| sunnybrook | U-Net | 160 | 0.4990 | 0.4629 | 0.3036–0.7822 | 0.2709 | 0.2716 |

MAE is dataset-scale dependent and should not be compared directly between PTB-XL, EchoNext, Sunnybrook, and LUDB. Pearson rank patterns are much more stable across cohorts. The early 19-model MSVAE subset appears stronger than U-Net on PTB-XL and EchoNext, but it is selectively sampled, includes models that later failed inference-readiness retries, and was evaluated only with the legacy semantics. It is hypothesis-generating, not an architecture result.

### Full 160-mask U-Net factorial effects on missing-lead signal Pearson

| Dataset | Change | Pairs | Mean Δr | Median Δr | Beneficial |
| --- | --- | --- | --- | --- | --- |
| ptb_xl | Correlation | 80 | 0.14960 | 0.10326 | 80/80 |
| ptb_xl | Derivative | 80 | -0.04246 | -0.02213 | 0/80 |
| ptb_xl | VCG | 80 | -0.00900 | 0.02130 | 43/80 |
| ptb_xl | Energy distance | 80 | -0.24258 | -0.25078 | 0/80 |
| ptb_xl | Lead consistency | 80 | -0.00659 | 0.00732 | 47/80 |
| ptb_xl | MMD-1 vs none | 32 | -0.02389 | -0.00465 | 7/32 |
| ptb_xl | MMD-2 vs none | 32 | -0.00773 | -0.00102 | 7/32 |
| ptb_xl | MMD-3 vs none | 32 | -0.00085 | -0.00004 | 10/32 |
| ptb_xl | MMD-4 vs none | 32 | -0.00325 | -0.00017 | 10/32 |
| echonext | Correlation | 80 | 0.13047 | 0.09754 | 80/80 |
| echonext | Derivative | 80 | -0.02363 | -0.01004 | 16/80 |
| echonext | VCG | 80 | -0.02854 | 0.00431 | 41/80 |
| echonext | Energy distance | 80 | -0.23116 | -0.23084 | 0/80 |
| echonext | Lead consistency | 80 | 0.00823 | 0.01617 | 57/80 |
| echonext | MMD-1 vs none | 32 | -0.01381 | -0.00332 | 11/32 |
| echonext | MMD-2 vs none | 32 | -0.00728 | -0.00126 | 6/32 |
| echonext | MMD-3 vs none | 32 | 0.00022 | -0.00010 | 9/32 |
| echonext | MMD-4 vs none | 32 | -0.00001 | -0.00003 | 13/32 |
| sunnybrook | Correlation | 80 | 0.13960 | 0.10080 | 80/80 |
| sunnybrook | Derivative | 80 | -0.04394 | -0.02480 | 0/80 |
| sunnybrook | VCG | 80 | 0.00696 | 0.03464 | 47/80 |
| sunnybrook | Energy distance | 80 | -0.20855 | -0.20695 | 0/80 |
| sunnybrook | Lead consistency | 80 | -0.00844 | 0.00619 | 47/80 |
| sunnybrook | MMD-1 vs none | 32 | -0.01652 | -0.00356 | 9/32 |
| sunnybrook | MMD-2 vs none | 32 | -0.00652 | -0.00070 | 9/32 |
| sunnybrook | MMD-3 vs none | 32 | -0.00030 | -0.00003 | 13/32 |
| sunnybrook | MMD-4 vs none | 32 | -0.00149 | -0.00009 | 12/32 |

Correlation is the cleanest positive factor: it improves all 80 matched U-Net pairs on each of PTB-XL, EchoNext, and Sunnybrook, by mean Δr +0.150, +0.130, and +0.140. Energy distance is the cleanest negative factor: it harms all 80 pairs on every cohort, by mean Δr −0.243, −0.231, and −0.209. Derivative loss is also consistently negative for PTB-XL and Sunnybrook and mostly negative on EchoNext. VCG and lead consistency have small, interaction-dependent effects. MMD-3 is closest to neutral; other MMD variants usually reduce Pearson, although some improve MAE slightly.

### Cross-dataset stability and strongest U-Nets

| Dataset A | Dataset B | Models | Spearman ρ | p |
| --- | --- | --- | --- | --- |
| ptb_xl | echonext | 160 | 0.9810 | 1.73e-114 |
| ptb_xl | sunnybrook | 160 | 0.9912 | 7.25e-141 |
| echonext | sunnybrook | 160 | 0.9599 | 3.37e-89 |

Rank correlation is exceptionally high (ρ=0.960–0.991). Within the U-Net family, loss-mask ranking therefore transfers across these three cohorts rather than being a PTB-XL-only artifact. The `110000x` family—MSE plus correlation, optionally with MMD—is consistently strongest.

| Dataset | Rank | Model | Missing-lead r | MAE | Mean R² |
| --- | --- | --- | --- | --- | --- |
| ptb_xl | 1 | f_1100003_s42 | 0.8598 | 0.1023 | 0.4574 |
| ptb_xl | 2 | f_1100000_s42 | 0.8596 | 0.1022 | 0.4461 |
| ptb_xl | 3 | f_1100004_s42 | 0.8594 | 0.1007 | 0.3979 |
| ptb_xl | 4 | f_1100002_s42 | 0.8340 | 0.1039 | 0.3128 |
| ptb_xl | 5 | f_1100010_s42 | 0.8215 | 0.1183 | 0.3505 |
| ptb_xl | 6 | f_1100014_s42 | 0.8206 | 0.1160 | 0.2289 |
| ptb_xl | 7 | f_1100013_s42 | 0.8185 | 0.1184 | 0.2980 |
| ptb_xl | 8 | f_1100012_s42 | 0.8087 | 0.1139 | 0.1783 |
| ptb_xl | 9 | f_1101000_s42 | 0.8026 | 0.1289 | 0.1727 |
| ptb_xl | 10 | f_1101003_s42 | 0.7990 | 0.1284 | 0.1800 |
| echonext | 1 | f_1100003_s42 | 0.7767 | 0.3491 | 0.3288 |
| echonext | 2 | f_1100004_s42 | 0.7728 | 0.3453 | 0.3365 |
| echonext | 3 | f_1100000_s42 | 0.7658 | 0.3487 | 0.3274 |
| echonext | 4 | f_1100002_s42 | 0.7602 | 0.3457 | 0.3241 |
| echonext | 5 | f_1100014_s42 | 0.7413 | 0.3515 | 0.3137 |
| echonext | 6 | f_1100010_s42 | 0.7384 | 0.3621 | 0.2866 |
| echonext | 7 | f_1100013_s42 | 0.7339 | 0.3601 | 0.2938 |
| echonext | 8 | f_1110004_s42 | 0.7336 | 0.3528 | 0.2875 |
| echonext | 9 | f_1100001_s42 | 0.7327 | 0.3412 | 0.3150 |
| echonext | 10 | f_1110003_s42 | 0.7313 | 0.3600 | 0.2717 |
| sunnybrook | 1 | f_1100000_s42 | 0.7822 | 0.2511 | 0.1357 |
| sunnybrook | 2 | f_1100003_s42 | 0.7809 | 0.2516 | 0.1426 |
| sunnybrook | 3 | f_1100004_s42 | 0.7754 | 0.2519 | 0.0473 |
| sunnybrook | 4 | f_1100002_s42 | 0.7528 | 0.2557 | -0.1179 |
| sunnybrook | 5 | f_1100014_s42 | 0.7341 | 0.2643 | -0.1938 |
| sunnybrook | 6 | f_1100010_s42 | 0.7327 | 0.2622 | 0.0479 |
| sunnybrook | 7 | f_1100013_s42 | 0.7295 | 0.2632 | -0.0522 |
| sunnybrook | 8 | f_1100012_s42 | 0.7280 | 0.2635 | -0.3882 |
| sunnybrook | 9 | f_1100001_s42 | 0.7241 | 0.2569 | -0.2980 |
| sunnybrook | 10 | f_1101000_s42 | 0.7217 | 0.2582 | 0.0196 |

### Sunnybrook dedicated signal endpoints

| Metric | Better | Mean | Min | Max | Top three |
| --- | --- | --- | --- | --- | --- |
| Signal_Missing_Leads_Pearson | higher | 0.5491 | 0.3733 | 0.8040 | f_1100000_s42=0.8040; f_1100003_s42=0.8028; f_1100004_s42=0.7979 |
| Signal_Missing_Leads_MSE | lower | 0.1702 | 0.1479 | 0.1841 | f_1100000_s42=0.1479; f_1100003_s42=0.1485; f_1100004_s42=0.1494 |
| Signal_Missing_Leads_SNR_dB | higher | 6.0242 | 4.5966 | 8.9525 | f_1100003_s42=8.9525; f_1100000_s42=8.9062; f_1100004_s42=8.6328 |
| Signal_Missing_Leads_DTW | lower | 0.0332 | 0.0199 | 0.0716 | f_1100011_s42=0.0199; f_1100012_s42=0.0201; f_1100101_s42=0.0235 |

`f_1100000_s42`, `f_1100003_s42`, and `f_1100004_s42` dominate Pearson, MSE, and SNR. DTW favors `f_1100011_s42` and `f_1100012_s42`, showing that temporal alignment and pointwise/morphologic fidelity are related but not identical objectives.

### Legacy ST-segment and ECGFounder observations

| Architecture | Rank | Model | Mean missing-lead ST r | Mean ST MAE |
| --- | --- | --- | --- | --- |
| MSVAE | 1 | factorial_msvae_1000013_s42 | 0.8730 | 0.0260 |
| MSVAE | 2 | factorial_msvae_1000012_s42 | 0.8544 | 0.0246 |
| MSVAE | 3 | factorial_msvae_1000003_s42 | 0.8345 | 0.0296 |
| MSVAE | 4 | factorial_msvae_1000004_s42 | 0.8313 | 0.0273 |
| MSVAE | 5 | factorial_msvae_1000014_s42 | 0.8230 | 0.0291 |
| MSVAE | 6 | factorial_msvae_1000010_s42 | 0.8109 | 0.0291 |
| MSVAE | 7 | factorial_msvae_1000011_s42 | 0.7837 | 0.0300 |
| MSVAE | 8 | factorial_msvae_1000002_s42 | 0.7803 | 0.0315 |
| U-Net | 1 | f_1100000_s42 | 0.6952 | 0.0736 |
| U-Net | 2 | f_1100003_s42 | 0.6903 | 0.0745 |
| U-Net | 3 | f_1100004_s42 | 0.6820 | 0.0712 |
| U-Net | 4 | f_1101000_s42 | 0.6571 | 0.0993 |
| U-Net | 5 | f_1101003_s42 | 0.6495 | 0.0982 |
| U-Net | 6 | f_1101004_s42 | 0.6454 | 0.0941 |
| U-Net | 7 | f_1100002_s42 | 0.6434 | 0.0762 |
| U-Net | 8 | f_1000000_s42 | 0.6250 | 0.0712 |

| Architecture | Models | Mean AUROC | Median AUROC | Best AUROC | Mean AUPRC | Best AUPRC | Best-AUROC model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSVAE | 19 | 0.8416 | 0.8502 | 0.8664 | 0.4073 | 0.4528 | factorial_msvae_1000013_s42 |
| U-Net | 159 | 0.8056 | 0.8082 | 0.8594 | 0.3469 | 0.4278 | f_1100003_s42 |

The legacy ECGFounder table lacks paired original-ECG predictions and patient-cluster deltas, so the apparent MSVAE advantage cannot be promoted to an architecture claim. It is nevertheless consistent with the signal and ST rankings: the early `factorial_msvae_1000013_s42` model leads all three.

## Dataset-by-dataset status

| Dataset | Corrected V2 status | Legacy status | Interpretation now |
| --- | --- | --- | --- |
| PTB-XL | 1 U-Net, partial post-processing | 160 U-Net + 19 early MSVAE | Only one-model paired clinical result is claimable |
| EchoNext | No rows | Signal/QRS rows only; no classifier comparison | Actual original-vs-reconstructed EchoNext classifier result is still missing |
| Sunnybrook | No rows | 160 U-Nets | Signal endpoints exploratory; old delineation endpoints invalid |
| LUDB | No rows | 7 U-Nets | Too incomplete for mask or architecture claims |
| ISP | No rows | No rows | Not evaluated |
| Zhejiang | No rows | No integrated rows | Not evaluated in this database |

## What can and cannot be claimed

### Supported now

- In the complete 160-mask U-Net factorial, correlation loss robustly improves missing-lead waveform correlation across PTB-XL, EchoNext, and Sunnybrook.
- In the same U-Net factorial, energy distance robustly harms missing-lead correlation across all three cohorts.
- For corrected PTB-XL `f_1000000_s42`, reconstruction significantly degrades paired ECGFounder macro discrimination and calibration.
- For that model, reconstructed missing leads systematically shorten QRS and attenuate Sokolow-Lyon voltage.

### Not supported yet

- MSVAE versus U-Net versus ECG-AIM superiority.
- Best loss mask across architectures.
- Any corrected EchoNext classifier preservation claim.
- Corrected external-cohort delineation or morphology preservation.
- Seed-robust factorial effects for MSVAE or ECG-AIM.
- Clinical equivalence or non-inferiority of reconstructed ECGs.

## Required next actions, in priority order

1. **Stop the evaluator retry loop and filter non-finite LVH values before all classification and bootstrap calls.** The completion sentinel is never written because the crash occurs before `ST_Lead_V6`, causing repeated 70-minute passes.
2. **Rebuild the V2 CSV from SQLite rather than appending per retry.** SQLite is deduplicated by primary key; the CSV is not.
3. **Resume corrected evaluation and verify the first model reaches its final sentinel, then advances to model 2.** Do not infer progress from process uptime.
4. **Add the actual EchoNext classifier comparison** with original-versus-reconstructed predictions, patient-level paired AUROC/AUPRC, Brier, ECE, and bootstrap deltas. The current EchoNext rows are only signal/QRS summaries.
5. **Complete all MSVAE and ECG-AIM masks and rerun the corrected evaluator** before any architecture statement.
6. **Add multiple seeds for shortlisted masks.** The current seed-42 MSVAE ranking has no reproducibility interval.
7. **Investigate GPU ECC/hardware failures before retrying the three failed MSVAE jobs.** Their failures are infrastructural, not interpretable outcomes.

## Appendix A: every fully logged MSVAE training model

| Mask | Corr | Deriv | VCG | ED | Lead | MMD | Epoch-1 r | Best r | Best epoch | Final r | Drop | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000000 | 0 | 0 | 0 | 0 | 0 | 0 | 0.7416 | 0.8501 | 7 | 0.6430 | 0.2071 | 47.9 |
| 1000001 | 0 | 0 | 0 | 0 | 0 | 1 | 0.6163 | 0.8134 | 9 | 0.7879 | 0.0255 | 48.2 |
| 1000002 | 0 | 0 | 0 | 0 | 0 | 2 | 0.7192 | 0.8470 | 9 | 0.8324 | 0.0146 | 48.5 |
| 1000003 | 0 | 0 | 0 | 0 | 0 | 3 | 0.7208 | 0.8522 | 10 | 0.8522 | 0.0000 | 49.4 |
| 1000010 | 0 | 0 | 0 | 0 | 1 | 0 | 0.7025 | 0.8587 | 10 | 0.8587 | 0.0000 | 47.6 |
| 1000011 | 0 | 0 | 0 | 0 | 1 | 1 | 0.6428 | 0.8185 | 10 | 0.8185 | 0.0000 | 47.9 |
| 1000013 | 0 | 0 | 0 | 0 | 1 | 3 | 0.7040 | 0.8351 | 10 | 0.8351 | 0.0000 | 48.8 |
| 1000014 | 0 | 0 | 0 | 0 | 1 | 4 | 0.7022 | 0.7562 | 10 | 0.7562 | 0.0000 | 49.2 |
| 1000101 | 0 | 0 | 0 | 1 | 0 | 1 | -0.0069 | 0.7216 | 10 | 0.7216 | 0.0000 | 48.5 |
| 1000102 | 0 | 0 | 0 | 1 | 0 | 2 | 0.4348 | 0.7622 | 10 | 0.7622 | 0.0000 | 49.1 |
| 1000103 | 0 | 0 | 0 | 1 | 0 | 3 | 0.4938 | 0.7677 | 10 | 0.7677 | 0.0000 | 49.9 |
| 1000104 | 0 | 0 | 0 | 1 | 0 | 4 | 0.2681 | 0.7574 | 10 | 0.7574 | 0.0000 | 49.9 |
| 1000110 | 0 | 0 | 0 | 1 | 1 | 0 | 0.0108 | 0.1095 | 2 | 0.0600 | 0.0495 | 48.0 |
| 1000111 | 0 | 0 | 0 | 1 | 1 | 1 | 0.4851 | 0.8149 | 10 | 0.8149 | 0.0000 | 48.7 |
| 1000112 | 0 | 0 | 0 | 1 | 1 | 2 | 0.0188 | 0.5598 | 10 | 0.5598 | 0.0000 | 49.4 |
| 1000113 | 0 | 0 | 0 | 1 | 1 | 3 | 0.0600 | 0.7175 | 10 | 0.7175 | 0.0000 | 50.1 |
| 1000114 | 0 | 0 | 0 | 1 | 1 | 4 | -0.0150 | 0.3947 | 10 | 0.3947 | 0.0000 | 50.1 |
| 1001000 | 0 | 0 | 1 | 0 | 0 | 0 | 0.7634 | 0.8600 | 9 | 0.8591 | 0.0009 | 47.6 |
| 1001001 | 0 | 0 | 1 | 0 | 0 | 1 | 0.7062 | 0.8580 | 9 | 0.8574 | 0.0006 | 48.2 |
| 1001002 | 0 | 0 | 1 | 0 | 0 | 2 | 0.3017 | 0.8004 | 9 | 0.7982 | 0.0022 | 48.4 |
| 1001003 | 0 | 0 | 1 | 0 | 0 | 3 | 0.7860 | 0.8615 | 10 | 0.8615 | 0.0000 | 49.3 |
| 1001004 | 0 | 0 | 1 | 0 | 0 | 4 | 0.7541 | 0.8615 | 10 | 0.8615 | 0.0000 | 49.1 |
| 1001010 | 0 | 0 | 1 | 0 | 1 | 0 | 0.7631 | 0.8602 | 10 | 0.8602 | 0.0000 | 47.6 |
| 1001011 | 0 | 0 | 1 | 0 | 1 | 1 | 0.6915 | 0.8568 | 10 | 0.8568 | 0.0000 | 47.9 |
| 1001100 | 0 | 0 | 1 | 1 | 0 | 0 | 0.0081 | 0.3046 | 4 | 0.0010 | 0.3036 | 48.1 |
| 1001101 | 0 | 0 | 1 | 1 | 0 | 1 | 0.1753 | 0.7853 | 10 | 0.7853 | 0.0000 | 48.3 |
| 1001102 | 0 | 0 | 1 | 1 | 0 | 2 | 0.2211 | 0.8066 | 10 | 0.8066 | 0.0000 | 48.9 |
| 1001103 | 0 | 0 | 1 | 1 | 0 | 3 | 0.1690 | 0.7808 | 10 | 0.7808 | 0.0000 | 49.9 |
| 1001104 | 0 | 0 | 1 | 1 | 0 | 4 | 0.5462 | 0.8079 | 10 | 0.8079 | 0.0000 | 49.8 |
| 1010000 | 0 | 1 | 0 | 0 | 0 | 0 | 0.7301 | 0.8634 | 9 | 0.8341 | 0.0293 | 47.3 |
| 1010001 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5865 | 0.8257 | 10 | 0.8257 | 0.0000 | 47.8 |
| 1010002 | 0 | 1 | 0 | 0 | 0 | 2 | 0.7250 | 0.8540 | 9 | 0.8053 | 0.0487 | 48.1 |
| 1010003 | 0 | 1 | 0 | 0 | 0 | 3 | 0.7203 | 0.8660 | 8 | 0.7773 | 0.0887 | 49.0 |
| 1010004 | 0 | 1 | 0 | 0 | 0 | 4 | 0.6743 | 0.7866 | 10 | 0.7866 | 0.0000 | 48.9 |
| 1010100 | 0 | 1 | 0 | 1 | 0 | 0 | 0.2098 | 0.7558 | 10 | 0.7558 | 0.0000 | 47.8 |
| 1010101 | 0 | 1 | 0 | 1 | 0 | 1 | 0.0007 | 0.6714 | 9 | 0.6687 | 0.0027 | 48.2 |
| 1010102 | 0 | 1 | 0 | 1 | 0 | 2 | 0.1695 | 0.8011 | 10 | 0.8011 | 0.0000 | 48.9 |
| 1010103 | 0 | 1 | 0 | 1 | 0 | 3 | 0.0405 | 0.1279 | 4 | 0.1111 | 0.0168 | 49.4 |
| 1010104 | 0 | 1 | 0 | 1 | 0 | 4 | 0.0434 | 0.6681 | 10 | 0.6681 | 0.0000 | 49.5 |
| 1011000 | 0 | 1 | 1 | 0 | 0 | 0 | 0.7356 | 0.8656 | 9 | 0.8599 | 0.0057 | 47.5 |
| 1011001 | 0 | 1 | 1 | 0 | 0 | 1 | 0.6850 | 0.8508 | 10 | 0.8508 | 0.0000 | 47.8 |
| 1011002 | 0 | 1 | 1 | 0 | 0 | 2 | 0.7860 | 0.8675 | 9 | 0.8593 | 0.0082 | 48.3 |
| 1011003 | 0 | 1 | 1 | 0 | 0 | 3 | 0.7479 | 0.8642 | 10 | 0.8642 | 0.0000 | 49.0 |
| 1011004 | 0 | 1 | 1 | 0 | 0 | 4 | 0.7478 | 0.8656 | 10 | 0.8656 | 0.0000 | 49.2 |
| 1011100 | 0 | 1 | 1 | 1 | 0 | 0 | 0.1470 | 0.7841 | 10 | 0.7841 | 0.0000 | 48.3 |
| 1011101 | 0 | 1 | 1 | 1 | 0 | 1 | 0.0433 | 0.8128 | 10 | 0.8128 | 0.0000 | 48.4 |
| 1011102 | 0 | 1 | 1 | 1 | 0 | 2 | 0.0250 | 0.7777 | 10 | 0.7777 | 0.0000 | 48.8 |
| 1011103 | 0 | 1 | 1 | 1 | 0 | 3 | 0.4722 | 0.8232 | 10 | 0.8232 | 0.0000 | 49.9 |
| 1011104 | 0 | 1 | 1 | 1 | 0 | 4 | 0.6281 | 0.8170 | 10 | 0.8170 | 0.0000 | 49.5 |
| 1100000 | 1 | 0 | 0 | 0 | 0 | 0 | 0.7877 | 0.8736 | 10 | 0.8736 | 0.0000 | 47.3 |
| 1100001 | 1 | 0 | 0 | 0 | 0 | 1 | 0.6739 | 0.8526 | 10 | 0.8526 | 0.0000 | 47.6 |
| 1100002 | 1 | 0 | 0 | 0 | 0 | 2 | 0.7600 | 0.8603 | 8 | 0.8548 | 0.0055 | 48.2 |
| 1100003 | 1 | 0 | 0 | 0 | 0 | 3 | 0.7903 | 0.8701 | 10 | 0.8701 | 0.0000 | 49.0 |
| 1100004 | 1 | 0 | 0 | 0 | 0 | 4 | 0.4251 | 0.7987 | 9 | 0.7869 | 0.0118 | 49.1 |
| 1100100 | 1 | 0 | 0 | 1 | 0 | 0 | 0.2807 | 0.8110 | 10 | 0.8110 | 0.0000 | 47.8 |
| 1100101 | 1 | 0 | 0 | 1 | 0 | 1 | 0.2060 | 0.8098 | 10 | 0.8098 | 0.0000 | 48.5 |
| 1100102 | 1 | 0 | 0 | 1 | 0 | 2 | 0.0000 | 0.4517 | 10 | 0.4517 | 0.0000 | 48.8 |
| 1100103 | 1 | 0 | 0 | 1 | 0 | 3 | 0.3829 | 0.7987 | 10 | 0.7987 | 0.0000 | 50.1 |
| 1100104 | 1 | 0 | 0 | 1 | 0 | 4 | 0.3453 | 0.8107 | 10 | 0.8107 | 0.0000 | 49.8 |
| 1101000 | 1 | 0 | 1 | 0 | 0 | 0 | 0.7575 | 0.8677 | 10 | 0.8677 | 0.0000 | 47.5 |

## Appendix B: every U-Net model on legacy signal-valid axes

| Model | Mask | PTB r | PTB MAE | EchoNext r | EchoNext MAE | Sunnybrook r | Sunnybrook MAE | PTB ST r | Legacy macro AUROC | Legacy macro AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_1000000_s42 | 1000000 | 0.7516 | 0.1055 | 0.6679 | 0.3490 | 0.6816 | 0.2552 | 0.6250 | — | — |
| f_1000001_s42 | 1000001 | 0.6125 | 0.1138 | 0.5714 | 0.3543 | 0.5708 | 0.2675 | 0.4806 | 0.8239 | 0.3705 |
| f_1000002_s42 | 1000002 | 0.6891 | 0.1096 | 0.6329 | 0.3536 | 0.6406 | 0.2618 | 0.5310 | 0.8376 | 0.3934 |
| f_1000003_s42 | 1000003 | 0.7454 | 0.1067 | 0.6749 | 0.3487 | 0.6792 | 0.2564 | 0.6138 | 0.8490 | 0.4012 |
| f_1000004_s42 | 1000004 | 0.7383 | 0.1062 | 0.6748 | 0.3458 | 0.6801 | 0.2575 | 0.6097 | 0.8474 | 0.3964 |
| f_1000010_s42 | 1000010 | 0.6114 | 0.1267 | 0.5570 | 0.3737 | 0.5192 | 0.2703 | 0.4410 | 0.8212 | 0.3711 |
| f_1000011_s42 | 1000011 | 0.5841 | 0.1208 | 0.5489 | 0.3558 | 0.5253 | 0.2714 | 0.4111 | 0.8041 | 0.3530 |
| f_1000012_s42 | 1000012 | 0.6262 | 0.1237 | 0.5753 | 0.3632 | 0.5521 | 0.2686 | 0.4652 | 0.8201 | 0.3691 |
| f_1000013_s42 | 1000013 | 0.6131 | 0.1277 | 0.5680 | 0.3696 | 0.5225 | 0.2716 | 0.4472 | 0.8203 | 0.3729 |
| f_1000014_s42 | 1000014 | 0.6121 | 0.1273 | 0.5741 | 0.3639 | 0.5292 | 0.2737 | 0.4624 | 0.8193 | 0.3662 |
| f_1000100_s42 | 1000100 | 0.3675 | 0.1356 | 0.3063 | 0.3841 | 0.3280 | 0.2810 | 0.2754 | 0.7658 | 0.2928 |
| f_1000101_s42 | 1000101 | 0.3718 | 0.1353 | 0.3164 | 0.3828 | 0.3346 | 0.2810 | 0.2779 | 0.7665 | 0.2917 |
| f_1000102_s42 | 1000102 | 0.3671 | 0.1356 | 0.3057 | 0.3841 | 0.3278 | 0.2810 | 0.2744 | 0.7656 | 0.2920 |
| f_1000103_s42 | 1000103 | 0.3674 | 0.1356 | 0.3060 | 0.3841 | 0.3280 | 0.2810 | 0.2753 | 0.7657 | 0.2919 |
| f_1000104_s42 | 1000104 | 0.3679 | 0.1355 | 0.3065 | 0.3840 | 0.3282 | 0.2810 | 0.2755 | 0.7660 | 0.2925 |
| f_1000110_s42 | 1000110 | 0.3755 | 0.1357 | 0.3306 | 0.3797 | 0.3396 | 0.2821 | 0.2822 | 0.7582 | 0.2905 |
| f_1000111_s42 | 1000111 | 0.3806 | 0.1355 | 0.3381 | 0.3791 | 0.3439 | 0.2824 | 0.2917 | 0.7601 | 0.2926 |
| f_1000112_s42 | 1000112 | 0.3754 | 0.1357 | 0.3303 | 0.3797 | 0.3399 | 0.2821 | 0.2802 | 0.7580 | 0.2898 |
| f_1000113_s42 | 1000113 | 0.3756 | 0.1358 | 0.3304 | 0.3797 | 0.3397 | 0.2821 | 0.2819 | 0.7583 | 0.2902 |
| f_1000114_s42 | 1000114 | 0.3756 | 0.1357 | 0.3304 | 0.3797 | 0.3396 | 0.2820 | 0.2820 | 0.7585 | 0.2901 |
| f_1001000_s42 | 1001000 | 0.5739 | 0.1424 | 0.4902 | 0.3829 | 0.5275 | 0.2676 | 0.4879 | 0.8328 | 0.3916 |
| f_1001001_s42 | 1001001 | 0.5106 | 0.1246 | 0.4441 | 0.3621 | 0.4774 | 0.2674 | 0.4175 | 0.8189 | 0.3637 |
| f_1001002_s42 | 1001002 | 0.5446 | 0.1296 | 0.4452 | 0.3702 | 0.5019 | 0.2655 | 0.4941 | 0.8297 | 0.3781 |
| f_1001003_s42 | 1001003 | 0.5719 | 0.1422 | 0.4887 | 0.3824 | 0.5288 | 0.2678 | 0.4877 | 0.8336 | 0.3881 |
| f_1001004_s42 | 1001004 | 0.5671 | 0.1390 | 0.4949 | 0.3763 | 0.5279 | 0.2669 | 0.4819 | 0.8340 | 0.3839 |
| f_1001010_s42 | 1001010 | 0.5695 | 0.1481 | 0.4955 | 0.3795 | 0.5188 | 0.2751 | 0.4703 | 0.8312 | 0.3887 |
| f_1001011_s42 | 1001011 | 0.5510 | 0.1275 | 0.5059 | 0.3589 | 0.5062 | 0.2683 | 0.4637 | 0.8231 | 0.3852 |
| f_1001012_s42 | 1001012 | 0.5678 | 0.1327 | 0.4937 | 0.3645 | 0.5129 | 0.2671 | 0.4682 | 0.8244 | 0.3831 |
| f_1001013_s42 | 1001013 | 0.5671 | 0.1478 | 0.4910 | 0.3796 | 0.5171 | 0.2748 | 0.4417 | 0.8309 | 0.3865 |
| f_1001014_s42 | 1001014 | 0.5599 | 0.1451 | 0.4930 | 0.3739 | 0.5135 | 0.2745 | 0.4441 | 0.8282 | 0.3856 |
| f_1001100_s42 | 1001100 | 0.3914 | 0.1308 | 0.3380 | 0.3780 | 0.3725 | 0.2748 | 0.2604 | 0.8020 | 0.3149 |
| f_1001101_s42 | 1001101 | 0.3876 | 0.1316 | 0.3351 | 0.3787 | 0.3698 | 0.2755 | 0.2534 | 0.8003 | 0.3106 |
| f_1001102_s42 | 1001102 | 0.3907 | 0.1309 | 0.3373 | 0.3780 | 0.3722 | 0.2748 | 0.2592 | 0.8014 | 0.3147 |
| f_1001103_s42 | 1001103 | 0.3916 | 0.1308 | 0.3381 | 0.3780 | 0.3727 | 0.2748 | 0.2606 | 0.8021 | 0.3151 |
| f_1001104_s42 | 1001104 | 0.3915 | 0.1308 | 0.3381 | 0.3779 | 0.3726 | 0.2748 | 0.2607 | 0.8019 | 0.3149 |
| f_1001110_s42 | 1001110 | 0.4237 | 0.1287 | 0.3709 | 0.3746 | 0.4021 | 0.2739 | 0.3246 | 0.7945 | 0.3163 |
| f_1001111_s42 | 1001111 | 0.4216 | 0.1294 | 0.3687 | 0.3753 | 0.3994 | 0.2747 | 0.3177 | 0.7928 | 0.3133 |
| f_1001112_s42 | 1001112 | 0.4234 | 0.1291 | 0.3696 | 0.3753 | 0.4008 | 0.2743 | 0.3206 | 0.7929 | 0.3142 |
| f_1001113_s42 | 1001113 | 0.4237 | 0.1287 | 0.3709 | 0.3746 | 0.4021 | 0.2739 | 0.3246 | 0.7946 | 0.3163 |
| f_1001114_s42 | 1001114 | 0.4237 | 0.1287 | 0.3709 | 0.3746 | 0.4020 | 0.2739 | 0.3248 | 0.7945 | 0.3162 |
| f_1010000_s42 | 1010000 | 0.5720 | 0.1251 | 0.5154 | 0.3845 | 0.5104 | 0.2632 | 0.4126 | 0.8466 | 0.3990 |
| f_1010001_s42 | 1010001 | 0.5293 | 0.1271 | 0.5185 | 0.3638 | 0.4612 | 0.2723 | 0.3467 | 0.8038 | 0.3512 |
| f_1010002_s42 | 1010002 | 0.5438 | 0.1304 | 0.5400 | 0.3753 | 0.4580 | 0.2719 | 0.3688 | 0.8151 | 0.3770 |
| f_1010003_s42 | 1010003 | 0.5648 | 0.1272 | 0.5092 | 0.3821 | 0.5055 | 0.2649 | 0.3854 | 0.8433 | 0.3963 |
| f_1010004_s42 | 1010004 | 0.5486 | 0.1293 | 0.4981 | 0.3808 | 0.4890 | 0.2668 | 0.4250 | 0.8406 | 0.3918 |
| f_1010010_s42 | 1010010 | 0.5190 | 0.1317 | 0.5342 | 0.3854 | 0.4376 | 0.2708 | 0.3892 | 0.8158 | 0.3616 |
| f_1010011_s42 | 1010011 | 0.4531 | 0.1365 | 0.4783 | 0.3735 | 0.3931 | 0.2779 | 0.3536 | 0.7947 | 0.3450 |
| f_1010012_s42 | 1010012 | 0.4889 | 0.1367 | 0.5093 | 0.3797 | 0.4087 | 0.2739 | 0.3970 | 0.8019 | 0.3478 |
| f_1010013_s42 | 1010013 | 0.5181 | 0.1325 | 0.5357 | 0.3830 | 0.4359 | 0.2718 | 0.3902 | 0.8140 | 0.3587 |
| f_1010014_s42 | 1010014 | 0.5022 | 0.1333 | 0.5227 | 0.3778 | 0.4256 | 0.2740 | 0.3929 | 0.8091 | 0.3546 |
| f_1010100_s42 | 1010100 | 0.3543 | 0.1377 | 0.3168 | 0.3821 | 0.3036 | 0.2832 | 0.2525 | 0.7240 | 0.2741 |
| f_1010101_s42 | 1010101 | 0.3605 | 0.1377 | 0.3234 | 0.3811 | 0.3094 | 0.2832 | 0.2514 | 0.7246 | 0.2765 |
| f_1010102_s42 | 1010102 | 0.3579 | 0.1377 | 0.3199 | 0.3818 | 0.3078 | 0.2831 | 0.2527 | 0.7225 | 0.2740 |
| f_1010103_s42 | 1010103 | 0.3546 | 0.1377 | 0.3168 | 0.3820 | 0.3038 | 0.2832 | 0.2528 | 0.7241 | 0.2742 |
| f_1010104_s42 | 1010104 | 0.3547 | 0.1376 | 0.3169 | 0.3820 | 0.3038 | 0.2832 | 0.2529 | 0.7242 | 0.2743 |
| f_1010110_s42 | 1010110 | 0.3547 | 0.1376 | 0.3124 | 0.3821 | 0.3181 | 0.2833 | 0.2583 | 0.7193 | 0.2677 |
| f_1010111_s42 | 1010111 | 0.3646 | 0.1374 | 0.3247 | 0.3809 | 0.3209 | 0.2837 | 0.2649 | 0.7213 | 0.2670 |
| f_1010112_s42 | 1010112 | 0.3553 | 0.1375 | 0.3118 | 0.3822 | 0.3189 | 0.2833 | 0.2584 | 0.7190 | 0.2670 |
| f_1010113_s42 | 1010113 | 0.3547 | 0.1376 | 0.3123 | 0.3821 | 0.3181 | 0.2833 | 0.2583 | 0.7192 | 0.2677 |
| f_1010114_s42 | 1010114 | 0.3554 | 0.1374 | 0.3128 | 0.3820 | 0.3183 | 0.2832 | 0.2589 | 0.7198 | 0.2677 |
| f_1011000_s42 | 1011000 | 0.5469 | 0.1501 | 0.4329 | 0.3986 | 0.5007 | 0.2715 | 0.4278 | 0.8409 | 0.3852 |
| f_1011001_s42 | 1011001 | 0.4723 | 0.1328 | 0.4005 | 0.3806 | 0.4659 | 0.2690 | 0.3739 | 0.8270 | 0.3812 |
| f_1011002_s42 | 1011002 | 0.5084 | 0.1360 | 0.3849 | 0.3876 | 0.4737 | 0.2685 | 0.3758 | 0.8349 | 0.3896 |
| f_1011003_s42 | 1011003 | 0.5423 | 0.1494 | 0.4275 | 0.3979 | 0.4970 | 0.2717 | 0.4426 | 0.8402 | 0.3873 |
| f_1011004_s42 | 1011004 | 0.5350 | 0.1479 | 0.4307 | 0.3938 | 0.4863 | 0.2710 | 0.4528 | 0.8399 | 0.3852 |
| f_1011010_s42 | 1011010 | 0.5172 | 0.1492 | 0.4909 | 0.3805 | 0.4677 | 0.2774 | 0.5131 | 0.8252 | 0.3704 |
| f_1011011_s42 | 1011011 | 0.5130 | 0.1294 | 0.4900 | 0.3624 | 0.4720 | 0.2714 | 0.4750 | 0.8204 | 0.3672 |
| f_1011012_s42 | 1011012 | 0.5182 | 0.1328 | 0.4790 | 0.3678 | 0.4759 | 0.2700 | 0.5132 | 0.8257 | 0.3703 |
| f_1011013_s42 | 1011013 | 0.5148 | 0.1488 | 0.4865 | 0.3804 | 0.4666 | 0.2772 | 0.5102 | 0.8261 | 0.3705 |
| f_1011014_s42 | 1011014 | 0.5023 | 0.1469 | 0.4775 | 0.3763 | 0.4574 | 0.2777 | 0.5011 | 0.8230 | 0.3667 |
| f_1011100_s42 | 1011100 | 0.3863 | 0.1321 | 0.3414 | 0.3779 | 0.3582 | 0.2738 | 0.2845 | 0.7920 | 0.3196 |
| f_1011101_s42 | 1011101 | 0.3849 | 0.1329 | 0.3391 | 0.3783 | 0.3568 | 0.2747 | 0.2792 | 0.7867 | 0.3161 |
| f_1011102_s42 | 1011102 | 0.3853 | 0.1322 | 0.3404 | 0.3781 | 0.3577 | 0.2740 | 0.2829 | 0.7912 | 0.3187 |
| f_1011103_s42 | 1011103 | 0.3862 | 0.1321 | 0.3412 | 0.3779 | 0.3581 | 0.2738 | 0.2844 | 0.7920 | 0.3197 |
| f_1011104_s42 | 1011104 | 0.3863 | 0.1320 | 0.3414 | 0.3779 | 0.3581 | 0.2738 | 0.2844 | 0.7920 | 0.3195 |
| f_1011110_s42 | 1011110 | 0.4017 | 0.1310 | 0.3619 | 0.3763 | 0.3775 | 0.2728 | 0.3266 | 0.7836 | 0.3086 |
| f_1011111_s42 | 1011111 | 0.4024 | 0.1316 | 0.3628 | 0.3766 | 0.3776 | 0.2736 | 0.3196 | 0.7828 | 0.3088 |
| f_1011112_s42 | 1011112 | 0.4020 | 0.1314 | 0.3620 | 0.3768 | 0.3781 | 0.2732 | 0.3217 | 0.7829 | 0.3081 |
| f_1011113_s42 | 1011113 | 0.4016 | 0.1310 | 0.3617 | 0.3763 | 0.3775 | 0.2728 | 0.3265 | 0.7836 | 0.3085 |
| f_1011114_s42 | 1011114 | 0.4017 | 0.1310 | 0.3618 | 0.3763 | 0.3774 | 0.2728 | 0.3265 | 0.7837 | 0.3090 |
| f_1100000_s42 | 1100000 | 0.8596 | 0.1022 | 0.7658 | 0.3487 | 0.7822 | 0.2511 | 0.6952 | 0.8573 | 0.4113 |
| f_1100001_s42 | 1100001 | 0.7976 | 0.1009 | 0.7327 | 0.3412 | 0.7241 | 0.2569 | 0.6136 | 0.8491 | 0.4097 |
| f_1100002_s42 | 1100002 | 0.8340 | 0.1039 | 0.7602 | 0.3457 | 0.7528 | 0.2557 | 0.6434 | 0.8483 | 0.4152 |
| f_1100003_s42 | 1100003 | 0.8598 | 0.1023 | 0.7767 | 0.3491 | 0.7809 | 0.2516 | 0.6903 | 0.8594 | 0.4120 |
| f_1100004_s42 | 1100004 | 0.8594 | 0.1007 | 0.7728 | 0.3453 | 0.7754 | 0.2519 | 0.6820 | 0.8593 | 0.4158 |
| f_1100010_s42 | 1100010 | 0.8215 | 0.1183 | 0.7384 | 0.3621 | 0.7327 | 0.2622 | 0.5694 | 0.8488 | 0.3993 |
| f_1100011_s42 | 1100011 | 0.7813 | 0.1067 | 0.7120 | 0.3418 | 0.7064 | 0.2621 | 0.5538 | 0.8336 | 0.3785 |
| f_1100012_s42 | 1100012 | 0.8087 | 0.1139 | 0.7151 | 0.3509 | 0.7280 | 0.2635 | 0.5975 | 0.8400 | 0.3813 |
| f_1100013_s42 | 1100013 | 0.8185 | 0.1184 | 0.7339 | 0.3601 | 0.7295 | 0.2632 | 0.5710 | 0.8463 | 0.3987 |
| f_1100014_s42 | 1100014 | 0.8206 | 0.1160 | 0.7413 | 0.3515 | 0.7341 | 0.2643 | 0.5831 | 0.8475 | 0.3999 |
| f_1100100_s42 | 1100100 | 0.4604 | 0.1299 | 0.3956 | 0.3766 | 0.4244 | 0.2760 | 0.3486 | 0.7788 | 0.3088 |
| f_1100101_s42 | 1100101 | 0.4567 | 0.1301 | 0.3964 | 0.3765 | 0.4241 | 0.2764 | 0.3445 | 0.7770 | 0.3028 |
| f_1100102_s42 | 1100102 | 0.4595 | 0.1299 | 0.3945 | 0.3767 | 0.4237 | 0.2760 | 0.3474 | 0.7785 | 0.3087 |
| f_1100103_s42 | 1100103 | 0.4603 | 0.1299 | 0.3954 | 0.3767 | 0.4244 | 0.2760 | 0.3485 | 0.7787 | 0.3089 |
| f_1100104_s42 | 1100104 | 0.4601 | 0.1298 | 0.3954 | 0.3766 | 0.4241 | 0.2759 | 0.3484 | 0.7787 | 0.3088 |
| f_1100110_s42 | 1100110 | 0.4699 | 0.1290 | 0.4114 | 0.3734 | 0.4352 | 0.2766 | 0.3642 | 0.7769 | 0.3065 |
| f_1100111_s42 | 1100111 | 0.4690 | 0.1292 | 0.4126 | 0.3734 | 0.4357 | 0.2771 | 0.3621 | 0.7759 | 0.3044 |
| f_1100112_s42 | 1100112 | 0.4694 | 0.1290 | 0.4106 | 0.3735 | 0.4349 | 0.2767 | 0.3624 | 0.7762 | 0.3058 |
| f_1100113_s42 | 1100113 | 0.4698 | 0.1290 | 0.4112 | 0.3735 | 0.4352 | 0.2766 | 0.3640 | 0.7769 | 0.3066 |
| f_1100114_s42 | 1100114 | 0.4697 | 0.1290 | 0.4112 | 0.3735 | 0.4350 | 0.2766 | 0.3641 | 0.7770 | 0.3066 |
| f_1101000_s42 | 1101000 | 0.8026 | 0.1289 | 0.7051 | 0.3649 | 0.7217 | 0.2582 | 0.6571 | 0.8588 | 0.4278 |
| f_1101001_s42 | 1101001 | 0.7403 | 0.1120 | 0.6396 | 0.3425 | 0.6703 | 0.2602 | 0.5399 | 0.8466 | 0.4139 |
| f_1101002_s42 | 1101002 | 0.7804 | 0.1170 | 0.6618 | 0.3517 | 0.7006 | 0.2577 | 0.5953 | 0.8488 | 0.4105 |
| f_1101003_s42 | 1101003 | 0.7990 | 0.1284 | 0.7025 | 0.3648 | 0.7186 | 0.2585 | 0.6495 | 0.8582 | 0.4238 |
| f_1101004_s42 | 1101004 | 0.7968 | 0.1255 | 0.7038 | 0.3580 | 0.7168 | 0.2578 | 0.6454 | 0.8573 | 0.4213 |
| f_1101010_s42 | 1101010 | 0.7803 | 0.1353 | 0.6747 | 0.3682 | 0.7085 | 0.2656 | 0.5915 | 0.8572 | 0.4267 |
| f_1101011_s42 | 1101011 | 0.7241 | 0.1151 | 0.6506 | 0.3459 | 0.6611 | 0.2606 | 0.5471 | 0.8451 | 0.4055 |
| f_1101012_s42 | 1101012 | 0.7630 | 0.1213 | 0.6688 | 0.3554 | 0.6883 | 0.2606 | 0.5493 | 0.8477 | 0.4104 |
| f_1101013_s42 | 1101013 | 0.7745 | 0.1354 | 0.6768 | 0.3666 | 0.7064 | 0.2653 | 0.5825 | 0.8551 | 0.4249 |
| f_1101014_s42 | 1101014 | 0.7743 | 0.1325 | 0.6790 | 0.3605 | 0.7086 | 0.2646 | 0.5818 | 0.8556 | 0.4256 |
| f_1101100_s42 | 1101100 | 0.4894 | 0.1255 | 0.4185 | 0.3704 | 0.4633 | 0.2716 | 0.3405 | 0.8112 | 0.3228 |
| f_1101101_s42 | 1101101 | 0.4850 | 0.1260 | 0.4148 | 0.3713 | 0.4605 | 0.2721 | 0.3331 | 0.8082 | 0.3211 |
| f_1101102_s42 | 1101102 | 0.4884 | 0.1256 | 0.4174 | 0.3705 | 0.4627 | 0.2716 | 0.3373 | 0.8107 | 0.3217 |
| f_1101103_s42 | 1101103 | 0.4892 | 0.1255 | 0.4183 | 0.3704 | 0.4632 | 0.2716 | 0.3403 | 0.8112 | 0.3228 |
| f_1101104_s42 | 1101104 | 0.4891 | 0.1255 | 0.4184 | 0.3704 | 0.4632 | 0.2716 | 0.3395 | 0.8111 | 0.3228 |
| f_1101110_s42 | 1101110 | 0.5046 | 0.1245 | 0.4363 | 0.3690 | 0.4767 | 0.2713 | 0.3801 | 0.8045 | 0.3244 |
| f_1101111_s42 | 1101111 | 0.4997 | 0.1251 | 0.4319 | 0.3698 | 0.4708 | 0.2720 | 0.3789 | 0.8022 | 0.3224 |
| f_1101112_s42 | 1101112 | 0.5022 | 0.1249 | 0.4333 | 0.3698 | 0.4724 | 0.2717 | 0.3835 | 0.8024 | 0.3234 |
| f_1101113_s42 | 1101113 | 0.5046 | 0.1245 | 0.4363 | 0.3690 | 0.4767 | 0.2713 | 0.3798 | 0.8045 | 0.3245 |
| f_1101114_s42 | 1101114 | 0.5044 | 0.1244 | 0.4363 | 0.3689 | 0.4766 | 0.2713 | 0.3792 | 0.8045 | 0.3244 |
| f_1110000_s42 | 1110000 | 0.7828 | 0.1112 | 0.7197 | 0.3602 | 0.6945 | 0.2581 | 0.6087 | 0.8533 | 0.4037 |
| f_1110001_s42 | 1110001 | 0.7563 | 0.1119 | 0.7015 | 0.3472 | 0.6890 | 0.2633 | 0.4609 | 0.8293 | 0.3833 |
| f_1110002_s42 | 1110002 | 0.7803 | 0.1155 | 0.7241 | 0.3548 | 0.6906 | 0.2661 | 0.4473 | 0.8369 | 0.3975 |
| f_1110003_s42 | 1110003 | 0.7876 | 0.1107 | 0.7313 | 0.3600 | 0.7019 | 0.2570 | 0.6140 | 0.8536 | 0.4060 |
| f_1110004_s42 | 1110004 | 0.7887 | 0.1092 | 0.7336 | 0.3528 | 0.7038 | 0.2577 | 0.6126 | 0.8549 | 0.4056 |
| f_1110010_s42 | 1110010 | 0.7679 | 0.1223 | 0.7299 | 0.3716 | 0.6928 | 0.2640 | 0.5661 | 0.8390 | 0.3845 |
| f_1110011_s42 | 1110011 | 0.7474 | 0.1162 | 0.7038 | 0.3490 | 0.6802 | 0.2693 | 0.4544 | 0.8211 | 0.3685 |
| f_1110012_s42 | 1110012 | 0.7651 | 0.1201 | 0.7183 | 0.3606 | 0.6861 | 0.2665 | 0.4912 | 0.8247 | 0.3642 |
| f_1110013_s42 | 1110013 | 0.7682 | 0.1225 | 0.7306 | 0.3693 | 0.6921 | 0.2647 | 0.5563 | 0.8383 | 0.3843 |
| f_1110014_s42 | 1110014 | 0.7653 | 0.1210 | 0.7292 | 0.3611 | 0.6922 | 0.2668 | 0.5481 | 0.8358 | 0.3841 |
| f_1110100_s42 | 1110100 | 0.4439 | 0.1300 | 0.3999 | 0.3736 | 0.4046 | 0.2779 | 0.3002 | 0.7523 | 0.2970 |
| f_1110101_s42 | 1110101 | 0.4452 | 0.1301 | 0.4003 | 0.3736 | 0.4056 | 0.2782 | 0.3000 | 0.7510 | 0.2947 |
| f_1110102_s42 | 1110102 | 0.4460 | 0.1301 | 0.4007 | 0.3739 | 0.4076 | 0.2780 | 0.3019 | 0.7510 | 0.2948 |
| f_1110103_s42 | 1110103 | 0.4439 | 0.1300 | 0.3999 | 0.3737 | 0.4046 | 0.2779 | 0.3001 | 0.7522 | 0.2970 |
| f_1110104_s42 | 1110104 | 0.4440 | 0.1300 | 0.4000 | 0.3736 | 0.4045 | 0.2779 | 0.3002 | 0.7524 | 0.2966 |
| f_1110110_s42 | 1110110 | 0.4532 | 0.1295 | 0.4096 | 0.3726 | 0.4108 | 0.2778 | 0.3259 | 0.7532 | 0.2953 |
| f_1110111_s42 | 1110111 | 0.4552 | 0.1297 | 0.4111 | 0.3723 | 0.4090 | 0.2782 | 0.3258 | 0.7544 | 0.2972 |
| f_1110112_s42 | 1110112 | 0.4527 | 0.1295 | 0.4087 | 0.3728 | 0.4101 | 0.2778 | 0.3252 | 0.7528 | 0.2951 |
| f_1110113_s42 | 1110113 | 0.4531 | 0.1295 | 0.4095 | 0.3726 | 0.4107 | 0.2778 | 0.3258 | 0.7531 | 0.2952 |
| f_1110114_s42 | 1110114 | 0.4530 | 0.1295 | 0.4095 | 0.3726 | 0.4105 | 0.2778 | 0.3259 | 0.7532 | 0.2952 |
| f_1111000_s42 | 1111000 | 0.7240 | 0.1416 | 0.6420 | 0.3761 | 0.6513 | 0.2657 | 0.6102 | 0.8423 | 0.3990 |
| f_1111001_s42 | 1111001 | 0.6986 | 0.1211 | 0.6153 | 0.3548 | 0.6452 | 0.2636 | 0.5008 | 0.8294 | 0.3965 |
| f_1111002_s42 | 1111002 | 0.7434 | 0.1205 | 0.6337 | 0.3638 | 0.6697 | 0.2592 | 0.5547 | 0.8476 | 0.4074 |
| f_1111003_s42 | 1111003 | 0.7280 | 0.1392 | 0.6335 | 0.3755 | 0.6541 | 0.2651 | 0.5912 | 0.8466 | 0.4109 |
| f_1111004_s42 | 1111004 | 0.7297 | 0.1361 | 0.6348 | 0.3700 | 0.6597 | 0.2640 | 0.5700 | 0.8462 | 0.4095 |
| f_1111010_s42 | 1111010 | 0.7412 | 0.1405 | 0.6617 | 0.3712 | 0.6627 | 0.2695 | 0.6130 | 0.8479 | 0.3985 |
| f_1111011_s42 | 1111011 | 0.7019 | 0.1177 | 0.6492 | 0.3482 | 0.6370 | 0.2638 | 0.5264 | 0.8334 | 0.3869 |
| f_1111012_s42 | 1111012 | 0.7354 | 0.1230 | 0.6571 | 0.3570 | 0.6635 | 0.2629 | 0.6062 | 0.8418 | 0.3949 |
| f_1111013_s42 | 1111013 | 0.7414 | 0.1404 | 0.6632 | 0.3714 | 0.6643 | 0.2697 | 0.6128 | 0.8470 | 0.3980 |
| f_1111014_s42 | 1111014 | 0.7366 | 0.1383 | 0.6609 | 0.3678 | 0.6638 | 0.2694 | 0.6035 | 0.8445 | 0.3983 |
| f_1111100_s42 | 1111100 | 0.4686 | 0.1260 | 0.4085 | 0.3721 | 0.4423 | 0.2703 | 0.3353 | 0.8007 | 0.3249 |
| f_1111101_s42 | 1111101 | 0.4659 | 0.1268 | 0.4024 | 0.3728 | 0.4381 | 0.2712 | 0.3274 | 0.7973 | 0.3219 |
| f_1111102_s42 | 1111102 | 0.4679 | 0.1262 | 0.4073 | 0.3722 | 0.4417 | 0.2704 | 0.3336 | 0.7999 | 0.3253 |
| f_1111103_s42 | 1111103 | 0.4686 | 0.1260 | 0.4084 | 0.3721 | 0.4423 | 0.2703 | 0.3352 | 0.8007 | 0.3260 |
| f_1111104_s42 | 1111104 | 0.4686 | 0.1260 | 0.4085 | 0.3721 | 0.4423 | 0.2703 | 0.3351 | 0.8008 | 0.3249 |
| f_1111110_s42 | 1111110 | 0.4860 | 0.1253 | 0.4293 | 0.3707 | 0.4600 | 0.2696 | 0.3654 | 0.7954 | 0.3200 |
| f_1111111_s42 | 1111111 | 0.4839 | 0.1260 | 0.4266 | 0.3711 | 0.4571 | 0.2705 | 0.3579 | 0.7944 | 0.3170 |
| f_1111112_s42 | 1111112 | 0.4848 | 0.1257 | 0.4272 | 0.3712 | 0.4586 | 0.2700 | 0.3605 | 0.7940 | 0.3172 |
| f_1111113_s42 | 1111113 | 0.4861 | 0.1253 | 0.4292 | 0.3707 | 0.4600 | 0.2696 | 0.3654 | 0.7954 | 0.3200 |
| f_1111114_s42 | 1111114 | 0.4859 | 0.1253 | 0.4292 | 0.3707 | 0.4598 | 0.2696 | 0.3654 | 0.7954 | 0.3201 |

## Appendix C: every early MSVAE model in the legacy evaluator

| Model | Mask | PTB r | PTB MAE | PTB R² | EchoNext r | EchoNext MAE | EchoNext R² | PTB ST r | Legacy macro AUROC | Legacy macro AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factorial_msvae_1000001_s42 | 1000001 | 0.8184 | 0.0735 | 0.4338 | 0.8017 | 0.3276 | 0.3450 | 0.7486 | 0.8434 | 0.4158 |
| factorial_msvae_1000002_s42 | 1000002 | 0.8032 | 0.0731 | 0.5233 | 0.7135 | 0.3181 | 0.3462 | 0.7803 | 0.8432 | 0.4068 |
| factorial_msvae_1000003_s42 | 1000003 | 0.8574 | 0.0667 | 0.6557 | 0.7936 | 0.2846 | 0.4660 | 0.8345 | 0.8652 | 0.4427 |
| factorial_msvae_1000004_s42 | 1000004 | 0.8513 | 0.0679 | 0.6133 | 0.7845 | 0.2904 | 0.4853 | 0.8313 | 0.8595 | 0.4393 |
| factorial_msvae_1000010_s42 | 1000010 | 0.8438 | 0.0678 | 0.6400 | 0.7804 | 0.2974 | 0.4613 | 0.8109 | 0.8622 | 0.4367 |
| factorial_msvae_1000011_s42 | 1000011 | 0.8295 | 0.0715 | 0.5181 | 0.8082 | 0.3231 | 0.3629 | 0.7837 | 0.8509 | 0.4281 |
| factorial_msvae_1000012_s42 | 1000012 | 0.8741 | 0.0630 | 0.7066 | 0.7988 | 0.3014 | 0.4511 | 0.8544 | 0.8663 | 0.4489 |
| factorial_msvae_1000013_s42 | 1000013 | 0.8834 | 0.0608 | 0.7259 | 0.8144 | 0.2927 | 0.5260 | 0.8730 | 0.8664 | 0.4528 |
| factorial_msvae_1000014_s42 | 1000014 | 0.8499 | 0.0685 | 0.6461 | 0.7947 | 0.2961 | 0.4684 | 0.8230 | 0.8644 | 0.4437 |
| factorial_msvae_1000100_s42 | 1000100 | 0.6726 | 0.0874 | 0.0886 | 0.6901 | 0.3457 | 0.2623 | 0.6536 | 0.8319 | 0.3906 |
| factorial_msvae_1000101_s42 | 1000101 | 0.7687 | 0.0818 | 0.3714 | 0.7428 | 0.3298 | 0.3117 | 0.6971 | 0.8407 | 0.4097 |
| factorial_msvae_1000102_s42 | 1000102 | 0.6216 | 0.0941 | -0.0644 | 0.6277 | 0.3532 | 0.2265 | 0.4755 | 0.8247 | 0.3760 |
| factorial_msvae_1000103_s42 | 1000103 | 0.7910 | 0.0784 | 0.4405 | 0.7576 | 0.3315 | 0.3093 | 0.7237 | 0.8508 | 0.4074 |
| factorial_msvae_1000104_s42 | 1000104 | 0.8386 | 0.0721 | 0.4953 | 0.7864 | 0.3143 | 0.3485 | 0.7659 | 0.8569 | 0.4286 |
| factorial_msvae_1000110_s42 | 1000110 | 0.6852 | 0.0889 | 0.1619 | 0.6834 | 0.3352 | 0.2853 | 0.7266 | 0.8372 | 0.3897 |
| factorial_msvae_1000111_s42 | 1000111 | 0.6380 | 0.0917 | 0.0363 | 0.6122 | 0.3540 | 0.2169 | 0.4908 | 0.8179 | 0.3617 |
| factorial_msvae_1000112_s42 | 1000112 | 0.1081 | 0.1600 | 0.0032 | 0.1093 | 0.4351 | 0.0034 | -0.0356 | 0.8502 | 0.4077 |
| factorial_msvae_1000113_s42 | 1000113 | -0.0215 | 0.1518 | -0.5869 | 0.0781 | 0.4174 | -0.0243 | -0.0790 | 0.7518 | 0.2920 |
| factorial_msvae_1000114_s42 | 1000114 | 0.6053 | 0.0954 | -0.1592 | 0.5908 | 0.3518 | 0.2094 | 0.5267 | 0.8076 | 0.3608 |

## Source artifacts

- `refine-logs/queue_3arch/queue_state.json` — Live 320-job queue state
- `refine-logs/queue_3arch/jobs` — Per-job MSVAE training logs
- `results/clinical_biomarkers_multids/clinical_metrics.db` — Authoritative clinical metrics and paired-inference SQLite store
- `results/clinical_biomarkers_multids/clinical_metrics_summary_missing_leads_v2.csv` — Append-only V2 CSV; audited for duplicates, not used as authority
- `results/clinical_biomarkers_multids/architecture_completeness.json` — Architecture claim gate
- `scripts/common_loss.py` — Factorial mask and loss definitions
- `scripts/evaluate_clinical_biomarkers_multids.py` — Evaluator semantics and failure location

Generated by `scripts/build_results_so_far_report.py` using the project virtual environment. Rebuild after queue or evaluator changes to refresh all tables.
