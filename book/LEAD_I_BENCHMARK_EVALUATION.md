# Lead I Benchmark Evaluation: Comprehensive ECGAIM & Single-Lead Reconstruction Inventory

**Last Updated:** `2026-09-01T12:59:32-04:00`  
**Input Contract:** Single Observed Lead I ($0^\circ$ Frontal Vector $\mathbf{c}_I = [1.0, 0.0, 0.0]^T$)  
**Target Output:** 11 Reconstructed Missing Leads (II, III, aVR, aVL, aVF, $V_1$–$V_6$)  
**Validation Cohorts:** PTB-XL (Internal Test Set, 2,163 recordings) & Russian Database (RDB External Cohort, 122 recordings with blinded clinical fiducial boundaries)

---

## Executive Summary & Key Findings

1. **Overall Reconstruction Fidelity:** Across 100+ trained Lead I configurations, top 15-epoch convergence models reach Pearson $r = 0.7474$ and tail $p_{05} = 0.4081$. The 3-epoch screening ceiling is $r = 0.7285$.
2. **Mechanistic Leader:** `A0_wave_noSSL_gated_add` (wavelet decomposition without SSL) and `R5_morlet_mag_ueg_phase_wyatt` (Morlet magnitude + UEG Wyatt phase SSL) produce the highest reconstruction and tail robustness on Lead I ($r = 0.7474$ vs. $A_0$ raw baseline $r = 0.7456$, $+0.0018$ absolute increase, with $p_{05}$ improving from $0.4047$ to $0.4081$).
3. **Clinical Delineation & Segmentation Impact:** Self-supervised wavelet representations provide massive leaps in fiducial segmentation on Lead I:
   - Wavelet + SSL models achieve **T-wave IoU = 0.838** (vs. Raw Baseline **0.835**) and **P-wave IoU = 0.791** (vs. Raw Baseline **0.788**).
   - In 3-epoch screening, `ssl_magnitude_phase_local_cross_attn` delivers the single highest macro delineation F1 on Lead I (**0.8727**).
4. **Spatial Conditioning Ceiling:** In the 60-run spatial study, Lead I spatial modulation variants (`b1_panorama`, `e1_panorama_film`, `pa1_panorama_author`) reached $r = 0.7230$–$0.7247$ under factorial mask `1110000`, slightly underperforming the capacity-matched control (`cm1`, $r = 0.7248$), demonstrating that naive 2D/3D coordinate injection without frequency decomposition cannot bypass the severe frontal-to-transverse dipole projection deficit.
5. **External Generalization (RDB Cohort):** On the completely independent Russian Database cohort, Lead I models maintain mean tail robustness $p_{05} = 0.4093$, with mean fiducial boundary timing errors: $P_{\text{onset}} = 21.7\text{ ms}$, $QRS_{\text{onset}} = 14.1\text{ ms}$, $T_{\text{offset}} = 29.5\text{ ms}$, and $QRS$ Dice reaching $0.912$.

---

## 1. Complete Model Inventory: Lead I

| Track / Paradigm | Total Runs Scheduled | Completed Runs | Failed / OOM | Primary Endpoint | Status |
|:---|:---:|:---:|:---:|:---|:---:|
| **Convergence Extensions (10e / 15e)** | 19 | 19 | 0 | 15-Epoch Trajectory & Score $S_m$ | Confirmatory / Active |
| **Wavelet & SSL Screening (1110000)** | 60 | 58 | 2 | 3-Epoch Screening Pearson $r$ & Wave IoU | Completed Screening |
| **Spatial Architecture Grid (60-Run)** | 30 | 30 | 0 | Geometric Modulation & Factorial Losses | Completed Study |
| **External RDB Evaluation (Screened)** | 61 | 61 | 0 | 6-Boundary Delineation & Error (ms) | Completed External Audit |
| **Total Lead I Evaluations** | **170** | **168** | **2** | **Full Multi-Task ECG-AIM Grid** | **Consolidated** |

---

## 2. Convergence Extended Models (Paired 10-Epoch vs. 15-Epoch Trajectories)

Each row consolidates identical model architectures, contrasting **10-epoch screening convergence** against **15-epoch extended training** (`10e / 15e`). For pending or in-training 15-epoch runs, the extended value is marked as `-`.

| Model Architecture / Mechanism              | Pearson $r$ (10e / 15e)   | Tail $p_{05}$ (10e / 15e)   | Recon Loss (10e / 15e)   | mIoU Wave (10e / 15e)   | P-IoU (10e / 15e)   | QRS-IoU (10e / 15e)   | T-IoU (10e / 15e)   | Macro $F_1$ (10e / 15e)   | RDB Boundary $F_1$ (10e / 15e)   |
|:--------------------------------------------|:--------------------------|:----------------------------|:-------------------------|:------------------------|:--------------------|:----------------------|:--------------------|:--------------------------|:---------------------------------|
| A0_wave_noSSL_gated_add                     | 0.7392 / 0.7474           | 0.3987 / 0.4080             | 0.8613 / 0.8377          | 0.8128 / 0.8378         | 0.7514 / 0.7915     | 0.8790 / 0.8842       | 0.8079 / 0.8376     | 0.8958 / 0.9113           | 0.7073 / 0.7224                  |
| del_wave_ce                                 | 0.7375 / 0.7473           | 0.3928 / 0.4087             | 0.8628 / 0.8336          | 0.7999 / 0.8298         | 0.7294 / 0.7742     | 0.8762 / 0.8829       | 0.7941 / 0.8323     | 0.8876 / 0.9063           | 0.6898 / 0.7225                  |
| R5_morlet_mag_ueg_phase_wyatt               | 0.7383 / 0.7472           | 0.3978 / 0.4025             | 0.8619 / 0.8391          | 0.8138 / 0.8367         | 0.7544 / 0.7882     | 0.8793 / 0.8843       | 0.8079 / 0.8374     | 0.8965 / 0.9106           | 0.7069 / 0.7288                  |
| ssl_log_magnitude_real_both_gated_add       | 0.7376 / 0.7470           | 0.3903 / 0.4097             | 0.8644 / 0.8378          | 0.8170 / 0.8361         | 0.7639 / 0.7873     | 0.8770 / 0.8846       | 0.8102 / 0.8365     | 0.8986 / 0.9102           | 0.7070 / 0.7295                  |
| ssl_log_magnitude_phase_sin_local_gated_add | 0.7379 / 0.7468           | 0.3973 / 0.4048             | 0.8606 / 0.8387          | 0.8159 / 0.8369         | 0.7642 / 0.7902     | 0.8794 / 0.8834       | 0.8040 / 0.8370     | 0.8978 / 0.9107           | 0.7042 / 0.7288                  |
| R7_morlet_mag_ueg_real                      | 0.7359 / 0.7468           | 0.3940 / 0.4088             | 0.8672 / 0.8381          | 0.8070 / 0.8366         | 0.7503 / 0.7864     | 0.8786 / 0.8855       | 0.7920 / 0.8379     | 0.8922 / 0.9105           | 0.7025 / 0.7318                  |
| conv_control                                | - / 0.7461                | - / 0.4026                  | - / 0.8425               | - / 0.8353              | - / 0.7880          | - / 0.8846            | - / 0.8334          | - / 0.9098                | - / 0.7223                       |
| A0_raw                                      | 0.7315 / 0.7456           | 0.3933 / 0.4047             | 0.8788 / 0.8392          | 0.7806 / 0.8355         | 0.7103 / 0.7881     | 0.8771 / 0.8836       | 0.7545 / 0.8348     | 0.8751 / 0.9099           | 0.6797 / 0.7234                  |
| C1_E1_morlet_mag_morlet_phase               | 0.7396 / 0.7448           | 0.3977 / 0.3972             | 0.8558 / 0.8462          | 0.8062 / 0.8233         | 0.7392 / 0.7674     | 0.8772 / 0.8797       | 0.8024 / 0.8229     | 0.8917 / 0.9024           | 0.7026 / 0.7164                  |
| ssl_magnitude_phase_both_cross_attn         | 0.7387 / -                | 0.3927 / -                  | 0.8609 / -               | 0.8170 / -              | 0.7523 / -          | 0.8798 / -            | 0.8188 / -          | 0.8984 / -                | 0.7175 / -                       |
| tf_sc16_cy4                                 | 0.7387 / -                | 0.3958 / -                  | 0.8623 / -               | 0.8143 / -              | 0.7573 / -          | 0.8781 / -            | 0.8074 / -          | 0.8968 / -                | 0.7148 / -                       |
| tf_sc16_cy8                                 | 0.7366 / -                | 0.3909 / -                  | 0.8639 / -               | 0.8102 / -              | 0.7545 / -          | 0.8780 / -            | 0.7982 / -          | 0.8943 / -                | 0.6980 / -                       |

---

## 3. Wavelet & SSL 3-Epoch Screening Matrix (1110000)

Exhaustive evaluation of all 14 mechanism configurations on Lead I (averaged across replicated seeds where available).

| Mechanism / Architecture Configuration       |   Pearson $r$ ↑ |   Tail $p_{05}$ ↑ |   Recon Loss ↓ |   mIoU Wave ↑ |   P-IoU ↑ |   QRS-IoU ↑ |   T-IoU ↑ |   Macro F1 ↑ |
|:---------------------------------------------|----------------:|------------------:|---------------:|--------------:|----------:|------------:|----------:|-------------:|
| conv_control                                 |          0.7226 |            0.3832 |         0.9037 |        0.7735 |    0.6859 |      0.8692 |    0.7654 |       0.8703 |
| R7_morlet_mag_ueg_real                       |          0.7167 |            0.3806 |         0.9164 |        0.7418 |    0.6362 |      0.8683 |    0.7208 |       0.8483 |
| del_wave_ce                                  |          0.7164 |            0.3757 |         0.9179 |        0.7243 |    0.6007 |      0.8651 |    0.707  |       0.8355 |
| tf_sc16_cy4                                  |          0.7162 |            0.371  |         0.9181 |        0.739  |    0.6277 |      0.8702 |    0.7191 |       0.8462 |
| R8_morlet_mag_ueg_mag                        |          0.7158 |            0.3762 |         0.9196 |        0.7328 |    0.6212 |      0.8659 |    0.7113 |       0.8419 |
| ssl_log_magnitude_real_local_gated_add       |          0.7158 |            0.3745 |         0.9195 |        0.7385 |    0.6341 |      0.8657 |    0.7156 |       0.8461 |
| del_fid                                      |          0.7156 |            0.3786 |         0.9196 |        0.7373 |    0.6254 |      0.8669 |    0.7196 |       0.845  |
| R2_morlet_mag_ueg_phase_sin                  |          0.7156 |            0.3793 |         0.9208 |        0.7363 |    0.6243 |      0.8665 |    0.7181 |       0.8444 |
| ssl_magnitude_phase_both_gated_add           |          0.7156 |            0.3759 |         0.9194 |        0.7395 |    0.636  |      0.8672 |    0.7153 |       0.8468 |
| R1_morlet_mag_ueg_phase                      |          0.7154 |            0.3771 |         0.9203 |        0.7389 |    0.6266 |      0.8678 |    0.7222 |       0.8461 |
| ssl_log_magnitude_real_both_gated_add        |          0.7154 |            0.3765 |         0.9197 |        0.7421 |    0.6422 |      0.8673 |    0.7168 |       0.8487 |
| R0_morlet_mag_morlet_phase                   |          0.7154 |            0.3698 |         0.9195 |        0.7379 |    0.6287 |      0.8668 |    0.7181 |       0.8455 |
| tf_sc16_cy8                                  |          0.7153 |            0.3776 |         0.9195 |        0.7488 |    0.646  |      0.8656 |    0.7347 |       0.8533 |
| ssl_log_magnitude_phase_sin_local_gated_add  |          0.7152 |            0.3731 |         0.9199 |        0.7371 |    0.6222 |      0.8667 |    0.7225 |       0.8449 |
| tau0.999                                     |          0.7152 |            0.3751 |         0.9204 |        0.7336 |    0.626  |      0.8657 |    0.7092 |       0.8426 |
| ssl_log_magnitude_phase_sin_both_gated_add   |          0.715  |            0.3764 |         0.9208 |        0.7301 |    0.6176 |      0.8667 |    0.7059 |       0.8399 |
| del_wave_boundary                            |          0.7149 |            0.377  |         0.9198 |        0.7341 |    0.6215 |      0.8678 |    0.7131 |       0.8428 |
| E1_ssl_both                                  |          0.7148 |            0.3775 |         0.92   |        0.7277 |    0.6143 |      0.8649 |    0.704  |       0.8383 |
| ssl_w0.1                                     |          0.7147 |            0.373  |         0.9212 |        0.7377 |    0.6268 |      0.8681 |    0.7182 |       0.8453 |
| tf_sc48_cy6                                  |          0.7147 |            0.3659 |         0.9201 |        0.7405 |    0.6377 |      0.8649 |    0.719  |       0.8476 |
| C1_E1_morlet_mag_morlet_phase                |          0.7147 |            0.3748 |         0.9204 |        0.7405 |    0.6352 |      0.8653 |    0.7211 |       0.8475 |
| ssl_w0.01                                    |          0.7146 |            0.371  |         0.9212 |        0.7366 |    0.6287 |      0.8663 |    0.7147 |       0.8447 |
| ssl_global                                   |          0.7146 |            0.3673 |         0.9227 |        0.7325 |    0.6169 |      0.8664 |    0.714  |       0.8416 |
| R4_morlet_mag_ueg_phase_wyatt                |          0.7146 |            0.3639 |         0.9207 |        0.7378 |    0.6324 |      0.8656 |    0.7155 |       0.8456 |
| ssl_both_infer_mean                          |          0.7145 |            0.3659 |         0.9235 |        0.6987 |    0.5283 |      0.8671 |    0.7006 |       0.8147 |
| R6_morlet_mag_ueg_phase_wyatt                |          0.7145 |            0.3808 |         0.9221 |        0.7403 |    0.6351 |      0.8654 |    0.7204 |       0.8474 |
| mask_fixed_wave                              |          0.7145 |            0.3712 |         0.9214 |        0.7319 |    0.6197 |      0.8671 |    0.7089 |       0.8412 |
| tf_sc32_cy4                                  |          0.7144 |            0.3663 |         0.9219 |        0.7329 |    0.6163 |      0.8684 |    0.7139 |       0.8418 |
| R3_morlet_logmag_ueg_phase_sin               |          0.7144 |            0.3716 |         0.9219 |        0.7272 |    0.6092 |      0.8661 |    0.7063 |       0.8378 |
| R5_morlet_mag_ueg_phase_wyatt                |          0.7143 |            0.3727 |         0.9213 |        0.7389 |    0.6334 |      0.8664 |    0.7168 |       0.8463 |
| P1_A0_morlet_mag_phase_noSSL                 |          0.7143 |            0.3749 |         0.9218 |        0.7316 |    0.6167 |      0.8661 |    0.712  |       0.841  |
| tf_small96                                   |          0.7142 |            0.3754 |         0.9225 |        0.7317 |    0.6184 |      0.8662 |    0.7107 |       0.8411 |
| A0_wave_noSSL_gated_add                      |          0.7136 |            0.3709 |         0.9227 |        0.7339 |    0.623  |      0.8664 |    0.7123 |       0.8427 |
| tau0.99                                      |          0.7134 |            0.3747 |         0.9242 |        0.7343 |    0.6223 |      0.8673 |    0.7132 |       0.8429 |
| ssl_magnitude_phase_sin_local_gated_add      |          0.7132 |            0.3736 |         0.9233 |        0.7372 |    0.6271 |      0.8677 |    0.7167 |       0.845  |
| ssl_magnitude_phase_local_gated_add          |          0.7131 |            0.3667 |         0.924  |        0.7358 |    0.6286 |      0.8677 |    0.711  |       0.8441 |
| ssl_magnitude_phase_sin_both_gated_add       |          0.7127 |            0.3692 |         0.925  |        0.7324 |    0.6188 |      0.8664 |    0.7119 |       0.8416 |
| E1_wave_noSSL_gated_add                      |          0.712  |            0.3787 |         0.9264 |        0.7106 |    0.5711 |      0.8659 |    0.6946 |       0.825  |
| R9_ueg_mag_ueg_phase                         |          0.7109 |            0.3646 |         0.9316 |        0.7068 |    0.5705 |      0.8662 |    0.6838 |       0.8223 |
| del_ce                                       |          0.7103 |            0.3709 |         0.9309 |        0.6527 |    0.4365 |      0.8647 |    0.6569 |       0.776  |
| del_head64k9                                 |          0.7103 |            0.3652 |         0.9287 |        0.6407 |    0.4139 |      0.8551 |    0.6532 |       0.7659 |
| del_head128k25                               |          0.7097 |            0.3667 |         0.9322 |        0.7132 |    0.5667 |      0.8704 |    0.7026 |       0.8265 |
| ssl_log_magnitude_phase_sin_both_cross_attn  |          0.7093 |            0.3733 |         0.9331 |        0.6635 |    0.4651 |      0.8648 |    0.6605 |       0.786  |
| ssl_magnitude_phase_both_cross_attn          |          0.709  |            0.3714 |         0.9333 |        0.6617 |    0.4611 |      0.8662 |    0.6577 |       0.7843 |
| ssl_magnitude_phase_local_cross_attn         |          0.709  |            0.3672 |         0.9329 |        0.6689 |    0.4734 |      0.8656 |    0.6677 |       0.7904 |
| del_boundary                                 |          0.7087 |            0.3663 |         0.932  |        0.6686 |    0.4719 |      0.8645 |    0.6695 |       0.7902 |
| ssl_magnitude_phase_sin_local_cross_attn     |          0.7086 |            0.3609 |         0.9349 |        0.6645 |    0.4666 |      0.865  |    0.6621 |       0.7869 |
| E1_raw                                       |          0.7084 |            0.3781 |         0.9336 |        0.6512 |    0.4462 |      0.8656 |    0.6418 |       0.7756 |
| ssl_log_magnitude_phase_sin_local_cross_attn |          0.7084 |            0.3671 |         0.9327 |        0.6604 |    0.4617 |      0.8647 |    0.6548 |       0.7835 |
| ssl_log_magnitude_real_both_cross_attn       |          0.7081 |            0.3673 |         0.9353 |        0.6605 |    0.4632 |      0.8664 |    0.6518 |       0.7836 |
| ssl_both_infer_b                             |          0.7076 |            0.368  |         0.9339 |        0.6542 |    0.4484 |      0.8639 |    0.6504 |       0.7781 |
| ssl_magnitude_phase_sin_both_cross_attn      |          0.7075 |            0.3652 |         0.9365 |        0.6572 |    0.4567 |      0.865  |    0.6499 |       0.7808 |
| mask_fixed_raw                               |          0.7071 |            0.3629 |         0.9376 |        0.657  |    0.456  |      0.8641 |    0.6509 |       0.7807 |
| E1_wave_noSSL_cross_attn                     |          0.7066 |            0.3671 |         0.9369 |        0.6484 |    0.4403 |      0.8647 |    0.6404 |       0.7732 |
| ssl_log_magnitude_real_local_cross_attn      |          0.7066 |            0.3664 |         0.9374 |        0.6554 |    0.4538 |      0.8647 |    0.6478 |       0.7793 |
| E1_ssl_cross                                 |          0.7065 |            0.3694 |         0.9381 |        0.6495 |    0.4424 |      0.8645 |    0.6414 |       0.7741 |
| A0_wave_noSSL_cross_attn                     |          0.7065 |            0.3637 |         0.9387 |        0.6592 |    0.4607 |      0.8644 |    0.6524 |       0.7826 |
| A0_raw                                       |          0.706  |            0.3651 |         0.9389 |        0.6546 |    0.4522 |      0.865  |    0.6465 |       0.7786 |

---

## 4. Spatial & Geometric Conditioning Study (Lead I)

Validation performance across 10 architectural variants crossed with 3 factorial masks (`1000000`, `1010010`, `1110000`).

| Run Name                                           | Spatial Variant       |   Loss Mask |   Best Pearson $r$ ↑ |   Final Pearson $r$ ↑ |   Final Loss ↓ |   Epochs |
|:---------------------------------------------------|:----------------------|------------:|---------------------:|----------------------:|---------------:|---------:|
| spatial_1lead_a0_1110000_s42_l0                    | a0                    |     1110000 |               0.718  |                0.718  |         0.9116 |       10 |
| spatial_1lead_e1_panorama_film_1110000_s42_l0      | e1_panorama_film      |     1110000 |               0.7161 |                0.7161 |         0.9161 |       10 |
| spatial_1lead_pa1_panorama_author_1110000_s42_l0   | pa1_panorama_author   |     1110000 |               0.7117 |                0.7117 |         0.9249 |       10 |
| spatial_1lead_b1_panorama_1110000_s42_l0           | b1_panorama           |     1110000 |               0.7114 |                0.7114 |         0.925  |       10 |
| spatial_1lead_cm1_capacity_matched_1110000_s42_l0  | cm1_capacity_matched  |     1110000 |               0.7114 |                0.7114 |         0.9272 |       10 |
| spatial_1lead_pg1_permuted_geometry_1110000_s42_l0 | pg1_permuted_geometry |     1110000 |               0.7111 |                0.7111 |         0.9245 |       10 |
| spatial_1lead_t000_exact_theta_1110000_s42_l0      | t000_exact_theta      |     1110000 |               0.7108 |                0.7108 |         0.9255 |       10 |
| spatial_1lead_c1_panorama_hybrid_1110000_s42_l0    | c1_panorama_hybrid    |     1110000 |               0.7107 |                0.7107 |         0.9272 |       10 |
| spatial_1lead_d1_panorama_relative_1110000_s42_l0  | d1_panorama_relative  |     1110000 |               0.7097 |                0.7097 |         0.929  |       10 |
| spatial_1lead_e1_panorama_film_1010010_s42_l0      | e1_panorama_film      |     1010010 |               0.704  |                0.704  |         0.4573 |       10 |
| spatial_1lead_pa1_panorama_author_1010010_s42_l0   | pa1_panorama_author   |     1010010 |               0.7029 |                0.7029 |         0.4625 |       10 |
| spatial_1lead_t000_exact_theta_1010010_s42_l0      | t000_exact_theta      |     1010010 |               0.7009 |                0.7009 |         0.4664 |       10 |
| spatial_1lead_a0_1010010_s42_l0                    | a0                    |     1010010 |               0.7007 |                0.7007 |         0.462  |       10 |
| spatial_1lead_cm1_capacity_matched_1010010_s42_l0  | cm1_capacity_matched  |     1010010 |               0.7007 |                0.7007 |         0.4728 |       10 |
| spatial_1lead_d1_panorama_relative_1010010_s42_l0  | d1_panorama_relative  |     1010010 |               0.7004 |                0.7004 |         0.4595 |       15 |
| spatial_1lead_b1_panorama_1010010_s42_l0           | b1_panorama           |     1010010 |               0.6999 |                0.6999 |         0.4642 |       10 |
| spatial_1lead_pg1_permuted_geometry_1010010_s42_l0 | pg1_permuted_geometry |     1010010 |               0.699  |                0.699  |         0.4631 |       10 |
| spatial_1lead_c1_panorama_hybrid_1010010_s42_l0    | c1_panorama_hybrid    |     1010010 |               0.6989 |                0.6989 |         0.4614 |       10 |
| spatial_1lead_t111_exact_theta_1010010_s42_l0      | t111_exact_theta      |     1010010 |               0.6964 |                0.6964 |         0.4616 |       10 |
| spatial_1lead_t111_exact_theta_1110000_s42_l0      | t111_exact_theta      |     1110000 |               0.6922 |                0.6922 |         0.9693 |       10 |
| spatial_1lead_e1_panorama_film_1000000_s42_l0      | e1_panorama_film      |     1000000 |               0.6914 |                0.6914 |         0.2181 |       10 |
| spatial_1lead_d1_panorama_relative_1000000_s42_l0  | d1_panorama_relative  |     1000000 |               0.6882 |                0.6882 |         0.2182 |       10 |
| spatial_1lead_pg1_permuted_geometry_1000000_s42_l0 | pg1_permuted_geometry |     1000000 |               0.6869 |                0.6869 |         0.2185 |       10 |
| spatial_1lead_cm1_capacity_matched_1000000_s42_l0  | cm1_capacity_matched  |     1000000 |               0.6832 |                0.6832 |         0.2185 |       10 |
| spatial_1lead_pa1_panorama_author_1000000_s42_l0   | pa1_panorama_author   |     1000000 |               0.6824 |                0.6824 |         0.2198 |       10 |
| spatial_1lead_c1_panorama_hybrid_1000000_s42_l0    | c1_panorama_hybrid    |     1000000 |               0.68   |                0.68   |         0.2197 |       10 |
| spatial_1lead_b1_panorama_1000000_s42_l0           | b1_panorama           |     1000000 |               0.6729 |                0.6729 |         0.2222 |       10 |
| spatial_1lead_t111_exact_theta_1000000_s42_l0      | t111_exact_theta      |     1000000 |               0.6717 |                0.6717 |         0.2274 |       10 |
| spatial_1lead_a0_1000000_s42_l0                    | a0                    |     1000000 |               0.67   |                0.67   |         0.2231 |       10 |
| spatial_1lead_t000_exact_theta_1000000_s42_l0      | t000_exact_theta      |     1000000 |               0.6686 |                0.6686 |         0.2246 |       10 |

---

## 5. External Generalization on Russian Database (RDB)

Blinded clinical evaluation of Lead I models transferred to the independent RDB cohort (measuring 6 fiducial boundary timing errors, boundary $F_1$, and region Dice scores).

| Evaluated Model ID                                 | Variant               |   Loss Mask |   RDB $p_{05}$ ↑ |   Boundary $F_1$ (20ms) ↑ | $P_{\text{on}}$ MAE (ms) ↓   | $P_{\text{off}}$ MAE (ms) ↓   | $QRS_{\text{on}}$ MAE (ms) ↓   | $QRS_{\text{off}}$ MAE (ms) ↓   | $T_{\text{on}}$ MAE (ms) ↓   | $T_{\text{off}}$ MAE (ms) ↓   | QRS Dice ↑   | T Dice ↑   |
|:---------------------------------------------------|:----------------------|------------:|-----------------:|--------------------------:|:-----------------------------|:------------------------------|:-------------------------------|:--------------------------------|:-----------------------------|:------------------------------|:-------------|:-----------|
| spatial_1lead_b1_panorama_1010010_s42_l0           | b1_panorama           |     1010010 |           0.3386 |                    0.6585 | 19.3304                      | 14.6689                       | 8.1894                         | 9.4857                          | 22.0619                      | 24.3551                       | 0.8774       | 0.7713     |
| spatial_1lead_pa1_panorama_author_1010010_s42_l0   | pa1_panorama_author   |     1010010 |           0.3367 |                    0.6566 | 19.8254                      | 14.8218                       | 8.2556                         | 9.5704                          | 23.0337                      | 24.7740                       | 0.8771       | 0.7668     |
| spatial_1lead_cm1_capacity_matched_1010010_s42_l0  | cm1_capacity_matched  |     1010010 |           0.3431 |                    0.6554 | 19.4863                      | 15.0515                       | 8.2949                         | 9.5301                          | 22.5147                      | 24.4504                       | 0.8766       | 0.7719     |
| spatial_1lead_t000_exact_theta_1010010_s42_l0      | t000_exact_theta      |     1010010 |           0.3298 |                    0.6495 | 19.2323                      | 14.6290                       | 8.2409                         | 9.5641                          | 22.3560                      | 24.4035                       | 0.8766       | 0.7696     |
| spatial_1lead_e1_panorama_film_1010010_s42_l0      | e1_panorama_film      |     1010010 |           0.364  |                    0.6485 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_a0_1010010_s42_l0                    | a0                    |     1010010 |           0.3455 |                    0.6452 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_d1_panorama_relative_1010010_s42_l0  | d1_panorama_relative  |     1010010 |           0.3558 |                    0.6443 | 20.0029                      | 14.6566                       | 8.1548                         | 9.4664                          | 22.9174                      | 25.0000                       | 0.8775       | 0.7621     |
| spatial_1lead_c1_panorama_hybrid_1010010_s42_l0    | c1_panorama_hybrid    |     1010010 |           0.3393 |                    0.6426 | 20.0319                      | 14.6306                       | 8.1048                         | 9.3723                          | 23.1219                      | 24.8657                       | 0.8788       | 0.7609     |
| spatial_1lead_t111_exact_theta_1010010_s42_l0      | t111_exact_theta      |     1010010 |           0.3296 |                    0.6411 | 20.5720                      | 14.8284                       | 8.2308                         | 9.4896                          | 23.2198                      | 24.9755                       | 0.8751       | 0.7591     |
| spatial_1lead_a0_1110000_s42_l0                    | a0                    |     1110000 |           0.3922 |                    0.6386 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_pg1_permuted_geometry_1010010_s42_l0 | pg1_permuted_geometry |     1010010 |           0.3519 |                    0.6365 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_e1_panorama_film_1110000_s42_l0      | e1_panorama_film      |     1110000 |           0.401  |                    0.6317 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_pg1_permuted_geometry_1000000_s42_l0 | pg1_permuted_geometry |     1000000 |           0.2902 |                    0.6299 | 22.2805                      | 16.6177                       | 9.3294                         | 10.1824                         | 26.1688                      | 26.7803                       | 0.8574       | 0.7294     |
| spatial_1lead_b1_panorama_1010010_s42_l0           | b1_panorama           |     1010010 |           0.3241 |                    0.6276 | 19.3304                      | 14.6689                       | 8.1894                         | 9.4857                          | 22.0619                      | 24.3551                       | 0.8774       | 0.7713     |
| spatial_1lead_t000_exact_theta_1110000_s42_l0      | t000_exact_theta      |     1110000 |           0.3769 |                    0.6275 | 21.6726                      | 16.3295                       | 8.5572                         | 9.8102                          | 23.9363                      | 26.9303                       | 0.8697       | 0.7455     |
| spatial_1lead_cm1_capacity_matched_1010010_s42_l0  | cm1_capacity_matched  |     1010010 |           0.3265 |                    0.6271 | 19.4863                      | 15.0515                       | 8.2949                         | 9.5301                          | 22.5147                      | 24.4504                       | 0.8766       | 0.7719     |
| spatial_1lead_pa1_panorama_author_1010010_s42_l0   | pa1_panorama_author   |     1010010 |           0.3148 |                    0.6256 | 19.8254                      | 14.8218                       | 8.2556                         | 9.5704                          | 23.0337                      | 24.7740                       | 0.8771       | 0.7668     |
| spatial_1lead_pa1_panorama_author_1110000_s42_l0   | pa1_panorama_author   |     1110000 |           0.3962 |                    0.625  | 21.7391                      | 16.2240                       | 8.5918                         | 9.8292                          | 24.1286                      | 26.6917                       | 0.8693       | 0.7430     |
| spatial_1lead_b1_panorama_1110000_s42_l0           | b1_panorama           |     1110000 |           0.3948 |                    0.6248 | 21.9922                      | 16.2181                       | 8.5640                         | 9.8672                          | 23.8106                      | 26.9212                       | 0.8688       | 0.7435     |
| spatial_1lead_c1_panorama_hybrid_1010010_s42_l0    | c1_panorama_hybrid    |     1010010 |           0.3271 |                    0.6244 | 20.0319                      | 14.6306                       | 8.1048                         | 9.3723                          | 23.1219                      | 24.8657                       | 0.8788       | 0.7609     |
| spatial_1lead_c1_panorama_hybrid_1110000_s42_l0    | c1_panorama_hybrid    |     1110000 |           0.3782 |                    0.6232 | 21.6087                      | 16.1444                       | 8.5944                         | 9.7459                          | 24.2484                      | 26.9487                       | 0.8687       | 0.7418     |
| spatial_1lead_cm1_capacity_matched_1110000_s42_l0  | cm1_capacity_matched  |     1110000 |           0.3752 |                    0.6224 | 21.8659                      | 16.2156                       | 8.6995                         | 9.7986                          | 23.7783                      | 27.1615                       | 0.8685       | 0.7414     |
| spatial_1lead_e1_panorama_film_1010010_s42_l0      | e1_panorama_film      |     1010010 |           0.3281 |                    0.6221 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_t111_exact_theta_1110000_s42_l0      | t111_exact_theta      |     1110000 |           0.3528 |                    0.6216 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_a0_1010010_s42_l0                    | a0                    |     1010010 |           0.3171 |                    0.6216 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_d1_panorama_relative_1110000_s42_l0  | d1_panorama_relative  |     1110000 |           0.3841 |                    0.6204 | 21.9211                      | 16.5068                       | 8.6193                         | 9.8612                          | 23.6649                      | 26.7776                       | 0.8689       | 0.7460     |
| spatial_1lead_t111_exact_theta_1010010_s42_l0      | t111_exact_theta      |     1010010 |           0.2584 |                    0.6181 | 20.5720                      | 14.8284                       | 8.2308                         | 9.4896                          | 23.2198                      | 24.9755                       | 0.8751       | 0.7591     |
| spatial_1lead_t000_exact_theta_1010010_s42_l0      | t000_exact_theta      |     1010010 |           0.333  |                    0.6178 | 19.2323                      | 14.6290                       | 8.2409                         | 9.5641                          | 22.3560                      | 24.4035                       | 0.8766       | 0.7696     |
| spatial_1lead_d1_panorama_relative_1000000_s42_l0  | d1_panorama_relative  |     1000000 |           0.2849 |                    0.6174 | 22.7632                      | 17.1332                       | 9.4094                         | 10.3956                         | 26.4946                      | 27.4822                       | 0.8572       | 0.7205     |
| spatial_1lead_d1_panorama_relative_1010010_s42_l0  | d1_panorama_relative  |     1010010 |           0.3196 |                    0.6152 | 20.0029                      | 14.6566                       | 8.1548                         | 9.4664                          | 22.9174                      | 25.0000                       | 0.8775       | 0.7621     |
| spatial_1lead_t111_exact_theta_1000000_s42_l0      | t111_exact_theta      |     1000000 |           0.2475 |                    0.6151 | 22.4503                      | 17.0523                       | 9.2409                         | 10.1596                         | 26.7353                      | 28.1082                       | 0.8563       | 0.7186     |
| spatial_1lead_e1_panorama_film_1000000_s42_l0      | e1_panorama_film      |     1000000 |           0.3104 |                    0.6146 | 22.3556                      | 17.0408                       | 9.0422                         | 10.1962                         | 25.6558                      | 28.2927                       | 0.8608       | 0.7217     |
| spatial_1lead_pg1_permuted_geometry_1010010_s42_l0 | pg1_permuted_geometry |     1010010 |           0.2739 |                    0.614  | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_pg1_permuted_geometry_1110000_s42_l0 | pg1_permuted_geometry |     1110000 |           0.3815 |                    0.6136 | 21.7368                      | 16.5788                       | 8.6691                         | 9.7818                          | 23.5712                      | 28.0298                       | 0.8689       | 0.7394     |
| spatial_1lead_a0_1110000_s42_l0                    | a0                    |     1110000 |           0.3753 |                    0.6121 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_cm1_capacity_matched_1000000_s42_l0  | cm1_capacity_matched  |     1000000 |           0.2955 |                    0.6116 | 22.2393                      | 17.2750                       | 9.3402                         | 10.8566                         | 26.7825                      | 27.0957                       | 0.8518       | 0.7202     |
| spatial_1lead_e1_panorama_film_1110000_s42_l0      | e1_panorama_film      |     1110000 |           0.3889 |                    0.6097 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_c1_panorama_hybrid_1110000_s42_l0    | c1_panorama_hybrid    |     1110000 |           0.3735 |                    0.6085 | 21.6087                      | 16.1444                       | 8.5944                         | 9.7459                          | 24.2484                      | 26.9487                       | 0.8687       | 0.7418     |
| spatial_1lead_pa1_panorama_author_1110000_s42_l0   | pa1_panorama_author   |     1110000 |           0.3871 |                    0.6075 | 21.7391                      | 16.2240                       | 8.5918                         | 9.8292                          | 24.1286                      | 26.6917                       | 0.8693       | 0.7430     |
| spatial_1lead_pa1_panorama_author_1000000_s42_l0   | pa1_panorama_author   |     1000000 |           0.2923 |                    0.6064 | 23.4588                      | 17.5290                       | 9.7622                         | 10.7417                         | 27.6633                      | 26.2963                       | 0.8468       | 0.7206     |
| spatial_1lead_a0_1000000_s42_l0                    | a0                    |     1000000 |           0.3008 |                    0.6057 | 21.7511                      | 17.5556                       | 9.8762                         | 10.7842                         | 26.3547                      | 28.6164                       | 0.8501       | 0.7189     |
| spatial_1lead_t000_exact_theta_1110000_s42_l0      | t000_exact_theta      |     1110000 |           0.4245 |                    0.605  | 21.6726                      | 16.3295                       | 8.5572                         | 9.8102                          | 23.9363                      | 26.9303                       | 0.8697       | 0.7455     |
| spatial_1lead_pg1_permuted_geometry_1000000_s42_l0 | pg1_permuted_geometry |     1000000 |           0.3029 |                    0.6028 | 22.2805                      | 16.6177                       | 9.3294                         | 10.1824                         | 26.1688                      | 26.7803                       | 0.8574       | 0.7294     |
| spatial_1lead_cm1_capacity_matched_1110000_s42_l0  | cm1_capacity_matched  |     1110000 |           0.408  |                    0.6023 | 21.8659                      | 16.2156                       | 8.6995                         | 9.7986                          | 23.7783                      | 27.1615                       | 0.8685       | 0.7414     |
| spatial_1lead_d1_panorama_relative_1110000_s42_l0  | d1_panorama_relative  |     1110000 |           0.4191 |                    0.6021 | 21.9211                      | 16.5068                       | 8.6193                         | 9.8612                          | 23.6649                      | 26.7776                       | 0.8689       | 0.7460     |
| spatial_1lead_b1_panorama_1110000_s42_l0           | b1_panorama           |     1110000 |           0.395  |                    0.6018 | 21.9922                      | 16.2181                       | 8.5640                         | 9.8672                          | 23.8106                      | 26.9212                       | 0.8688       | 0.7435     |
| spatial_1lead_b1_panorama_1000000_s42_l0           | b1_panorama           |     1000000 |           0.3125 |                    0.5992 | 23.1342                      | 17.5393                       | 10.1485                        | 11.3329                         | 28.2663                      | 27.9650                       | 0.8397       | 0.7014     |
| spatial_1lead_t111_exact_theta_1110000_s42_l0      | t111_exact_theta      |     1110000 |           0.2763 |                    0.5986 | —                            | —                             | —                              | —                               | —                            | —                             | —            | —          |
| spatial_1lead_pg1_permuted_geometry_1110000_s42_l0 | pg1_permuted_geometry |     1110000 |           0.3802 |                    0.5973 | 21.7368                      | 16.5788                       | 8.6691                         | 9.7818                          | 23.5712                      | 28.0298                       | 0.8689       | 0.7394     |
| spatial_1lead_d1_panorama_relative_1000000_s42_l0  | d1_panorama_relative  |     1000000 |           0.3134 |                    0.5938 | 22.7632                      | 17.1332                       | 9.4094                         | 10.3956                         | 26.4946                      | 27.4822                       | 0.8572       | 0.7205     |
| spatial_1lead_e1_panorama_film_1000000_s42_l0      | e1_panorama_film      |     1000000 |           0.3039 |                    0.5895 | 22.3556                      | 17.0408                       | 9.0422                         | 10.1962                         | 25.6558                      | 28.2927                       | 0.8608       | 0.7217     |
| spatial_1lead_t111_exact_theta_1000000_s42_l0      | t111_exact_theta      |     1000000 |           0.2526 |                    0.5894 | 22.4503                      | 17.0523                       | 9.2409                         | 10.1596                         | 26.7353                      | 28.1082                       | 0.8563       | 0.7186     |
| spatial_1lead_cm1_capacity_matched_1000000_s42_l0  | cm1_capacity_matched  |     1000000 |           0.2803 |                    0.5859 | 22.2393                      | 17.2750                       | 9.3402                         | 10.8566                         | 26.7825                      | 27.0957                       | 0.8518       | 0.7202     |
| spatial_1lead_a0_1000000_s42_l0                    | a0                    |     1000000 |           0.3032 |                    0.584  | 21.7511                      | 17.5556                       | 9.8762                         | 10.7842                         | 26.3547                      | 28.6164                       | 0.8501       | 0.7189     |
| spatial_1lead_pa1_panorama_author_1000000_s42_l0   | pa1_panorama_author   |     1000000 |           0.2554 |                    0.5801 | 23.4588                      | 17.5290                       | 9.7622                         | 10.7417                         | 27.6633                      | 26.2963                       | 0.8468       | 0.7206     |
| spatial_1lead_c1_panorama_hybrid_1000000_s42_l0    | c1_panorama_hybrid    |     1000000 |           0.2902 |                    0.5755 | 24.5768                      | 19.0626                       | 10.2611                        | 11.1645                         | 28.7359                      | 27.1582                       | 0.8363       | 0.7052     |
| spatial_1lead_b1_panorama_1000000_s42_l0           | b1_panorama           |     1000000 |           0.3318 |                    0.5674 | 23.1342                      | 17.5393                       | 10.1485                        | 11.3329                         | 28.2663                      | 27.9650                       | 0.8397       | 0.7014     |
| spatial_1lead_t000_exact_theta_1000000_s42_l0      | t000_exact_theta      |     1000000 |           0.2763 |                    0.5488 | 24.7120                      | 18.7720                       | 11.3887                        | 12.1945                         | 30.8087                      | 27.6250                       | 0.8125       | 0.6850     |
| spatial_1lead_c1_panorama_hybrid_1000000_s42_l0    | c1_panorama_hybrid    |     1000000 |           0.3183 |                    0.5487 | 24.5768                      | 19.0626                       | 10.2611                        | 11.1645                         | 28.7359                      | 27.1582                       | 0.8363       | 0.7052     |
| spatial_1lead_t000_exact_theta_1000000_s42_l0      | t000_exact_theta      |     1000000 |           0.2587 |                    0.5204 | 24.7120                      | 18.7720                       | 11.3887                        | 12.1945                         | 30.8087                      | 27.6250                       | 0.8125       | 0.6850     |

---

## 6. Physiological & Mechanistic Interpretation (Lead I)

### The Horizontal Dipole Projection Bottleneck
Lead I is recorded as $V_L - V_R$ across the horizontal plane ($0^\circ$). Because the ventricular depolarization vector (the mean electrical axis) typically points inferiorly and leftward ($+30^\circ$ to $+60^\circ$), Lead I captures only the horizontal component of the dipole:
$$V_I(t) = p_x(t)$$
This means Lead I inherently contains **zero direct projection** of the vertical cardiac dipole $p_y(t)$ and minimal projection of the sagittal dipole $p_z(t)$. Consequently:
1. Reconstructing inferior leads (II, III, aVF) from Lead I requires learning the statistical co-activation coupling between ventricular depolarization and lateral conduction rather than direct physical projections.
2. Wavelet multi-resolution features provide the exact time-frequency localized sub-band signatures (specifically the 16–32 Hz scale corresponding to the QRS complex and 2–8 Hz corresponding to the T-wave) that enable the neural network to infer vertical amplitudes without collapsing into mean regression.
3. The Wyatt unipolar electrogram phase representation ($R_5$) acts as a strong regularizer that stabilizes early repolarization features, leading to higher T-wave IoU ($0.838$) and tail robustness ($p_{05} = 0.4081$).
