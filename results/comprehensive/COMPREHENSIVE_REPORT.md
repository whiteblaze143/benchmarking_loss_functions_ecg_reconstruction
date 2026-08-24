# Comprehensive ECG Reconstruction Loss Benchmark

## Evaluation protocol

- Dataset: PTB-XL test split (2198 records).
- Primary reconstruction metrics are computed only on leads not supplied to each model.
- Diagnostic labels, age, and sex are joined from PTB-XL metadata using ECG IDs.
- Clinical metrics use the same frozen 150-task ECGFounder classifier for every reconstruction.
- Robustness covers synthetic Gaussian noise and drift plus NSTDB baseline-wander, electrode-motion, and muscle-artifact records at 24/12/6/0 dB SNR.
- Results are single-seed unless stated otherwise; no inferential significance claim is made.

Every model receives the same I–II–V2 observed-lead basis. Cross-family absolute
comparisons remain descriptive because architectures and optimization differ; loss conclusions
primarily use within-family comparisons against that family's baseline.

## Primary missing-lead reconstruction results

| Model | Family | Loss | Observed leads | MSE ↓ | RMSE ↓ | Pearson ↑ | R² ↑ |
|---|---|---|---|---:|---:|---:|---:|
| unet_mse_full | unet | mse | I, II, V2 | 0.022344 | 0.132195 | 0.8649 | 0.6582 |
| unet_pearson_full | unet | mse+pearson | I, II, V2 | 0.021445 | 0.129179 | 0.9091 | 0.6781 |
| unet_mmd_full | unet | mse+mmd | I, II, V2 | 0.021762 | 0.130973 | 0.8640 | 0.6619 |
| unet_deriv_full | unet | mse+derivative | I, II, V2 | 0.021370 | 0.129450 | 0.8713 | 0.6714 |
| unet_all_full | unet | mse+pearson+mmd+derivative | I, II, V2 | 0.022081 | 0.131606 | 0.9041 | 0.6491 |
| cnvae_mse_full | cnvae | elbo | I, II, V2 | 0.045109 | 0.157493 | 0.4444 | 0.4428 |
| cnvae_pearson_full | cnvae | elbo+pearson | I, II, V2 | 0.045111 | 0.157498 | 0.4452 | 0.4427 |
| cnvae_mmd_full | cnvae | elbo+mmd | I, II, V2 | 0.045111 | 0.157497 | 0.4445 | 0.4427 |
| cnvae_all_full | cnvae | elbo+pearson+mmd+derivative | I, II, V2 | 0.045117 | 0.157509 | 0.4450 | 0.4425 |
| msvae_mse_full | multiscale_vae | mse | I, II, V2 | 0.018415 | 0.110858 | 0.9118 | 0.7986 |
| msvae_pearson_full | multiscale_vae | mse+pearson | I, II, V2 | 0.018563 | 0.111067 | 0.9175 | 0.8057 |
| msvae_mmd_full | multiscale_vae | mse+mmd | I, II, V2 | 0.018319 | 0.110365 | 0.9124 | 0.8006 |
| msvae_all_full | multiscale_vae | mse+pearson+mmd+derivative | I, II, V2 | 0.017963 | 0.107089 | 0.9219 | 0.8140 |
| alitok_mse_full | alitok_vae | mse | I, II, V2 | 0.018719 | 0.103726 | 0.9165 | 0.8069 |
| alitok_pearson_full | alitok_vae | mse+pearson | I, II, V2 | 0.018131 | 0.100647 | 0.9245 | 0.8200 |

## Diagnostic utility and calibration

| Model | Macro AUROC ↑ | Macro AP ↑ | Macro F1 ↑ | Sensitivity ↑ | Specificity ↑ | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| unet_mse_full | 0.8681 | 0.4425 | 0.0776 | 0.4221 | 0.9739 | 0.0391 | 0.0269 |
| unet_pearson_full | 0.8721 | 0.4595 | 0.0764 | 0.4043 | 0.9762 | 0.0353 | 0.0255 |
| unet_mmd_full | 0.8689 | 0.4629 | 0.0794 | 0.4058 | 0.9755 | 0.0383 | 0.0261 |
| unet_deriv_full | 0.8661 | 0.4473 | 0.0750 | 0.4017 | 0.9753 | 0.0382 | 0.0260 |
| unet_all_full | 0.8708 | 0.4509 | 0.0738 | 0.4020 | 0.9755 | 0.0365 | 0.0256 |
| cnvae_mse_full | 0.8473 | 0.4343 | 0.0694 | 0.4308 | 0.9671 | 0.0504 | 0.0322 |
| cnvae_pearson_full | 0.8473 | 0.4343 | 0.0694 | 0.4308 | 0.9671 | 0.0504 | 0.0322 |
| cnvae_mmd_full | 0.8473 | 0.4343 | 0.0694 | 0.4308 | 0.9671 | 0.0504 | 0.0322 |
| cnvae_all_full | 0.8473 | 0.4343 | 0.0694 | 0.4309 | 0.9670 | 0.0504 | 0.0323 |
| msvae_mse_full | 0.8795 | 0.4648 | 0.0750 | 0.4183 | 0.9746 | 0.0357 | 0.0264 |
| msvae_pearson_full | 0.8781 | 0.4589 | 0.0754 | 0.4218 | 0.9753 | 0.0342 | 0.0258 |
| msvae_mmd_full | 0.8790 | 0.4626 | 0.0756 | 0.4195 | 0.9747 | 0.0356 | 0.0263 |
| msvae_all_full | 0.8803 | 0.4638 | 0.0760 | 0.4266 | 0.9749 | 0.0351 | 0.0262 |
| alitok_mse_full | 0.8784 | 0.4609 | 0.0758 | 0.4021 | 0.9763 | 0.0341 | 0.0252 |
| alitok_pearson_full | 0.8802 | 0.4701 | 0.0785 | 0.4197 | 0.9759 | 0.0349 | 0.0255 |

## Within-family loss deltas

Deltas are relative to the MSE/ELBO baseline in the same family. Negative ΔMSE and
positive ΔPearson/ΔAUROC indicate improvement.

| Model | Baseline | Δ missing-lead MSE | Δ Pearson | Δ macro AUROC |
|---|---|---:|---:|---:|
| unet_pearson_full | unet_mse_full | -0.000898 | 0.0442 | 0.0039 |
| unet_mmd_full | unet_mse_full | -0.000582 | -0.0008 | 0.0008 |
| unet_deriv_full | unet_mse_full | -0.000973 | 0.0064 | -0.0020 |
| unet_all_full | unet_mse_full | -0.000263 | 0.0393 | 0.0027 |
| cnvae_pearson_full | cnvae_mse_full | 0.000002 | 0.0007 | -0.0000 |
| cnvae_mmd_full | cnvae_mse_full | 0.000002 | 0.0000 | -0.0000 |
| cnvae_all_full | cnvae_mse_full | 0.000008 | 0.0005 | -0.0000 |
| msvae_pearson_full | msvae_mse_full | 0.000148 | 0.0057 | -0.0014 |
| msvae_mmd_full | msvae_mse_full | -0.000096 | 0.0006 | -0.0005 |
| msvae_all_full | msvae_mse_full | -0.000452 | 0.0101 | 0.0008 |
| alitok_pearson_full | alitok_mse_full | -0.000588 | 0.0080 | 0.0018 |

## Robustness, morphology, and fairness

| Model | Gaussian 6 dB ΔMSE | NSTDB-BW 6 dB ΔMSE | NSTDB-EM 6 dB ΔMSE | NSTDB-MA 6 dB ΔMSE | R-peak timing MAE (ms) | Gender AUROC gap | Age AUROC gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| unet_mse_full | 0.005953 | 0.004945 | 0.005478 | 0.004709 | 35.93 | 0.0047 | 0.0250 |
| unet_pearson_full | 0.004666 | 0.003916 | 0.004366 | 0.004148 | 31.91 | 0.0061 | 0.0173 |
| unet_mmd_full | 0.009092 | 0.004984 | 0.005583 | 0.004909 | 36.27 | 0.0100 | 0.0292 |
| unet_deriv_full | 0.005615 | 0.005103 | 0.005367 | 0.004722 | 41.47 | 0.0191 | 0.0369 |
| unet_all_full | 0.005938 | 0.004927 | 0.005311 | 0.004957 | 33.85 | 0.0072 | 0.0245 |
| cnvae_mse_full | 0.003894 | 0.003943 | 0.003958 | 0.004027 | N/A | 0.0201 | 0.0310 |
| cnvae_pearson_full | 0.003894 | 0.003943 | 0.003958 | 0.004027 | N/A | 0.0200 | 0.0311 |
| cnvae_mmd_full | 0.003894 | 0.003943 | 0.003958 | 0.004027 | N/A | 0.0200 | 0.0311 |
| cnvae_all_full | 0.003894 | 0.003943 | 0.003958 | 0.004027 | N/A | 0.0199 | 0.0311 |
| msvae_mse_full | 0.000477 | 0.005488 | 0.005445 | 0.004468 | 40.93 | 0.0009 | 0.0224 |
| msvae_pearson_full | 0.000865 | 0.005394 | 0.005857 | 0.004870 | 39.68 | 0.0018 | 0.0256 |
| msvae_mmd_full | 0.000536 | 0.005466 | 0.005488 | 0.004321 | 40.55 | 0.0007 | 0.0229 |
| msvae_all_full | 0.001006 | 0.005641 | 0.006086 | 0.005004 | 37.47 | 0.0024 | 0.0269 |
| alitok_mse_full | 0.005238 | 0.005862 | 0.006474 | 0.006323 | 41.50 | 0.0024 | 0.0244 |
| alitok_pearson_full | 0.005825 | 0.005131 | 0.006266 | 0.006308 | 40.17 | 0.0043 | 0.0233 |

## Interpretation guardrails

- A lower composite training objective is not itself evidence of better reconstruction; the tables above use held-out metrics.
- Observed-lead copying can inflate all-lead scores, so missing-lead scores are primary.
- Cross-family comparisons are descriptive because architectures and optimization differ.
- Single-seed differences require multi-seed confirmation before strong claims.
- Subgroup gaps are descriptive and should be accompanied by uncertainty intervals in publication claims.

Machine-readable task exports and per-model plots are stored beside this report.
