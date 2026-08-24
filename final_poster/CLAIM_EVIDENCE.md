msvae

# Claim-to-Evidence Map

| Poster claim                                                          | Locked source                                                                 |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 48 primary models; 3 architectures × 16 masks                        | `data/all_48_models_master.csv`; `data/completeness_verification.json`    |
| 18 confirmation runs                                                  | `../results/factorial_v4_2x4/COMPLETENESS_REPORT.md`                        |
| 2,198 PTB-XL ECGs / 1,904 patients / 2,000 BCa resamples              | `data/familywise_endpoint_tests.csv`                                        |
| Full composite improves all six QRS/ST contrasts at α=0.0167         | `data/familywise_endpoint_tests.csv`, QRS and ST rows                       |
| No significant ECGFounder AUROC change in the three full-vs-MSE tests | `data/familywise_endpoint_tests.csv`, diagnostic utility rows               |
| MultiScale-VAE 0100: Pearson 0.874, R² −14.984                      | `data/all_48_models_master.csv`                                             |
| Best R² masks: U-Net 1000; MultiScale-VAE 1101; ECG-AIM 0111         | `data/poster_summary.csv`                                                   |
| MSE-on conditional R² component effects and BCa intervals            | `data/factorial_effects_mse_on_conditional.csv`                             |
| 17 paired noise conditions                                            | `data/completeness_verification.json`; master noise columns                 |
| EchoNext n=5,442; smartwatch n=719; Sunnybrook n=20                   | comprehensive package raw protocol files and`data/all_48_models_master.csv` |
| Full ECG-AIM Sunnybrook Pearson 0.849                                 | `data/all_48_models_master.csv`                                             |
| Full-loss ECGFounder, quality, artifact, and fidelity values          | `data/all_48_models_master.csv`                                             |
| 27/48 exploratory ECGFounder non-inferiority passes                   | `../results/factorial_v4_2x4/statistics.json`                               |
| 22/22 comprehensive-package checks                                    | `data/completeness_verification.json`                                       |

All generated figure source data are copied into `data/`. Figure and template
asset hashes are recorded in `ASSET_MANIFEST.json`.

## Author-only declaration

No authoritative funding or conflict-of-interest wording was present in the
provided paper/project materials. `poster.tex` therefore exposes editable
macros and visibly marks the statement as requiring author confirmation.
