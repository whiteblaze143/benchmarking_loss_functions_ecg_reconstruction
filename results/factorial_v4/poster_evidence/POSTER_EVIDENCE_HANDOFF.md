# Factorial v4 Poster Evidence Handoff

Machine gates: core `complete`, clinical `PASS`, final `9/10`.

The only unresolved final check is the independent postrun audit. Every statement below is therefore labeled empirical and provisional for publication wording.

## Selected architecture results

| Family | Mask | PTB-XL R² | PTB-XL Pearson | EchoNext R² | EchoNext SHD AUROC |
|---|---:|---:|---:|---:|---:|
| U-Net | `110` | 0.662 | 0.908 | -35.243 | 0.724 |
| MultiScale-VAE | `100` | 0.816 | 0.922 | -0.966 | 0.767 |
| ECG-AIM | `110` | 0.820 | 0.925 | 0.593 | 0.775 |

## Proposed empirical poster statements

1. Among validation-selected models, ECG-AIM has the highest PTB-XL R2 and is the only family with positive EchoNext R2.
2. Correlation loss improves R2 in MultiScale-VAE and ECG-AIM but reduces R2 in U-Net.
3. The full objective improves QRS and ST correlations in every family, without a significant ECGFounder AUROC improvement.
4. At 0 dB NSTDB, U-Net's worse clean reconstruction can become competitive because it degrades less than the latent models.
5. All selected family-device smartwatch evaluations have negative missing-eleven-lead R2.

## Required visual framing

- Show absolute clean performance and degradation under noise together.
- Keep morphology and diagnostic-utility endpoints visually separate.
- Label smartwatch classifier outputs as probability-fidelity proxies.
- State that EchoNext SHD inference also uses unchanged tabular metadata.
- Do not display the quarantined MultiScale-VAE MSE-off cells as valid effects.
