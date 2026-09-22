# Frozen PTB-XL latent-space figures

The five comparison arms are GraphECG, the three P07
SetOperator variants: subset/robust, subset/robust with auxiliary response
reconstruction, and full-lead-only; and FixedTensor P02. The apparent fourth GraphECG file under
the configuration-shift output is byte-identical to the canonical GraphECG
checkpoint, so it is deliberately not plotted as a separate model.

Extraction uses held-out PTB-XL Fold 8 (`n=2,173`) and full/Q8 input only. No
encoder or diagnostic head is fitted. Run after GPU queue idleness:

```bash
/home/mithunmanivannan/.venv/bin/python extract_frozen_ptbxl_latents.py --device cuda
/home/mithunmanivannan/.venv/bin/python gen_fig1_fold8_latent_space.py
/home/mithunmanivannan/.venv/bin/python gen_fig2_fold8_latent_spectrum.py
/home/mithunmanivannan/.venv/bin/python gen_fig3_ptbxl_shift_auroc.py
```

CPU extraction is allowed but materially slower:

```bash
/home/mithunmanivannan/.venv/bin/python extract_frozen_ptbxl_latents.py --device cpu
```

The PCA panels are fitted separately per encoder and therefore cannot support
claims about absolute locations, inter-model distances, separation quality, or
clinical superiority. The variance spectrum is a compactness diagnostic, not
a performance metric. Figure 3 reads the already frozen, exact macro-AUROC
matrix and prints every plotted value; it is the performance complement to the
descriptive latent-space figures.
