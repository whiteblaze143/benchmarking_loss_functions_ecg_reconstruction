# Benchmarking Loss Functions for ECG Reconstruction

This folder contains the code and paper-supporting material for the loss-function benchmarking project:

**Benchmarking Loss Functions for 12-Lead ECG Reconstruction from Limited Leads: A Multi-Objective Framework for Morphological Fidelity**

## Contents

- `scripts/`: M0/M1-HPO training, loss ablation, HPO/Pareto analysis, morphology, diagnostic utility, robustness, external validation, and figure/table utilities.
- `notebooks/`: exploratory notebooks for baseline evaluation, ablations, and paper metrics.
- `docs/`: draft methods/intro/discussion notes plus loss-design and Mason-alignment notes.
- `figures/`: generated or paper-supporting figures that were previously mixed into root-level folders.
- `src_notes/`: reference notes moved out of shared `src/` locations.
- `CHECKPOINT_STORAGE.md`: exact checkpoint archival, verification, cache, and
  inference instructions for the 480-cell factorial queue.

## Runtime Assumptions

The current factorial training, book EDA, and inference paths use the
project-local `data/`, `checkpoints/`, and `results/` directories. Some legacy
evaluation scripts still import shared modules under
`/home/mithunmanivannan/src`; treat those scripts as legacy until their imports
have been migrated and verified.

Use the project virtual environment and run scripts from the repository root,
for example:

```bash
/home/mithunmanivannan/.venv/bin/python scripts/evaluate_nstdb.py
```

## Split Boundary

I left the foundation-model/VAE project in its existing locations, including:

- `scripts/launch_fm_vae_recovery.sh`
- `ecg_fm_integration/`
- `reports/fm_vae_recovery/`
- `reports/manuscripts/wearecg_fm_*`
- `reports/wearecg_fm_*`
- `checkpoints/wearecg_fm/`

## Known Gaps Preserved From Original Workspace

Some HPO launcher scripts refer to `m1_multiobj_hpo.py` or `m1_multiobj_hpo_v2_rigorous.py`. Those target files were not present in the workspace at the time of the split, so the launchers are preserved as historical/runbook material rather than verified runnable entry points.

Several legacy scripts import `src.reconstruction.learn_functions.mason_mmd_variants`, which was also not present as a source file in the current workspace. I did not fabricate or rewrite that model implementation during the folder split.
