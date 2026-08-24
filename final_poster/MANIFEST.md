# Final Poster Manifest

## Primary deliverables

- `poster.tex` — editable A0 source of truth.
- `poster.pdf` — verified print PDF.
- `poster_preview.png` — full-page review image.
- `README.md` — build and editing instructions.
- `VERIFICATION.json` — 24/24 local verification checks.
- `SHA256SUMS.txt` — primary-deliverable hashes.

## Evidence and review

- `CONTENT_PLAN.md`
- `CLAIM_EVIDENCE.md`
- `VISUAL_REVIEW.md`
- `POSTER_STATE.json`
- `ASSET_MANIFEST.json`

## Deterministic scripts

- `scripts/build_poster_assets.py`
- `scripts/verify_poster.py`
- `Makefile`

## Locked data snapshots

- `data/all_48_models_master.csv`
- `data/familywise_endpoint_tests.csv`
- `data/factorial_effects_mse_on_conditional.csv`
- `data/poster_summary.csv`
- `data/completeness_verification.json`

## Poster figures

Every quantitative figure is available in vector PDF and high-resolution PNG:

- `assets/figures/factorial_atlas.*`
- `assets/figures/clinical_effects.*`
- `assets/figures/conditional_r2_effects.*`
- `assets/figures/noise_stress.*`
- `assets/figures/external_scorecard.*`
- `assets/figures/echonext_representative_reconstructions.*`

## Official conference assets

- `assets/logos/embc2026.png`
- `assets/logos/ieee_embs.png`
- `assets/template/IEEE-EMBSPosterTemplate2026.pptx`
- `assets/template/image1.png` through `image5.png`
- `assets/qr/project_qr.png`

## Build intermediates

`build/` contains the LuaLaTeX log/auxiliary files and the rendered preview
used during verification. They can be removed with `make clean` and recreated
with `make`.
