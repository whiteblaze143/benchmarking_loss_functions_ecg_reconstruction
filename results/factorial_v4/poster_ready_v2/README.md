# Poster-ready factorial-v4 package

This directory is an isolated, regenerable figure and interpretation layer. It
does not overwrite the audited factorial-v4 result tables or earlier figures.

- `REVISED_INTERPRETATION.md`: poster-level scientific interpretation
- `FIGURE_PLAN.md`: visual intent and source mapping
- `CAPTIONS.md`: compact captions
- `figures/`: PDF, SVG, PNG, and per-figure provenance
- `scripts/build_all.py`: deterministic figure build
- `verify_figures.py`: source/output integrity and coverage checks
- `VERIFICATION.json`: verifier result
- `MANIFEST.md`: file hashes

Regenerate with:

```bash
/home/mithunmanivannan/.venv/bin/python \
  results/factorial_v4/poster_ready_v2/scripts/build_all.py
```

Then verify with:

```bash
/home/mithunmanivannan/.venv/bin/python \
  results/factorial_v4/poster_ready_v2/verify_figures.py
```
