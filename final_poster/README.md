# EMBC 2026 Final Poster — Native LaTeX Edition

This folder is a self-contained, editable A0 portrait poster package for:

> **Benchmarking Loss Functions for 12-Lead ECG Reconstruction from Limited
> Leads: A Multi-Objective Framework for Morphological Fidelity**

The source of truth is [`poster.tex`](poster.tex). The current compiled output
is [`poster.pdf`](poster.pdf), and [`poster_preview.png`](poster_preview.png)
provides a quick full-page review.

## Build

From the repository root:

```bash
make -C final_poster
```

This command:

1. regenerates every quantitative figure from locked machine-readable results;
2. compiles the poster twice with LuaLaTeX;
3. renders the preview;
4. verifies the page geometry, assets, evidence cells, text, and LaTeX overflow
   log.

The build requires Python 3 with pandas/matplotlib/Pillow, LuaLaTeX, and the
TeX packages imported by `poster.tex`. Poppler binaries bundled under
`poster_html/tools/` are used for preview and PDF verification.

## Author edits before printing

Edit these three macros near the top of `poster.tex`:

```tex
\newcommand{\FundingStatement}{...}
\newcommand{\ConflictStatement}{...}
\newcommand{\PosterNumber}{...}
```

The supplied project materials did not contain authoritative funding or
conflict wording, so the poster intentionally displays an author-confirmation
notice instead of inventing a declaration. This is the only print-facing
content item still requiring author input.

## Folder structure

- `poster.tex` — native, highly editable A0 LaTeX poster.
- `poster.pdf` — one-page print output at exactly 841 × 1189 mm.
- `poster_preview.png` — low-resolution review rendering.
- `assets/logos/` — official EMBC 2026 and IEEE EMBS artwork extracted from the
  supplied PowerPoint.
- `assets/template/` — official template motifs and the original supplied PPTX.
- `assets/figures/` — vector PDF and high-resolution PNG visualizations.
- `data/` — locked poster-level evidence snapshots.
- `scripts/build_poster_assets.py` — deterministic figure generator.
- `scripts/verify_poster.py` — geometry, evidence, hash, and overflow checks.
- `ASSET_MANIFEST.json` — file-level provenance and SHA-256 hashes.
- `CLAIM_EVIDENCE.md` — poster claim-to-source map.
- `VERIFICATION.json` — latest 24-check verification ledger.
- `SHA256SUMS.txt` — hashes of the primary deliverables.

## Venue compliance

- True A0 portrait: **841 × 1189 mm**.
- One printed page, English language.
- Title and all authors/organisations appear at the top.
- Approximate typography: 76 pt title, 40 pt section headings, 25 pt body.
- Financial support/collaboration and conflict fields are visibly reserved.
- Official conference and society assets come from the supplied template.

The official presenter page was checked on 27 July 2026:
<https://embc.embs.org/2026/presenter-guidelines/>.

## Evidence scope

The poster includes the complete seed-42 \(2^4\) E/C/M/D factorial benchmark:
48 primary cells across U-Net, MultiScale-VAE, and ECG-AIM, plus 18
confirmation runs. Evaluations cover PTB-XL clean reconstruction and
morphology, frozen ECGFounder, PTB-XL superclasses, signal quality,
17 deterministic noise conditions, EchoNext, four smartwatches, and
Sunnybrook Cath Lab ECGs.

Sunnybrook AUROC is explicitly labelled as a non-adjudicated
Philips-statement proxy. Smartwatch results are simulator/device transfer, not
human diagnostic validation.

## Review status

Local visual, geometry, evidence, and print checks pass. The selected ARIS
poster skill normally requests an independent Gemini multimodal review; that
bridge is not available in this environment, so no independent-review claim is
made. See [`VISUAL_REVIEW.md`](VISUAL_REVIEW.md).
