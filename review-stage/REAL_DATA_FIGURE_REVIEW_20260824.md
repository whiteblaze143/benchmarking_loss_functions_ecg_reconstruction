# Manual review — curated real-dataset figures

Reviewer: primary Codex agent. Review date: 2026-08-24. Each PNG was opened at
original resolution after deterministic regeneration. This is a visual and
scientific-communication audit, not an independent clinical-label audit.

## Shared acceptance checks

- Three panels are visible without overlap: UMAP, original-versus-projected
  neighborhood check, and a purpose-linked raw measurement.
- Group counts are visible; axes, units, diagnostic values, and the scientific
  question are legible.
- The raw panel does not hide sparse groups behind a boxplot.
- UMAP is not used as the sole evidence for separation.
- No RDB acquisition/patient identifier is displayed.

## Figure decisions

| Dataset | Manual observation | Revision made | Final narrow reading | PNG SHA-256 |
|---|---|---|---|---|
| PTB-XL | Train/validation/test points visually overlap; one large RMS outlier compressed the first draft. | Panel C changed to a log RMS scale while retaining the outlier. | Neighbor agreement 0.670 versus imbalance expectation 0.659 and negative silhouette do not support split-specific clusters. | `44088d0bd201ab42964a0df43aa0780e4567d534d5c38d8648b7bbe47aa065f9` |
| EchoNext | SHD groups overlap broadly; LVEF is lower and more dispersed in the SHD-positive sample. | Kept SHD as the endpoint color but placed LVEF in an independent raw panel. | Original-space enrichment is modest (0.563 versus 0.506); the plot is phenotype exploration, not an SHD classifier. | `786ee96b6a2eae1e9161d639981960234359aa5bf86b0a574f1a4e3d03089117` |
| LUDB | Fifteen AF-header records are sparse across the map; raw age differs visibly. | Shortened group names and made imbalance baseline explicit. | Same-group agreement equals the 0.861 imbalance expectation; the map provides no AF cluster claim. | `86fe20cd2fc11a6e4ee36eee2c0038686f15cf581417b80aeea76604b3472377` |
| ISP | Test points are interspersed with training; interval burdens overlap. | Figure purpose changed from generic embedding to split-shift audit. | Agreement is within 0.009 of imbalance expectation and silhouette is 0.017; no strong test island is shown. | `cad9c1b37853a4412ca9a8ce55183d793881a91b9c26dd9db27bc4481164bff3` |
| Sunnybrook | Two amplitude outliers are visible, but n=20 makes a boxplot and cluster language misleading. | Sparse outlier group now uses individual points plus a median rather than a box. | Trustworthiness 0.693 and seed stability 0.752 are weak; use the figure for QC/influence only. | `e5711471576d3eb20a24265bae7ac13c6ca1f8c1b12520ef8b539401964e0764` |
| Zhejiang | QRS-coverage quartiles show local gradients but extensive overlap. | Replaced long interval labels with ordered Q1–Q4 labels and kept raw percentages. | Original kNN enrichment is real (0.411 versus 0.247) but silhouette 0.014 rejects clean global clusters. | `8cefbeb5db901f8fab8a75f67e95290f62d847807ed72ed81c65c6b82760788c` |
| RDB | Eight retained rhythm codes overlap globally; P-mask coverage differs sharply by code. | Imposed stable code order and refused to expand `AFIB`, `SA`, or `SVT` beyond the cache mapping. | Original enrichment is modest (0.223 versus 0.147) with negative silhouette; P-mask coverage—not UMAP separation—is the mechanistic result. | `ba4f2b9bd84765b605492db7f7a45e481277295728ee8be2ccac4ffb3168dceb` |

## Final visual verdict

All seven figures pass the communication checks above. They do not support a
claim of seven clean phenotype clusters. Their value is diagnostic: they expose
split mixing, label imbalance, projection amplification, raw endpoint behavior,
and small-cohort instability. Any inferential follow-up must use original-space
features and patient-grouped validation where patient keys are verified.

## Rendered-artifact verification

The final Quarto-extracted PNGs were opened individually at original resolution
after the successful chapter render. Titles, legends, group counts, panel
letters, axis units, log-scale disclosure, sparse-group points, and diagnostic
bars remained legible; no panel was clipped or overlapped.

| Dataset | Final rendered PNG SHA-256 |
|---|---|
| PTB-XL | `268af6706bf5180e532d635ca32c56ce74f61cb2cd047d044bccde4262008066` |
| EchoNext | `b1097e1e6ff163a425810b34745c20300d3edd3bf13a11f2bc7dc7228322b9ea` |
| LUDB | `fb549b8af3f1aac6facd52650ef260ad3a779fc02a2f8122b2a7912179badc8c` |
| ISP | `a1883b80385ca609abf37059a4dc51af78c154a8db51b7e53c28c62f0b12167a` |
| Sunnybrook | `eb3bb5b9eebe9f1c062174c899d5df7c5f7a41f082ceae32816195601bcc39f8` |
| Zhejiang | `49b22232c1e74641e37cad65433db5a1de903cf71bcabdb97d2311f5e4c0f595` |
| RDB | `439966bebbacf22afb2c0153a00b6c59591acfda425a32ee629a22ed790587c4` |
