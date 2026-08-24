# Quarto book improvement plan — 2026-08-24

Status: author-fix round 1 in progress. Public release remains blocked.

## Evidence base

- Fresh foundations review: `BOOK_FOUNDATIONS_REVIEW_20260824.md` (4 critical, 41 major, 15 minor).
- Fresh benchmark review: `BOOK_BENCHMARKS_REVIEW_20260824.md` (6 critical, 29 major, 12 minor).
- Fresh live/system review: `BOOK_LIVE_SYSTEM_REVIEW_20260824.md` (pending persistence at plan creation).
- Mechanical audit: 19 configured QMDs, 148 executable blocks after label repair, no duplicate labels or missing local links in the pre-fix snapshot.
- Execution sweep round 1: 15/19 chapters returned zero; Chapters 07, 08, 10, and 13 failed on missing legacy artifacts. This sweep used Quarto 1.4.555 and is diagnostic only, not a release build.

## Non-negotiable release gates

1. Render all configured chapters sequentially with repository-pinned Quarto 1.10.18 and the registered venv kernel.
2. Never promote old/no-execute HTML after a failed render; render to staging and validate before promotion.
3. Bind claim-bearing tables to current, generation-identified artifacts and show digest/as-of/completeness status.
4. Distinguish confirmatory results, exploratory results, operational status, simulations/tutorials, and unavailable/pending evidence.
5. Block downstream claims when a sanity gate fails (including the Zhejiang ceiling).
6. Do not use a checkpoint-catalog fallback as evidence for a different locked model generation.
7. Use patient as the inferential unit where identity is available; otherwise label the limitation and do not call record-held-out analyses patient-held-out.
8. Preserve detector failures and missingness in denominators; never turn unavailable endpoint flags into negative labels.
9. Vendor or explicitly gate external runtime dependencies; verify every deployed resource.
10. Generate a site-wide release manifest covering source, helpers, data snapshots, environment, renderer, output inventory, and build time.

## One-at-a-time author-fix order

### Phase A — scientific correctness

1. Chapter 03: repair the malformed energy-distance equation; expose the IMQ shared-kernel mismatch; expand factorial Cartesian-product and branch test gates.
2. Chapter 04: replace finite-model/clinical guarantees with testable hypotheses; correct MMD/W1, agreement, clustering, and multiplicity language.
3. Zhejiang tutorial: enforce the ceiling gate, clarify detector-versus-mask terminology, and block millisecond claims until sampling provenance is authoritative.
4. Chapters 02 and index: replace the cross-generation registry fallback with explicit generation status; make executive evidence status machine-derived.
5. Chapters 05 and 06: clearly segregate didactic simulation from empirical evidence; correct detector/SQI implementation descriptions and executable contracts.

### Phase B — current artifact migration

6. Chapter 08: migrate the locked 48-cell analysis to `results/factorial_v4/`; remove hard-coded pseudo-ANOVA and render the actual BCa/familywise tables.
7. Chapter 07: migrate smartwatch analysis to `results/factorial_v4_clinical/`; make frozen-classifier task fidelity explicit and show the incomplete final audit gate.
8. Chapter 10: migrate EchoNext figures/audits to current `factorial_v4` and clinical artifacts; suppress unavailable record-level analyses instead of retaining static claims.
9. Chapter 13: convert absent mixed-level artifacts into explicit pending panels or remove dependent claims; add waveform/failure panels only when digest-bound artifacts exist.
10. Chapters 09, 11, and 12: fix unsupported static claims, invalid ISP target handling, current-generation audit filtering, and snapshot timestamps.

### Phase C — live analyses and book system

11. Chapter 15: report every declared source, WAL-aware freshness, normalized lifecycle states, and one snapshot/render ID.
12. Chapter 16: typed endpoint/metric applicability; registered ECG-AIM comparisons; deterministic checkpoint selection; correct Bland–Altman and seed grouping; privacy-safe output.
13. Chapter 17: screening banner, explicit contrast estimability, queue health, and suppression of decorative UMAP until sample-size/stability gates pass.
14. Chapter 14: retain the safety boundary and complete the statistical-analysis-plan requirements.
15. Build system: pinned renderer, repository-relative command, staged atomic build, release manifest, warning policy, offline/runtime dependency audit.

## Verification sequence

For each source file: syntax/static audit → pinned single-chapter render → expected-output check → local-resource audit → claim/status inspection. After all files pass: fresh zero-context review round 2, severity triage, author fixes, pinned full sequential render, and final release audit. No score or acceptance claim will be invented for the unavailable Gemini backend.
