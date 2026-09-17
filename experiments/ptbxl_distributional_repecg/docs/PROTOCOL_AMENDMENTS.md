# Protocol Amendments

## A001 — Nyström fidelity thresholds

Timestamp: 2026-09-16, before any full development training or fold-8 model
comparison.

Original gate:

- Spearman correlation at least `0.95`;
- median relative error below `0.10`;
- promote 128 to 256 landmarks if 128 fails.

Observed pre-training audit on 1,000 frozen development pairs:

| Landmarks | Spearman rho | Median relative error | Original result |
|---:|---:|---:|---|
| 128 | 0.94543 | 0.18332 | FAIL |
| 256 | 0.95698 | 0.12645 | FAIL |

User-authorized amendment after seeing only approximation fidelity—not model
performance:

- Spearman correlation at least `0.90`;
- median relative error below `0.15`.

Under the amended gate, 128 remains `FAIL` and 256 becomes `PASS`. Therefore
all representations using this audit must use 256 landmarks. The amendment may
not be revisited after development model results are observed. The original
audit is retained as `nystrom_audit_pre_amendment.json` in the Paper-2 runtime
evidence directory.
