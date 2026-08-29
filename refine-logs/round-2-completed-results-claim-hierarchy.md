# Round 2 — Completed-results claim hierarchy

## Frozen problem anchor

The current book exposes extensive completed ECG reconstruction results but
does not consistently distinguish internal validation, external evaluation,
linear accessibility, incremental utility, matched objective effects, and
clinical utility. The refinement target is a reproducible analysis layer that
makes those distinctions explicit and derives claims only from compatible,
stored evidence.

### Bottleneck

Leaderboards and UMAPs can be overinterpreted when the underlying evidence is
single-seed, selected on a composite checkpoint score, evaluated with weak
comparators, or missing matched controls. The highest-risk case is the ECG-AIM
representation study, where a significant probe can be mistaken for objective
superiority or prospective clinical validity.

### Non-goals

- No new training or retrospective relabelling.
- No claim that one observed lead identifies the true missing 12-lead ECG.
- No conversion of released rhythm codes into incident or recurrent AF.
- No description of UMAP separation as proof of biological manifolds.
- No pooling across incompatible masks, budgets, seeds, or evaluator generations.

## Claim hierarchy

1. **Descriptive:** report stored values, splits, sample sizes, completion and
   checkpoint-selection rules.
2. **Matched internal:** compare the same lead, mask, seed, budget and evaluator;
   show direction and magnitude without inferential language if per-record
   uncertainty is unavailable.
3. **Paired inferential:** use shared held-out records, bootstrap intervals and
   multiplicity correction where persisted.
4. **External validity:** restrict to RDB/LUDB test protocols and retain
   endpoint-specific rather than universal rankings.
5. **Clinical utility:** not currently supported; requires a strong native-ECG
   comparator, calibration/decision analysis, external replication and a
   prospectively defined use case.

## Current representation verdict

- Frozen ECG-AIM latents contain AF/AFIB-code information beyond heart rate,
  RMS and spectral entropy.
- `1011011` does not significantly outperform `1000000` on paired AF probe
  metrics; it must not be called an improved AF base architecture.
- `1011011` improves several latent-only QRS-onset–to–T-offset timing metrics,
  but waveform summaries remain substantially better and combined gains are
  negligible.
- Rhythm labels are locally enriched but not globally separated in latent
  space; UMAP is descriptive.
- Latent outlier score predicts an internally defined reconstruction-failure
  endpoint, but requires external calibration before operational use.

## Current convergence verdict

- Thirty-three terminal run artifacts supersede the stale queue ledger's ten
  completed labels.
- Longer training disproportionately improves P/T and boundary endpoints;
  short-run rankings confound architecture with convergence rate.
- R5 has the strongest broad lead-II 15-epoch profile, while the simpler A0
  wavelet/no-SSL model remains strongest on several reconstruction endpoints.
- The principal A0 raw → A0+wavelet → A0+wavelet+SSL contrast remains incomplete
  at 15 epochs, so an SSL effect is not identified.

## Acceptance criteria for the remaining refinement

- Completed spatial and external chapters contain matched controls, endpoint
  conflicts, failure modes and claim ledgers.
- Every affected QMD executes from current databases without hidden state.
- A full-book render succeeds.
- Final refinement artifacts distinguish supported, contradicted, unresolved
  and explicitly out-of-scope claims.

External independent review is unavailable in this environment. This round is
therefore a local evidence audit and is not represented as independent review.
