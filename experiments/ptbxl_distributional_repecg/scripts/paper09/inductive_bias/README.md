# Paper 09 inductive-bias gates

Paper 09 may not enter a clinical development grid until the frozen synthetic
mechanism gate passes. It is an architectural test, not evidence of clinical
benefit.

## Synthetic contract

The audit samples `S in R^2`, a fixed `L in R^(8x2)`, and externally assigned
unit operators `q` independent of `S`. Its raw synthetic response is
`F_i(q) = (q^T L S_i) C`. This permits exact sign, homogeneity, and
superposition checks. Those checks do **not** apply to the nonlinear Phase-KME
features in the clinical pipeline.

The recovery task observes an unordered context of three `(q, F(q))` pairs and
predicts a held-out `F(q*)`. A three-context result is eligible only when
`rank(Q L)=2`; the artifact reports `sigma_min(Q L)` and an observability curve
instead of pooling impossible and observable cases.

## Required gates

- Jointly permuting all context pairs leaves the prediction unchanged.
- A within-record response-only derangement preserves the `q` and `F`
  marginals exactly but materially damages recovery.
- The original paired relative-MSE qualification threshold remains `< 0.50`.
- An incorrect query is substantially worse, while a q-independent negative
  control remains insensitive to its target query.
- The raw synthetic measurement obeys sign, homogeneity, and superposition.
- Geometry-only and target-only ridge controls remain q-blind because `Q` is
  exogenous to patient state.
- Rank-one duplicate contexts are explicitly declared unidentifiable, and
  end-to-end gradients are finite and nonzero.

The artifact additionally reports the analytic pseudoinverse oracle, q-blind
error, oracle-gap closure, query-angle degradation, and context-count
diagnostics. It preserves all failed pilot artifacts; diagnostics never relax
the frozen `0.50` qualification rule.

## Architecture under test

The model receives both the paired cross-moment
`C_qF = sum_j q_j F_j^T` and the context geometry Gram matrix
`G_Q = sum_j q_j q_j^T`. Both are invariant to a joint permutation of pairs;
only `C_qF` changes under a response-only permutation. The clinical decoder is
still nonlinear in Phase-KME space, so this does not claim a linear KME
measurement law.
