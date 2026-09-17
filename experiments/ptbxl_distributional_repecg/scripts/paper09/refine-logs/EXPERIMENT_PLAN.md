# Experiment Plan: Paper 09

## Objective
Validate that the operator-conditioned set architecture actually uses the
operator-response pairing before any clinical downstream comparison.

## Pre-clinical mechanism gate

The frozen `inductive_bias/audit_operator_biases.py` must pass before a
development grid can start. It uses an exogenous raw linear synthetic world
`F_i(q) = (q^T L S_i) C`, with `S in R^2`, `L in R^(8x2)`, and reports recovery
only conditional on `rank(Q L)=2`. It tests:

- exact joint permutation invariance of an identical operator-response set;
- a surgical within-context response derangement which preserves both context
  marginals exactly and changes only the q-response association;
- paired target-response recovery under the unchanged relative-MSE `<0.50`
  qualification gate, against both a q-blind baseline and analytic
  pseudoinverse oracle;
- query specificity, q-independent negative control behavior, and q-only /
  geometry-only exogeneity controls;
- raw-response sign, homogeneity, and superposition laws; and
- rank-deficient duplicate-context declaration, observability diagnostics,
  query-angle degradation, and finite end-to-end gradients.

The architecture under test exposes both `G_Q = sum q q^T` and
`C_qF = sum q F^T`. This is a measurement-geometry addition to the paired
cross-moment, not an assertion that nonlinear Phase-KME features are linear in
q.

The audit is an architectural mechanism test, not diagnostic evidence. A
failure leaves the Paper 09 clinical experiment ineligible; it does not permit
post-hoc changes to the operator bank, loss weights, or synthetic threshold.
The existing 100-epoch bilinear result remains diagnostic only and cannot
retroactively qualify the architecture.

## Variants to Test
- **full**: Evaluates the specific component or serves as a baseline/sham control.
- **diagnosis_only**: Evaluates the specific component or serves as a baseline/sham control.
- **mismatched_q**: Evaluates the specific component or serves as a baseline/sham control.

## Must-Prove Claims
- Counterfactual prediction uses the pairing between `q` and `F(q)`.
- The representation is invariant to context-set ordering.
- Operator conditioning affects predictions only when the response depends on
  the target operator.

## Validation Strategy
After the mechanism gate, compare `full`, `diagnosis_only`, and `mismatched_q`
on the frozen clinical development artifact. Report diagnostic, held-out
counterfactual, and normalized-state metrics separately; no model is selected
from fold 8.
