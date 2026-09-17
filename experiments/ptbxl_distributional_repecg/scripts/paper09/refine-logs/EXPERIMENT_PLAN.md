# Experiment Plan: Paper 09

## Objective
Validate that the operator-conditioned set architecture actually uses the
operator-response pairing before any clinical downstream comparison.

## Pre-clinical mechanism gate

The frozen `inductive_bias/audit_operator_biases.py` must pass before a
development grid can start. It uses held-out synthetic records and tests:

- exact permutation invariance of an identical operator-response context set;
- paired target-response recovery against a zero-response predictor;
- the mismatched-operator kill test;
- worse recovery for an incorrect target operator; and
- no material target-operator sensitivity in an operator-independent world.

The audit is an architectural mechanism test, not diagnostic evidence. A
failure leaves the Paper 09 clinical experiment ineligible; it does not permit
post-hoc changes to the operator bank, loss weights, or synthetic threshold.

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
