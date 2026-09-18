# Research Proposal: Paper 11 — Predictive State Compression for Ordered ECG Beats

## Problem Anchor

- **Bottom-line problem:** determine whether a compact state computed from prior ECG beats predicts later beats, rather than merely being a diagnostic feature cluster.
- **Must-solve bottleneck:** diagnosis-trained recurrent states have no obligation to represent the conditional future.
- **Non-goals:** biological-causal discovery, a classical epsilon-machine minimality theorem, and record-level inputs that already average away beat chronology.
- **Constraints:** patient-disjoint PTB-XL folds; preserve ordered raw beats; fit 1–6, select 7, pseudo-test 8, do not read 9–10.
- **Success condition:** held-out future prediction improves with a finite causal-prefix state; synthetic recovery succeeds only where identifiable and fails under pair destruction and IID data.

## Why the Existing Paper Fails

The current model is a phase-feature GRU trained only with diagnosis BCE. It has no future-prediction target, conditional-future equivalence test, state-minimality mechanism, or complexity estimator. Its `shuffled_futures` variant simply permutes inputs. The existing Phase-KME representation averages all beats before classification, so it cannot support a beat-history claim.

## Method Thesis

**A finite categorical state inferred only from an ordered beat prefix is a predictive compression only if it improves held-out next-beat prediction at a reported effective state complexity.**

For beat descriptors \(r_{1:T}\), a causal GRU produces \(h_t\) from \(r_{\leq t}\), assigns \(q_t=\operatorname{softmax}(W h_t)\), and a state embedding predicts \(r_{t+1}\):

\[
L=L_{\rm diagnosis}+\lambda_{\rm future}\|D(q_tE)-r_{t+1}\|_2^2.
\]

The paper makes no epsilon-machine, minimum-sufficient-state, or biological-causality claim. Effective state complexity is descriptive occupancy entropy. The new decoder is the only added trainable component.

## Required Controls and Gates

- `diag_only`, `continuous_predictor`, `wrong_future`, and within-record `beat_shuffle`.
- Synthetic finite-HMM recovery, wrong-pair falsification, and IID negative control.
- Exact beat/target/patient/fold provenance and causal-prefix masking before real data.

**Initial verdict: RETHINK.** No current Paper 11 grid result is interpretable for this proposal.
