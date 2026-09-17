## Verdict: REVISE — 8.86/10

The proposal is now scientifically coherent and largely implementation-ready. The Problem Anchor is preserved, the contribution is focused, and all eight branches have independent evidence boundaries. Three locally repairable—but genuine—method defects remain, so it does not yet satisfy READY.

Progress: **6.30 → 8.35 → 8.86**.

| Dimension | Score | Weight | Weighted |
|---|---:|---:|---:|
| Problem Fidelity | 9.7 | 15% | 1.455 |
| Method Specificity | 8.4 | 25% | 2.100 |
| Contribution Quality | 9.3 | 25% | 2.325 |
| Frontier Leverage | 8.8 | 15% | 1.320 |
| Feasibility | 8.5 | 10% | 0.850 |
| Validation Focus | 8.2 | 5% | 0.410 |
| Venue Readiness | 8.0 | 5% | 0.400 |
| **OVERALL** |  |  | **8.860/10** |

## What is now resolved

- The anchor is correctly narrowed to diagnostic utility on PTB-XL, with no physiological-state or generic-method novelty claims ([proposal](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/refine-logs/round-2-refinement.md:68)).
- Branches are judged independently; negative and inconclusive outcomes are retained without a portfolio-winner claim ([proposal](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/refine-logs/round-2-refinement.md:73)).
- Whitening, Nyström coordinates, near-zero audit errors, exact finite moment matching, Paper 6 conditioning, and endpoint-specific eligibility are now concrete.
- `INCONCLUSIVE` is terminal, eliminating adaptive reseeding after observing fold 8 ([gate matrix](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/docs/CLAIM_GATE_MATRIX.md:17)).
- Paper 7 has frozen operator banks and diagnosis-only primary training ([contract](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/docs/MATHEMATICAL_CONTRACT.md:218)).
- Paper 8 now separates patients for vocabulary construction and equivalence calibration and attempts simultaneous familywise uncertainty control.

## Genuine blockers

### 1. Paper 3 applies PCA after full whitening

The contract whitens the full path descriptor and then selects 64 PCA components ([contract](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/docs/MATHEMATICAL_CONTRACT.md:147)). After full whitening, the training covariance is approximately identity, so “top” PCA directions are arbitrary and may be determined by numerical noise.

This is not merely a coding detail: it makes the retained path subspace unstable and scientifically uninterpretable.

**Fix:** perform one truncated PCA-whitening operation: center the raw path descriptors, retain the 64 leading training-covariance eigenvectors, then scale those retained scores by their eigenvalues. No additional experiment is required.

### 2. Paper 7’s unseen categorical comparator is undefined

The primary mechanism statistic is categorical-lead-ID MSE minus continuous-operator MSE on unseen operators ([gate matrix](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/docs/CLAIM_GATE_MATRIX.md:35)). But the contract does not specify how a categorical model represents an operator absent from its training vocabulary.

Possible choices—one `<UNK_q>` embedding, nearest known ID, or an untrained new ID—produce materially different baselines. Additionally, response MSE comes from the auxiliary reconstruction model even though that model is called secondary.

**Fix:** freeze one `<UNK_q>` category for every unseen operator, use the same response encoder, decoder, depth, and width as the continuous model, and state explicitly that the separately trained reconstruction pair supplies the mechanism gate while the BCE-only pair supplies `Delta_task`.

### 3. Paper 8’s simultaneous upper-bound direction is not justified

The contract uses

\[
T_b=\max_j(D^*_{bj}-\hat D_j),\qquad
U_j=\hat D_j+q_{0.95}(T).
\]

That is a percentile-style positive estimator-deviation band. For an upper confidence bound on the true distance, the required error is instead \(D_j-\hat D_j\). Without an explicit symmetry/pivotal argument—which MMD need not satisfy near equivalence—the stated construction does not establish the claimed familywise upper coverage ([contract](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/docs/MATHEMATICAL_CONTRACT.md:250)).

**Fix:** use the basic one-sided simultaneous construction

\[
T_b=\max_j(\hat D_j-D^*_{bj}),\qquad
U_j=\hat D_j+q_{0.95}(T),
\]

or a frozen studentized equivalent.

Paper 8’s primary “odd/even token agreement” also needs an exact formula. A suitable fixed definition is patient-level

\[
1-\operatorname{JSD}(p_{\mathrm{odd}},p_{\mathrm{even}})/\log 2
\]

between odd- and even-beat token-frequency vectors, followed by patient-equal aggregation.

## Implementation polish—not blockers

- Replace the undefined `K_AA` in the Nyström ridge expression with `κ(A,A)`.
- Freeze Paper 5’s PCA dimension; the refinement says PCA sizes are frozen, but only Paper 3’s size appears in the contract.
- Rename Paper 5’s estimator “record-specific” unless all records from a patient are explicitly combined.
- Define zero-vector behavior for cosine mechanism statistics.
- State whether mechanism statistics are averaged first within record and then patient, especially for patients with multiple records.
- Replace the two surviving phrases “causal tests” and “causally used” with “mechanism-use tests” ([proposal](/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/refine-logs/round-2-refinement.md:225)).
- Flatten the refinement document instead of recursively embedding Round 1 inside Round 2. This is editorial only.

## Drift Warning

**NONE.**

Paper 7 remains a within-lead-span operator study, Paper 8 remains an equivalence-vocabulary study, and none of the branches has displaced the distribution-valued phase-cell primitive.

## Simplification Opportunities

- Implement PCA and whitening as a single train-fitted transform rather than two sequential objects.
- Add a short metric glossary containing the exact eight mechanism-statistic formulas, aggregation units, and undefined-value rules.
- Make the precedence chain canonical in the proposal summaries rather than relying on annex overrides.
- Keep Paper 7’s diagnostic and reconstruction models separate in code and reporting.

## Modernization Opportunities

**NONE required.**

The techniques are modern and appropriately placed. Additional neural modules or larger pretrained models would weaken attribution.

## Remaining action items

1. **P0:** Replace Paper 3’s whitening-then-PCA with frozen truncated PCA whitening.
2. **P0:** Define Paper 7’s unseen categorical operator and the role of the auxiliary reconstruction pair.
3. **P0:** Correct or formally justify Paper 8’s simultaneous upper-bound construction.
4. **P0:** Freeze the exact odd/even token-agreement statistic.
5. **P1:** Specify Paper 5 PCA size and patient/record terminology.
6. **P1:** Add aggregation and zero-vector rules for all mechanism statistics.
7. **P2:** Remove stale causal language and flatten the refinement document.
8. **P2:** Replace provisional resource estimates with measurements from the already-planned smoke runs.

No additional experiments or branch deletions are needed. Once the four P0 contract edits are made, this should plausibly clear the READY threshold.

Review basis: direct audit of the four supplied files; the external cross-model reviewer backend was unavailable.
