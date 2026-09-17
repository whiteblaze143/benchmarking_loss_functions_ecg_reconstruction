# Research Proposal: Distribution-Valued ECG Objects

## Problem Anchor

- **Bottom-line problem:** Determine whether a diagnostically useful PTB-XL ECG representation can be built from distributions of locally contiguous cardiac electrical states, rather than producing another feature-engineering pipeline whose apparent gain comes from architecture capacity or test-set selection.
- **Must-solve bottleneck:** Prior repStat work showed that learned states were not reliably interpretable as P/QRS/T states, inferential non-rejection was conservative, and connected-component merging did not define a valid state ontology. The new program must preserve local structure and dependence without equating non-rejection with biological equivalence.
- **Non-goals:** Do not force named cardiac-wave semantics; do not use clinical metadata; do not claim generic recurrence, MMD, DMD, Koopman analysis, or ECG tokenization is itself novel; do not select a winner by AUROC alone; do not share learned objects across papers.
- **Constraints:** PTB-XL `records500` for training and validation; evaluation spans all 9 available datasets (EchoNext, LUDB, RDB, ISP, Kingston-ICU, Emory-MUSE, Sunnybrook, Zhejiang); official folds; an additional fold-8 development firewall; one local A100 40 GB; physical-mV information must be preserved; fold 10 stays locked until configurations are frozen; all fifteen specified branches remain independently runnable.
- **Success condition:** Each branch is judged independently and survives only if it (1) beats or stabilizes against its strongest matched representation control, (2) responds to a mechanism-use falsification in the predicted direction, and (3) retains diagnostic information under paired patient-level evaluation. There is no post-hoc portfolio-winner claim; negative and inconclusive branches remain reportable outcomes.

## Technical Gap

Raw waveform networks entangle morphology, sampling geometry, phase alignment,
lead identity, and classifier capacity. Ordinary summary statistics discard
higher-order local distributions. The previous repStat route added distribution
tests but exposed three failures: local statistical states did not acquire the
assumed physiological semantics, non-rejection was too conservative to serve as
equivalence, and transitive graph components overstated sameness.

The smallest shared correction is not a larger neural network. It is a frozen
measurement layer:

1. establish cardiac phase and locally contiguous cells;
2. represent each cell's empirical distribution with a train-fitted kernel mean;
3. preserve physical amplitude and dependence;
4. let each paper change exactly one mathematical object downstream;
5. require a matched control and a mechanism-use falsification for that change.

## Method Thesis

- **One-sentence thesis:** A cardiac phase cell should be treated as an empirical distribution in a shared RKHS, from which recurrence, path, dynamical, conditional, operator-valued, and discrete ECG objects can be derived and falsified without imposing wave labels.
- **Dominant program contribution:** A leakage-safe distribution-valued ECG measurement layer with branch-specific mathematical objects and mechanism tests.
- **Supporting contribution:** Dependence-preserving inference and practical equivalence replace invalid non-rejection graph semantics.
- **Explicit non-contributions:** A new generic kernel, generic CNN, generic DMD/Koopman method, or generic transformer.

## Complexity Budget

- **Frozen and reused:** PTB-XL folds and label aggregation; zero-phase filtering; WFDB R detection; PCHIP phase normalization; exact biased IMQ MMD; Nyström approximation; standard small probes; patient-clustered bootstrap.
- **New trainable components:** Each paper may introduce only its specified representation encoder and classification head. Shared learned kernel landmarks are fitted separately inside every paper run, never imported between paper outputs.
- **Rejected complexity:** P/QRS/T pseudo-labels, connected-component state merging, a monolithic network containing all eight ideas, clinical metadata, fold-10 architecture search, and simultaneous full-seed execution before mechanism gates.

## System Overview

```text
PTB-XL records500 (Training)
  -> canonical lead order + physical-mV copy
  -> 0.5--40 Hz zero-phase filter
  -> independent 8-lead basis
  -> train-only scaling / whitening
  -> R peaks -> valid RR cycles -> 256 phase samples
  -> locally contiguous phase or beat cells
  -> exact MMD audit + independently fitted Nyström kernel means
  -> one paper-specific mathematical object
  -> matched probe and mechanism-use control
  -> validation selection -> frozen manifest -> locked fold-10 evaluation + Multi-Dataset OOD evaluation (9 datasets)
```

## Shared Data and Representation Contract

- Development: folds 1--7 train and fold 8 representation/hyperparameter selection.
- Final: folds 1--8 train, fold 9 early stopping/model selection, fold 10 locked test.
- Primary labels: NORM, MI, STTC, CD, HYP using PTB-XL superclass aggregation.
- Independent basis: I, II, V1--V6; conventional 12-lead raw baseline remains mandatory.
- Beat eligibility: at least two valid full cycles, with an at-least-three sensitivity.
- Phase cells: 16 cells of 16 samples, except Paper 4's eight cells of 32 samples.
- IMQ audit: whitened descriptors and source-faithful c-squared=1 primary; scales 0.5, 1, 2, and train median are sensitivities.
- Nyström: 128 train-only landmarks, promoted to 256 unless 1,000 development pairs achieve Spearman at least 0.95 and median relative error below 10 percent.
- Training: AdamW, cosine schedule, maximum 100 epochs, patience 10, five fixed seeds after gates.
- Evaluation: paired patient bootstrap with 2,000 replicates; no test tuning.

## Fifteen Independent Branches

### Paper 1: Distributional Recurrence Operator

- **Object:** A 16-by-16 symmetrically normalized affinity between phase-cell kernel means, excluding cyclic neighbors.
- **Models:** Direct upper triangle, spectral descriptors, and a fixed small 2-D CNN.
- **Matched control:** Identical pipeline using distances between cell means.
- **Falsification:** Phase permutation and removal of local-neighbor suppression.
- **Kill rule:** Stop if distributional recurrence is neither more predictive nor materially more split-half/perturbation stable than mean recurrence.

### Paper 2: Kernel-Mean ECG

- **Object:** A 16-by-128 sequence of phase-cell RKHS means.
- **Model:** Three-block width-128 circular phase CNN with global pooling.
- **Matched controls:** Mean; mean plus SD; diagonal moments; linear-kernel mean.
- **Falsification:** Nonlinear IMQ must separate from matched moment/linear controls.
- **Kill rule:** Stop the distribution-level claim if nonlinear KME is indistinguishable from moments.

### Paper 3: Path-Signature repECG

- **Object:** Train-standardized start, end, mean, and depth-3 log-signature for every beat-cell, pooled distributionally by phase.
- **Model:** The same phase CNN as Paper 2.
- **Matched control:** Unordered waveform KME under the same downstream capacity.
- **Falsification:** Time reversal and sample-order destruction must change the representation; monotone warp should be comparatively stable.
- **Kill rule:** Stop if the claimed path-order bias is not used.

### Paper 4: Local Hankel Operators

- **Object:** Distributions of local delay/SVD/DMD descriptors from eight 32-sample phase regions, delay six and rank at most eight.
- **Model:** The same phase CNN family.
- **Matched controls:** Whole-ECG DMD, whole-beat DMD, local mean pooling, and raw-waveform KME.
- **Falsification:** Within-cell temporal scrambling must destroy the dynamical advantage.
- **Kill rule:** Stop if local distributional dynamics do not beat or stabilize over existing DMD-style alternatives.

### Paper 5: Recurrent-State Koopman ECG

- **Object:** A record-specific Koopman transition operator in a globally aligned 32-anchor soft-state coordinate system.
- **Model:** PCA of vectorized transition operator plus spectral/error descriptors and a fixed MLP.
- **Matched control:** State occupancy alone versus occupancy plus transition operator.
- **Falsification:** Chronological shuffling preserves occupancy but must harm the transition representation.
- **Kill rule:** Stop the dynamics claim if chronology destruction has no material effect.

### Paper 6: Conditional Residual repStat

- **Object:** Recurrence between residual lead-space distributions conditioned on a train-fitted rank-3 dominant electrical macrostate.
- **Model:** Paper 1 recurrence representation and matched probe.
- **Mandatory controls:** Full signal, macro component, residual marginal, and residual conditional on macrostate.
- **Falsification:** Within-record residual-to-macrostate permutation preserves both marginals while destroying their conditional relationship.
- **Kill rule:** Stop if the conditional model survives the pairing destruction unchanged.

### Paper 7: Continuous Measurement-Operator ECG

- **Object:** A function from normalized lead operator q to 16 phase-cell kernel responses of waveform and derivative.
- **Primary model:** Operator MLP plus circular response CNN, a two-block Set Transformer, and diagnosis head trained with BCE only. The target-operator reconstruction head belongs to a secondary auxiliary-loss model.
- **Matched controls:** Discrete lead identity, raw waveform response, and canonical-only training.
- **Falsification:** Held-out derived leads, dense random operators, sign symmetry, and continuous interpolation.
- **Kill rule:** Stop if unseen-operator response is discontinuous or no better than categorical lead identity.

### Paper 8: Statistical ECG Tokens

- **Object:** Tokens defined by complete-linkage distributional equivalence, using a bootstrap upper confidence bound below a practical within-state margin.
- **Model:** A four-layer, width-128 masked-token transformer.
- **Matched controls:** Equal-size K-means, learned VQ, waveform patches, median-beat tokens, and continuous kernel means.
- **Falsification:** Token quality must exceed equal-size quantization and not collapse to phase identity.
- **Kill rule:** Stop general-purpose claims if fold-10 unknown rate exceeds 20 percent or phase nearly determines token identity.

### Paper 9: Counterfactual Measurement-Operator ECG

- **Object:** Continuous counterfactual states of measurement operators based on causal representation learning.
- **Model:** Operator MLP plus causal-regularized response CNN.
- **Matched controls:** Standard continuous measurement-operator ECG without counterfactual regularizers.
- **Falsification:** Randomize the counterfactual intervention angle; must destroy the counterfactual prediction ability while maintaining observational prediction.
- **Kill rule:** Stop if counterfactual regularizers offer no material stability gain over the baseline operator model under distribution shifts.

### Paper 10: Interventional repStat ECG

- **Object:** Stability of local distributional representations under direct intervention on phase cells.
- **Model:** Phase CNN with intervention-aware stability training.
- **Matched controls:** Observational repStat without interventional stability.
- **Falsification:** Apply random non-causal perturbation (sham intervention) instead of the controlled intervention; stability should degrade.
- **Kill rule:** Stop if interventional stability does not outperform observational models on out-of-distribution paired sets.

### Paper 11: repStat Causal-State ECG

- **Object:** Causal states defined via predictive future equivalence rather than historical correlation, mapped to RKHS.
- **Model:** Predictive state RNN/CNN predicting future distribution embeddings.
- **Matched controls:** Past-history predictive state (standard Koopman/DMD).
- **Falsification:** Break the temporal link between past and future distributions (chronological shuffle); predictive state collapse required.
- **Kill rule:** Stop if the forward predictive states collapse to simple occupancy or fail to predict future RKHS embeddings.

### Paper 12: Structural-Innovation repECG

- **Object:** The structural innovation (unpredictable residual) from the predictable cyclic component of the ECG.
- **Model:** Cyclic-predictive model with innovation-extraction layer.
- **Matched controls:** Raw residual prediction without causal ordering.
- **Falsification:** Destroy the causal ordering of structural innovations; the predictive benefit of the innovation stream must vanish.
- **Kill rule:** Stop if structural innovations are indistinguishable from standard unstructured residuals.

### Paper 13: Counterfactual Distribution Surgery

- **Object:** Splicing/surgery of local phase distributions across counterfactual states to isolate causal features.
- **Model:** Distributional surgery module applied to phase-cell embeddings.
- **Matched controls:** Standard feature masking or attention without surgical intervention.
- **Falsification:** Apply mismatched/random distribution surgeries; must perform worse than causally aligned surgeries.
- **Kill rule:** Stop if distribution surgery yields no improvement over standard masking in robustness or explainability.

### Paper 14: Invariant-Mechanism Discovery with repStat

- **Object:** Invariant risk minimization (IRM) applied across all available real-world dataset domains (e.g., PTB-XL, EchoNext, LUDB, Emory, Sunnybrook) rather than relying on synthetic shifts.
- **Model:** IRM-regularized classification head over phase-cell KMEs.
- **Matched controls:** Empirical Risk Minimization (ERM) pooling across domains.
- **Falsification:** Shuffle domain assignments across the 9 available real-world datasets to destroy invariant structures; IRM advantage must disappear.
- **Kill rule:** Stop if IRM provides no out-of-domain generalization benefit over ERM.

### Paper 15: Causal Mechanism Factorization of the Cardiac Cycle

- **Object:** Factorization of the ECG generative process into independent causal mechanisms (e.g., depolarization vs. repolarization).
- **Model:** Independent-mechanism autoencoder with distributional disentanglement.
- **Matched controls:** Entangled representation learning (standard VAE or KME).
- **Falsification:** Force mechanism entanglement via random basis mixing; downstream causal queries must fail.
- **Kill rule:** Stop if the factorized components do not exhibit statistical independence or fail to align with known physiological boundaries.

## Training and Inference Plan

All preprocessing and train-fitted objects are materialized with provenance,
split membership, configuration hash, and source-data fingerprints. Each paper
fits its own scaler, whitening transform, kernel landmarks, prototypes, PCA,
and model weights from its own allowed training folds. No learned artifact is
read from a sibling paper directory.

Development runs use folds 1--7/8 and only one seed until the architecture's
invariant and falsification smoke tests pass. Hyperparameters are then frozen.
Final runs refit from scratch on folds 1--8, use fold 9 for stopping/selection,
and execute five seeds at full GPU throughput. Fold 10 is opened only when the
freeze manifest proves the configuration and comparison were predeclared.

## Failure Modes and Diagnostics

- **R-peak selection bias:** Report exclusion and beat counts by fold/class; run the frozen all-record fixed-window sensitivity unconditionally.
- **Nyström distortion:** Enforce exact-MMD rank/error gates before any representation run.
- **Phase surrogate:** Measure phase mutual information and use phase-permutation controls.
- **Capacity confounding:** Use identical downstream architectures for matched representations and include linear probes.
- **Patient leakage:** Assert patient disjointness and stop on any violation.
- **Amplitude erasure:** Retain physical mV and prohibit record/beat amplitude normalization.
- **Compute bloat:** Mechanism smoke, single-seed development, and explicit kill criteria precede full seeds.

## Novelty and Elegance Argument

The novelty is not the ingredients in isolation. It is the change in primitive
from waveform sample or named lead to a local empirical electrical process,
combined with matched mechanism-use tests that can reject each proposed object. The
shared trunk is deliberately small; the branches are separate papers rather
than stacked modules. Papers 4, 5, 7, and 8 make deliberately narrow claims
because DMD, Koopman ECG, flexible-lead models, and ECG tokenizers already
exist.

No foundation-model component is forced into the program. Paper 8 uses a small
masked-token transformer only because token prediction is the object under
test; larger language models would confound token quality with scale.

## Claim-Driven Validation Sketch

### Claim 1: Distribution-valued phase cells contain information beyond matched moments

- **Minimal experiment:** Paper 2 IMQ kernel means versus linear KME and mean/variance with the identical phase CNN.
- **Metric:** Fold-8 macro AUROC plus odd/even representation agreement.
- **Required evidence:** Nonlinear KME advantage or substantial stability gain without material predictive loss.

### Claim 2: Distant distributional recurrence adds information beyond mean recurrence

- **Minimal experiment:** Paper 1 direct MMD recurrence versus direct mean recurrence with identical classifier.
- **Metric:** Fold-8 macro AUROC and odd/even normalized Frobenius similarity.
- **Required evidence:** Apply the frozen task ROPE, stability threshold, and scalar phase-destruction statistic in `docs/CLAIM_GATE_MATRIX.md`.

### Portfolio branch claims

Papers 3--15 proceed only after their mechanism-destroying smoke tests behave as
specified. This gate does not remove any paper from the implementation scope;
it determines whether expensive final-seed runs are scientifically warranted.

## Experiment Handoff Inputs

- **Must-prove:** Kernel direction beyond moments; recurrence beyond means; each branch-specific mechanism is measurably used by its pipeline.
- **Must-run ablations:** Linear kernel/moments, phase shuffle, path reversal/order shuffle, Hankel time shuffle, Koopman chronology shuffle, conditional pairing shuffle, unseen operator tests, matched tokenizer controls.
- **Critical metrics:** Macro AUROC/AUPRC, calibration, paired patient bootstrap, representation stability, and branch-specific mechanism endpoints.
- **Highest-risk assumptions:** Reliable phase extraction; useful higher-order cell distributions with few beats; Nyström fidelity; Paper 8 vocabulary scalability.

## Compute and Timeline Estimate

- Shared preprocessing/QC and representation cache: CPU/I/O dominated, expected hours rather than GPU-days.
- Shared raw baselines plus Papers 1--2 development: approximately 8--20 A100 GPU-hours after cached representations.
- Papers 3--6 gated development: approximately 12--30 A100 GPU-hours total, excluding failed branches stopped at gates.
- Paper 7 and Paper 8: highest cost, provisionally 20--50 A100 GPU-hours each if they reach full five-seed execution.
- Papers 9--15 causal/interventional development: approximately 30--60 A100 GPU-hours total, contingent on passing causal mechanism gates.
- Final five-seed portfolio cost is not committed until measured throughput from the first development runs replaces these estimates.





## Binding Implementation Annexes

- `docs/MATHEMATICAL_CONTRACT.md`
- `docs/CLAIM_GATE_MATRIX.md`
- `docs/SPEC_RECONCILIATION.md`

The annexes are authoritative for estimator, eligibility, comparison, and gate details.

