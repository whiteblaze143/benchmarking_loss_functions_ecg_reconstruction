# Paper 13 Inductive-Bias Audit

**Protocol:** `P13-SURGERY-ELIGIBILITY-v1` (IMMUTABLE — failure verdict locked)  
**Pivot branch:** `P13-REPLACEMENT-SENSITIVITY-V2`  
**Status:** current architecture fails causal surgery eligibility; production training remains blocked.

---

## Scientific object

The implemented object is conditional phase-feature **replacement sensitivity**,
not anatomical equivariance and not an identified Pearl intervention. The
maximum defensible claim from the present operator is:

> A trained classifier's prediction changes when one Phase-KME cell is
> replaced by a specified reference feature.

This is a model-behaviour statement. It is not evidence that the phase cell
causes disease or that the waveform structure is necessary or sufficient.

---

## Decisive findings (corrected)

**1.** `full`, `no_propagation`, and `wrong_phase_surgery` use the same forward
   algorithm during BCE training. The surgery operator is never trained or
   evaluated by the grid.

**2.** `wrong_phase_surgery` is metadata only and has no executable effect on
   the tensor passed to the classifier.

**3.** `no_structural_propagation` is a 90/10 blend of two classifier outputs. It
   is not a no-propagation structural control. The 90/10 blend has no principled
   interpretation and should be deleted.

**4.** Circular convolution followed by global phase averaging makes the
   diagnostic forward pass invariant to cyclic phase rotation.
   Consequently, **absolute phase identity is unavailable to the classifier**.
   This does not mean that a local replacement-sensitivity field cannot exist.
   Replacing cell $Z_k \neq Z_k^{\text{id}}$ can still change the prediction
   because the information around that cell changes:
   $$f(X^{k \leftarrow r}) - f(X) \neq 0.$$
   The precise conclusion is: absolute phase address is unusable, but a
   **rotation-equivariant sensitivity field** is coherent:
   $$S_{k+s}(R_s X) = S_k(X).$$

**5.** Replacing a feature cell does not regenerate downstream phase states.
   The feature-only replacement captures the direct path $X_1 \to Y$ but
   leaves the mediated path $X_1 \to X_2 \to X_3 \to Y$ stale.
   The counterfactual recovery error
   $E\bigl[|\hat{Y}_{\text{replace}} - Y_{do}|\bigr]$
   substantially exceeds the irreducible propagated baseline
   (World A, `run_inductive_bias_gates.py`).

**6.** Conditioning the proposed reference on `Y=NORM` requires care.
   Two cases must be distinguished:
   - **Training-set NORM reference bank (deployable):** A reference bank
     constructed from training-fold NORM labels and applied identically to
     every evaluation patient does not require knowing the evaluation label.
     It is still supervised, class-conditional, not an identified patient
     counterfactual, and not suitable for a Pearl causal interpretation — but
     it is *not* an evaluation-time oracle.
   - **Evaluation-label oracle (not deployable):** If the target patient's
     evaluation label determines which reference is selected at inference time,
     that is oracle-label leakage and must be reported as such.
   Any deployable reference sampler must be trained exclusively on folds 1–6
   and applied without access to fold-7 or hold-out labels.

---

## V1 gate summary (immutable failure record)

| Gate | Question | V1 result |
|---|---|---|
| G0 objective execution | Do named controls execute different algorithms? | **FAIL** |
| G1 structural recovery | Does replacement recover a known interventional effect? | **FAIL** — feature-only effect nonzero but wrong (stale mediated path) |
| G2 false causality | Does a confounded proxy receive zero causal score? | **FAIL** in principle for prediction sensitivity |
| G3 support | Are reference replacements conditionally on-support? | **UNTESTED** |
| G4 phase identity | Can the model address the absolute operated phase? | **FAIL** — invariant classifier |
| G5 cyclic boundary | Is a forward intervention order identified? | **FAIL** — cyclic representation, no SCM cut-point |
| G6 real-data admissibility | Are replacements realizable as valid ECGs? | **BLOCKED** |

Status: **`P13-CAUSAL-SURGERY = INELIGIBLE`**

---

## V2 gate design (P13-REPLACEMENT-SENSITIVITY-V2)

The pivot estimand is:
$$S_k(X) = E_{\tilde{Z}_k}\bigl[d\bigl(f(X),\, f(X^{k \leftarrow \tilde{Z}_k})\bigr)\bigr],$$
where $\tilde{Z}_k \sim q_\phi(Z_k \mid C_k(X))$ is sampled from a
**train-only** context-conditional reference and no causal claim is asserted.

| Gate | Question | Target |
|---|---|---|
| **G0** operator execution | Do every named replacement and control actually execute a different operator on the tensor? | All variants alter the tensor; no metadata-only controls |
| **G1** identity | Does replacing $Z_k$ by itself give exactly zero sensitivity? | $S_k^{\text{id}}(X) = 0$ to machine precision |
| **G2** equivariance | Does rotating the ECG rotate the sensitivity field without changing its values? | $S_{k+s}(R_s X) = S_k(X)$ verified numerically |
| **G3** conditional support | Are generated / retrieved replacements on-support relative to train data? | Report $d_{\text{support}}$, donor reuse, context distance, replacement magnitude, coverage |
| **G4** phase specificity | Does target-phase replacement exceed magnitude-matched wrong-phase replacement where synthetic relevance is known? | Decisive contrast on synthetic ground-truth world |
| **G5** predictive-not-causal falsification | Does the confounded-proxy world demonstrate that $S_k > 0$ does not require a structural effect? | **Claim boundary**, not a failure: $S_k \neq 0 \not\Rightarrow \text{causal effect}$ |
| **G6** real-data stability | Are sensitivity maps reproducible across seeds / reference draws / patients? | Confidence intervals over folds and donor samples |

---

## Required controls for the V2 grid

Every control must alter the tensor passed to the classifier:

| Control | Operator |
|---|---|
| Identity / no-op | $Z_k \leftarrow Z_k$ |
| Conditional same-phase replacement | $Z_k \leftarrow \tilde{Z}_k \sim q_\phi(Z_k \mid C_k(X))$ |
| Unconditional replacement | $Z_k \leftarrow \tilde{Z}_k \sim p_{\text{train}}(Z_k)$ |
| Magnitude-matched wrong-phase replacement | $Z_k \leftarrow \alpha \cdot Z_j / \|Z_j\|$ where $\alpha = \|Z_k\|$, $j \neq k$ |
| Off-support replacement | $Z_k \leftarrow \tilde{Z}_k$ from a distribution disjoint from training support |
| Deterministic different-patient donor | $Z_k \leftarrow Z_k^{(m)}$ for patient $m \neq n$, bijective assignment |

The 90/10 `no_structural_propagation` blend is deleted.

---

## Reference sampler specification

- Trained exclusively on folds 1–6. Never uses evaluation labels.
- Preferred form: $q_\phi(Z_k \mid Z_{k-w:k-1}, Z_{k+1:k+w})$ — small
  masked conditional network, or train-only nearest-neighbour donor rule.
- Per-replacement diagnostics: $d_{\text{support}}$, donor reuse,
  context distance, replacement magnitude, coverage.
- If using empirical donors: near-bijective deterministic assignment required
  (Paper 12 established why unrestricted donor reuse distorts controls).

---

## Required redesign before real-data testing

- Rename the estimand to *phase-replacement sensitivity* in all code, paper
  text, and tracker entries.
- Delete or isolate all current "surgery" / "counterfactual" language unless
  an explicit SCM is learned and independently validated on synthetic worlds
  first.
- Supply phase positional encoding if absolute phase identity is required for
  a future claim. Do not mix that model with the current invariant classifier.
- Report training-set NORM reference bank results separately from any analysis
  that uses evaluation labels to select donors.
- Validate G5 (confounded proxy = claim boundary) explicitly in paper text
  as a limitation, not as an anomaly to suppress.

The existing smoke artifacts are execution artifacts only and must not be cited
as evidence for the Paper 13 thesis.

---

## Tracker update

| Branch | Status |
|---|---|
| `P13-CAUSAL-SURGERY` | **INELIGIBLE** (immutable, all G0–G6 fail or are blocked) |
| `P13-REPLACEMENT-SENSITIVITY-V2` | **OPEN** — V2 gate design above; no real-data runs yet |
