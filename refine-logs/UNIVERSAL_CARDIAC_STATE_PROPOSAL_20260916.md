# Universal Cardiac State Field: refined research proposal

Date: 2026-09-16  
Status: design proposal only; no training authorized by this document

## 1. Problem anchor

The target is not another deterministic one-lead-to-twelve-lead reconstructor. It is a reusable encoder that estimates **what is knowable about a patient's cardiac electrical state from an arbitrary observed lead set**, represents uncertainty about what is not knowable, and supports both continuous-view ECG generation and clinically meaningful downstream inference.

The core object is therefore a posterior, not a point embedding:

\[
q_\phi(z_{\mathrm{card}},z_{\mathrm{amb}}\mid\{(x_l,g_l)\}_{l\in S}),
\]

where `z_card` is the cardiac state consistently supported by the observed leads and `z_amb` is the conditional ambiguity remaining after those observations. A continuous observation operator generates a signal at requested geometry `g_q`:

\[
\hat x_q=D_\theta(z_{\mathrm{card}},z_{\mathrm{amb}},g_q,a),
\]

with a separate acquisition variable `a` for device/filter/electrode effects. Clinical prediction heads may consume `z_card`, but not `a`.

The intended claim is bounded: **observation-set-universal across the ECG lead geometries, devices, populations, and labels represented by the admitted cohorts**. It is not a claim of universal human cardiac physiology.

## 2. Why Nef-Net v2 is not this model

Nef-Net v2 is a strong direct query-conditioned view transformer. Its authors deliberately avoid explicit field reconstruction to obtain efficient view-to-view synthesis. Their paper and reviews expose the remaining gap:

- direct deterministic transformation does not represent the distribution of valid unobserved cardiac states;
- PanoBench is spatially dense but small and predominantly young/healthy, so it cannot anchor broad pathology;
- device identity and dataset identity are entangled in the five-dataset design;
- static geometric attention does not express time-varying spatial activation;
- evaluation is dominated by pointwise fidelity, with limited disease/cardiologist validation;
- five-second per-patient calibration assumes sufficient temporal stability and is not a principled uncertainty estimate;
- the dipole motivation does not itself establish that complex pathological fields are identifiable from sparse leads.

The new work should preserve the useful continuous-query geometry while changing the scientific object from `source -> target waveform` to `arbitrary observations -> posterior cardiac state -> arbitrary observations and clinical phenotypes`.

## 3. What each available cohort is allowed to teach

| Cohort | Admitted role | Information it uniquely contributes | What it must not be used to claim |
|---|---|---|---|
| PanoBench train (3,440) | spatial anchor | 42 independent Wilson-terminal torso views plus I/II and CT-derived coordinates | broad disease coverage or a literal 48-to-standard-12 correspondence |
| PanoBench sealed test (1,030) | final spatial evaluation only | dense held-out view reconstruction | model selection |
| PTB-XL folds 1-8 | clinical/morphology development | 18,885 patients, expert SCP statements, standard leads, demographics, noise flags | dense torso geometry |
| PTB-XL fold 9 | validation/model selection | patient-separated development inference | final external performance |
| PTB-XL fold 10 | sealed final evaluation | standard-12 pathology and morphology retention | tuning |
| HEEDB | scale and diagnostic diversity, conditional on audit | metadata describes 1,061,598 ECGs from 349,548 patients plus GE 12SL labels/overreads | usable pretraining scale until waveform membership, patient split, licensing, and exact labels pass audit |
| EchoNext | privileged structural phenotype evaluation under the current artifact contract | 5,442 official test ECGs with echo-derived structural endpoints | pretraining; the locally retained waveform artifact is a holdout set |
| LUDB | temporal morphology supervision/evaluation | lead-specific P/QRS/T boundaries across 200 patients | population-scale clinical representation |
| RDB | arrhythmia-aware delineation | multi-arrhythmia, multi-lead wave boundaries | geometric field supervision |
| ISP/Zhejiang | secondary delineation only after provenance gates | additional interval/mask supervision | confirmatory evidence while subject semantics/provenance remain unresolved |
| Sunnybrook | exploratory clinical transfer | fully measured leads and machine morphometrics | population generalization; current sample is only 20 records |

No dataset is silently substituted when another is absent. Missing HEEDB waveforms, a failed license gate, or an unavailable training portion of EchoNext reduces scope and must be reported; it does not trigger synthetic data or holdout reuse.

## 4. Proposed model: Cardiac State Field (CSF)

### 4.1 Geometry-aware observation-set encoder

Each observed lead is tokenized jointly with:

- its waveform patches;
- lead definition (bipolar pair or unipolar electrode/reference), not merely a channel index;
- spatial coordinate where known;
- sampling/filter metadata where known;
- a missingness mask.

A permutation-equivariant set transformer accepts any number and ordering of leads. Geometry is continuous; standard lead names are metadata, not a fixed input width. PanoBench and standard-12 samples can therefore train the same encoder without inventing a one-to-one lead mapping.

The output is multiscale rather than one pooled vector:

- beat/phase tokens for rhythm and P-QRS-T dynamics;
- time-resolved field tokens for changing cardiac activation;
- a global state token for stable morphology and phenotype;
- an acquisition token kept explicitly separate from cardiac state.

### 4.2 Observable-ambiguous posterior factorization

For every complete record available in a cohort, create two views of the same patient-time window:

- a full-available teacher input `A` (all leads genuinely present in that dataset);
- a randomly masked observed subset `S`.

The teacher encodes the richest state that the record supports. The partial encoder must reproduce the teacher's **observable** subspace. A conditional flow learns the residual distribution needed to reproduce the full state:

\[
z_A = z_{\mathrm{obs}}(S) + r,\qquad r\sim p_\psi(r\mid z_{\mathrm{obs}},S).
\]

This split is essential. A deterministic embedding forces the mean of mutually plausible missing-lead morphologies; an unconstrained generator can hallucinate. CSF instead identifies what is stable across lead subsets and represents the remainder as calibrated uncertainty.

### 4.3 Continuous, time-varying field decoder

The decoder queries `D(z,g_q,t)` at an anatomical location and time. Unlike a shared static angular attention map, spatial mixing may change across P, QRS, ST, and T phases. Use a low-rank spatiotemporal basis plus a learned residual field rather than assuming a pure dipole:

\[
x(g,t)=\sum_{k=1}^{K} c_k(t)b_k(g)+\epsilon_\theta(z,g,t).
\]

`K` is selected on validation data and the residual is ablated. This admits higher-order spatial structure while retaining an interpretable low-rank cardiac-vector component.

For standard limb outputs, exact Einthoven/Goldberger relations are enforced by generating an independent lead basis and deriving dependent leads. Observed leads are copied exactly when the task is completion, so the model cannot improve a score by altering its inputs.

### 4.4 Clinical sufficiency without label leakage

Dataset-specific heads supervise only records with corresponding ground truth:

- PTB-XL: diagnostic, rhythm, form, and direct ECG measurement targets;
- HEEDB, if admitted: diagnostic labels, demographics, longitudinal/patient structure, and overread-aware targets;
- LUDB/RDB: P/QRS/T boundary and phase targets;
- paired echo development data only if a non-test training cohort becomes legitimately available: continuous and categorical structural phenotypes.

The heads are discarded after pretraining. Cross-cohort minibatches are patient-balanced and task-balanced; a million-record cohort must not erase the dense geometry or morphology anchors. Labels are never used as decoder inputs, so generated waveforms remain conditioned on observed ECG evidence rather than requested diagnoses.

The acquisition token predicts device/filter/noise metadata. A gradient-reversal audit may test whether `z_card` carries device identity, but invariance is not imposed blindly: some device and population variables are statistically inseparable without paired cross-device patients. The paper must report this identifiability boundary.

## 5. Training curriculum

### Stage D0: executable data/provenance gate

Before modeling, emit one immutable manifest with record ID, patient ID, cohort, split, lead definitions, units, sampling rate, labels present, geometry source, device metadata, and file hash.

Hard failures:

- patient overlap across train/validation/test;
- EchoNext test membership in training;
- PanoBench test membership in training or checkpoint selection;
- derived limb leads counted as independent supervision;
- HEEDB metadata rows without verified waveform files counted as samples;
- unknown units or inferred label semantics admitted silently.

### Stage D1: representation sanity study

Use PTB-XL folds 1-8 plus PanoBench train and the provenance-resolved delineation cohorts. Compare:

1. fixed 12-lead encoder;
2. arbitrary-lead set encoder;
3. set encoder plus continuous field decoder;
4. full CSF with observable-ambiguous posterior.

Do not begin million-record HEEDB pretraining until the arbitrary-lead encoder reconstructs held-out true views, preserves observed leads, and passes angle-permutation and patient-leakage negative controls.

### Stage D2: scale enrichment

If HEEDB passes D0, pretrain with patient-equal sampling. Repeated ECGs from a patient may support longitudinal consistency, but the unit of sampling and all confidence intervals remain the patient. Keep PanoBench batches explicitly oversampled as geometry anchors and LUDB/RDB batches as morphology anchors; report effective sampling weights.

### Stage D3: posterior completion

Train the residual flow after the deterministic state/decoder reaches its preregistered reconstruction and clinical-retention gates. Then optionally fine-tune end-to-end at a lower learning rate. The point estimate is the posterior mean; samples are used only for uncertainty-aware metrics and clinical scenario analysis.

## 6. Minimal decisive experiment before scale-up

The first paper-quality experiment is not a full foundation-model run. It is a falsifiable 4-arm study on the frozen PTB-XL/PanoBench development contract:

| Arm | Encoder | Decoder | Posterior |
|---|---|---|---|
| U0 | fixed-channel A0 | standard-12 | deterministic |
| U1 | geometry-aware arbitrary-lead set | standard-12 | deterministic |
| U2 | same | continuous time-varying field | deterministic |
| U3 | same | continuous time-varying field | observable + conditional residual flow |

All arms use the same patients, updates, source-subset schedule, optimizer family, and validation opportunities. U3 is allowed to consume randomness only through its posterior; no best-of-K reconstruction metric is primary.

Primary gates:

1. **Arbitrary-set utility:** U1 must outperform U0 under prespecified 1-, 2-, 3-, and 6-independent-lead masks without degrading full-12 clinical probing.
2. **Geometry utility:** U2 must improve true held-out PanoBench coordinate reconstruction over U1 and fail appropriately under permuted coordinates.
3. **Uncertainty utility:** U3 must improve proper scoring rules and empirical coverage over U2 without reducing point reconstruction or clinical-retention performance.
4. **State utility:** frozen `z_card` must improve patient-level linear/low-shot probes over reconstruction-only baselines on PTB-XL validation, while device/dataset predictability is disclosed.

If U1 fails, stop: the universal observation-set premise is unsupported. If U2 fails, do not claim a continuous cardiac field. If U3 fails calibration, retain a deterministic encoder and reject the probabilistic claim. Only a passing U3 proceeds to HEEDB scale-up.

## 7. Evaluation matrix

### Reconstruction and field fidelity

- patient-equal Pearson paired with MAE, RMSE, amplitude/variance retention, spectral and morphology errors;
- per-lead and lower-tail results, especially V1-V6;
- exact observed-lead identity and limb-law residuals;
- held-out PanoBench views and coordinate interpolation, with distance-stratified errors;
- phase-specific spatial error for P, QRS, ST, and T.

### Clinical state

- frozen linear and low-shot probes on PTB-XL fold 10 only after development is frozen;
- morphology/delineation on LUDB/RDB with beat-aware matching;
- EchoNext 5,442 holdout records once, using paired patient-level uncertainty and separating waveform contribution from unchanged tabular covariates;
- subgroup and pathology-stratified results, not only macro averages.

### Probabilistic validity

- CRPS and energy score;
- 50/80/95% interval coverage and width by lead, pathology, and input cardinality;
- calibration of clinically measured intervals/amplitudes;
- posterior diversity versus error (to detect both collapse and gratuitous variation);
- never use oracle/best-of-K as the primary result.

### Required negative controls

- shuffled lead geometry;
- shuffled clinical labels within cohort;
- device/dataset prediction from `z_card` versus `z_acq`;
- duplicated-patient and split-membership audit;
- remove PanoBench geometry anchor;
- remove clinical heads;
- remove morphology heads;
- static versus time-varying field;
- deterministic versus posterior completion;
- input lead identity/count stress tests.

## 8. Novelty and nearest-neighbor boundary

Arbitrary-lead encoders alone are no longer novel: K-MERL, TolerantECG, and LAEF already target missing/arbitrary standard leads. Large clinically supervised encoders are also occupied by ECGFounder and ECG-LFM. Conditional 12-lead generation and panoramic view synthesis are occupied by MCMA, clinical three-lead reconstruction, Nef-Net, and Nef-Net v2.

The defensible novelty is their conjunction around a different object:

1. a **posterior cardiac state** rather than a classification embedding or direct waveform mapping;
2. a **continuous time-varying observation field** grounded by dense true torso views;
3. an explicit **observable-versus-ambiguous factorization** for sparse-lead identifiability;
4. **asymmetric heterogeneous supervision**, where geometry, pathology, delineation, and structural phenotype datasets teach different state axes without pretending they are paired;
5. evaluation of both representation utility and calibrated generative completion under sealed clinical cohorts.

This is strong only if the posterior and field claims survive the negative controls. Merely combining more datasets or adding a flow to ECG-AIM is not sufficient novelty.

## 9. Major risks and decisions

1. **No cross-dataset patient pairing.** The shared state is learned through common ECG observations and task supervision, not direct cross-cohort latent matching. Dataset adversarial objectives cannot prove causal disentanglement.
2. **EchoNext governance.** Under the current artifact contract it remains test-only. Structural phenotype training requires a separately authorized non-test paired cohort and a new external holdout.
3. **HEEDB readiness.** Its metadata scale is not evidence that all raw waveforms are locally complete or usable. D0 decides admission.
4. **PanoBench population.** It anchors geometry, not clinical epidemiology. Geometry learned in young/healthy subjects must be stress-tested on standard-lead pathological cohorts and described as transfer.
5. **Identifiability.** One lead cannot uniquely determine twelve. The model must widen uncertainty when evidence is insufficient rather than emit falsely precise pathology.
6. **Compute.** Develop U0-U3 at modest scale first. No full HEEDB run until all three scientific gates pass.

## 10. Recommended next action

Do not restart C1/C2 as the principal experiment. Preserve their artifacts as a deterministic native-component pilot.

Next, implement only D0 and produce:

- `UNIVERSAL_DATA_LEDGER.parquet`;
- `UNIVERSAL_SPLIT_MANIFEST.json`;
- `UNIVERSAL_COHORT_ADMISSION.md` with explicit eligible/ineligible reasons;
- an exact U0-U3 experiment contract with frozen masks, seeds, metrics, and stopping gates.

Only after D0 passes should code for the geometry-aware set encoder begin.

## 11. Internal adversarial review

**Strongest objection:** this can look like a kitchen-sink combination of set transformers, neural fields, flows, and multitask supervision.

**Response:** the architecture must remain two scientific modules: (1) an arbitrary-observation cardiac-state field autoencoder, and (2) a conditional residual posterior. Dataset-specific heads are training measurements, not architectural contributions. The staged U0-U3 study makes each claim independently falsifiable.

**Strongest clinical objection:** a plausible sampled ECG may contain a diagnostic pattern not supported by the input.

**Response:** clinical outputs are distributions, observed leads are immutable, proper scores replace best-of-K, and the interface must distinguish invariant findings from sample-dependent findings. High posterior disagreement becomes an abstention signal, not a diagnosis.

**Strongest novelty objection:** lead-agnostic ECG foundation models already exist.

**Response:** do not sell lead agnosticism. Sell inference of a calibrated, queryable cardiac-state posterior whose spatial observation model is grounded by true dense-body measurements and whose latent sufficiency is tested across morphology, diagnosis, and structural phenotype.

