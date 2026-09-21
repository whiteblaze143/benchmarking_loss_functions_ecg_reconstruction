# PRD: Operator-Conditioned Braid-Field ECG Representation Suite

## 1. Objective

Build and evaluate ECG encoders that preserve ordinary waveform/field morphology while adding dynamic topological summaries of an operator-conditioned canonical electrical field.

The core research question is whether dynamic field topology adds configuration-shift robustness beyond GraphECG, SetOperator, and a canonical-field-only baseline.

Core map:

    D_O = {(q_i, x_i)} -> Phi_hat(q,t) -> [z_field, z_topology] -> y

The implementation must support 1--8 observed independent leads without changing model parameters or relying on a fixed channel vocabulary.

## 2. Claims the suite is allowed to test

1. Field hypothesis: an operator-conditioned canonical field is more robust to lead deletion than a fixed-channel representation.
2. Braid residual hypothesis: braid/event descriptors add information beyond the canonical field itself.
3. Cross-operator invariance hypothesis: paired sparse views of the same ECG can be encouraged to yield consistent braid embeddings without collapse.
4. Uncertainty hypothesis: posterior variation in latent, field, and braid states can expose ambiguity under severe sparsity.

Do not describe a result as beating GraphECG unless task, patient split, target configuration, covariates, and statistical analysis are matched.

## 3. Model variants

| Variant | Field | Braid | Event/Birth-Death | Paired Invariance | Posterior |
|---|---:|---:|---:|---:|---:|
| field | yes | no | no | no | deterministic |
| field_braid | yes | yes | no | no | deterministic |
| field_braid_event | yes | yes | yes | no | deterministic |
| field_braid_inv | yes | yes | yes | yes | deterministic |
| field_braid_prob | yes | yes | yes | no | diagonal Gaussian |

All variants use the same continuous-operator context encoder and canonical-field decoder. This keeps the principal ablation field_braid* minus field interpretable.

### Differentiable braid surrogate

SoftBraidReadout tracks several soft maxima and minima over a supplied 2-D query chart. Pairwise relative trajectories yield smooth winding summaries. This is a differentiable braid surrogate for optimization, not proof of an Artin or surface-braid invariant. Exact critical-point and surface-braid analysis remains a post-hoc scientific audit.

### Birth/death surrogate

Track concentration is converted to soft occupancy and temporal occupancy changes yield soft birth/death summaries. This avoids imposing one fixed-strand braid group across creation and annihilation events.

## 4. Data contract

build_braid_field_dataset.py consumes:
- PTB-XL production phase caches for train and validation;
- the frozen physical scaler;
- Paper-07 response_fit.npz.

It emits train.npz and val.npz containing:
- context_operators: canonical Q8 operators;
- context_responses: frozen response features;
- query_operators and query_coords: canonical query bank;
- target_field: physical phase-mean voltage projections normalized by the frozen voltage scale;
- labels, ECG IDs, and patient IDs.

### Critical manifold limitation

The default RBF-interpolated 2-D query bank is only a diagnostic smoke manifold. It must not be presented as a validated torso surface or real wearable geometry. Before a topology paper, replace or independently validate it with dense Nef/PanoBench view geometry, body-surface potential coordinates, or another prespecified physiological manifold.

## 5. Training protocol

### Strict-zero-shot configuration arm

Use --training-regime full_only.

- Context is full canonical Q8 for field, field_braid, field_braid_event, and field_braid_prob.
- No lead-drop augmentation is used.
- Checkpoint selection uses PTB-XL validation only.
- Reduced configurations and unseen operators are never used for model selection.

field_braid_inv is the explicit paired-view training variant and therefore samples two observation subsets by design. It must be reported separately from strict full-input-only variants.

### Robustness-trained arm

Use --training-regime subset_aug.

- Sample observed Q8 subsets during training.
- Report separately from strict-zero-shot arms.
- Compare only against equally augmentation-aware baselines such as masked/random-lead ResNet and, if trained that way, GraphECG.

### Loss

    L = L_BCE + lambda_f L_field + lambda_b L_braid_inv
        + lambda_c L_anti_collapse + lambda_s L_separation + lambda_KL L_KL

Only terms appropriate to the selected variant are active. Hyperparameters must be selected on training-domain validation, never using LUDB, Zhejiang, ISP, Kingston, EchoNext, or held-out configuration outcomes.

## 6. Evaluation integration

### E1: Standard Q8 configuration shift

Use evaluate_braid_field_shift.py.

Required named configurations include full Q8, six precordials, independent I/II, I/II/V1, I/II/V5, Lead I, and Lead II.

Run --all-subsets to evaluate all 255 non-empty Q8 subsets.

Report absolute macro AUROC/AUPRC, delta from Q8, retention, mean by cardinality, worst case by cardinality, and between-configuration dispersion. Absolute performance is primary; retention alone is insufficient because a weak full-input model can show deceptively high or greater-than-100-percent retention.

### E2: Unseen in-span operators

Use evaluate_braid_field_ood.py.

- Fixed bank of 100 unit operators with seed 2026.
- Operators are absent from standard Q8.
- Synthesize the corresponding projection from the held-out 8-D basis.
- No parameter updates.

Label this unseen in-span measurement-operator generalization, not real wearable/device validation.

### E3: Task-native external datasets

Integrate braid models into the existing task-native evaluation registry after the local task-native queue/evaluator code is synchronized to GitHub.

For each external task:
1. train only the task-native head on the task's full/native training configuration;
2. freeze encoder and head;
3. evaluate reduced/novel configurations;
4. bootstrap by patient.

Kingston-ICU must use only configurations supported by its native telemetry channels. Do not fabricate unavailable Q8 or precordial conditions.

### E4: EchoNext / GraphECG matched comparison

When EchoNext is available, use the same train/validation/test patients, SHD labels, covariates or a matched no-covariate ablation, reduced-lead configurations, and novel ICM proxy. Use paired patient bootstrap and DeLong where appropriate.

This is the decisive GraphECG comparison.

## 7. Required baselines and ablations

Required baselines:
- GraphECG;
- full-lead fixed-channel 1D ResNet with zero-fill at test;
- the same ResNet plus an explicit missingness mask;
- the same ResNet with random lead masking during training;
- SetOperator full-lead-only;
- SetOperator subset-trained;
- field;
- field_braid;
- field_braid_event;
- field_braid_inv;
- field_braid_prob.

Mechanistic operator ablations:
- response set without q;
- categorical lead-ID embedding;
- continuous q;
- shuffled q assignment.

The main topology contrast is field_braid* versus field. Beating GraphECG without improving over field does not establish a topological contribution.

## 8. Statistical contract

Primary comparisons:
1. field_braid_inv minus field;
2. field_braid_inv minus GraphECG.

Use patient-level paired resampling. Do not treat multiple configurations or OOD operators from one patient as independent observations.

For each primary configuration report point estimate, paired 95% CI, absolute score, and change from full input. Predeclare multiplicity handling for any inferential family spanning many configurations.

## 9. Scientific gates

### Gate A: field validity
Dense field reconstruction must preserve held-out physical projection structure.

### Gate B: topological richness
The field must exhibit more than one persistent dynamic critical structure often enough for braid dynamics to be non-degenerate. If it is effectively dipolar, fail closed and stop the braid branch.

### Gate C: invariance without collapse
Same-record/different-operator braid distance should decrease while between-state discriminability is preserved.

### Gate D: incremental utility
At least one braid variant must improve over field under the prespecified metric. Otherwise topology has not added useful information.

### Gate E: configuration generalization
Any claimed advantage must survive severe 1--3 lead conditions and unseen in-span operators.

## 10. Repository layout

    src/repecg/braid_field/
      __init__.py
      geometry.py
      features.py
      losses.py
      model.py

    scripts/braid_field/
      build_braid_field_dataset.py
      train_braid_field.py
      evaluate_braid_field_shift.py
      evaluate_braid_field_ood.py

    tests/
      test_braid_field.py

## 11. Example commands

Build dataset:

    python scripts/braid_field/build_braid_field_dataset.py \
      --train-cache <train_phase_cache> \
      --validation-cache <val_phase_cache> \
      --scaler <physical_scaler.json> \
      --response-fit <paper07/response_fit.npz> \
      --output <braid_dataset>

Train a variant:

    python scripts/braid_field/train_braid_field.py \
      --dataset <braid_dataset> \
      --output <outputs/braid_field> \
      --variant field_braid_inv \
      --training-regime full_only

Evaluate all Q8 subsets:

    python scripts/braid_field/evaluate_braid_field_shift.py \
      --checkpoint <checkpoint.pt> \
      --dataset <braid_dataset> \
      --split val \
      --all-subsets \
      --output <shift.json>

Evaluate OOD in-span operators:

    python scripts/braid_field/evaluate_braid_field_ood.py \
      --checkpoint <checkpoint.pt> \
      --dataset <braid_dataset> \
      --phase-cache <val_phase_cache> \
      --scaler <physical_scaler.json> \
      --response-fit <paper07/response_fit.npz> \
      --output <ood.json>

## 12. Definition of done

- Unit tests pass on CPU.
- Every model variant trains and emits a reproducible checkpoint.
- All 255 Q8 subset evaluations run without architecture changes.
- 100 OOD in-span operator evaluations run from the same frozen checkpoint.
- Patient IDs are retained for paired bootstrap analysis.
- The task-native evaluator registers the five braid variants after its local code is synchronized.
- No external outcome is used to select topology thresholds, loss weights, or checkpoints.

## 13. Fail-closed interpretation

These implementations are candidate models, not evidence that braids improve ECG inference. If the field-richness gate fails, the braid suite should be reported as a negative result rather than rescued by additional architecture complexity. If braid variants do not beat field-only under matched training, the canonical field, not topology, is the supported mechanism.