# SetOperator: end-to-end implementation note

## Purpose and scope

This note explains the implemented Paper-07 `OperatorSetModel` and its four
training variants:

1. continuous primary;
2. continuous auxiliary;
3. categorical primary; and
4. categorical auxiliary.

It separates **implemented facts** from **the intended inductive-bias claim**.
In particular, giving a model a continuous operator coordinate makes an unseen
measurement representable; it does not itself establish that the model will
generalize accurately to that measurement.

The latent-space figures use the continuous primary, continuous auxiliary, and
full-lead-only continuous-primary frozen checkpoints. The categorical variants
are architecture-matched controls used to test the measurement-geometry part
of the proposal.

## 1. Measurement model

Let `x(t)` denote an unobserved eight-coordinate cardiac measurement basis.
An observed channel is defined by a normalized linear operator `q`:

\[
  y_q(t) = \langle q, x(t)\rangle,
  \qquad q\in\mathbb{R}^8,\quad \lVert q\rVert_2=1.
\]

The eight canonical coordinates are:

\[
  [\mathrm{I},\ \mathrm{II},\ \mathrm{V1},\ \mathrm{V2},\ \mathrm{V3},\ \mathrm{V4},\ \mathrm{V5},\ \mathrm{V6}].
\]

A canonical lead uses a one-hot operator. For example:

\[
q_{\mathrm{I}}=[1,0,0,0,0,0,0,0],\qquad
q_{\mathrm{V3}}=[0,0,0,0,1,0,0,0].
\]

A derived measurement is another normalized coefficient vector. For example,
the oblique difference used by the shift battery is:

\[
q_{\mathrm{V3-V2}}
= \frac{q_{\mathrm{V3}}-q_{\mathrm{V2}}}{\sqrt{2}},
\qquad
y_{\mathrm{V3-V2}}(t)
= \frac{y_{\mathrm{V3}}(t)-y_{\mathrm{V2}}(t)}{\sqrt{2}}.
\]

This is a modelling convention, not a claim that the eight coordinates fully
recover the physical cardiac field. It defines the operator space available to
the encoder.

## 2. The record-level input is a set, not a fixed tensor

For a record with `m` observed measurements, the model receives:

\[
  \mathcal S=\{(q_j,r_j)\}_{j=1}^{m}.
\]

`q_j` says **what was measured**. `r_j` says **what response was observed
through that measurement**. The input order is not supposed to encode clinical
meaning. A reordering of all pairs should leave the final patient latent
unchanged.

The set size is variable. In the shared-grid training routine, a batch-wide
context size is sampled uniformly from 1 through 6, and a separate candidate
operator is held out as target. Thus the training objective can use a short
context even though each training record provides eight candidate pairs.

## 3. How each response `r_j` is built

The response construction happens before the neural model.

1. A physical beat tensor has shape `(beats, 256, 8)`.
2. The chosen operator is normalized to unit Euclidean norm.
3. The corresponding waveform is projected:

   \[
   y_q = xq/\text{voltage-scale}.
   \]

4. A temporal derivative is computed:

   \[
   \dot y_q = \nabla_t y_q.
   \]

5. Each time point becomes a two-dimensional response atom:

   \[
   a_q(t)=[y_q(t),\dot y_q(t)].
   \]

6. The 256 samples are divided into 16 phase cells of 16 atoms each.
7. Atoms are transformed with frozen whitening parameters, then a frozen
   Nyström feature map with 128 landmarks.
8. Features are averaged within each phase cell.

Therefore the response supplied to the model has shape:

\[
 r_j\in\mathbb{R}^{16\times128}.
\]

The 16 rows preserve coarse phase order. They should not be described as
clinically annotated P, QRS, and T segments: the implementation partitions a
resampled beat uniformly in time.

## 4. Shared neural backbone

All four variants use the same response branch, set backbone, pooling path,
latent width, diagnosis head, and decoder shape. This is what makes the
continuous-versus-categorical contrast meaningful.

### 4.1 Response encoder

Each `16 x 128` response is passed through `PhaseCNN` with width 128 and two
blocks:

\[
  r_j\in\mathbb R^{16\times128}
  \longmapsto e_{r,j}\in\mathbb R^{128}.
\]

This branch receives waveform-derived information only. It has no direct lead
name or operator coordinate.

### 4.2 Operator encoder

Each variant produces a 64-dimensional operator embedding:

\[
  e_{q,j}\in\mathbb R^{64}.
\]

How it produces this vector is the only structural difference between the
continuous and categorical controls.

### 4.3 Channel token

The two embeddings are concatenated:

\[
 h_j=[e_{q,j};e_{r,j}]\in\mathbb R^{192}.
\]

For `m` observed channels, this creates a token matrix:

\[
 H\in\mathbb R^{m\times192}.
\]

### 4.4 Set interaction

`H` passes through a two-layer Transformer encoder with:

- model width 192;
- four attention heads; and
- feed-forward width 384.

The Transformer makes each channel token contextual: its representation can
depend on the other observed `(operator, response)` pairs in the same record.
Masks are supported for padded variable-cardinality batches.

### 4.5 Patient pooling and diagnostic head

A learned 192-dimensional pooling query attends over the encoded set. Its
output passes through a linear layer and GELU:

\[
 \tilde H\longrightarrow u\in\mathbb R^{192}
 \longrightarrow z=\operatorname{GELU}(W_zu+b_z)\in\mathbb R^{256}.
\]

`z` is the patient-level latent visualized by the latent-space figures.

The diagnostic head maps `z` to five logits:

\[
 \ell=W_{\mathrm{diag}}z+b_{\mathrm{diag}}\in\mathbb R^5,
 \qquad \hat p=\sigma(\ell).
\]

The labels are the five PTB-XL superdiagnostic labels. The head is multilabel,
not a single mutually exclusive diagnosis classifier.

## 5. Continuous variants

### 5.1 Continuous operator interface

The continuous model passes the actual operator vector through an MLP:

\[
q_j\in\mathbb R^8
\rightarrow \operatorname{Linear}(8,64)
\rightarrow \operatorname{GELU}
\rightarrow \operatorname{Linear}(64,64)
\rightarrow e_{q,j}.
\]

The continuous model is not analytically linear in `q`; the MLP learns a
nonlinear embedding of the coordinates. "Continuous" means the input domain
is a numeric vector space rather than a finite list of names.

That matters because a novel normalized `q` has a defined input:

\[
q_\star\in\mathbb R^8\quad\Rightarrow\quad\phi_q(q_\star)\in\mathbb R^{64}.
\]

It does **not** mean that nearby `q` values are guaranteed to have nearby
embeddings, or that interpolation is guaranteed. Those properties must be
tested on an independently frozen unseen-operator bank.

### 5.2 Continuous primary

Continuous primary uses the shared backbone and diagnosis loss only:

\[
 \mathcal L_{\mathrm{continuous-primary}}
 =\mathcal L_{\mathrm{BCE}}.
\]

The implementation uses prevalence-weighted binary cross entropy; positive
weights are formed from the training-label prevalence and capped at 10.

It is optimized to solve the five-label diagnosis task from a randomly chosen
context set. It is not directly required to predict a withheld response.

### 5.3 Continuous auxiliary

Continuous auxiliary adds a target-operator decoder. After forming the latent
from the context set, it embeds a distinct held-out target operator:

\[
[z;\phi_q(q_\star)]\in\mathbb R^{320}.
\]

The decoder is:

\[
\mathbb R^{320}
\rightarrow \operatorname{Linear}(320,256)
\rightarrow \operatorname{GELU}
\rightarrow \operatorname{Linear}(256,16\cdot128)
\rightarrow \hat r_\star\in\mathbb R^{16\times128}.
\]

The target is the phase-resolved RKHS/Nyström response, not raw ECG voltage.
The reconstruction component is mean squared error:

\[
 \mathcal L_{\mathrm{response}}
 =\operatorname{MSE}(\hat r_\star,r_\star).
\]

The implementation also creates explicit negative-orientation responses by
computing the features for `-q`. It evaluates the same record through:

\[
(q,r_q)\quad\text{and}\quad(-q,r_{-q}).
\]

It averages reconstruction loss over the two orientations and adds symmetric
Bernoulli KL between their diagnostic probabilities:

\[
\mathcal L_{\mathrm{orient}}
=\tfrac12\left[
D_{\mathrm{KL}}(p\Vert p^-)+D_{\mathrm{KL}}(p^-\Vert p)
\right].
\]

The implemented loss weights are:

\[
 \mathcal L_{\mathrm{continuous-aux}}
 =\mathcal L_{\mathrm{BCE}}
 +0.1\mathcal L_{\mathrm{response}}
 +0.05\mathcal L_{\mathrm{orient}}.
\]

The intended effect is to discourage a latent that retains only information
needed by the five diagnostic logits. It asks the latent to support a second
query: "what response would this record produce through `q_star`?"

## 6. Categorical variants

### 6.1 Categorical operator interface

The categorical model does not pass numeric `q` to the model. It maps each
operator row to an integer ID using exact byte equality against a vocabulary
constructed from the training operators:

\[
q_j\longrightarrow \operatorname{ID}(q_j)\longrightarrow
E[\operatorname{ID}(q_j)]\in\mathbb R^{64}.
\]

The vocabulary is not just eight named clinical leads. The training
construction gives each record four randomly selected canonical operators and
four record-specific sparse random operators. Consequently, the categorical
vocabulary contains exact operator vectors seen during training, including
many sampled sparse vectors.

This is important: categorical is an exact-operator lookup control, not merely
a conventional `I/II/V1/...` lead-name lookup.

### 6.2 The unseen-operator fallback

If an operator is absent from the exact training vocabulary, its ID is `-1`.
The model then uses `unknown_operator`, a persistent 64-dimensional all-zero
buffer:

\[
q_\star\notin\mathcal V
\quad\Rightarrow\quad
e_{q,\star}=0\in\mathbb R^{64}.
\]

It still receives the corresponding waveform-response feature `r_star`; it
does not lose the waveform. What it loses is explicit information about which
operator generated that waveform.

This makes categorical a deliberately stringent control for unseen operators:
it can test whether the response branch plus generic set aggregation is enough
when the measurement geometry is unavailable.

### 6.3 Categorical primary

Categorical primary changes only the operator interface:

\[
 \mathcal L_{\mathrm{categorical-primary}}=\mathcal L_{\mathrm{BCE}}.
\]

It shares the response encoder, Transformer, pooling, latent dimension,
diagnostic head, optimizer family, seed handling, and search grid with the
continuous primary control.

### 6.4 Categorical auxiliary

Categorical auxiliary adds the same target-response MSE objective:

\[
 \mathcal L_{\mathrm{categorical-aux}}
 =\mathcal L_{\mathrm{BCE}}+0.1\mathcal L_{\mathrm{response}}.
\]

It does **not** receive the continuous model's explicit `F(-q)` responses and
does **not** receive the 0.05 orientation-consistency term. Therefore the
comparison of continuous auxiliary against categorical auxiliary jointly tests
continuous operator coordinates plus the explicit orientation mechanism; it
does not isolate only the MLP-versus-embedding-table choice.

That is a design limitation to state explicitly in a paper. A stricter
categorical control would need an equally specified orientation mechanism.

## 7. What each comparison can support

| Contrast | Directly tests | Does not establish alone |
|---|---|---|
| Continuous primary vs categorical primary | Numeric operator input versus exact lookup input under diagnosis-only learning | Generalization to unseen operators without a frozen test |
| Continuous auxiliary vs continuous primary | Added response reconstruction and orientation consistency | Whether reconstruction, orientation, or both caused a difference |
| Continuous auxiliary vs categorical auxiliary | Full implemented continuous-aux mechanism versus implemented categorical-aux mechanism | Geometry-only benefit, because orientation supervision differs |
| Subset-trained continuous primary vs full-lead-only continuous primary | Benefit of variable context exposure | Universal robustness under every deployment measurement |
| SetOperator vs GraphECG | Two different variable-measurement inductive biases | A pure causal architecture effect unless training/evaluation contracts are matched |

## 8. What the latent visualizations mean

The figure scripts extract `z` from `encode_context`, before the diagnosis head
and before the optional decoder. PCA is then fit separately within each model.

The plots can describe within-model structure, label overlap, and variance
concentration. They cannot support claims based on:

- absolute latent coordinates across different encoders;
- apparent visual separation alone;
- a PCA axis being a named clinical factor; or
- a latent plot being a substitute for a pre-specified quantitative test.

The numerical Fold-8 configuration-shift heatmap is a separate performance
artifact. It should be interpreted only under its frozen evaluation contract.

## 9. Compact end-to-end pseudocode

```text
for record in dataset:
    candidate_operators = record-specific canonical and sparse q vectors
    context, target = choose distinct context q's and target q_star

    for q in context:
        y_q = normalized_projection(physical_beats, q)
        atoms = [y_q, temporal_derivative(y_q)]
        r_q = frozen_whitening_and_nystrom_phase_means(atoms)

        response_embedding = PhaseCNN(r_q)                  # 128 dims
        operator_embedding = MLP(q) or embedding_table[id(q)] # 64 dims
        token = concatenate(operator_embedding, response_embedding)

    z = learned_attention_pool(two_layer_set_transformer(tokens)) # 256 dims
    logits = diagnostic_head(z)                                  # 5 dims
    loss = weighted_multilabel_BCE(logits, labels)

    if auxiliary:
        target_response = response_for(q_star)
        predicted_response = decoder(concatenate(z, operator_embedding(q_star)))
        loss += 0.1 * MSE(predicted_response, target_response)
        if continuous:
            loss += 0.05 * symmetric_KL(predictions(q), predictions(-q))
```

## 10. Source pointers

- Model definition: `src/repecg/paper07_operator/model.py`
- Measurement/operator construction: `src/repecg/paper07_operator/measurements.py`
- Response-set build and frozen preprocessing contract:
  `scripts/paper07/build_paper07_representations.py`
- Shared continuous/categorical training and objective definitions:
  `scripts/paper07/train_paper07_shared_grid.py`
- Full-lead-only ablation training:
  `scripts/paper07/train_paper07_fulllead_only.py`
