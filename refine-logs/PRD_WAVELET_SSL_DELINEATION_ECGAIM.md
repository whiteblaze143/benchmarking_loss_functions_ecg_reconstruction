# PRD — Wavelet-SSL + Delineation ECG-AIM Smoke Program

**Date:** 2026-08-22  
**Last operational update:** 2026-08-23 America/Toronto
**Task:** exactly one observed ECG lead → 11 reconstructed leads + dense delineation  
**Development plan:** broad 3-epoch screen → selected 5-epoch runs from scratch → selected 10-epoch confirmation from scratch  
**Deployment heads:** exactly two: (1) waveform reconstruction, (2) delineation  
**Active reconstruction objective:** `1110000`, selected after the completed
1-lead spatial controls showed that the stronger objective dominated the
architecture-only MSE gains. The earlier `1000000` manifest is retained as a
superseded, unlaunched design artifact.

---

## 1. Scientific question

Current 1-lead results indicate that geometry helps most when the reconstruction objective is simple. On Lead I with pure MSE, the current spatial sequence improved validation missing-lead Pearson approximately:

`A0 0.6700 → B1 0.6729 → C1 0.6800 → D1 0.6882 → E1 0.6914`

whereas under `1010010`, A0–E1 were nearly flat. The next architecture question should therefore target **temporal morphology representation**, not simply add more geometry.

The CinC 2026 abstract “Self-Supervised Learning for ECG Representation Using Physiology-Informed Wavelet Transforms” motivates:
- two time-frequency ECG views,
- BYOL,
- TimeSformer encoding,
- downstream T-wave delineation.

The proposed ECG-AIM extension asks:

> Does local/global physiology-informed wavelet SSL improve missing-lead P/QRS/T morphology and delineation while preserving reconstruction, and is it complementary to source-target lead geometry?

### Claim hierarchy for the active `1110000` study

The principal architectural comparison is:

`A0 -> A0 + standard-Morlet wavelet branch -> A0 + the same wavelet branch + SSL`.

E1 is **not** an assumed improved base architecture. The current `1110000`
evidence does not support that interpretation. E1 is reserved for the matched
complementarity contrast:

`A0 + wavelet + SSL` versus `E1 + the same wavelet + SSL`.

The principal second-view mechanism contrast is:

`Morlet magnitude + Morlet phase` versus
`Morlet magnitude + Potse/Stoks-derived UEG-repolarization phase`.

These cells must match TimeSformer, fusion, BYOL mode and weight, optimizer,
seed, observed lead, epoch budget, and reconstruction objective. Only view B's
bank may differ. A favorable UEG result supports the limited claim that the
physiology of the second SSL view matters under this controlled screen.

RDB supplies lead-specific inclusive `START/END` annotations. The cache maps
these released boundaries into six supervised channels: P-on/P-off,
QRS-on/QRS-off, and T-on/T-off. The head contains no peak channels because the
release contains no explicit peak column. Therefore
T-onset and T-offset localization are real-GT endpoints; T-peak is unavailable
and must not be inferred or claimed as RDB ground truth.

---

## 2. Critical reproducibility boundary

The CinC abstract reports T-start/T-peak/T-end sensitivities of approximately 99.96%, 99.96%, and 99.92% within a stated AAMI tolerance window.

**Do not claim replication yet.**

The abstract does not provide enough information to reconstruct:
- the exact custom ECG-inspired mother-wavelet equation,
- exact scale/frequency choices,
- exact CWT normalization,
- exact TimeSformer configuration,
- exact fiducial-matching rule,
- full AAMI tolerance implementation.

The supplied code therefore contains:
- a deterministic analytic Morlet-like engineering bank for controlled smokes,
- an explicit `custom_asset` hook,
- a hard failure if an exact custom-wavelet asset is requested but absent.

Until the exact asset is obtained, label these experiments **abstract-inspired**, not exact reproduction.

---

## 3. Existing implementation freeze

Current base:
`unified_latents/engineering/experimental/aim_1_lead.py`

Current trainer behavior to preserve:
- PTB-XL 500 Hz
- 5000 samples / 10 s
- patch size 25 samples = 50 ms
- encoder depth 8
- decoder depth 4
- current external `CombinatorialCompositeLoss`
- AdamW
- initial LR `1e-4`
- betas `(0.9, 0.95)`
- weight decay `1e-4`
- OneCycle max LR `5e-4`
- `pct_start=0.2`
- gradient clip 1.0
- missing-lead validation Pearson from zero-centered flattened missing-lead waveforms

### Important parity discrepancy

`AliTokECGAIM` displays class defaults `width=384, heads=8`, but the active builder defaults are `encoder_width=768, encoder_heads=12`, and the current trainer does not override them.

Therefore the current effective ECG-AIM training path is **width 768 / heads 12**.

The new scripts intentionally default to 768/12. Do not silently switch to 384/8.

### Existing artificial-mask issue

Current grid mask-type embedding uses `inherited` rather than `inherited | artificial`.

Do not silently fix this during the wavelet experiment.

The new model exposes:
- `mask_type_mode=legacy`
- `mask_type_mode=all_masked`

Treat the correction as its own named ablation.

---

## 4. New files

Place:

1. `wavelet_ssl_ecg_aim.py` at  
   `unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py`

2. `train_1lead_wavelet_ssl_mtl.py` at  
   `scripts/train_1lead_wavelet_ssl_mtl.py`

3. this PRD at  
   `refine-logs/PRD_WAVELET_SSL_DELINEATION_ECGAIM.md`

Do not edit the existing ECG-AIM or factorial trainer in place.

---

## 5. Architecture

```text
                            observed source ECG
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
          raw ECG-AIM patches                    wavelet/scalogram views
                 │                              /                     \
                 ▼                             ▼                       ▼
        masked ECG-AIM encoder            view A                  view B
                 │                              \                     /
                 │                               ▼                   ▼
                 │                         TimeSformer-like encoder
                 │                                  │
                 └──────── gated/cross-attention fusion ─────────────┘
                                                    │
                                                    ▼
                                          source morphology tokens
                                                    │
                                        + existing lead geometry
                                                    │
                                                    ▼
                                          12 × 200 axial grid
                                          /                 \
                                         ▼                   ▼
                              reconstruction head      delineation head
                                  [12,5000]             [12,4,5000]
                                                         + optional
                                                      6 boundary maps
```

Geometry answers **where the electrical view is measured**.

Wavelet SSL answers **what temporal morphology / cardiac phase is present**.

Raw ECG-AIM retains responsibility for **voltage fidelity**.

---

## 6. Wavelet branch

### Engineering Morlet bank

Initial smoke factors:
- scales: 16 / 32 / 48
- frequency range: nominal 0.5–45 Hz
- cycles: 4 / 6 / 8
- complex coefficients

Views:
- magnitude
- log magnitude
- phase
- `sin(phase)`
- `cos(phase)`
- real
- imaginary

The 5000-sample transform is block-averaged inside the exact ECG-AIM 25-sample patch boundaries to generate 200 aligned temporal positions.

### Exact custom ECG-inspired wavelet

BLOCKED until a documented asset is available.

Required metadata:
- source / paper
- formula or released code
- sampling convention
- normalization
- scale/frequency settings
- license
- SHA256

Never fabricate it to keep the queue moving.

---

## 7. Wavelet encoder

Primary:
- factorized TimeSformer-like block
- temporal attention within each frequency
- frequency attention within each time index
- pool frequency only
- retain 200 temporal embeddings

Default:
- dim 192
- depth 2
- heads 6
- project to ECG-AIM width 768

Controls:
- dim 96
- depth 4
- convolutional scalogram encoder

If the convolutional control matches TimeSformer, cut TimeSformer complexity.

---

## 8. SSL objective

Candidate modes:
- `none`
- `global`
- `local`
- `both`

Global BYOL aligns record/source summaries.

Local BYOL aligns corresponding 50-ms temporal embeddings:

`hA(t) ↔ hB(t)`

This is the most relevant extension for delineation because it directly encourages the two physiological views to agree about local morphology.

Initial SSL weights:
- 0.01
- 0.05
- 0.10

EMA:
- 0.99
- 0.996
- 0.999

Important control:
- wavelet representation with SSL off

This separates **time-frequency features** from **BYOL learning**.

---

## 9. Leakage prevention — hard gate

The existing AIM model artificially hides observed temporal patches.

A wavelet branch could accidentally bypass that masking if it receives the complete source waveform and feeds its features into reconstruction.

Required behavior:

### Reconstruction-fusion path
Before wavelet computation, zero all temporal samples belonging to:

`inherited OR artificial`

masked patches.

### SSL-only path
The separate SSL objective may use the complete genuinely observed source lead.

Those full-source SSL features must **not** be routed into the reconstruction grid.

Required dedicated test:
1. force an artificial mask,
2. alter only the waveform samples under the hidden patch,
3. recompute fusion-source wavelet features,
4. features must remain identical within numerical tolerance.

Failure blocks all GPU sweeps.

---

## 10. Delineation branch

Exactly one delineation task branch.

Do not simply interpolate 50-ms logits.

Instead:
1. each 768-d patch feature projects to `hidden × 25`,
2. reshape to 25 distinct sample-level feature vectors,
3. run compact Conv1d refinement,
4. classify each sample.

Semantic output:
- background
- P
- QRS
- T

Supervised boundary channels in the same branch:
- P onset
- P offset
- QRS onset
- QRS offset
- T onset
- T offset

This is still conceptually one delineation branch, not a third interval-regression task.

---

## 11. Delineation cache contract

Each `.pt` record:

```text
waveform             float [12,5000]
segmentation          long  [12,5000]
seg_valid             bool  [12,5000] optional
fiducial_heatmaps     float [12,6,5000] optional
fiducial_valid        bool  [12,6] optional
annotation_weight     scalar optional
annotation_type       metadata
record_id             metadata
patient_id            metadata
source_dataset        metadata
```

Segmentation:
- 0 background
- 1 P
- 2 QRS
- 3 T
- -1 invalid

No silent resampling is allowed inside the trainer.

Generate 500-Hz cache labels from original annotation timestamps upstream.

---

## 12. Lead-specific versus integrated labels

Do not treat all datasets as equivalent dense 12-lead truth.

Maintain:
- `annotation_type`
- validity mask
- explicit `annotation_weight`

If integrated annotations are duplicated across leads, document that they are integrated timing supervision, not true independent lead-specific delineation.

Use lead-specific expert labels as the highest-quality endpoint where possible.

---

## 13. Dataset leakage rule

If LUDB is currently the frozen external morphology evaluation set, do not train on those same patients/records and then report LUDB as external validation.

Use:
- non-overlapping patient-level training/test splits, or
- other delineation datasets for training and LUDB as external test.

Patient IDs must be disjoint across train/validation/test.

---

## 14. Joint training objective

Initial:

`L = λR Lrecon + λSSL LBYOL + λCE LCE + λD LDice + λB Lboundary + λF Lfid`

Broad screen:
- reconstruction mask: `1110000`
- reconstruction weight: 1
- CE: 1
- Dice: 0 or 0.5
- boundary: 0 or 0.1
- fiducial: 0 or 0.1 only when actual fiducial labels exist
- SSL: 0 / 0.01 / 0.05 / 0.10

Missing-lead delineation is the primary target; observed-lead segmentation can be excluded with `seg_missing_only`.

A delineation improvement with reconstruction non-inferiority is a success. Pearson itself does not have to improve.

---

# 15. Verification gates

## G0 — syntax

```bash
python -m py_compile \
  unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py \
  scripts/train_1lead_wavelet_ssl_mtl.py
```

## G1 — synthetic model self-test

```bash
python unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py \
  --self-test
```

Required:
- waveform output shape correct
- segmentation shape correct
- fiducial shape correct
- finite SSL
- gradient reaches online wavelet encoder
- EMA target receives no gradient
- EMA actually updates
- deterministic fixed wavelet bank

## G2 — delineation cache audit

```bash
python scripts/train_1lead_wavelet_ssl_mtl.py \
  --audit-delineation-dir data/rdb_wavelet_delineation_cache/train \
  --audit-output refine-logs/delineation_train_audit.json
```

Repeat for validation.

Required:
- no nonfinite waveforms
- only legal class IDs
- expected source-dataset counts
- patient IDs available where possible
- file SHA256 recorded

Separately assert zero patient overlap.

## G3 — one-batch real-data GPU smokes

Run `--quick-verify` for at least:
1. A0 raw
2. E1 raw
3. wavelet no SSL
4. local SSL
5. global+local SSL
6. cross-attention fusion
7. conv wavelet encoder
8. 48 scales
9. boundary loss
10. fiducial loss if labels exist

PASS:
- no OOM
- no NaN/Inf
- gradients finite
- validation metrics produced
- no malformed output/checkpoint

## G4 — old/new A0 parity

Before trusting the sweep:
- same seed
- same weights
- same width/head count
- same lead mask
- eval mode

Compare first forward pass between old A0 and new wavelet-disabled A0.

Any unexplained discrepancy blocks the queue.

## G5 — wavelet leakage

Run the artificial-mask invariance test described above.

---

# 16. Canonical 120-model three-epoch screen

The active screen is a frozen, hash-bound manifest rather than an approximate
smoke design. It contains exactly:

- 60 curated method cells,
- two observed-source conditions (`--observed-leads 0` and `1`),
- reconstruction loss mask `1110000`,
- seed 42,
- three epochs from scratch,
- 120 total jobs.

The two source conditions are separate jobs, not independent training seeds.
Any aggregate analysis must retain observed source as a paired/blocking factor.

Canonical manifest:
`refine-logs/wavelet_ssl_1110000/full/manifest.json`

Canonical identities at launch:

```text
manifest SHA256             008a7d9b49a8d912cc9bdd132a93c493128f17a9dcb944d2984e40112466aec6
model source SHA256         8df118fb2bd004d4ae37ee51aa921a8f97d6dc949eae481b2c13d6d0fc5f034e
trainer source SHA256       6a2221fd2a3487d8cbc5181150e25e41e56b3b8e7ee779a4a7b799bb8d4ac7b0
PTB-XL content SHA256       87adfbc339de7faf1fdff96b0728b3ad3a4d484808b8fa123d92c8b6114f0541
RDB delineation SHA256      f37f48fb68f0210233c9c4f4efef4305e9930e8102f803799d71afffea1e594f
```

Regenerate only when intentionally creating a new protocol identity:

```bash
python scripts/train_1lead_wavelet_ssl_mtl.py \
  --data-dir data/ptb_xl/tensors \
  --data-manifest refine-logs/ptbxl_tensor_content_manifest.json \
  --delineation-dir data/rdb_wavelet_delineation_cache \
  --factorial-mask 1110000 \
  --batch-size 32 \
  --delineation-batch-size 32 \
  --num-workers 4 \
  --checkpoint-policy none \
  --rolling-resume \
  --require-cuda \
  --emit-sweep-manifest refine-logs/wavelet_ssl_1110000/full/manifest.json \
  --sweep-output-root refine-logs/wavelet_ssl_1110000/full/runs \
  --sweep-epochs 3 \
  --sweep-leads 0 1 \
  --sweep-masks 1110000 \
  --seed 42
```

The guarded supervisor, rather than the trainer's simple manifest runner, is
the canonical launcher:

```bash
tmux new-session -d -s wavelet_ssl_6boundary \
  "cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction && \
   python scripts/run_wavelet_ssl_after_spatial.py --poll-seconds 10 \
   >> refine-logs/wavelet_ssl_1110000/supervisor_6boundary.log 2>&1"
```

The canonical live state is SQLite-backed at
`refine-logs/wavelet_ssl_1110000/full/queue.sqlite`. A CSV summary is optional,
derived output and must not replace the database as queue state.

---

# 17. Canonical screen dimensions

The included screen covers:

- A0 and E1 geometry anchors
- time-frequency fusion without SSL
- magnitude vs log magnitude
- phase vs `sin(phase)`
- real-component alternative
- global/local/both BYOL
- gated-add vs cross-attention fusion
- 16/32/48 scales
- different Morlet cycle counts
- TimeSformer-like vs convolutional encoder
- SSL weights
- EMA coefficients
- CE vs CE+Dice vs boundary loss
- delineation-head width/kernel
- optional fiducial loss
- E1 + SSL complementarity
- legacy vs corrected mask-type semantics

The queue now includes a hashed Potse/Stoks-derived UEG asset. Its primary
controlled use is the matched R0-versus-R1 second-view contrast; broader Wyatt
and representation variants remain secondary ablations.

## 17.1 Exact 60-cell inventory

Each cell below is instantiated once for observed-source condition 0 and once
for condition 1, producing the 120 jobs. Names are stable analysis keys.

### Anchors and wavelet-without-SSL controls

1. `A0_raw`
2. `E1_raw`
3. `A0_wave_noSSL_gated_add`
4. `A0_wave_noSSL_cross_attn`
5. `E1_wave_noSSL_gated_add`
6. `E1_wave_noSSL_cross_attn`

### Principal claim and physiology-view cells

7. `P1_A0_morlet_mag_phase_noSSL`
8. `R0_morlet_mag_morlet_phase`
9. `R1_morlet_mag_ueg_phase`
10. `C1_E1_morlet_mag_morlet_phase`
11. `R2_morlet_mag_ueg_phase_sin`
12. `R3_morlet_logmag_ueg_phase_sin`
13. `R7_morlet_mag_ueg_real`
14. `R8_morlet_mag_ueg_mag`
15. `R9_ueg_mag_ueg_phase`
16. `R4_morlet_mag_ueg_phase_wyatt`
17. `R5_morlet_mag_ueg_phase_wyatt`
18. `R6_morlet_mag_ueg_phase_wyatt`

The primary matched sequence is `A0_raw` ->
`P1_A0_morlet_mag_phase_noSSL` -> `R0_morlet_mag_morlet_phase`.
The physiology test is R0 versus R1. The complementarity test is R0 versus
C1. R2-R9 and Wyatt variants are secondary mechanism/representation screens
and must not replace the primary contrasts post hoc.

### SSL view, scope and fusion screen

19. `ssl_magnitude_phase_local_gated_add`
20. `ssl_magnitude_phase_local_cross_attn`
21. `ssl_magnitude_phase_both_gated_add`
22. `ssl_magnitude_phase_both_cross_attn`
23. `ssl_magnitude_phase_sin_local_gated_add`
24. `ssl_magnitude_phase_sin_local_cross_attn`
25. `ssl_magnitude_phase_sin_both_gated_add`
26. `ssl_magnitude_phase_sin_both_cross_attn`
27. `ssl_log_magnitude_phase_sin_local_gated_add`
28. `ssl_log_magnitude_phase_sin_local_cross_attn`
29. `ssl_log_magnitude_phase_sin_both_gated_add`
30. `ssl_log_magnitude_phase_sin_both_cross_attn`
31. `ssl_log_magnitude_real_local_gated_add`
32. `ssl_log_magnitude_real_local_cross_attn`
33. `ssl_log_magnitude_real_both_gated_add`
34. `ssl_log_magnitude_real_both_cross_attn`
35. `ssl_global`
36. `ssl_both_infer_b`
37. `ssl_both_infer_mean`

### Capacity and encoder controls

38. `tf_sc16_cy4`
39. `tf_sc16_cy8`
40. `tf_sc32_cy4`
41. `tf_sc48_cy6`
42. `tf_small96`
43. `tf_deep4`
44. `conv_control`

### SSL weight and EMA controls

45. `ssl_w0.01`
46. `ssl_w0.1`
47. `tau0.99`
48. `tau0.999`

### Delineation controls

49. `del_ce`
50. `del_boundary`
51. `del_wave_ce`
52. `del_wave_boundary`
53. `del_head64k9`
54. `del_head128k25`
55. `del_fid`

All 120 jobs produce waveform reconstruction and four-class dense semantic
segmentation. The 118 non-`del_fid` jobs disable the direct fiducial head and
derive boundaries from semantic transitions if needed. The two `del_fid`
jobs additionally train the six-channel RDB boundary-heatmap head with
`fiducial_weight=0.1`.

### Geometry-complementarity and mask-semantics controls

56. `E1_ssl_local`
57. `E1_ssl_both`
58. `E1_ssl_cross`
59. `mask_fixed_raw`
60. `mask_fixed_wave`

## 17.2 Preflight and operational contract

Before the full queue, a 15-cell real-data GPU preflight covers A0, E1,
wavelet/no-SSL, local/both SSL, R0/R1/C1, global SSL, 48 scales, convolution,
boundary loss and the six-boundary `del_fid` path. The canonical preflight
completed 15/15 with manifest SHA256
`213cd173d607642a67ce078bb896d2b572d9df043f61a639c309c548a33e0b21`.
The `del_fid` preflight produced finite `train_fid=0.2311684291`.

Queue safety requirements:

- batch size 32 for reconstruction and delineation,
- four data-loader workers,
- one GPU job at a time,
- minimum 8 GiB free disk and 5 GiB available RAM,
- wait for GPU quiescence before launch,
- abort the active phase after bounded consecutive/total failures,
- hard fail if uncorrected GPU ECC increases,
- record but do not hard fail solely on corrected ECC increments,
- rolling `resume.pt` only for interrupted/active jobs,
- delete rolling resume state after validated success,
- retain compact SQLite queue state rather than per-step CSV exports,
- never resume a manifest after model, trainer or dataset identity changes.

The three-architecture queue remains blocked by a hash-bound barrier until the
wavelet queue reaches validated completion. Graceful stop retains the barrier
and resumable state.

## 17.3 Launch-state snapshot

At the 2026-08-23 operational update:

- tmux session `wavelet_ssl_6boundary` was alive,
- all 15 preflight cells had completed successfully,
- the full queue had 120 jobs,
- one job was complete, one was running and 118 were pending,
- the supervisor phase was `running_full_sweep`,
- no OOM, NaN or queue failure had occurred,
- uncorrected ECC remained at the launch baseline,
- peak preflight allocation was below the enforced 36-GiB safety limit.

This subsection is a dated audit snapshot, not a live status API. Query
`full/queue.sqlite` and `supervisor_state.json` for current state.

---

# 18. Three-epoch promotion rule

Do not rank by one number.

A configuration is eligible for 5 epochs only if:
1. no crash / OOM / NaN,
2. validation missing Pearson is no worse than A0 by more than ~0.01,
3. it improves at least one meaningful delineation endpoint:
   - wave-class mIoU
   - boundary F1 smoke
   - QRS IoU
   - T IoU,
4. it is not obviously a one-source artifact unless that is the intended hypothesis.

Promote roughly 10–12 diverse configurations.

Include:
- best raw control
- best wavelet/no-SSL control
- best local SSL
- best both SSL
- best E1+SSL
- best simple encoder
- best delineation-heavy model
- best reconstruction-preserving model

---

# 19. Five-epoch stage

Train selected models **from scratch** for five epochs.

Do not continue the three-epoch checkpoints because OneCycle scheduling and total optimization path would differ.

Use the same seed 42 and full train/validation data.

Examine:
- whether early smoke gains persist
- learning-curve separation
- reconstruction/delineation conflict
- Lead I vs Lead II consistency

Promote 3–4 models.

---

# 20. Ten-epoch stage

Train from scratch:
- A0
- E1
- best wavelet SSL without E1
- best E1 + wavelet SSL

Only after the architecture is frozen should the winner be tested with the stronger existing reconstruction objective.

This separates:
- architecture benefit
- loss benefit
- interaction between them

---

# 21. More abstract-faithful transfer experiment

After architecture selection:

### Stage A — unlabeled pretraining
PTB-XL:
- reconstruction on
- wavelet SSL on
- delineation loss off

### Stage B — delineation-head adaptation
Load Stage A:
- freeze shared network
- reconstruction weight 0
- SSL weight 0
- train delineation head only

Use:

```bash
--train-head-only --reconstruction-weight 0 --ssl-weight 0
```

### Stage C — full joint fine-tuning
Load Stage B:
- unfreeze
- joint reconstruction + delineation

This should be compared with the same two-head model trained from scratch.

---

# 22. Final delineation metrics

Smoke transition sensitivity is not sufficient.

Final evaluation must report:
- P/QRS/T IoU
- sensitivity
- PPV
- F1
- onset/offset timing errors
- RDB: P-on/P-off, QRS-on/QRS-off and T-on/T-off only
- LUDB or another peak-annotated dataset: P onset/peak/offset,
  QRS onset/R/QRS offset and T onset/peak/offset
- PR duration MAE
- QRS duration MAE
- QT MAE

Do not infer RDB peaks from interval centers, extrema or another delineator.
RDB duration endpoints may use its released START/END boundaries, but any
peak-based endpoint requires a separately documented peak-annotated dataset.

For the 99% comparison:
- recover exact tolerance windows
- recover exact matching
- report PPV, not sensitivity only
- freeze same split
- verify whether their model uses pre-cropped T-wave segments

---

# 23. Label-efficiency experiment

After architecture freeze, repeat delineation fine-tuning with:
- 100%
- 50%
- 20%
- 10%

of labeled **training patients**.

Compare:
- scratch
- existing masked reconstruction pretraining
- wavelet SSL pretraining

This is the cleanest test of the SSL contribution suggested by the CinC abstract.

---

# 24. Full clinical reconstruction evaluation

For finalists, retain the existing reconstruction panel:
- p05/p01 missing-lead Pearson
- p95 MSE
- event amplitude error
- QRS RMSE
- QRS area error
- T RMSE
- T area error
- ST-J error

For ST claims, add per-lead J/ST60/ST80 threshold analysis.

For QT/QRS claims, use actual interval-measurement error.

---

# 25. Statistics

Final comparisons:
- patient-clustered paired bootstrap
- 10,000 replicates
- preserve all records/leads/beats for each sampled patient
- confidence intervals on paired model differences

Do not treat beat × lead observations as independent.

---

# 26. Interpretation rules

### Wavelet no-SSL > A0
Time-frequency representation itself is useful.

### SSL > wavelet no-SSL
Evidence supports representation learning beyond extra input features.

### Local > global
Supports cardiac-phase/local morphology alignment.

### E1+SSL > E1 and SSL
Supports complementary spatial and temporal inductive biases.

### Conv ≈ TimeSformer
Cut the TimeSformer complexity.

### mask-type correction wins broadly
Separate the bug fix from the method contribution and rerun key baselines.

### delineation improves but waveform collapses
Reject as a reconstruction model even if sensitivity is high.

---

# 27. Execution checklist and current disposition

Completed before launch:

1. New model and trainer placed at their canonical paths.
2. Syntax and focused unit tests passed.
3. CPU model self-test passed, including shapes, finite SSL, online gradients,
   no target gradients, EMA update, A0 parity and artificial-mask leakage.
4. Effective parent width/heads confirmed as 768/12.
5. Patient-disjoint RDB train/validation/test assignment retained.
6. All 2,398 RDB cache tensors converted and audited under the six-boundary
   contract; all removed historical peak slots were asserted zero and invalid.
7. Canonical cache SHA256 bound into the full manifest.
8. Full manifest counted and inspected: 60 unique cells x two sources = 120.
9. Fifteen-cell GPU preflight passed, including `del_fid`.
10. Guarded supervisor launched in tmux after spatial completion.

Required after the 120-job screen:

11. Query the SQLite queue and validate every success artifact; do not infer
    completion from tmux absence alone.
12. Build the comparison table around the prespecified P1/R0/R1/C1 contrasts.
13. Treat source leads as paired blocks and report both source-specific values.
14. Select about 10–12 Pareto-diverse cells for fresh five-epoch training.
15. Select three or four models for fresh ten-epoch confirmation.
16. Freeze the architecture before held-out test evaluation.
17. Run label-efficiency and clinical reconstruction panels only after freeze.

---

## Primary paper-quality claim to test

> Physiology-informed time-frequency self-supervision improves temporal morphology in sparse-lead ECG reconstruction, yielding better missing-lead delineation without materially degrading waveform fidelity, and provides information complementary to source-target lead geometry.

A high tolerance-window sensitivity alone does not establish that claim.
