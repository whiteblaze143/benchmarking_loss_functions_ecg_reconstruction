# LVCG vs. Current ECG-AIM Latent Representation

Date: 2026-09-10

## Bottom line

The current ECG-AIM encoder produces a useful **rhythm-sensitive representation**, but the available evidence does not support calling it a general-purpose ECG embedding. It is weak on at least one morphology-duration probe, and the current "3D-theta" mechanism is target-lead conditioning rather than a recovered three-dimensional cardiac field.

LVCG has a better-defined representation interface and separates beat structure, temporal dynamics, and rhythm. Its physical VCG construction, however, cannot be copied directly into the strict one-lead-to-twelve-lead setting: one observed lead supplies a rank-one measurement of a three-dimensional field, so the inverse problem is not identifiable without a learned prior or additional observed leads.

## What each model actually represents

| Property | Current ECG-AIM / 3D-theta | LVCG |
|---|---|---|
| Input used by core setup | One observed ECG lead | Three visible leads during pretraining; all leads for released embedding inference |
| Initial latent | Temporal patches, `[B, 200, 768]` | Deterministic three-channel VCG time series |
| Spatial mechanism | A target-lead code modulates copies of the source latent | Fixed lead-direction matrix lifts ECG to VCG and projects VCG back to leads |
| Record embedding | Ad hoc mean/std/max pooling of encoder activations, 2304 dimensions in the existing study | Explicit 640-D concatenation: 256 structural + 256 dynamic + 128 rhythm |
| Beat structure | Implicit in transformer patches; optional delineation loss | Explicit R-peak beat segmentation and CNN beat tokens |
| Rhythm | Implicit | Explicit RR-interval embedding plus sequential state model |
| Reconstruction constraint | Learned shared decoder; optional Kors-VCG loss | Fixed geometric projection followed by learned refinement |
| Physical interpretation | Target codes are metadata/identifiers, not a latent VCG | Approximate physical VCG, subject to the fixed lead-vector model |

## Evidence from our completed studies

The sample-level checkpoint embedding study is complete for 11 models. For the two primary models, a linear probe on the pooled encoder latent achieved AF/AFIB test AUROC of 0.760 and 0.787 (N=360), compared with 0.713 from three simple waveform-summary features. For the stronger primary model, the latent improvement over waveform summaries was +0.0738 AUROC (95% bootstrap CI +0.0157 to +0.1345, p=0.011).

That result demonstrates useful rhythm information, but it is not sufficient evidence of a general embedding. On annotation-derived QRS-onset-to-T-offset duration, latent-only regression was worse than waveform summaries: MAE was higher by 13.3 ms for the stronger primary model and 15.7 ms for the other (both bootstrap p<0.001). Adding the latent to waveform summaries did not materially improve that duration endpoint.

The representational-geometry audit also undercuts a literal physical interpretation of the theta model:

- True theta, permuted theta, learned IDs, and random fixed codes all reorganize toward similar downstream functional lead geometry.
- The best physical alignment is strongest at the conditioner and is partly overwritten by the decoder.
- Therefore theta primarily breaks symmetry between output slots; reconstruction supervision learns the usable lead relationships.

Important audit caveat: `REPRESENTATIONAL_GEOMETRY_REPORT.md` contains narrative numbers that conflict with its generated CSV (for example, the prose calls the "Goldilocks" effective-rank range 40–58 while the listed best-model ranks are mostly much smaller). Treat the CSV/database as evidence and the prose report as an unverified interpretation until regenerated and audited.

## What LVCG improves

1. It exposes a stable downstream API (`ext_ecg_emb`) instead of requiring an arbitrary hook and pooling rule.
2. It creates separate morphology, beat-dynamics, and RR-rhythm components, making failures measurable and permitting component ablations.
3. It uses a low-dimensional bottleneck before reconstruction, which makes "representation learning" a clearer claim than a large decoder-conditioned feature grid.
4. It supplies a fixed geometric reconstruction path that can be evaluated separately from the learned refinement decoder.

## LVCG limitations relevant to us

1. **Strict one-lead identifiability:** recovering three VCG coordinates from one scalar lead is rank deficient. LVCG's default three-visible-lead training avoids this; our task does not.
2. **No released weights in the cloned repository:** the repository contains training and probing code but no checkpoint, so a numerical head-to-head comparison cannot yet be run from this checkout.
3. **Domain/protocol mismatch:** LVCG pretrains on MIMIC ECG at 100 Hz with z-score normalization; our principal setup uses PTB-XL at 500 Hz and retains physical millivolts. Scores are not directly comparable without the same records, preprocessing, and probe.
4. **Idealized geometry:** fixed lead vectors do not model patient anatomy or electrode placement variation.
5. **Implementation mismatch:** `VCGPseudoInverse` is described as SVD-based but currently computes an explicit regularized inverse of `U^T U`; conditioning should be tested.
6. **Structural summary:** the released GRU baseline uses the first complete beat as its structural embedding, which may be brittle for ectopy or noisy early beats.
7. **Learned bypass:** the refinement decoder can repair or bypass errors from the fixed physical projection. Geometric-only and refined reconstructions must be reported separately.

## Recommended hybrid for strict 1-to-12 reconstruction

Do not replace ECG-AIM with LVCG wholesale. Test a hybrid representation:

- Retain the one-lead temporal encoder.
- Add an explicit beat-token branch and RR embedding.
- Produce three named record components: morphology, dynamics, and rhythm.
- Replace spherical theta with either fixed lead projection constraints or the empirically derived functional lead graph already supported by our audit.
- Keep hard limb-lead algebra as a deterministic output projection.
- If using a three-channel latent field, infer it with a learned prior conditioned on the single lead and label it a **learned latent cardiac field**, not an identifiable VCG.

## Fair comparison required before changing the main model

Run frozen linear probes on identical patient-disjoint PTB-XL splits and identical input availability:

1. Current encoder pooling (mean/std/max).
2. Current encoder plus explicit beat/RR heads.
3. LVCG with all 12 leads (upper-bound representation comparison).
4. LVCG with the same single observed lead (stress test, expected to expose rank deficiency).
5. Hybrid learned-field model with one lead.

Report AF/AFIB, superclass and diagnostic multilabel AUROC/AUPRC, morphology-duration error, lead-view invariance, missing-lead reconstruction, and cross-dataset transfer. Match embedding dimension or include a dimension-matched projection, and use the same preprocessing and probe hyperparameter selection for every model.

## Evidence locations

- Current model: `unified_latents/engineering/models/three_d_theta_reconstruction.py`
- Current VCG regularizer: `scripts/common_loss.py` (`KorsVCGLoss`)
- Current sample-level study: `results/checkpoint_embeddings/compact.sqlite`
- Current geometry audit: `results/representational_geometry/model_geometry_summary.csv`
- LVCG model: `external/LVCG/lvcg/models/lvcg.py`
- LVCG geometry: `external/LVCG/lvcg/models/vcg.py`
- LVCG probing wrapper: `external/LVCG/probing/encoders/lvcg_encoder.py`

