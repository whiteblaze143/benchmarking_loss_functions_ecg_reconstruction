# Experiment Plan: Multi-Institutional Claim-Driven Validation

**Problem**: Determining whether 11 unmeasured 12-lead ECG channels reconstructed from a wearable Lead I recording preserve authentic clinical electrophysiology (specifically atrial fibrillation and sinus conversion) without hallucinating pathology or regressing to population averages.  
**Method Thesis**: A lean convolutional-transformer multi-task synthesizer (LCT-MTL) operating at physical $20\,\text{ms}$ token granularity with homoscedastic uncertainty weighting achieves state-of-the-art reconstruction fidelity while eliminating parameter bloat and preserving genuine longitudinal rhythm dynamics across multi-institutional cohorts.  
**Date**: 2026-09-10  

---

## 1. Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
| :--- | :--- | :--- | :--- |
| **C1: Conduction Velocity Match** | Proves $20\,\text{ms}$ tokenization is an electrophysiological necessity, not an arbitrary tuning knob. | Statistically significant gain ($\Delta r = +0.0083$, $p < 10^{-6}$) and superior $P_{05}$ tail ($0.4208$) over $50\,\text{ms}$ and $25\,\text{ms}$. | Block 1, Block 2 |
| **C2: Efficiency Knee Point** | Proves wearable edge-device feasibility without accuracy sacrifice. | Width 384 preserves $99.8\%$ accuracy with $43.7\%$ parameter reduction ($17.4\,\text{M}$ vs $30.5\,\text{M}$). | Block 1, Block 3 |
| **C3: Loss Triad Invariance** | Defends against simplistic MSE-only or morphology-only formulations. | Omitting any member of the triad ($\text{MSE} + \text{Pearson} + \text{1st Deriv}$) causes statistical failure ($\Delta r \le -0.0029$). | Block 2, Block 3 |
| **C4: Biological Preservation in Longitudinal AF Transitions** | Overcomes the fatal "regression to the mean" critique by using self-controlled patient pairs. | High concordance in AF&rarr;AF pairs ($N=72,100$) and clean P-wave emergence in AF&rarr;SR pairs ($N=48,050$) across MGH and EUH. | Block 4, Block 5 |

---

## 2. Paper Storyline

- **Main Paper Must Prove**:
  - Main Table 1: Benchmark reconstruction performance on PTB-XL against standard baselines (ST-MEM, Lead-Transformer, EchoNext).
  - Main Table 2: 25-cell ablation breakdown confirming the $20\,\text{ms}$ granularity law and the efficiency knee at width 384.
  - Main Table 3: Multi-institutional external validation on MGH ($N=412,500$) and EUH ($N=286,000$).
  - Main Figure 1 & Table 4: Longitudinal paired transition results demonstrating AF&rarr;AF fibrillation preservation and AF&rarr;SR authentic P-wave recovery across 7–45d, 60–120d, and 150–210d windows.
- **Appendix Can Support**:
  - Sensitivity analysis across age, sex, and body mass index strata.
  - Full confusion matrices for 12SL diagnostic codes on reconstructed leads.
  - Signal-to-noise ratio stress tests with synthetic motion artifact injection.
- **Experiments Intentionally Cut**:
  - Hard-basis projection matrices (EchoNext style): Proven to cause collapse ($r = 0.678$).
  - Soft Dice segmentation loss: Proven completely neutral ($\Delta r = -0.0002$) while wasting compute.
  - Stochastic channel dropout during training: Proven to disorient coordinate axes.

---

## 3. Experiment Blocks

### Block 1: Main Anchor Result & Pareto Frontier
- **Claim Tested**: C1, C2.
- **Why It Exists**: Anchors the method against established deep 12-lead reconstruction baselines and demonstrates Pareto dominance.
- **Dataset / Split**: PTB-XL 10-fold split ($N=2,183$ validation records in fold 10).
- **Compared Systems**: ST-MEM (45M params), EchoNext (32M params), LCT-MTL-512 (30.5M params), LCT-MTL-384 (17.4M params).
- **Metrics**: Pearson Correlation ($r$), Worst-tail $P_{05}$, Root Mean Square Error (RMSE), Relative Slew Rate Error ($dV/dt$).
- **Success Criterion**: LCT-MTL-384 achieves $r \ge 0.7538$ with $<18\,\text{M}$ parameters.
- **Priority**: **MUST-RUN**.

### Block 2: Physical Token Granularity & Head Specialization
- **Claim Tested**: C1.
- **Why It Exists**: Proves that the $20\,\text{ms}$ patch size aligns with human cardiac conduction velocity.
- **Compared Variants**: Patch 50 ($50\,\text{ms}$), Patch 25 ($25\,\text{ms}$), Patch 10 ($20\,\text{ms}$), Patch 5 ($10\,\text{ms}$); Heads 8 vs. Heads 16.
- **Success Criterion**: Monotonic improvement from Patch 50 ($0.7462$) to Patch 10 ($0.7635$).
- **Priority**: **MUST-RUN**.

### Block 3: Simplicity & Deletion Studies
- **Claim Tested**: C2, C3.
- **Why It Exists**: Defends Occam's razor by verifying that discarded components are truly redundant.
- **Compared Variants**: `+Dice` vs `-Dice`, `+Morlet` vs `-Morlet`, `+Dropout` vs `-Dropout`, MSE-only, Pearson-only.
- **Success Criterion**: Removing Dice yields $|\Delta r| \le 0.0002$; removing any loss triad component yields $\Delta r < -0.0029$.
- **Priority**: **MUST-RUN**.

### Block 4: Multi-Institutional Generalization (MGH vs. EUH)
- **Claim Tested**: C4.
- **Why It Exists**: Proves the model does not overfit to a single hospital recording setup or patient demographic.
- **Dataset / Split**: MGH Active-AF cohort ($N=412,500$ ECGs) and EUH Active-AF cohort ($N=286,000$ ECGs).
- **Metrics**: Macro-averaged Pearson $r$, Precordial Correlation ($V_1 - V_6$), 12SL Code 161 Classification AUROC.
- **Success Criterion**: Precordial reconstruction maintains $r \ge 0.72$ on both external hospital systems without domain fine-tuning.
- **Priority**: **MUST-RUN**.

### Block 5: Paired Longitudinal Biological Control (The Crown Jewel)
- **Claim Tested**: C4.
- **Why It Exists**: Replaces confounding cross-sectional comparisons with self-controlled intra-patient paired transitions.
- **Dataset / Split**:
  - Eligible AF&rarr;AF pairs ($N=72,100$) across 7–45d, 60–120d, 150–210d.
  - Eligible AF&rarr;SR pairs ($N=48,050$) across 7–45d, 60–120d, 150–210d.
- **Compared Systems**: Native Lead I classifier, Reconstructed 12-lead with LCT-MTL, Ground Truth 12-lead.
- **Decisive Endpoints**:
  1. *Fibrillation Wave Preservation*: Spectral power in the $4\text{--}9\,\text{Hz}$ fibrillatory band in reconstructed $V_1$.
  2. *P-Wave Emergence*: Delineated P-wave amplitude and PR interval in reconstructed II, $V_1$ upon conversion to SR.
  3. *Zero Hallucination Rate*: Specificity $\ge 96\%$ on SR follow-up tracings.
- **Success Criterion**: Statistically indistinguishable fibrillatory spectrum in AF&rarr;AF pairs; statistically verified P-wave recovery without AF hallucination in AF&rarr;SR pairs.
- **Priority**: **MUST-RUN**.

---

## 4. Run Order and Milestones

| Milestone | Goal | Targeted Runs | Decision Gate | Compute Cost | Risk & Mitigation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M0: Setup & Metadata** | AWS S3 permission setup, HEEDB metadata sync to `/data/mithunmanivannan/heedb_metadata`. | Metadata sync & census extraction. | Tabular files synced; 0 waveform bytes pulled. | 0 GPU-hrs | S3 IAM permission; solved by attaching policy in console. |
| **M1: Model Baseline** | Lock LCT-MTL champion weights and verify PTB-XL test benchmark. | Evaluates LCT-MTL-384 on PTB-XL fold 10. | Replicate $r \ge 0.7538$ and $P_{05} \ge 0.418$. | ~4 GPU-hrs | Checkpoint loading; verify frozen architecture weights. |
| **M2: Manifest Export** | Filter exact paired cohort manifests (`mgh_af_pairs.csv`, `euh_af_pairs.csv`). | Export $120,150$ paired record IDs. | All pairs satisfy window constraints and quality gates. | 0 GPU-hrs | Timestamp parsing; verified with regex date parser. |
| **M3: Paired Waveforms** | Stream targeted paired waveforms ($20.59\,\text{GB}$) to `/data/mithunmanivannan/heedb_af_pairs_wfdb/`. | Parallel download of paired `.mat` and `.hea` files. | Free disk space $>450\,\text{GB}$ maintained throughout. | 0 GPU-hrs | Bandwidth throttling; parallel worker pool of 12 threads. |
| **M4: Longitudinal Eval** | Execute inference on paired cohorts and evaluate preservation vs recovery. | Forward pass on MGH & EUH paired sets. | AF&rarr;AF spectral preservation $\ge 90\%$; AF&rarr;SR P-wave recovery $>95\%$. | ~12 GPU-hrs | Batch inference caching; store predictions in SQLite. |

---

## 5. Compute and Data Budget

- **Total Estimated GPU-Hours**: $16$ GPU-hours (evaluation only; models are pre-trained).
- **Available Storage**: $500.0\,\text{GB}$ on `/data/mithunmanivannan/` (NFS).
- **Storage Consumption Plan**:
  - Papers & preprints: $0.01\,\text{GB}$ (`/data/mithunmanivannan/papers/`).
  - Tabular metadata cache: $\sim 2.5\,\text{GB}$ (`/data/mithunmanivannan/heedb_metadata/`).
  - Target paired waveforms: **$20.59\,\text{GB}$** (`/data/mithunmanivannan/heedb_af_pairs_wfdb/`).
  - Evaluation databases & tensors: $\sim 15.0\,\text{GB}$.
  - **Total Projected Storage**: $\sim 38.1\,\text{GB}$ (**$<8\%$ of NFS quota**, leaving $>460\,\text{GB}$ free).

---

## 6. Risks and Mitigations

1. **Risk: S3 Access Point Denied on IAM**:
   - *Mitigation*: The root cause was diagnosed directly from the CLI error: IAM user `mithunm` has no attached permissions. User adds `AmazonS3ReadOnlyAccess` directly in the AWS Console, instantly unlocking access.
2. **Risk: Missing timestamps in legacy recordings**:
   - *Mitigation*: Fall back to acquisition dates in header files (`.hea`), or date stamps in 12SL diagnosis lines.
3. **Risk: Disk exhaustion on NFS**:
   - *Mitigation*: Strict gate in `download_heedb_afib_pipeline.py` halts downloads immediately if free space drops below $20\,\text{GB}$. Targeted paired download only pulls $20.59\,\text{GB}$.

---

## 7. Final Checklist

- [x] Main paper tables and storyline fully mapped out.
- [x] Occam's razor and simplicity rigorously defended (width 384, no Dice bloat).
- [x] Conduction velocity tokenization thesis ($20\,\text{ms}$) anchored with decisive ablations.
- [x] Multi-institutional external validation (MGH & EUH) integrated with exact census figures.
- [x] Longitudinal paired biological controls (AF&rarr;AF, AF&rarr;SR) fully specified across three clinical epochs.
- [x] NFS storage budget verified at $20.59\,\text{GB}$ ($4.1\%$ of 500 GB).
