# Experiment Plan: Quotient Vectorcardiographic Representation via Recurrent Distributional Motifs (Q-VCG)

**Problem**: Traditional ECG representations treat whole waveforms or individual heartbeats as monolithic atomic units, failing to distinguish local functional electrophysiological dynamics from global anatomical position and entangling electrode geometry with cardiac pathology.
**Method Thesis**: Cardiac electrical trajectories contain recurrent, distributionally equivalent local dynamical motifs in spatially disconnected regions of VCG state space; quotienting VCG phase space by this distributional equivalence factorizes function from location ($[Z_{\text{motif}}, Z_{\text{where}}]$) and yields a compact, clinically accessible representation without requiring diagnosis labels or deep neural architectures.
**Date**: 2026-09-11

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| **C1 (Recurrent Functional Equivalence)**: Spatially separated regions of 3D VCG phase space exhibit reproducible, distributionally equivalent local dynamics ($\xi = [s, \rho, \kappa]$) that can be merged into quotient motifs ($K < M$). | Falsifies the assumption that representation units must be contiguous beats or ad-hoc tokens; proves the existence of recurrent dynamical motifs. | Nontrivial repeated motifs identified ($K < M$), reproducible across patient half-splits ($\text{AMI} > 0.60$), with MMD below self-reproducibility floor $\epsilon_{\text{MMD}}$. | B1, B2, B5 |
| **C2 (Function/Location Factorization)**: Factorizing representation into functional motif identity and physical VCG coordinates ($R_3 = [Z_{\text{motif}}, Z_{\text{where}}]$) outperforms both motif-only ($Z_{\text{motif}}$) and location-only ($Z_{\text{where}}$) representations across clinical diagnostics. | Demonstrates that the repSpat quotient provides beneficial regularized functional compression while explicit spatial position provides essential complementary information ($R_3 > R_1 > R_0$). | Factorization test showing $R_3 > R_1$ and $R_3 > Z_{\text{motif}}$ and $R_3 > Z_{\text{where}}$ across PTB-XL superclass, subclass, form, and rhythm linear probes. | B3, B4 |

---

## Paper Storyline

- **Main Paper Must Prove**:
  1. The VCG phase space quotient $\mathcal{Q} = \mathcal{V} / \sim$ discovers genuine spatially disconnected, distributionally equivalent motifs that compress local domain vocabulary ($K \ll M$) without clinical labels.
  2. Quotient motif representation ($R_2$) is non-inferior to unmerged spatial domains ($R_1$) despite high compression ($C_Q > 0.40$), acting as structural regularization.
  3. Factorizing function and location ($R_3 = [Z_{\text{motif}}, Z_{\text{where}}]$) yields superior linear accessibility and low-shot sample efficiency compared to raw VCG summaries ($R_0$) and unmerged spatial domains ($R_1$).
  4. Representations produce clinically structured nearest-neighbor retrieval neighborhoods on sealed patient splits.

- **Appendix Can Support**:
  1. Sensitivity to 3D lattice discretization ($16^3, 24^3, 32^3$).
  2. Kernel invariance analysis (Gaussian RBF vs. Inverse Multiquadric IMQ).
  3. Non-dipolar residual analysis $r(t) = E(t) - A v(t)$ across diagnostic categories.
  4. Post-hoc electrophysiological alignment with classical P/QRS/T phases.

- **Experiments Intentionally Cut**:
  - Complex transformer / diffusion backbones (PRD Section 3 & 53 explicitly forbids neural models before transparent gate passes).
  - Multi-lead reconstruction optimization (v1 focuses strictly on representation learning).
  - End-to-end backpropagation into the lead direction matrix $A$.

---

## Experiment Blocks

### Block 1: Discovery of Recurrent VCG Motifs (Sanity & Novelty Isolation)
- **Claim Tested**: C1 (Existence and non-triviality of recurrent distributional motifs in 3D VCG phase space).
- **Why This Block Exists**: Verifies whether disconnected regions in physical electrical space genuinely share identical dynamical distributions.
- **Dataset / Split**: PTB-XL Folds 1–7 (patient-balanced: 1 ECG per patient, capped microstates, $N_{\text{patient}} \approx 2,500$).
- **Compared Systems**:
  1. Unconstrained spatial clustering (K-Means on $\xi$).
  2. Spatially Contiguous Domains ($M=64$ domains via CAHC).
  3. Q-VCG Quotient Motifs ($K$ classes via calibrated MMD block permutation).
- **Metrics**:
  - Number of spatial domains $M$ vs. quotient classes $K$.
  - Number of `REPEATED` motifs (containing $\ge 2$ disconnected spatial components).
  - Compression ratio $C_Q = 1 - K/M$.
  - Median component separation in physical mV.
- **Setup Details**:
  - Lead direction matrix $A \in \mathbb{R}^{12 \times 3}$ (LVCG Table 7).
  - SVD Tikhonov pseudoinverse with $\lambda = 10^{-6}$.
  - Savitzky-Golay polynomial derivatives at $500\text{ Hz}$, $P=64$ phase points per beat.
  - Voxel grid: $24^3$, 26-neighbor adjacency.
  - MMD: Gaussian RBF, block-permutation swapping patient blocks $\Xi_{p, i} \leftrightarrow \Xi_{p, j}$, tolerance $\epsilon_{\text{MMD}} = Q_{0.95}(\mathcal{D}_{\text{self}})$.
- **Success Criterion**: $K < M$, at least 5 robust `REPEATED` motifs identified, zero false merges on synthetic null test.
- **Failure Interpretation**: If $K = M$, repSpat discovers no repeated structure; kill the motif claim.
- **Table / Figure Target**: Figure 2 (3D VCG phase space colored by quotient motifs) and Table 1 (Motif registry statistics).
- **Priority**: MUST-RUN.

### Block 2: Transparent Representation Qualification Suite (Main Anchor Result)
- **Claim Tested**: C1, C2 (Clinical accessibility and compression value of quotient representation).
- **Why This Block Exists**: Proves that quotienting preserves clinical diagnostic information while reducing feature dimensionality.
- **Dataset / Split**:
  - Training: PTB-XL Folds 1–7 ($N = 17,441$).
  - Development / Validation: PTB-XL Fold 8 ($N = 2,183$).
  - Confirmation: PTB-XL Fold 9 ($N = 2,183$).
  - Sealed: PTB-XL Fold 10 ($N = 2,183$).
- **Tasks**:
  1. Superclass (5 broad diagnostic classes: NORM, MI, STTC, CD, HYP).
  2. Subclass (21 clinical categories).
  3. Form (12 morphology-specific classes).
  4. Rhythm (12 arrhythmia classes).
- **Compared Systems**:
  - $R_0$: Basic VCG summary statistics ($D=25$).
  - $R_1$: Spatial domain features ($M=64$: occupancy, dwell, transitions).
  - $R_2$: Quotient motif features ($K$ motifs: occupancy, dwell, transitions).
  - $R_3$: Quotient + Location features ($[Z_{\text{motif}}, Z_{\text{location}}]$).
- **Metrics**: Macro AUROC, Macro AUPRC, Sensitivity@95% Specificity, Specificity@95% Sensitivity.
- **Setup Details**: Frozen representation vectors $\to$ Logistic Regression with $L_2$ regularization sweep ($C \in [10^{-4}, 10^2]$) tuned on Fold 8.
- **Success Criterion**: $R_2 \ge R_1 - 0.01$ AUROC (non-inferiority under compression), $R_3 > R_1$ and $R_3 > R_0$.
- **Failure Interpretation**: If $R_2 \ll R_1$, quotienting discards essential clinical morphology; stop and do not add neural capacity.
- **Table / Figure Target**: Main Paper Table 1 (Multi-task AUROC/AUPRC across R0, R1, R2, R3).
- **Priority**: MUST-RUN.

### Block 3: Function vs. Location Factorization Test (Novelty & Elegance Check)
- **Claim Tested**: C2 (Decoupling functional dynamics from physical field coordinates).
- **Why This Block Exists**: Determines whether the repSpat quotient contributes distinct information beyond absolute spatial coordinate.
- **Dataset / Split**: PTB-XL Folds 1–7 (train), Fold 8 (val).
- **Compared Systems**:
  1. $Z_{\text{motif}}$ only (occupancy, dwell, and transition dynamics of quotient motifs).
  2. $Z_{\text{where}}$ only (mean physical coordinates $\bar v_{p, k}$ of visited motifs).
  3. Full Factorized $[Z_{\text{motif}}, Z_{\text{where}}]$ ($R_3$).
- **Metrics**: Macro AUROC across Superclass and Subclass tasks.
- **Success Criterion**: Outcome A: $[Z_{\text{motif}}, Z_{\text{where}}] > Z_{\text{motif}} > Z_{\text{where}}$.
- **Failure Interpretation**: If $Z_{\text{where}} \approx [Z_{\text{motif}}, Z_{\text{where}}]$, local dynamics add nothing over spatial location; kill the motif claim.
- **Table / Figure Target**: Main Paper Figure 3 (Factorization ablation bar chart).
- **Priority**: MUST-RUN.

### Block 4: Sample Efficiency & Low-Shot Linear Evaluation
- **Claim Tested**: C1, C2 (Unsupervised representation richness under scarce clinical labels).
- **Why This Block Exists**: Since motifs are discovered completely unsupervised, good representations should separate pathologies with minimal labeled training data.
- **Dataset / Split**: PTB-XL Folds 1–7 with train subsampling: 1%, 5%, 10%, 25%, 50%, 100% of labels.
- **Compared Systems**: $R_0$, $R_1$, $R_2$, $R_3$.
- **Metrics**: Macro AUROC vs. label percentage.
- **Success Criterion**: $R_3$ maintains $\ge 85\%$ of full-data AUROC with only 10% labels, consistently beating $R_0$ by $> 0.05$ AUROC at 1% and 5% regimes.
- **Table / Figure Target**: Main Paper Figure 5 (Sample efficiency curves).
- **Priority**: MUST-RUN.

### Block 5: Stability, Invariance & Retrieval Diagnostics
- **Claim Tested**: C1 (Stability and physical validity of the representation object).
- **Why This Block Exists**: Rules out clustering artifacts, patient overfitting, and kernel sensitivity.
- **Evaluation Protocols**:
  1. **Split-Half Patient Stability**: Partition Folds 1–7 into disjoint halves $H_1, H_2$; repeat motif discovery; compute Adjusted Mutual Information (AMI) and Variation of Information (VI).
  2. **Kernel Invariance**: Compare Gaussian RBF vs. IMQ quotient graphs.
  3. **Lattice Invariance**: Compare $16^3, 24^3, 32^3$ spatial discretization.
  4. **Clinical Retrieval**: On Fold 8, evaluate $P@K$, $Recall@K$, $nDCG@K$ for diagnostic multi-label Jaccard similarity.
- **Success Criterion**: $\text{AMI}(H_1, H_2) > 0.60$, major repeated motifs persist across kernels, retrieval $nDCG@10$ outperforms $R_0$ by $\ge 10\%$.
- **Failure Interpretation**: If motifs reorganize completely under patient splits, the representation is an artifact of sampling noise; halt pipeline.
- **Table / Figure Target**: Appendix Tables S1–S3 and Figure S2.
- **Priority**: MUST-RUN.

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk & Mitigation |
|---|---|---|---|---|---|
| **M0: Unit Verification** | Pass 22/22 unit tests for geometry, dynamics, synthetic repSpat (A–F), and representations. | Synthetic tests & contracts | All tests PASS | < 2 min CPU | Strict contracts; loud assertions prevent silent fallback. |
| **M1: Microstate Census** | Build patient-balanced microstates on Folds 1–7 ($N \approx 2,500$ patients, $1.5 \times 10^5$ samples). | `build_vcg_microstates.py` | Occupancy census verified, zero NaNs, $500\text{ Hz}$ physical derivatives. | ~ 5 min CPU | Insufficient R peaks $\to$ adaptive threshold with physiological bounds. |
| **M2: Spatial Domains** | Fit CAHC on $24^3$ grid to obtain $M=64$ contiguous spatial domains. | `discover_vcg_domains.py` | $M=64$ contiguous domains, zero unassigned voxels in supported region. | ~ 10 min CPU | Isolated voxels $\to$ enforce 26-connectivity component cleanup. |
| **M3: RepSpat Quotient** | Run calibrated MMD matrix and patient-block permutation to obtain $K$ motifs. | `discover_repeated_motifs.py` | $\epsilon_{\text{MMD}}$ calibrated from split-half self-reproducibility; $K < M$. | ~ 30 min CPU/GPU | Patient dependence $\to$ strict block permutation swapping patient sets. |
| **M4: Feature Extraction** | Extract $R_0, R_1, R_2, R_3$ across Folds 1–9. | `build_qvcg_features.py` | Feature tables serialized to Parquet, OOD rate logged. | ~ 15 min CPU | High OOD rate $\to$ inspect grid bounds, flag points explicitly. |
| **M5: Linear & Low-Shot Probing** | Train frozen linear probes on 4 PTB-XL tasks across 6 label percentages. | `evaluate_qvcg.py` | $R_2$ non-inferior to $R_1$, $R_3 > R_1 > R_0$. | ~ 20 min CPU | Fold 10 strictly sealed; all tuning on Fold 8. |
| **M6: Stability & Gate Review** | Compute split-half stability, kernel agreement, retrieval metrics, output `QVCG_TRANSPARENT_GATE.json`. | Split-half & retrieval scripts | All 7 transparent gate conditions PASS. | ~ 30 min CPU | Transparent gate halts pipeline if any kill condition triggers. |

---

## Compute and Data Budget

- **Hardware**: Local CPU (multiprocessing with 8–16 workers) + NVIDIA A100 GPU for fast matrix operations.
- **Total Estimated Compute**: ~ 1.5–2.0 hours end-to-end.
- **Data Footprint**:
  - PTB-XL raw records: ~ 15 GB (already local in `data/ptb_xl`).
  - Extracted microstates: ~ 80 MB Parquet.
  - Feature vectors ($R_0, R_1, R_2, R_3$): ~ 45 MB Parquet.
- **Bottleneck**: Pairwise MMD computation across domain pairs $\to$ accelerated via vectorized PyTorch kernels.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **No repeated motifs found ($K = M$)** | Calibrated MMD on continuous dynamics $\xi$ detects distributional similarity missed by mean clustering. If truly $K=M$, stop and honestly report negative finding (PRD Section 75). |
| **Over-merging ($K \to 1$)** | Self-reproducibility floor $\epsilon_{\text{MMD}} = Q_{0.95}(\mathcal{D}_{\text{self}})$ ensures domains are only merged if their MMD is within finite-sample noise of identical domains. |
| **Patient leakage / pseudo-replication** | Block permutation swaps entire patient trajectories $(\Xi_{p, i} \leftrightarrow \Xi_{p, j})$, preventing sample-level independence violations. |
| **Out-of-distribution test points** | Explicit OOD flag $q(v) = \text{OOD}$; points outside training support are never silently mapped to arbitrary motifs. |

---

## Final Checklist

- [x] Main paper tables and figures are mapped to specific blocks.
- [x] Methodological novelty (repSpat quotient in physical VCG space) is isolated from architecture.
- [x] Simplicity is defended (transparent representations before any neuralization).
- [x] Clinical labels are strictly excluded from tokenizer discovery.
- [x] Fold 9 is reserved for confirmation and Fold 10 is sealed.
- [x] Synthetic test contracts (Tests A–F) defined to guarantee statistical validity before PTB-XL execution.
