# Source provenance

- Nef-Net v2 source: copied byte-for-byte from `external/NEFNET-v2-main` on 2026-09-13 into `author_code/nefnet_v2`.
- Copy verification: `diff -qr` returned no differences immediately after copying.
- Aggregate SHA-256 of the sorted 67-file source checksum list: `ad8baf335a16ea7f363e52cf91b3e0c86f641cda4a8c234721b532633460aaad`.
- Author paper SHA-256: `b8ebbb5b309e47511cd9387a84dc038231cfb1bc960498e1ce21d035ddc06b01`.
- repSpat is not copied again. The frozen dependency is the existing `external/repspat-main` tree.
- Aggregate SHA-256 of its sorted 29-file checksum list: `ff66a6e82431bc1f19592dfe24e38312845bd2eb3fee0b30120c537eb0cdeb9e`.
- Repository snapshot containing both external trees: `e3a5969554d253de3908b29128b70356681d1101`.

The author Nef-Net tree is kept unchanged. All experiment-specific code lives in `src/`, `scripts/`, `configs/`, and `tests/`.

---

## N001 — PILOT_HYBRID_CONFIG (archived, not a candidate)

- Run: `results/pilot_hybrid_lr0.1_batch512_seed123/`
- Architecture: `nefnet_plus.layer` (GeoVT branch) ✓
- Optimizer: SGD lr=0.1, B=512 — sourced from `nef_net.yml` (wrong lineage for this architecture)
- Result: L1 plateau ~0.486 through epoch 28. Archived diagnostic only.

---

## N002 — panobench_3view_variablethird_geovt_seed123

- Run: `results/panobench_3view_variablethird_geovt_seed123/` (symlinked from `panobench_fixed_triplet_randview3_geovt_seed123`)
- Checkpoint SHA-256: `9174e0fef5fc9e446299944802bbc72f39da8a3cd54b632ac13e900d3625624c`
- Architecture: `nefnet_plus.layer` — PASS
- Optimizer: AdamW lr=1e-3 wd=0.01, MultiStep [50,100,150] γ=0.5, B=32, 200 epochs — PASS
- Training L1: 0.068 (epoch 0) → 0.004 (epoch 199) — PASS
- **Scientific contract** (verified against source):
  - `MODEL = official nefnet_plus.layer`
  - `CARDINALITY = 3` (fixed slot count = 3: two anchor slots I, II and one third slot)
  - `ANCHORS = I, II`
  - `VARIABLE_CARDINALITY_ANY_PAIRS = NO` (model constructor fixes `lead_num=3` under `super_mode='optimization'`)
  - `ANGLE_CONDITIONED VARIABLE_VIEW = YES`: `fixed_cardinality ≠ fixed_identity`. The third waveform goes through `W_encoder_list[0]`, while its actual $(\theta, \phi)$ goes through `mlp_list[0]`. Those angle embeddings participate in `view_transformer(encoded_theta, query_theta, w)`. The model was trained across random third views ($r \sim \{2,\dots,43\}$), incentivizing separation of waveform content from observation direction.
  - Normalization: global min-max across all 44 PanoBench channels before selecting 3 inputs (author-faithful benchmark).
- **CUDA/cuDNN note**: Convolutions required `torch.backends.cudnn.enabled = False` on the training GPU. 33 volatile uncorrected ECC errors were logged. Cause unconfirmed.
- **Status**: ARCHITECTURE=PASS, OPTIMIZATION=PASS, CONVERGENCE=PASS, FIXED_CARDINALITY_VAR_THIRD=PASS.

---

## G001-A — Held-out PanoBench test evaluation (author normalization)

- Script: `scripts/eval_g001.py`
- Test split: `/data/mithunmanivannan/panobench/test/` (1030 records, strictly held-out)
- Eval seed: 42 (distinct from training seed 123)
- Samples per record: 5 random (third-view, query) pairs (5,150 total evaluations)
- Output: `results/g001_eval/` (symlinked as `results/g001_a_eval/`)
- **Status**: COMPLETE / PASS
  - L1: mean = 0.00915, std = 0.00687, p5 = 0.00234, p95 = 0.02517
  - PSNR: mean = 38.08 dB, std = 6.52 dB
  - SSIM: mean = 0.9773, std = 0.0281
  - Angular stratification shows monotonic or stable quality across angular distance bins (35.1 dB at <33° to 41.2 dB at >164°).

---

## G001-B — Held-out PanoBench test evaluation (observed-only normalization)

- Script: `scripts/eval_g001_b.py`
- Preprocessing: `m_obs = min_{j in {I,II,r}, t} V_j(t)`, `M_obs = max_{j in {I,II,r}, t} V_j(t)`. Strictly observation-only; no leakage from unobserved views.
- Model: Same frozen N002 checkpoint (`model_final.pt`). No retraining.
- Test split: `/data/mithunmanivannan/panobench/test/` (1030 records x 5 samples = 5,150 paired evaluations)
- Output: `results/g001_b_eval/`
- **Status**: COMPLETE / EMPIRICAL FINDING
  - Absolute metrics (author-equivalent scale):
    - L1: mean = 0.01404, std = 0.01009
    - PSNR: mean = 34.59 dB, std = 6.16 dB
    - SSIM: mean = 0.9683, std = 0.0285
    - L1 physical: mean = 0.3556, std = 0.2796
  - Paired comparison vs G001-A (5,150 identical triples):
    - Delta L1 (B - A): +0.00489 (+0.5% scale shift)
    - Delta PSNR (B - A): -3.50 dB (38.08 -> 34.59 dB)
    - Delta SSIM (B - A): -0.0090 (0.9773 -> 0.9683; morphology preserved)
  - **Verdict**: SENSITIVE_TO_ALL_VIEW_LEAKAGE in absolute amplitude scaling (-3.5 dB), but structural morphology remains virtually intact (SSIM > 0.968).

---

## G001-C — Clinical-angle zero-shot transfer (Real 12-lead ECGs)

- Script: `scripts/eval_g001_c.py`
- Input: Exactly `{I, II, V3}` with author PTB-XL canonical angles: `a_I=(90°, 90°)`, `a_II=(150°, 90°)`, `a_V3=(95°, 15°)`.
- Query: Precordial leads `{V1, V2, V4, V5, V6}` with their respective canonical angles.
- Ground truth: Real measured leads from 1,000 held-out clinical 12-lead ECGs from HEEDB Emory WFDB dataset (`heedb_emory/WFDB/2010`).
- Model: Same frozen N002 checkpoint (`model_final.pt`). Zero-shot: no fine-tuning, no calibration.
- Normalization: Strictly observation-only (`m_obs, M_obs` of `{I, II, V3}`).
- Output: `results/g001_c_eval/`
- **Status**: COMPLETE / DIAGNOSTIC
  - Overall (5,000 lead queries):
    - L1 (norm): 0.04658 ± 0.03889
    - L1 (physical): 0.1049 ± 0.1002 mV (~0.10 mV absolute error)
    - PSNR: 23.72 ± 4.65 dB
    - SSIM: 0.9015 ± 0.0719
    - Pearson r: 0.4239 ± 0.5122
  - Per-lead breakdown (ordered laterally from query V6 to septal V1):
    - V6: r = 0.7229, L1 = 0.0798 mV, PSNR = 26.67 dB, SSIM = 0.9375
    - V5: r = 0.6552, L1 = 0.0965 mV, PSNR = 24.79 dB, SSIM = 0.9247
    - V4: r = 0.4998, L1 = 0.1137 mV, PSNR = 22.90 dB, SSIM = 0.9007
    - V2: r = 0.2445, L1 = 0.1276 mV, PSNR = 21.41 dB, SSIM = 0.8716
    - V1: r = -0.0030, L1 = 0.1069 mV, PSNR = 22.80 dB, SSIM = 0.8730
  - **Verdict**: Strong lateral transfer (V6/V5 correlation 0.66–0.72, SSIM > 0.92, physical error < 0.09 mV), but septal degradation at V1/V2 (r < 0.25). Corroborates that PanoBench-only torso training suffers from domain/torso-geometry transport across the septum, requiring either canonicalization or author multi-dataset pretraining as predicted in the decision tree.

---

## G002 — State-Space Geometry & Metric Distortion Benchmark

- Scripts: [`scripts/eval_g002.py`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/theta_repspat_nefnetv2/scripts/eval_g002.py), [`scripts/audit_metric_distortion.py`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/theta_repspat_nefnetv2/scripts/audit_metric_distortion.py), [`scripts/eval_g002_c_kernel_alignment.py`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/theta_repspat_nefnetv2/scripts/eval_g002_c_kernel_alignment.py)
- Model: Frozen N002 checkpoint (`model_final.pt`).
- Four-Arm Architectural Hierarchy:
  - **$R1$ (Observational 8-Lead Reference Geometry $\in \mathbb{R}^8$)**: $x_t^{GT} \in \mathbb{R}^8$ ($\rho_D=1.0000$ by construction; observational target reconstructed by GeoVT, not assumed to be true biological ground truth).
  - **$R0$ (Standard Kors VCG in $\mathbb{R}^3$)**: $K_{\text{Kors}} \cdot x_t^{GT}$.
  - **$R2$ (Nef Panoramic Completion in $\mathbb{R}^8$)**: Inputs $\{I, II, V3\}$ completed to 8 leads using frozen $N002$.
  - **$R3$ (Kors VCG on Nef Completion in $\mathbb{R}^3$)**: $K_{\text{Kors}} \cdot E^{R2}(t)$.
- Cohorts Evaluated:
  - Clinical 8-lead: 500 real patient recordings from HEEDB Emory WFDB (`heedb_emory/WFDB/2010`).
  - Dense 44-channel panorama: 100 held-out PanoBench test records (48 viewpoints, 42 precordial views with CT-derived spherical coordinates).
- Outputs:
  - `results/g002_eval/g002_summary.json`
  - `results/g002_eval/g002_clinical_records.json`
  - `results/g002_eval/g002_panobench_records.json`
  - `results/g002_eval/g002_ablation_triangle.json`
  - `results/g002_eval/g002_metric_distortion.json`
  - `results/g002_eval/g002_c_kernel_alignment.json`

### Empirical Metric Summary

#### Part A: Clinical 8-Lead State-Space Geometry (HEEDB Emory, $N=500$)
| Arm | Rank Correlation $\rho_D$ | Metric Distortion $\epsilon_D$ (95% CI) | Local Distortion $\epsilon_{D, 10}$ (95% CI) | $k$-NN Jaccard $J_{10}$ | CAHC Partition $\text{ARI}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$R1$ (Reference)** | $1.0000$ (def.) | $0.0000$ (def.) | $0.0000$ (def.) | $1.0000$ | $1.0000$ |
| **$R0$ (Kors VCG)** | $0.9439 \pm 0.0401$ | $0.5282$ $[0.5148, 0.5421]$ | $0.5384$ $[0.5286, 0.5480]$ | $0.3939 \pm 0.1239$ | $0.7921 \pm 0.2378$ |
| **$R2$ (Nef Panorama)** | $0.9103 \pm 0.0597$ | $0.3295$ $[0.3112, 0.3494]$ | $0.4632$ $[0.4353, 0.4950]$ | $0.2909 \pm 0.0914$ | $0.6934 \pm 0.2509$ |
| **$R3$ (Kors on Nef)** | $0.8479 \pm 0.0824$ | $0.6260$ $[0.6042, 0.6487]$ | $0.5995$ $[0.5828, 0.6151]$ | $0.1934 \pm 0.0642$ | $0.6826 \pm 0.2614$ |

#### Part B: Dense 44-Channel Panorama Geometry (PanoBench, $N=100$)
- **Rank Correlation $\rho_D(44)$**: mean = $0.8961 \pm 0.1260$, median = $0.9651$
- **Frobenius Metric Distortion $\epsilon_D(44)$**: mean = $0.4036$ (95% CI: $[0.3735, 0.4342]$)
- **Local $k$-NN Distortion $\epsilon_{D, 10}(44)$**: mean = $0.4598$ (95% CI: $[0.4244, 0.4951]$)
- **Local $k$-NN Jaccard $J_{10}(44)$**: mean = $0.3290 \pm 0.0999$, median = $0.3510$
- **CAHC Partition Agreement $\text{ARI}(44)$**: mean = $0.7226 \pm 0.2480$, median = $0.7434$

### Paired Contrasts (Bootstrap 95% Confidence Intervals, $N=500$)
1. **$R2 \text{ vs } R0$ (Nef 8D Panorama vs Kors 3D VCG)**:
   - $\Delta \rho_D = -0.0336$ (95% CI: $[-0.0386, -0.0283]$): Kors preserves distance rank slightly better.
   - $\Delta \epsilon_D = -0.1987$ (95% CI: $[-0.2163, -0.1812]$): Relative to the 8-lead observational coordinate system, $R2$ has lower unaligned distance-matrix error than Kors ($\epsilon_D = 0.3295 \text{ vs } 0.5282$).
   - $\Delta \epsilon_{D, 10} = -0.0752$ (95% CI: $[-0.1054, -0.0422]$).
   - $\Delta J_{10} = -0.1030$ (95% CI: $[-0.1150, -0.0909]$): Local neighborhood overlap is lower for $R2$ due to septal zero-shot drop ($r_{V1} \approx 0$).
   - $\Delta \text{ARI}_{\text{CAHC}} = -0.0986$ (95% CI: $[-0.1222, -0.0743]$).
2. **$R2 \text{ vs } R3$ (Nef 8D Panorama vs Kors-on-Nef 3D)**:
   - $\Delta \rho_D = +0.0625$ (95% CI: $[+0.0576, +0.0675]$) ($p < 10^{-15}$): **Preserving the 8D panorama significantly outperforms linear 3D Kors compression**.
   - $\Delta \epsilon_D = -0.2964$ (95% CI: $[-0.3148, -0.2785]$): Kors compression inflates metric distortion from $0.33$ to $0.63$.
   - $\Delta J_{10} = +0.0975$ (95% CI: $[+0.0938, +0.1014]$).
3. **$R3 \text{ vs } R0$ (Kors-on-Nef vs True Kors in $\mathbb{R}^3$)**:
   - $\Delta \rho_D = -0.0961$ (95% CI: $[-0.1026, -0.0900]$): Isolates canonicalization transfer error when dimension is fixed at 3.

---

## Scientific Ledger & Decision Status

```
N002_TRAINING            = PASS
PANOBENCH_SYNTHESIS      = PASS
DENSE_PANORAMA_GEOMETRY  = PASS
CLINICAL_TRANSFER        = PARTIAL
R2_VS_R3                 = STRONG PASS (Kors compression destroys useful panoramic structure)
R2_VS_R0                 = MIXED (R2 lower unaligned distance error; R0 better rank, J10, CAHC)
G002                     = MIXED
FINAL REPRESENTATION FREEZE = NO (N002 not ready for confirmatory freeze)
N003                     = JUSTIFIED
```

### Pre-Registered Expectations for `PHASE1_N002_EXPLORATORY`
- Current running job (`repspat_phase1`) is labeled `PHASE1_N002_EXPLORATORY` (evaluated with $B=500$, exploratory baseline rather than prespecified confirmatory $B=1000$).
- **Pre-registered qualitative expectation**:
  - $R2$ may improve some global recurrence-distribution behavior because it retains more of the full lead field.
  - However, its CAHC segmentation and local graph structure are expected to be less stable than $R0$ in cohorts whose septal morphology matters, because $J_{10}^{R2} < J_{10}^{R0}$ and $\text{ARI}_{\text{CAHC}}^{R2} < \text{ARI}_{\text{CAHC}}^{R0}$ on clinical leads.

---

## N003 Design Specification

Because $N002$'s PanoBench-only training regime exhibits incomplete clinical transport across the septum, $N003$ implements the authors' intended two-stage architecture workflow:

1. **Architecture**:
   - Identical official `nefnet_plus.layer` (GeoVT). No speculative architectural redesign.
2. **Stage I (Clinical Any-Pairs Pretraining)**:
   - Multi-dataset clinical Any-Pairs pretraining on standard published ECG datasets:
     $$\text{PTB-XL}, \quad \text{CPSC2018}, \quad \text{ChinaDB}, \quad \text{Tianchi}$$
   - Variable-cardinality input slots learning robust cross-lead clinical correlations.
3. **Stage II (Dense Angular Supervision)**:
   - Fine-tuning/calibration on PanoBench dense CT-localized torso-view geometry to acquire continuous angular interpolation.
4. **Data Separation Contract**:
   $$\mathcal{D}_{\text{N003 train}} \cap \mathcal{D}_{\text{repSpat eval}} = \varnothing$$
   - LUDB, ISP, RDB, and HEEDB Emory are strictly excluded from N003 pretraining, remaining pure external evaluation cohorts.
5. **Deployment-Consistent Normalization**:
   - Training normalization computed strictly from observed input views $\mathcal{O}$:
     $$m_{\text{obs}} = \min_{j \in \mathcal{O}, t} x_j(t), \quad M_{\text{obs}} = \max_{j \in \mathcal{O}, t} x_j(t)$$
     $$\tilde{x}_q = \frac{x_q - m_{\text{obs}}}{M_{\text{obs}} - m_{\text{obs}}}$$
     preventing test-time target statistics leakage.


