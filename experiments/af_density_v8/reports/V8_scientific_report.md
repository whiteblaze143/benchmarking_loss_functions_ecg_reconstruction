# AF-DENSITY V8: Reference Rhythm Semantics, Mathematical Conditioning, and ECG Observability of Temporal AF Phenotypes

**Program:** T-CAIREM Physiological Dynamics & Ambulatory ECG Research  
**Program Stage:** AF-DENSITY V8 — Final Adjudicated Measurement-Science Synthesis (`Stage 0`, `V8-E01`–`E07`, `M1–M3`, Experiments `A–D`, `Rev1–Rev3`)  
**Primary Report:** [V8_scientific_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/reports/V8_scientific_report.md)  
**Canonical Mirror:** [reports/V8_scientific_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/V8_scientific_report.md)  
**Evidence Ledger & Paper Outline:** [V8_final_evidence_ledger_and_outline.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/reports/V8_final_evidence_ledger_and_outline.md)  
**Results & Figures:** [`experiments/af_density_v8/results/`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/), [`experiments/af_density_v8/figures/`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/figures/)  

---

## Executive Summary: Three Levels of Measurement Failure

AF-DENSITY V8 establishes a **three-level measurement-science framework** demonstrating that measured Charitos AF density ($D = 2A / (1 - B)$) is not solely an intrinsic property of a patient's arrhythmia history: it depends jointly on **(1) reference-rhythm semantics**, **(2) ECG temporal discretization and detector fidelity**, and **(3) mathematical metric conditioning** near unit AF burden. Throughout, we pair $D$ with **Butkuvienė/Petrėnas AF Aggregation** ($G = 2A = (1 - B) D$) and **Episode Continuity** ($S_{\max} = L_{\max} / M$). Strictly enforced state nomenclature separates **`NATIVE_ALL_AF`** (`R0` unperturbed), **`ALL_KNOWN_AF`** (`100%` of retained known rhythm is AF while unresolved windows remain `UNKNOWN`), and **`OPERATIONAL_ALL_AF` / `ASSUMED_ALL_AF`** (relabeled or bridged timelines).

| Investigation | Core Quantitative Evidence | Scientific Verdict |
|:---|:---|:---|
| **Level 1 — Reference Semantics (`Exp A`, `Rev1`, `Rev3`)** | Strictly inside exact `12h` driving windows of the `19` high-burden ($B \ge 0.80$) `MIXED` LTAFDB patients (`1,097` internal gaps; `1,078 < 10s`), **native WFDB ventricular tags cover `704 / 1,078` (`65.31%` gap-weighted, `71.66%` patient-weighted mean)** while automated `r1_class` ventricular overrides cover `710` (`65.86%` / `74.08%`), alongside `353` (`32.75%` / `24.03%`) `NATIVE_SBR` and `16` (`1.48%` / `4.03%`) `NATIVE_N`. Masking `VT/IVR/B/T` (`0.1979%` of Mixed-19 time) shifts **`12/19`** to **`ALL_KNOWN_AF`** (`16/19` under `VT+SBR<60s`, `0.4836%` masked, `11.81 s` median unresolved) vs `0/4` SHDB & `0/6` IRIDIA. Exact `2^M` whole-interval partial ID (`29/38`, $M \le 12$) proves ranges up to `[0.1637, 0.9682] + ALL_AF`. | **Primary Finding (Sensitivity Analysis)** *(Blinded clinical EP packet exported for validation)* |
| **Level 3 — Mathematical Conditioning & Bounds (`Exp B`, `Rev1`, `Rev3`)** | Bridging $K-1$ periodic gaps leaves $\left\|\frac{\Delta D}{\Delta B}\right\| = \frac{1}{1-B}$ (`500x` at $B=0.998$) while **$\Delta G = 0$ identically** (`max <= 9.93e-15`), proving $G$'s high-burden stability comes from consolidation blindness. Localized equal-burden FP/FN perturbations (`66/66` $\delta \in [10^{-6}, 10^{-2}]$) satisfy non-trivial bounds $\|G-\hat{G}\| \le \delta/B < 1-B$ and $\|D-\hat{D}\| \le \delta/[B(1-B)] < 1$ (`max tightness 0.380`). Fractional discretization satisfies uniform bounds $\|G-\hat{G}_h\| \le 4h/T$ and $\|D-\hat{D}_h\| \le 4h/[T(1-B)]$ (`592/592` real tests, `506/592` non-trivial). $G$ is continuous at $B=1$ yet discontinuous at $B=0$. | **Strong Analytical & Empirical Contribution** |
| **Level 2 — Oracles, Fractional Mass & Matched Detectors (`Exp B`, `Exp C`, `Rev3`)** | Fixed binary oracles preserve `12/12 ALL_AF` and `22/22 ZERO_AF` (`0` spurious `MIXED`), yet lose defined density in `7.4%–41.9%` of `148` mixed recordings across `10s–60m`. All 4 fractional-occupancy placements (`LeftAligned, RightAligned, Centered, SplitEnds`) preserve **`148/148` (`100%`)** up to `60m` (`10s` paired MAE `0.000006–0.000007` vs `0.010383` binary), while splitting bin-edge `n_runs`. `RR_GBDT_AutoQRS` turns `23/34` (`67.65%`) undefined records `MIXED` (`2,553` false gaps, `precision 0.304 -> 0.242`); matched `25×6h` `ECGFounder` (`Raw/ZNorm`) emits `2,091–2,339` false gaps (`precision <= 0.067`). On **`(dataset, record_id)`-keyed `Common_All6_Defined` (`n=124`)**, `Agg` yields `ICC 0.733 [0.611, 0.832]` vs `D` `0.558 [0.398, 0.699]`, reversing in `LOW_LT20` (`n=74`: `D 0.454` vs `Agg 0.355`). | **Primary Supporting Evidence** |
| **Continuity Concealment (`Exp C`)** | Conditional on accurate burden and density (`<0.01`), `\|LS_err\| > {0.15, 0.20, 0.30, 0.40}` holds across `R0` (`40.6% / 37.5% / 25.0% / 15.6%`), `R2` (`34.4% / 34.4% / 21.9% / 12.5%`), and `R3_gap30s` (`32.4% / 29.4% / 23.5% / 17.6%`) under `AutoQRS RR-GBDT`, while `Oracle_60sGrid_Fixed` drops to `<= 2.1%`. | **Strong Supporting Finding** |
| **Secondary: `V8-E05` Continuity & `P02` Survival (`Exp D`, `Rev2`)** | Strict non-peeking `6h -> 6h` Poisson OOF deviance improves with `longest_share` under `R0` (`Δ = -16.487 [-31.932, -3.151], boot_p = 0.0190`), attenuating under `R2` (`-6.444, p=0.307`) and `R3` (`p >= 0.498`). `P02` causal propensity holds (`0.8273 P02_All`); on identical `n=41` patients `Hist+RR` lifts `0.478 (R0) -> 0.663 (R2/R3)`, yet pretrained `ECGFounder` matches 3-seed `RandomInit_Net1D` (`11/12` `p >= 0.106`). | **Secondary / Exploratory Only** |
| **Stopped / Retracted** | Cross-cohort fragmentation matching (`0/88` IRIDIA); `P01` density increment & propensity (`0.4936`); `>70%` false-termination wording; `CONFIRMED_ALL_AF` label for masked/bridged states; `ΔG = ΔB` bridging claim; defective oracle `ALL_AF -> MIXED` artifact; naive `record_id`-only common join; pretrained `ECGFounder` superiority. | **Stop / Retract** |

---

## 1. Stage 0: Reconciled Sample Inventory, SHDB Provenance & Unified Denominators

Sources: [`V8_stage0_sample_inventory.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_stage0_sample_inventory.csv), [`V8_m1_reconciled_denominators.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_m1_reconciled_denominators.csv).

- Enforcing `raw_samples = known_samples + unknown_samples` across all 379 files (`max_sample_discrepancy = 0`, resolving `0.519029 h` left-censored overlap in 34 IRIDIA files):
  - **IRIDIA-AF (`167` recs, `152` annotated patients):** `6,690.4760 h` raw = `6,687.0510 h` known + `3.4250 h` unknown (`1,608.3916 h` AF, `5,078.6594 h` non-AF; `200.0 Hz` `.h5`).
  - **LTAFDB (`84` recs, `84` patients):** `1,960.6000 h` raw = `1,934.9233 h` known + `25.6767 h` unknown (`1,030.8932 h` AF, `904.0301 h` non-AF; `128.0 Hz`).
  - **SHDB-AF (`128` raw / `98` annotated recs, `122` / `93` patients):** `3,050.3769 h` raw = `2,330.4926 h` known + `719.8843 h` unknown (`455.9402 h` AF, `1,874.5524 h` non-AF; `v1.0.1` distributed at `200.0 Hz`, `794` native `(AFIB` tags vs `809` in `v1.0.0`).
  - **COMBINED (`379` raw / `349` annotated recs, `358` / `329` unique patients):** **`11,701.4529 h`** raw = **`10,952.4669 h`** known + **`748.9860 h`** unknown (`3,095.2250 h` [`28.26%`] AF; `7,857.2419 h` non-AF).
- At fixed `12.0 h` known horizon (`E01/E02` vs `E03`), `326` unique patients qualify (`IRIDIA 152` [`87` 1st-rec defined vs `89` any-rec], `LTAFDB 82` [`59` defined], `SHDB-AF 92` [`52` defined]). Because numeric `record_id` strings collide across `LTAFDB` and `SHDB-AF` (e.g., `'103'`), all multi-cohort joins and intersections strictly key on **`(dataset, str(record_id))`**.

---

## 2. Level 1 — Reference Semantics, Exact 12h Gap Taxonomy & Rigorous Partial Identification (`Experiment A`, `Rev1`, `Rev3`)

Sources: [`V8_rev3_hb19_12h_taxonomy_weighted.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_hb19_12h_taxonomy_weighted.csv), [`V8_rev3_hb19_gap_contributions.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_hb19_gap_contributions.csv), [`V8_rev3_ep_review_packet_top60.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_ep_review_packet_top60.csv), [`V8_rev1_masking_denominators.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev1_masking_denominators.csv), [`V8_rev3_partial_id_rigorous.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_partial_id_rigorous.csv), [`V8_rev3_partial_id_extrema_witnesses.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_partial_id_extrema_witnesses.csv), [`V8_exp_a_reference_layers.png`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/figures/V8_exp_a_reference_layers.png).

### 2.1 Exact `12h` Driving-Gap Taxonomy: Native WFDB Labels vs Automated `r1_class` Overrides (`n = 1,097` / `1,078 < 10s`)
Restricting strictly to the exact first `12h` known windows of the `19` high-burden ($B \ge 0.80$) `MIXED` LTAFDB patients (`1,116` AF runs $\to$ **`1,097` internal gaps**, of which **`1,078` are `< 10 s`**; distinct from `3,021` / `2,738` full-24h counts) separates native WFDB rhythm tags from automated heuristic overrides across both gap-weighted and patient-weighted summaries:

| Classification Axis | Category (`EXACT_12H_LT10S_1078`) | Gaps (`n = 1,078`) | Gap-Weighted % | Patient-Weighted Mean % (`n=19`) | Patient Median % |
|:---|:---|---:|---:|---:|---:|
| **Native WFDB** | `NATIVE_VT` | `537` | `49.81%` | `65.10%` | `100.00%` |
| **Native WFDB** | `NATIVE_SBR` | `353` | `32.75%` | `24.03%` | `0.00%` |
| **Native WFDB** | `NATIVE_IVR` | `125` | `11.60%` | `2.11%` | `0.00%` |
| **Native WFDB** | `NATIVE_B` / `NATIVE_T` | `29` / `13` | `2.69%` / `1.21%` | `2.63%` / `1.82%` | `0.00%` |
| **Native WFDB** | `NATIVE_N` | `16` | `1.48%` | `4.03%` | `0.00%` |
| **Native WFDB** | `NATIVE_OTHER` (`N+SBR`, `AB`) | `4` / `1` | `0.37%` / `0.09%` | `0.25%` / `0.03%` | `0.00%` |
| *Subtotal Native Ventricular* | *`VT + IVR + B + T`* | **`704`** | **`65.31%`** | **`71.66%`** | *—* |
| **Automated `r1_class`** | `VENTRICULAR_OVERRIDE_PROB_CONTINUATION` | `710` *(+6 `N`-vent)* | `65.86%` | `74.08%` | `100.00%` |
| **Automated `r1_class`** | `ORGANIZED_NON_AF_PROB_TERMINATION` | `301` | `27.92%` | `21.55%` | `0.00%` |
| **Automated `r1_class`** | `INDETERMINATE_SHORT_OR_MIXED` / `IRREG_FWAVE` | `59` / `8` | `5.47%` / `0.74%` | `4.15%` / `0.22%` | `0.00%` |

*(Across all `1,097` exact-12h gaps regardless of length, native ventricular tags still account for `705/1,097` [`64.26%` gap-weighted, `71.63%` pt mean]: `537 VT`, `125 IVR`, `29 B`, `14 T`, alongside `357 SBR`, `25 N`, `10` mixed/other.)*  
- Automated QRS-cancelled/pre-QRS features remain **unvalidated screening proxies**, NOT ground truth. To enable formal clinical validation without selection bias, [**`V8_rev3_ep_review_packet_top60.csv`**](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_ep_review_packet_top60.csv) exports **`58` highest-impact sub-10s gaps** across all `19` patients with pre-populated gap timestamps, beat counts, and blinded adjudication fields (`ep_blinded_atrial_verdict`, `ep_confidence`, `ep_notes`).

### 2.2 State Nomenclature (`NATIVE_ALL_AF` vs `ALL_KNOWN_AF` vs `OPERATIONAL_ALL_AF`) & Certified Partial Identification
Masking suspect intervals marks them `UNKNOWN` (`ALL_KNOWN_AF`: `100%` of retained known samples are AF) and never establishes **`NATIVE_ALL_AF`**. Bridging/relabeling forces **`OPERATIONAL_ALL_AF` (`ASSUMED_ALL_AF`)**:

| Reference Layer | LTAFDB `MIXED` / `ALL_KNOWN_AF` / `NATIVE_or_OP_ALL_AF` | Mean Masked % (`All 36` / `Mixed 19`) | Cond. Density (Defined) | Cond. `Agg` / 0-Ext `Agg` | SHDB / IRIDIA `MIXED` Retained |
|:---|:---:|:---:|---:|---:|:---:|
| **`R0_Native`** | `19 / 0 / 17` *(NATIVE)* | `0.00% / 0.00%` | `0.5319` | `0.00017 / 0.00002` | `4/4` & `6/6` |
| **`R2_VT_Unknown`** | `7 / 12 / 17` *(NATIVE)* | `0.1045% / 0.1979%` | `0.4481` | `0.00056 / 0.00000` | `4/4` & `6/6` |
| **`R1_Waveform_ProbAF`** | `7 / 0 / 29` *(12 OPERATIONAL)* | `0.00% / 0.00%` | `0.7851` | `0.00056 / 0.00000` | `4/4` & `6/6` |
| **`R1_Waveform_ConsUnknown`** | `6 / 13 / 17` *(NATIVE)* | `0.1457% / 0.2762%` | `0.6011` | `0.00222 / 0.00000` | `4/4` & `6/6` |
| **`R2_VT_SBR10_Unknown`** | `3 / 16 / 17` *(NATIVE)* | `0.2103% / 0.3984%` | `0.6049` | `0.00080 / 0.00000` | `4/4` & `6/6` |
| **`R2_VT_SBR60_Unknown`** | `3 / 16 / 17` *(NATIVE)* | `0.2552% / 0.4836%` | `0.6469` | `0.00044 / 0.00000` | `4/4` & `6/6` |
| **`R3_gap10s`** | `3 / 0 / 33` *(16 OPERATIONAL)* | `0.00% / 0.00%` | `0.6992` | `0.00070 / 0.00000` | `4/4` & `6/6` |
| **`R3_gap30s` / `R3_gap60s`** | `1 / 0 / 35` *(18 OPERATIONAL)* | `0.00% / 0.00%` | `0.9667 / 0.9868` | `0.0964 / 0.00000` | `4/4` & `6/6` |

- **Exhaustive (`M <= 12`) vs Sampled (`M > 12`) Partial Identification (`V8_rev3_partial_id_rigorous.csv`):** On the fixed `43,200 s` physical clock under `constant_rhythm_per_masked_interval`:
  - **`29 / 38`** `(record, layer)` pairs have $M \le 12$ and are **`EXACT_EXHAUSTIVE_WHOLE_INTERVAL`** (`2^M` enumeration; e.g., under `R2_VT_SBR60_Unknown`: `13/19` exact exhaustive cases [`12` `ALL_KNOWN_AF` + `1` `MIXED`], median exact density width `0.0948`, max exact width `0.8045` in `LTAFDB 62` where `M=10` yields $D \in [0.1637, 0.9682] \cup \{\text{ALL\_AF}\}$ and certified burden `[0.999500, 1.000000]`).
  - **`9 / 38`** pairs have $M > 12$ and are explicitly labeled **`SAMPLED_ATTAINABLE_EXTREMA_ONLY`** (`6/19` under `R2_VT_SBR60_Unknown`: e.g., `LTAFDB 33` [`M=59`, `224.99 s` masked] shows attainable $D \in [0.1079, 0.9647] \cup \{\text{ALL\_AF}\}$ with certified burden `[0.994792, 1.000000]`; `LTAFDB 204` [`M=290`, `510.63 s`]: attainable `[0.1868, 0.9906] + ALL_AF`). Across the `16` `ALL_KNOWN_AF` cases under `R2_VT_SBR60_Unknown`, median unresolved time is only **`11.805 s` (`0.0273%`)**. Exact witness gap subsets attaining min/max extrema are exported in [`V8_rev3_partial_id_extrema_witnesses.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_partial_id_extrema_witnesses.csv).

---

## 3. Level 3 — Mathematical Conditioning ($\Delta G = 0$), Non-Trivial $\delta$ & $4h/T$ Bounds (`Experiment B`, `Rev1`, `Rev3`)

Sources: [`V8_rev1_theorem1_corrected_agg.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev1_theorem1_corrected_agg.csv), [`V8_rev3_nontrivial_delta_bounds.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_nontrivial_delta_bounds.csv), [`V8_rev3_fractional_bin_bounds.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_fractional_bin_bounds.csv), [`V8_rev1_theorem1_and_partial_id.png`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/figures/V8_rev1_theorem1_and_partial_id.png).

Let $B = M/T \in (0, 1)$, $D = \frac{2A}{1-B}$, and Butkuvienė/Petrėnas AF Aggregation $G = 2A = (1 - B) D \in [0, 1 - B]$.

1. **Theorem 1 ($\left|\Delta D / \Delta B\right| = 1/(1-B)$ vs Exact $\Delta G = 0$):** On the periodic $K$-cell family ($B = 1 - \epsilon$, $K \ge 2$ uniform gaps), bridging $K-1$ gaps produces $\left|\frac{\Delta D}{\Delta B}\right| = \frac{1}{1-B}$ (`50.0x` at $B=0.98$, `500.0x` at $B=0.998$) while **$\Delta G = 0$ identically** (`max |Delta_G| = 9.925e-15` across all `32/32` configs). $G$ avoids amplification near $B=1$ because it is blind to consolidating $K$ episodes into 1.
2. **Equal-Burden Disagreement Bounds — Global & Non-Trivial Localized (`66 / 66` Verified):**
   $$|G - \hat{G}| \le \frac{\delta}{B}, \qquad |D - \hat{D}| \le \frac{\delta}{B(1 - B)} \qquad (\hat{B} = B).$$
   Beyond global swaps (`426/426`, where $\delta > B(1-B)$ often pushes the $D$-bound above `1`), **`66 / 66` localized balanced FP/FN perturbations** ($\delta \in \{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$) sit strictly inside the non-trivial regime (`100%` satisfy $\delta/[B(1-B)] < 1$ and $\delta/B < 1-B$), satisfy both inequalities (`100%`), and reach tightness up to **`0.380`** of the theoretical bound.
3. **Fractional-Bin Error Bounds (`592 / 592` Verified):** Replacing binary labels with fractional bin occupancy $b_k \in [0, 1]$ at scale $h \ll T$ preserves burden identically ($\hat{B}_h = B$) and guarantees:
   $$|G - \hat{G}_h| \le \frac{4h}{T}, \qquad |D - \hat{D}_h| \le \frac{4h}{T(1 - B)}.$$
   Across all `148` mixed recordings × `{10s, 60s, 5m, 60m}` (`592` checks, cross-checked against `metrics.charitos_density` to `<= 1e-11`), **`592 / 592` (`100%`)** obey both inequalities and **`506 / 592` (`85.5%`)** lie strictly in the non-trivial `<1` regime for $D$.
4. **Boundary Asymmetry ($B \to 1^-$ vs $B \to 0^+$):** $D$ is discontinuous at both endpoints; $G \to 0$ continuously at $B = 1$, yet $G$ remains **path-dependent / discontinuous at $B \to 0^+$** ($\lim = 1$ for one short block vs $0$ for dispersed blocks).

---

## 4. Level 2 — Oracles, Fractional Placements, Detectors & Common-Population Reliability CIs (`Experiments B, C`, `Rev1`, `Rev3`)

Sources: [`V8_rev3_fractional_multiscale_summary.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_fractional_multiscale_summary.csv), [`V8_rev3_fractional_patient_level.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_fractional_patient_level.csv), [`V8_rev3_reliability_ci.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev3_reliability_ci.csv), [`V8_exp_c_full182_summary.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_exp_c_full182_summary.csv), [`V8_exp_c_matched25_summary.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_exp_c_matched25_summary.csv).

### 4.1 Fixed Binary Oracles vs Four Fractional Within-Bin Placements (`10s–60m`, `n = 148` Mixed)
All 8 fixed binary oracle implementations yield **`0 / 12 ALL_AF`** and **`0 / 22 ZERO_AF`** spurious transitions, yet majority-voting still loses defined density in `7.4%` (`11/148` at `10s`) up to `41.9%` (`62/148` at `60m`). All four fractional placements (`LeftAligned, RightAligned, Centered, SplitEnds`) keep **`148 / 148` (`100%`)** defined at every scale:

| Scale $h$ | Paired Common `n` | `Majority_Binary` Dens MAE (`Max`) | `Frac_RightAligned` Dens MAE (`Max`) | `Frac_Centered` Dens MAE | `Frac_SplitEnds` Dens MAE | `med \|Δn_runs\|` (`Frac` vs `Bin`) |
|:---:|---:|---:|---:|---:|---:|---:|
| **`10 s`** | `137` | `0.010383 (0.30294)` | `0.000006 (0.00008)` | `0.000006` | `0.000007` | `6.0–13.0` vs `1.0` |
| **`30 s`** | `127` | `0.014872 (0.37399)` | `0.000020 (0.00016)` | `0.000020` | `0.000027` | `3.0–8.0` vs `2.0` |
| **`60 s`** | `124` | `0.026821 (0.48697)` | `0.000046 (0.00029)` | `0.000040` | `0.000061` | `3.0–6.0` vs `4.0` |
| **`5 m`** | `108` | `0.025762 (0.28734)` | `0.000227 (0.00178)` | `0.000199` | `0.000296` | `2.0–3.0` vs `6.5` |
| **`15 m`** | `100` | `0.036769 (0.76497)` | `0.000812 (0.00451)` | `0.000786` | `0.000966` | `2.0–3.0` vs `7.0` |
| **`60 m`** | `88` | `0.041496 (0.40581)` | `0.005169 (0.01667)` | `0.005633` | `0.007466` | `3.0` vs `6.5` |

Across burden strata (`V8_rev3_fractional_patient_level.csv`), `Frac_RightAligned` achieves near-machine accuracy even inside `MOD_20_80` (`n=37`: `60s` MAE `0.000009` vs Binary `0.001722`) and `LOW_LT20` (`n=78`: `60s` MAE `0.000067` vs `0.032469` Binary `n=74`), whereas `HIGH_GE80` (`n=33`) drops to `0.000291` at `60s` and `0.020344` at `60m` (vs Binary `0.071780` on `n=9` surviving high-burden records).

### 4.2 Matched Detectors, `Common_All6_Defined` (`n=124`) Bootstrap CIs & Continuity Concealment
- **False Gaps Across Matched Timelines:** Under `R0`, `RR_GBDT_AutoQRS` converts **`23/34` (`67.65%`: `16/22 ZERO_AF + 7/12 ALL_AF`)** undefined histories to spurious `MIXED` (`2,553` false gaps, `precision 0.304 -> 0.242` across `R0->R3`). On identical `25×6h` timelines (`V8_exp_c_matched25_summary.csv`), `ECGFounder_ZNorm_10s` produces `2,183–2,339` false gaps (`precision <= 0.047`, `dens_mae 0.249–0.278`) and `Raw_10s` `2,091–2,267` (`precision <= 0.067`).
- **`B = 2,000` Patient-Cluster Bootstrap CIs (`Common_All6_Defined`, `n = 124` keyed by `(dataset, record_id)`):**

| Comparator & Stratum | Metric | `ICC(2,1) [95% CI]` | `Spearman [95% CI]` | `NMAE/SD [95% CI]` |
|:---|:---|---:|---:|---:|
| `Oracle_10sGrid_Fixed` (`ALL`, `n=124`) | `Charitos_Density` / `AF_Agg` | `0.997 [0.994, 0.999]` / `1.000 [0.999, 1.000]` | `0.972 [0.939, 0.991]` / `0.999 [0.998, 1.000]` | `0.024 [0.015, 0.037]` / `0.009 [0.006, 0.014]` |
| `Oracle_60sGrid_Fixed` (`ALL`, `n=124`) | `Charitos_Density` / `AF_Agg` | `0.921 [0.837, 0.971]` / `0.976 [0.945, 0.994]` | `0.849 [0.752, 0.923]` / `0.932 [0.855, 0.983]` | `0.140 [0.086, 0.207]` / `0.071 [0.037, 0.118]` |
| `RR_GBDT_AutoQRS` (`ALL`, `n=124`) | `Charitos_Density` / `AF_Agg` | `0.558 [0.398, 0.699]` / `0.733 [0.611, 0.832]` | `0.632 [0.499, 0.739]` / `0.694 [0.552, 0.805]` | `0.709 [0.519, 0.978]` / `0.435 [0.323, 0.576]` |
| `RR_GBDT_AutoQRS` (`LOW_LT20`, `n=74`) | **`Charitos_Density`** vs `AF_Agg` | **`0.454 [0.217, 0.674]`** vs `0.355 [0.124, 0.582]` | **`0.579 [0.375, 0.750]`** vs `0.418 [0.188, 0.634]` | **`0.826 [0.532, 1.309]`** vs `0.947 [0.624, 1.477]` |
| `RR_GBDT_AutoQRS` (`MOD_20_80`, `n=37`) | `Charitos_Density` vs **`AF_Agg`** | `0.379 [0.042, 0.716]` vs **`0.672 [0.422, 0.873]`** | `0.360 [0.016, 0.650]` vs **`0.725 [0.464, 0.899]`** | `0.988 [0.464, 5.850]` vs **`0.500 [0.258, 0.826]`** |
| `RR_GBDT_AutoQRS` (`HIGH_GE80`, `n=13`) | `Charitos_Density` vs `AF_Agg` | `0.542 [0.074, 0.842]` / `0.508 [0.166, 0.842]` | `0.555 [-0.048, 0.915]` / `0.775 [0.310, 0.994]` | `0.739 [0.398, 1.315]` / `0.985 [0.437, 1.939]` |

*(In `Each_Comp_Defined` `n=145`, `RR_GBDT_AutoQRS` gives `ICC 0.512 [0.364, 0.637]` vs `Agg 0.803 [0.720, 0.871]`. Common-cohort `n=124` eliminates denominator attrition and confirms the exact `LOW_LT20` vs `MOD_20_80` crossover.)*
- **Continuity Concealment:** Conditional on accurate $B$ and $D$ (`<0.01`), `|LS_err| > {0.15, 0.20, 0.30, 0.40}` holds across **`R0` (`40.6% / 37.5% / 25.0% / 15.6%`)**, **`R2` (`34.4% / 34.4% / 21.9% / 12.5%`)**, and **`R3_gap30s` (`32.4% / 29.4% / 23.5% / 17.6%`, `8/34` accurate or `8/130` defined)** under `AutoQRS`, vs `<= 2.1%` for `Oracle_60sGrid_Fixed` under `R3_gap30s`.

---

## 5. Exploratory Secondary Continuity & Survival Controls (`Experiment D`, `Rev2`)

Sources: [`V8_rev2_e05_causal_oof_ci.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev2_e05_causal_oof_ci.csv), [`V8_rev2_e05_causal_nb.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev2_e05_causal_nb.csv), [`V8_exp_d_p02_causal_maturity.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_exp_d_p02_causal_maturity.csv), [`V8_rev2_p02_same_patient_multiseed.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/results/V8_rev2_p02_same_patient_multiseed.csv).

1. **`V8-E05` Strict Non-Peeking `6h -> 6h` Transitions Across `R0`, `R2`, `R3`:** Bridging strictly inside `[0, 6h]` before feature extraction gives paired 5-fold OOF Poisson deviance reductions (`B=2,000` bootstrap) of **`-16.487 [-31.932, -3.151]` (`boot_p = 0.0190`)** at `R0_Native` (`n=128`), attenuating to **`-6.444 [-19.408, +4.132]` (`p = 0.3068`)** at `R2_VT_SBR60_Unknown` (`n=112`) and `-2.438 (p = 0.498)` / `-0.666 (p = 0.680)` at `R3_gap30s / gap60s` (`n=111`).
2. **`P02` Causal Maturity & Same-Patient `N_common = 41` Multi-Seed Null:** Strict `maturity_time <= landmark_time` confirms patient-level persistence propensity (`0.8273 P02_All`, `0.7882 ValidRR`, vs `P01 0.4936`). On identical `n=41` patients, `Hist + RR` within-patient logistic AUROC lifts `0.4779 (R0) -> 0.6633 (R2) -> 0.6634 (R3)` (`ECGFounder_Alone 0.6347 p=0.008` at `R2`), yet pretrained `ECGFounder` vs the 3-seed mean of `RandomInit_Net1D` yields `p >= 0.106` across `11/12` contrasts (`R2 Common LGBM p=0.0480` isolated). **Pretraining-specific benefit is unestablished.**

---
*Authoritative AF-DENSITY V8 Final Adjudicated Scientific Report (`Stage 0`, `Exp A–D`, `Rev1–Rev3`). See [V8_final_evidence_ledger_and_outline.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_v8/reports/V8_final_evidence_ledger_and_outline.md) for the strict 7-category Evidence Ledger, Paper 1 Manuscript Outline, and Blinded EP Review Packet specification.*
