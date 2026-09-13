# Experiment Audit Report: Cross-Model Integrity Verification

**Date**: 2026-09-13  
**Auditor**: Cross-Model Adversarial Integrity Auditor (Ultra Reasoning Mode)  
**Project**: `benchmarking_loss_functions_ecg_reconstruction` (Temporal RepSpat ECG Extension)  
**Scope**: 9 Empirical Clinical Cohorts (PTB-XL, EchoNext, LUDB, RDB, ISP, Kingston-ICU, Emory-MUSE, Sunnybrook, Zhejiang).

---

## Overall Verdict: ⚠️ WARN (Core Methodological Audit Maintained at WARN)

## Integrity Status: `warn`

> **Executive Summary**:  
> The codebase enforces a strict empirical mandate: **no fully synthetic ECGs and no synthetic labels/ground truth are used**. Controlled simulated measurement perturbations (noise ladder in Gate 4) are employed strictly for stability analysis.  
> An overall verdict of **WARN** is firmly maintained to distinguish verified implementations from completed multi-cohort validation, and to enforce eight verified methodological contracts:
> 1. **CAHC Exact Author Parity Verified**: `tit_ecg.src.cahc` was tested directly against the official author code (`repspat-main/src/repspat/clustering.py`). On both synthetic benchmarks and real clinical ECG waveforms, `tit_ecg` reproduces the author's spatial silhouette scores and cluster partitions at **exact machine precision ($\Delta = 0.00\times 10^0$)**.
> 2. **Gate 1 Implementation Pass; Frozen-Grid Result Mixed**: The resampling pipeline is parallel-from-original. Across 10 records, physical geometry beats discrete $m$ at 500-to-250 Hz (**0.825 vs 0.569**) and 1000-to-250 Hz (**0.806 vs 0.615**), but loses at 1000-to-500 Hz (**0.767 vs 0.844**).
> 3. **Gate 2A Scale-Equivariance Multi-Record Regression**: Analytical scale-equivariance ($\widehat{\text{MMD}}'^2 = \frac{1}{|a|} \widehat{\text{MMD}}^2$ with invariant permutation ordering) was empirically confirmed across **10 heterogeneous clinical records** from LUDB, ISP, and PTB-XL under $1\times, 10\times, 1000\times$ voltage scalings: **$\max \Delta p = 0.00\times 10^0$ and $\max \Delta q = 0.00\times 10^0$ across all 10 records**.
> 4. **Gate 3B Completed and Failed**: Exact phase/beat membership eliminates tunable purity thresholds. Across 34 eligible patients, mean $r_p$ is **0.136** for attribute blocks and **0.175** for temporal blocks, above the 0.08 calibration target; LUDB and Zhejiang show substantial excess rejection.
> 5. **RDB Provenance Explicitly Categorized as Algorithmic**: Documentation confirms RDB delineations are **wavelet-derived** (`data/rdb_wavelet_delineation_cache`), not manual cardiologist annotations. It is categorized under `algorithmic_delineation`.
> 6. **Three-Object Hierarchy for Cliques**: Disentangles the similarity graph $\mathcal{G}_{\text{sim}}$, the maximal cliques $\{\mathcal{Q}_1, \ldots, \mathcal{Q}_K\}$, and the hard partition heuristic $Y_{\text{resolved}}$, reporting graph/clique stability separately from partition heuristics.
> 7. **Candidate Topology Terminology Enforced**: The similarity graph is designated as a **candidate electrophysiological topology** or **derived recurrence-graph phenotype**.
> 8. **Pathology Associations are Exploratory**: Downstream density differences ($p_{\text{omnibus}, \rho} = 0.00246$) remain exploratory until the final frozen pipeline is executed end-to-end.

---

## Methodological Status Ledger

$$
\boxed{\texttt{CORE\_METHOD\_AUDIT = WARN}}
$$

$$
\boxed{\texttt{GATE1A\_IMPLEMENTATION = PASS;\ GATE1B\_FROZEN\_GRID = MIXED}}
$$

$$
\boxed{\texttt{GATE2A\_ANALYTIC\_EQUIVARIANCE = PASS\ (CONFIRMED\ ON\ 10\ RECORDS)}}
$$

$$
\boxed{\texttt{GATE3A\_RECURRENCE\_CONTROL = RUNNING}}
$$

$$
\boxed{\texttt{GATE3B\_POST\_CAHC\_CALIBRATION = FAIL}}
$$

$$
\boxed{\texttt{METHOD\_FREEZE\_DECISION = NO\ GO}}
$$

$$
\boxed{\texttt{CLIQUE\_GRAPH\_LOGIC = SUPPORTED\ (PREVENTS\ TRANSITIVE\ CHAINS)}}
$$

$$
\boxed{\texttt{CLIQUE\_OVERLAP\_RESOLUTION = ECG\ EXTENSION\ (SEPARATE\ OBJECT)}}
$$

$$
\boxed{\texttt{9\_COHORT\_BENCHMARK = RUNNING\ IN\ TMUX}}
$$

$$
\boxed{\texttt{PATHOLOGY\_CLAIMS = EXPLORATORY}}
$$

---

## Systematic Integrity Checks

### A. Ground Truth Provenance: ✅ PASS (Real Clinical Cohorts; Refined Provenance Taxonomy)

1. **Synthetic Data Policy Statement**:
   $$\boxed{\text{No fully synthetic ECGs and no synthetic labels/ground truth are used.}}$$
   Controlled simulated measurement perturbations (additive Gaussian noise ladder in Gate 4) are used solely for perturbation robustness analysis.

2. **Dataset Provenance & Annotation Category Taxonomy**:
   - `wave_delineation_gt` (Manual Expert Physician Annotations):
     - **LUDB** (`data/ludb/`): 200 12-lead ECGs with cardiologist-annotated P, QRS, T boundaries.
     - **ISP** (`data/isp/`): High-resolution (1000 Hz) clinical ECG records with manual wave boundary delineations.
     - **Zhejiang** (`data/zhejiang/`): 12-lead hospital ECG records with clinical rhythm and wave delineations.
   - `algorithmic_delineation` (Algorithmic / Automated Wave Delineation):
     - **RDB** (`data/rdb_wavelet_delineation_cache`): 2,398 clinical patient ECGs with **wavelet-derived** wave segmentations. Verified as algorithmic; not manual cardiologist annotations.
   - `diagnostic_labels` (Supervised Institutional Diagnoses):
     - **PTB-XL** (`data/ptbxl/`): PhysioNet diagnostic database (Wagner et al., 2020) with clinical SCP diagnostic statements.
     - **EchoNext** (`data/echonext/`): Pierre Elias / IntroECG EchoNext release (Columbia University Irving Medical Center, version `15233e9392e9dc10c136a5624abf10199207bce9`), 250 Hz 12-lead voltage waveforms with echocardiographic diagnoses.
   - `unlabeled_topology` (Real Clinical Signals for Candidate Topology Characterization):
     - **Kingston-ICU** (`data/kingston_icu/`): High-acuity ICU telemetry records (240 Hz).
     - **Emory-MUSE** (`data/emory_muse/`): Hospital 12-lead ECG database (500 Hz).
     - **Sunnybrook** (`data/sunnybrook/`): Academic medical center 12-lead ECG records (500 Hz).

---

### B. CAHC Exact Parity with Author Code: ✅ PASS (Machine Precision Verification)

1. **Parity Testing (`tit_ecg/tests/verify_cahc_against_author.py`)**:
   - Directly tested against `external/repspat-main/src/repspat/clustering.py`.
   - On synthetic 2D/1D grids across $G \in [3, 4, 5, 6]$:
     - Mean silhouette difference: **$\Delta = 0.00\times 10^0$**
     - Max per-sample silhouette difference: **$\Delta = 0.00\times 10^0$**
   - On real clinical ECG waveforms (LUDB Record 1 across $m \in [4, 8, 16], G \in [4, 5, 6, 7]$):
     - Author average silhouette: `0.85460243`
     - `tit_ecg` average silhouette: `0.85460243`
     - Absolute difference: **$0.00\times 10^0$**
2. **Singleton Handling Alignment**:
   - The paper's Equation 2 has $n_g - 1$ in the denominator. The author's code (`repspat/clustering.py`, line 114) sets $a_i = 0.0$ when $|C_g| \le 1$ and continues if a cluster has no spatial neighbors. `tit_ecg.src.cahc` now defaults to this exact author behavior while preserving an optional strict validity check.

---

### C. Gate 1 Resampling Chain & Empirical Equivariance: ✅ PASS (Parallel Design Verified)

1. **Parallel Resampling Architecture**:
   $$X_{1000} \xrightarrow{\text{resample\_poly}(1, 2)} X_{500}, \qquad X_{1000} \xrightarrow{\text{resample\_poly}(1, 4)} X_{250}$$
   Both branches resample directly from the original 1000 Hz waveform.
2. **Direct vs Cascaded Filtering Sanity Check**:
   - Evaluated on real 1000 Hz signal:
     - Direct: $X_{1000} \xrightarrow{(1,4)} X_{250}$
     - Cascaded: $X_{1000} \xrightarrow{(1,2)} X_{500} \xrightarrow{(1,2)} X_{250}^{\text{cascaded}}$
     - **Relative RMSE = 0.068%** ($< 0.1\%$). Demonstrates that filtering distortion is negligible compared to true temporal discretization effects.
3. **Frozen-grid empirical results (`same_signal_resampling_summary.json`, 10 ISP records)**:
   - **Clinical 500 Hz $\to$ 250 Hz Transition**:
     - Physical Geometry ARI: **0.825**
     - Uncorrected Discrete $m$ ARI: **0.569**
     - **Net Advantage**: $+0.256$ ARI gain for physical geometry.
   - **1000 Hz $\to$ 500 Hz Transition**:
     - Physical Geometry ARI: **0.767** (Discrete: **0.844**).
   - **1000 Hz $\to$ 250 Hz transition**:
     - Physical Geometry ARI: **0.806** (Discrete: **0.615**).
   - The frozen physical grid improves two of three comparisons but not 1000-to-500 Hz; Gate 1B is therefore mixed rather than a universal pass.

---

### D. Score Normalization & Multi-Record Scale Equivariance: ✅ PASS (10 Records Verified)

1. **Analytical Scale Equivariance**:
   $$k(ax, ay; ac) = \frac{1}{|a|} k(x, y; c) \implies \widehat{\text{MMD}}'^2 = \frac{1}{|a|} \widehat{\text{MMD}}^2$$
   Permutation test statistics scale by the exact same positive scalar $\frac{1}{|a|}$, guaranteeing invariant permutation rankings, $p$-values, and graph edges.
2. **Multi-Record Empirical Verification**:
   - Evaluated across 10 heterogeneous clinical records (4 LUDB, 3 ISP, 3 PTB-XL) across $1\times, 10\times, 1000\times$ voltage scalings:
     - Max $\Delta p = 0.00\times 10^0$ across all 10 records.
     - Max $\Delta q = 0.00\times 10^0$ across all 10 records.
     - All 10 records passed with exact machine-precision invariance.

---

### E. Gate 3 Formulation: Split into 3A and 3B

1. **Gate 3A (Prespecified Recurrence Controls)**:
   - Evaluates MMD permutation behavior directly on expert-delineated wave intervals (Beat 1 QRS vs Beat 2 QRS) without data-dependent clustering.
   - Reports: $P(\text{reject} \mid \text{prespecified same-phase recurrence})$.
2. **Gate 3B (Post-CAHC Recurrence Controls; completed 120/120)**:
   - Evaluates whether data-dependent CAHC selection induces excess rejection.
   - Protocol: admits a CAHC cluster only if every sample has the same annotated phase and QRS-defined beat (exact purity 1.0; no tunable purity cutoff). Null pairs share phase and have distinct beat IDs separated by $\ge 200\text{ ms}$.
   - Primary patient-equal mean $r_p$ across 34 eligible patients: **0.136** (attribute k-means blocks), **0.175** (temporal blocks), and **0.597** (naive unblocked comparator).
   - Cohort failures are material: LUDB mean $r_p=0.250$ for both blocked schemes; Zhejiang mean $r_p=0.315$ (attribute) and $0.411$ (temporal). The prespecified calibration target $\le 0.08$ is not met.
   - Exact-purity filtering leaves no alternative pairs in ISP, LUDB, or RDB, so power is not established by this run.

---

### F. Clique Logic & Partition Resolution: Three Distinct Objects

To separate graph-theoretic properties from partition heuristics:
1. $\mathcal{G}_{\text{sim}}$: Binary/weighted non-rejection graph. An edge means the BH-adjusted MMD test did not reject $P_g=P_h$ ($q>0.05$); it is not significant evidence of recurrence or equality.
2. $\{\mathcal{Q}_1, \ldots, \mathcal{Q}_K\}$: Maximal cliques of $\mathcal{G}_{\text{sim}}$, which strictly prevent unsupported transitive chaining ($A-B, B-C, A \not\sim C$).
3. $Y_{\text{resolved}}$: Resolved hard partition, where overlapping clique memberships are resolved by assigning nodes to their primary clique. Stability is evaluated and reported separately for all three objects.

### G. Gate 3B Cohort-Heterogeneity Diagnostic

1. **Decision consistency**: recurrence and positive controls use rejection at BH-adjusted $q<0.05$, exactly matching the quantity used to decide whether the graph retains an edge.
2. **Patient-level cohort contrast**: patient-bootstrap 95% intervals are ISP **0.000–0.060**, LUDB **0.000–0.625**, RDB **0.000–0.067**, and Zhejiang **0.095–0.560**. Only Zhejiang's interval excludes the 0.08 operational target, while LUDB is too small and imprecise to localize confidently.
3. **Apparent predictors are cohort-confounded**: raw patient-level associations with temporal separation ($\rho=0.377$) and heart rate ($\rho=-0.463$) weaken after within-dataset ranking ($\rho=0.174$ and $-0.264$). No examined predictor has a within-dataset exploratory $p<0.05$.
4. **Annotation/duration signal**: Zhejiang has much longer median control separation (**2799 ms**) and an implausibly low annotation-derived median heart rate (**21.3 bpm**), compared with **899–1038 ms** and **67–125 bpm** elsewhere. This points to cohort-specific annotation coverage or record-window structure as the first diagnostic target; it is not proof of a causal mechanism.
5. **Power remains incomplete**: allowing exact-purity clusters from different phases without a distinct-beat requirement produces 7/7 rejected positive controls, but all are from Zhejiang. All admissible recurrence controls are QRS, so phase heterogeneity and cross-cohort power remain unidentifiable.

---

## Action Items & Status

| ID | Issue | Required Action | Status |
|:---|:------|:----------------|:-------|
| **ACT-1** | Author code parity | Directly verify `cahc.py` against `repspat/clustering.py`. | ✅ **PASSED ($\Delta = 0.00\times 10^0$)** |
| **ACT-2** | Gate 1 resampling chain | Implement parallel-from-original design; run direct vs cascaded check. | ✅ **PASSED (RMSE = 0.068%)** |
| **ACT-3** | Gate 1 frozen-grid results | Collect aggregate metrics across real 1000/500/250 Hz signals. | ⚠️ **MIXED (physical geometry wins 2/3 comparisons)** |
| **ACT-4** | Multi-record scale equivariance | Test $1\times, 10\times, 1000\times$ across 10 heterogeneous records. | ✅ **PASSED ($\Delta p = 0.00\times 10^0$)** |
| **ACT-5** | Gate 3B patient-level calibration | Run exact phase/beat-pure post-CAHC controls and compute patient-level $r_p$. | ❌ **FAIL (blocked mean $r_p=0.136$–$0.175$)** |
| **ACT-6** | RDB provenance correction | Classify RDB as algorithmic/wavelet-derived delineation. | ✅ **CORRECTED** |
| **ACT-7** | Three-object clique reporting | Maintain $\mathcal{G}_{\text{sim}}$, $\{\mathcal{Q}_k\}$, and $Y_{\text{resolved}}$ as separate objects. | ✅ **ADOPTED** |
| **ACT-8** | Scope wording | Enforce "no fully synthetic ECGs or synthetic labels". | ✅ **ENFORCED** |
| **ACT-9** | Gate 3B terminology | Replace null/Type-I artifact fields with recurrence-control terminology. | ✅ **COMPLETED** |
| **ACT-10** | Cohort heterogeneity | Stratify the faithful attribute-block result at patient level without retuning. | ✅ **COMPLETED; annotation/window heterogeneity prioritized** |
