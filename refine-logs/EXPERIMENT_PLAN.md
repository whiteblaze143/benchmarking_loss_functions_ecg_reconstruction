# Experiment Plan: Validation Roadmap for Temporal repSpat (`temporal_rep_stat_ecg`)

**Date**: 2026-09-12  
**Document**: `refine-logs/EXPERIMENT_PLAN.md`  
**Pipeline Orchestrator**: `temporal_rep_stat_ecg/scripts/run_comprehensive_evaluation.py`  

---

## 1. Core Validation Claims

### Claim 1 (Simulation Superiority over Baselines)
*Under temporal autoregressive correlation ($\eta \in \{0.3, 0.8\}$), Temporal repSpat achieves significantly higher Adjusted Rand Index (ARI $\approx 0.95\text{--}1.00$) in recovering Repeated Temporal Patterns compared to:*
1. **CAHC alone**: Over-segments recurring intervals into disjoint labels (ARI $\le 0.60$).
2. **Unconstrained K-Means / HAC**: Fails to preserve temporal contiguity, fragmenting segments and confusing background noise with patterns.

### Claim 2 (Clinical Utility & Metric Sensitivity on Multi-Lead ECGs)
*On real patient multi-lead ECGs from the PTB-XL database, Temporal repSpat discovers clinically interpretable Repeated Temporal Patterns (RTPs) under both Continuous Morphology (Euclidean distance) and Binary Clinical Indicators (Jaccard distance), capturing recurrent arrhythmias and repolarization dynamics.*

---

## 2. Experiment Execution Matrix

### Block 1: Autoregressive Simulation Benchmark (Table 4 Replication in 1D)
- **Grid configurations**:
  - Autocorrelation $\eta \in \{0.3, 0.8\}$
  - Attribute dimensions $p \in \{5, 10\}$
  - Sequence lengths $n \in \{300, 600\}$
  - Random seeds: 5 seeds per configuration
- **Primary Metrics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Fowlkes-Mallows Index (FMI).
- **Ablation comparisons**:
  - Baseline A: Standard CAHC alone
  - Baseline B: Unconstrained K-Means
  - Baseline C: Unconstrained Agglomerative Clustering
  - Champion: Temporal repSpat (CAHC + IMQ MMD + Block Permutation + Maximal Cliques)

### Block 2: Clinical Multi-Lead ECG Benchmark (PTB-XL)
- **Database**: PTB-XL (`data/ptb_xl/records500`)
- **Evaluated representations**:
  - Representation 1: Continuous Morphology features (RR intervals, QRS width, ST deviation, T amplitude) with Euclidean distance.
  - Representation 2: Binary Clinical Markers (ST elevation, T inversion, premature beat, etc.) with Jaccard distance.
- **Outputs**:
  - Episode timelines with overlaid ECG waveforms
  - Episode similarity graphs $G_{\text{sim}}$ with edge weights ($\hat{MMD}^2_{\text{obs}}$) and highlighted maximal cliques
  - Structured JSON evaluation metrics.

---

## 3. Decision Gates & Stopping Criteria
- **Gate 1**: Unit test suite passes 100% (12/12 tests).
- **Gate 2**: Simulation benchmark confirms Temporal repSpat median ARI $\ge 0.85$ across configs.
- **Gate 3**: Clinical benchmark successfully executes without errors across test patient records.
