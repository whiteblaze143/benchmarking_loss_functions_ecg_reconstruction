# Pipeline Summary: Temporal repSpat for ECG Signal Analysis

**Problem**: Over-segmentation of recurring physiological states by temporally constrained clustering versus unconstrained noise contamination in electrocardiograms.  
**Final Method Thesis**: Nonparametric pairing of Constrained Agglomerative Hierarchical Clustering with Inverse Multiquadratic Maximum Mean Discrepancy, attribute-based block permutation, and maximal clique graph reassignment faithfully isolates Repeated Temporal Patterns (RTPs) in ECGs.  
**Final Verdict**: **READY**  
**Date**: 2026-09-12  

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Core Package: `temporal_rep_stat_ecg/`
- Authors' Original Code Preserved: `temporal_rep_stat_ecg/repspat/` and `temporal_rep_stat_ecg/author_repspat/`

## Contribution Snapshot
- **Dominant contribution**: The exact mathematical translation of repSpat to 1D temporal biomedical signals, establishing nonparametric discovery of Repeated Temporal Patterns (RTPs).
- **Supporting contribution**: Dual continuous (Euclidean) and binary clinical marker (Jaccard) representations for multi-lead ECGs.
- **Explicitly rejected complexity**: Stochastic black-box neural clusterers and uncalibrated heuristic thresholds.

## Must-Prove Claims
1. **Simulation Superiority**: Under temporal autoregression ($\eta \in \{0.3, 0.8\}$), Temporal repSpat achieves higher ARI ($\approx 0.95\text{--}1.00$) than CAHC alone ($\le 0.60$) and unconstrained clustering.
2. **Clinical Utility**: Discovers valid recurring rhythm and repolarization patterns across PTB-XL records under both Euclidean and Jaccard metrics.

## Next Action
- Launch full benchmark suite inside detached `tmux` session via `/run-experiment`.
