# Experiment Tracker: Paper 11

| Run ID | Gate | Purpose | Decisive metrics | Status |
|---|---|---|---|---|
| P11-CAT-G1 | categorical G1 | finite-HMM recovery and predictive value | cardinality, future MSE, copy/mean baselines | FAIL (seed 44: K=8 materially improves over K=4, and K=4 does not beat beat-copy) |
| P11-CONT-G1-v2 | continuous G1 | continuous predictive compression | retained gain, conditional-prefix residual, all false controls | PASS (fresh seeds 45/46/47; 1,000 updates) |
| P11-CONT-G0.0 | feature causality | eligibility of standard Phase-KME preprocessing | raw-sample provenance and acausal preprocessing audit | FAIL (full-record zero-phase filter and R-to-next-R tokens) |
| P11-CONT-G0 | causal-token provenance | `P11-CAUSAL-TOKEN-v1` artifact | exact lineage, suffix/target interventions, deterministic hashes | PASS (15,245 records; 13,142 patients; suffix and target interventions pass; fold 8 unread) |
| P11-CONT-G3 | real-data feasibility | causal-token coverage and selection bias | coverage, diagnosis-stratified eligibility, targets/patient | TODO (unblocked by G0) |
| P11-CONT-G4 | clinical pilot | held-out future loss and diagnosis | patient-equal future loss and diagnosis | BLOCKED by P11-CONT-G3 |
