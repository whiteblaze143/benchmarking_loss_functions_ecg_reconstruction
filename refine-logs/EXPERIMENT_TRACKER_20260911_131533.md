# Experiment Tracker: M3R and M3I

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| M3_ADAPTED_REPSPAT_PRIMARY | M3-F | Preserve completed primary evidence | Adapted repSpat, 999 permutations | PTB-XL train domains | checksums, 2,016 pairs | MUST | COMPLETE_FROZEN | SHA-256 manifest verified; one 64-node `AMBIGUOUS_NON_CLIQUE`, never a motif |
| QVCG_CC_MOTIF_GATE | M3-F | Categorical quotient decision | Frozen M3 | PTB-XL train domains | component/clique/rejection invariants | MUST | FAIL | Pairwise recurrence claim remains unresolved, not failed |
| M3R_REFERENCE_IMPLEMENTATION_SENSITIVITY | M3R | Full released-code sensitivity | floor blocks, standard KMeans, 200 permutations | same 64 domains | edge/rejection agreement and topology | MUST | RUNNING | `tmux:m3r_reference`; separate output directory; restart-safe |
| M3I_FUNCTIONAL_ATLAS_AUDIT | M3I | Label-free relational/geometry analysis | frozen primary M3 | same 64 domains | topology, distance association, nonlocal/adjacent pairs, MDS | MUST | COMPLETE | 330 distant non-rejected and 414 adjacent rejected pairs; no labels created |
| M3I_DEPENDENCE_CALIBRATION | M3I-CAL | Null calibration and power | synthetic patient/temporal dependence | synthetic | type-I error and power | NICE | DEFERRED | Follow after requested M3R/M3I |
