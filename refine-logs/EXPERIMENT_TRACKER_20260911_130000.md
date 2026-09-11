# Experiment Tracker: M3R and M3I

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| M3_ADAPTED_REPSPAT_PRIMARY | M3-F | Preserve completed primary evidence | Adapted repSpat, 999 permutations | PTB-XL train domains | checksums, 2,016 pairs | MUST | COMPLETE_PENDING_FREEZE | One 64-node `AMBIGUOUS_NON_CLIQUE`; never a motif |
| QVCG_CC_MOTIF_GATE | M3-F | Categorical quotient decision | Frozen M3 | PTB-XL train domains | component/clique/rejection invariants | MUST | FAIL | Pairwise recurrence claim remains unresolved, not failed |
| M3R_REFERENCE_IMPLEMENTATION_SENSITIVITY | M3R | Full released-code sensitivity | floor blocks, standard KMeans, 200 permutations | same 64 domains | edge/rejection agreement and topology | MUST | TODO | Separate output directory; restart-safe |
| M3I_FUNCTIONAL_ATLAS_AUDIT | M3I | Label-free relational/geometry analysis | frozen primary M3 | same 64 domains | topology, distance association, nonlocal/adjacent pairs, MDS | MUST | TODO | No community or cluster labels |
| M3I_DEPENDENCE_CALIBRATION | M3I-CAL | Null calibration and power | synthetic patient/temporal dependence | synthetic | type-I error and power | NICE | DEFERRED | Follow after requested M3R/M3I |

