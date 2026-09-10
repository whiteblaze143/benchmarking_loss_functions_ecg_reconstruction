# Experiment Tracker: Beat-Structured ECG-AIM

| ID | Status | Gate | Artifact |
|---|---|---|---|
| I0 | DONE | Wrapper imports unchanged upstream beat module | `unified_latents/engineering/models/ecg_aim_lvcg_variant.py` |
| I1 | DONE | Shape, gradient, embedding normalization, mask failure | `tests/test_ecg_aim_lvcg_variant.py` |
| D0 | DONE | 21,799/21,799 boundaries; 0 failures; cross-lead QA passed | `beat_bounds/audit.json`, `beat_bounds/quality_audit.json` |
| E0 | TODO | Frozen ECG-AIM reconstruction equivalence on 100 records | — |
| E1 | TODO | Current 640-D dimension-matched pooling baseline | — |
| E2 | TODO | Proposed auxiliary embedding pilot | — |
| E3 | TODO | Triplication capacity control | — |
| E4 | TODO | Shuffled-boundary control | — |
| E5 | TODO | Component deletion probes | — |
| E6 | TODO | Seeds 43/44 and diagnostic breadth | — |

No experiment result is claimed by this tracker until its artifact exists and its status is changed to DONE.
