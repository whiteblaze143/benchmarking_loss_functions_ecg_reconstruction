# Experiment Tracker: Beat-Structured ECG-AIM

| ID | Status | Gate | Artifact |
|---|---|---|---|
| I0 | DONE | Wrapper imports unchanged upstream beat module | `unified_latents/engineering/models/ecg_aim_lvcg_variant.py` |
| I1 | DONE | Shape, gradient, embedding normalization, mask failure | `tests/test_ecg_aim_lvcg_variant.py` |
| I2 | DONE | Label-free auxiliary pretrainer and executable smoke run | `scripts/train_ecg_aim_lvcg_embedding.py`, `pretrain_smoke/` |
| I3 | DONE | Frozen component-wise embedding extractor and executable smoke run | `scripts/extract_ecg_aim_lvcg_embeddings.py`, `extraction_smoke/` |
| D0 | DONE | 21,799/21,799 boundaries; 0 failures; cross-lead QA passed | `beat_bounds/audit.json`, `beat_bounds/quality_audit.json` |
| E0 | PARTIAL | Bitwise equivalence passed on real T_patch10 checkpoint/record; 100-record batch remains | `frozen_equivalence.json` |
| E1 | TODO | Current 640-D dimension-matched pooling baseline | — |
| E2 | TODO | Proposed auxiliary embedding pilot | — |
| E3 | READY | Triplication capacity control implemented and smoke-tested | `--field-mode triplicate` |
| E4 | READY | Count-matched uniform-window control implemented and smoke-tested | `--boundary-control uniform` |
| E5 | TODO | Component deletion probes | — |
| E6 | TODO | Seeds 43/44 and diagnostic breadth | — |

No experiment result is claimed by this tracker until its artifact exists and its status is changed to DONE.

Paper improvement is blocked: the mandated independent reviewer backend is unavailable, and no production pilot results exist yet.
