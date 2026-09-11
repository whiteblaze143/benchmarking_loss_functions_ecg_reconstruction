# Experiment Tracker: GRAIL-ECG v2

**Last Updated**: 2026-09-11 01:03 UTC  
**Master Session**: `grail_v2_pipeline` (tmux, NVIDIA A100 GPU)  

---

## Active Phase Status Table

| Phase | Description | Target Artifacts | Status |
|---|---|---|:---:|
| **R0** | Contract Corrections & Verification | `tests/test_grail_v2_contracts.py` (18 tests) | **COMPLETED (18/18 PASS)** |
| **R0** | Concept Hierarchy & Transfer Tiers | `configs/ptbxl_concept_tiers.yaml` | **COMPLETED** |
| **R1/R2** | Model `UB` (Supervised Upper Bound) | `checkpoints/grail_v2/ub_best.pt` | **IN PROGRESS (Epoch 6/12)** |
| **R1/R2** | Model `B0` (True Plain SSL) | `checkpoints/grail_v2/b0_best.pt` | QUEUED |
| **R1/R2** | Model `B1` (Clinical SSL) | `checkpoints/grail_v2/b1_best.pt` | QUEUED |
| **R1/R2** | Model `B3_geom` (Geometry alone) | `checkpoints/grail_v2/b3_geom_best.pt` | QUEUED |
| **R1/R2** | Model `B2_slots` (Slots alone) | `checkpoints/grail_v2/b2_slots_best.pt` | QUEUED |
| **R1/R2** | Model `Model_001` (View Aux alone) | `checkpoints/grail_v2/model_001_best.pt` | QUEUED |
| **R1/R2** | Model `Model_110` (Geometry + Slots) | `checkpoints/grail_v2/model_110_best.pt` | QUEUED |
| **R1/R2** | Model `Model_101` (Geometry + View Aux) | `checkpoints/grail_v2/model_101_best.pt` | QUEUED |
| **R1/R2** | Model `Model_011` (Slots + View Aux) | `checkpoints/grail_v2/model_011_best.pt` | QUEUED |
| **R1/R2** | Model `Model_M` (Full GRAIL Factorial) | `checkpoints/grail_v2/model_m_best.pt` | QUEUED |
| **R3** | Representation Qualification Battery | `results/grail_v2/representation/` | QUEUED |
| **R3** | Factorial ANOVA Main Effects ($\Delta_G, \Delta_S, \Delta_V$) | `results/grail_v2/factorial_qualification_summary.json` | QUEUED |
| **R3** | LVCG Downstream Probing Suite | `results/grail_v2/lvcg_probing_results.csv` | **ADAPTER BUILT & VERIFIED** |
| **R8/R9** | Exhaustive 4,095 Subset Lattice Evaluation | `results/grail_v2/subsets/subset_representation_metrics.parquet` | QUEUED |
| **R10** | Exact Lead Shapley Values | `results/grail_v2/subsets/lead_shapley.parquet` | QUEUED |
| **R10** | Pairwise Lead Synergy / Redundancy Graph | `results/grail_v2/subsets/lead_pair_interactions.parquet` | QUEUED |
| **R10** | Minimal Sufficient Lead Sets | `results/grail_v2/subsets/minimal_sufficient_sets.parquet` | QUEUED |
| **R10** | Pareto Information Frontiers | `results/grail_v2/subsets/information_frontier.parquet` | QUEUED |
