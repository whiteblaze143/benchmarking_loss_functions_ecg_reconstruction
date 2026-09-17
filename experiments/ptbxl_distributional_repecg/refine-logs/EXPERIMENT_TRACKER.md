# Experiment Tracker: 15-Paper Suite Ledger

| Paper ID | Architecture | Model Class | Status | Peak AUROC (PTB-XL) | OOD Status | Artifact Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Paper 02** | Local Phase KME | `PhaseCNN` | **IN PROGRESS** (epoch 14+) | **0.8716** (kernel complete) | Queued for 9 datasets | `outputs/paper02_kernel_mean/` |
| **Paper 01** | Distributional Recurrence | `RecurrenceCNN` | **PREPARED** (Smoke tested) | Pending execution | Queued | `outputs/paper01_distributional_recurrence/` |
| **Paper 03** | Phase-Cell Signature Path | `PathSignatureClassifier` | **PREPARED** | Pending execution | Queued | `outputs/paper03_signature_path/` |
| **Paper 04** | Hankel Dynamics & DMD | `HankelDynamicsModel` | **PREPARED** | Pending execution | Queued | `outputs/paper04_hankel_dynamics/` |
| **Paper 05** | Koopman Operator | `KoopmanOperatorModel` | **PREPARED** | Pending execution | Queued | `outputs/paper05_koopman_operator/` |
| **Paper 06** | Conditional RepStat | `ConditionalRepStatModel` | **PREPARED** | Pending execution | Queued | `outputs/paper06_conditional_repstat/` |
| **Paper 07** | Operator Reconstruction | `OperatorReconstructionAuxiliary` | **PREPARED** | Pending execution | Queued | `outputs/paper07_operator_reconstruction/` |
| **Paper 08** | Local Token Attention | `LocalTokenCrossAttention` | **PREPARED** | Pending execution | Queued | `outputs/paper08_token_attention/` |
| **Paper 09** | Counterfactual Measurement | `CounterfactualMeasurementOperator`| **PREPARED** | Pending execution | Queued | `outputs/paper09_counterfactual_measurement/`|
| **Paper 10** | Interventional RepStat | `InterventionalRepStatModel` | **PREPARED** | Pending execution | Queued | `outputs/paper10_interventional_repstat/` |
| **Paper 11** | Causal-State ECG | `CausalStateECGModel` | **PREPARED** | Pending execution | Queued | `outputs/paper11_causal_state_ecg/` |
| **Paper 12** | Structural Innovation | `StructuralInnovationModel` | **PREPARED** | Pending execution | Queued | `outputs/paper12_structural_innovation/` |
| **Paper 13** | Counterfactual Surgery | `CounterfactualSurgeryModel` | **PREPARED** | Pending execution | Queued | `outputs/paper13_counterfactual_surgery/` |
| **Paper 14** | Invariant Mechanism | `InvariantMechanismDiscoveryModel` | **PREPARED** | Pending execution | Queued | `outputs/paper14_invariant_mechanism/` |
| **Paper 15** | Causal Factorization | `CausalMechanismFactorizationModel`| **PREPARED** | Pending execution | Queued | `outputs/paper15_causal_factorization/` |

---

## Log of Completed Grid Cell Milestones
- **2026-09-17 04:30**: `smoke_train_kme.py` executed on NVIDIA A100-SXM4-40GB. VRAM footprint: 40MB. Verified Nyström kernel representations.
- **2026-09-17 04:45**: `build_paper02_ood_representations.py` generated OOD representations for EchoNext, LUDB, ISP, Kingston, Emory, Sunnybrook, Zhejiang.
- **2026-09-17 04:55**: `rdb_wavelet_tensor_cache.tgz` unzipped and verified. OOD representations built for all 9 external datasets.
- **2026-09-17 05:00**: `repecg_p2_pipeline` launched in detached tmux session.
- **2026-09-17 05:10**: `paper02` `kernel` variant finished 100 epochs. Best validation Macro AUROC = **0.8716** (learning rate `3e-4`, weight decay `1e-4`).
- **2026-09-17 05:15**: `train_paper01_shared_grid.py` dry-run smoke test verified dynamically computed $16 \times 16$ MMD operator forward/backward pass with zero OOM errors.
