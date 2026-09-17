# Experiment Plan: Paper 05
## The Koopman Operator in Lifted RKHS

### 1. Locked Scientific Variant Hierarchy

| Variant | $K_{\text{phase}}$ | $K_{\text{cycle}}$ | Lifting | Scientific Role |
| :--- | :---: | :---: | :---: | :--- |
| **`full_dual_koopman`** | Yes (RBF) | Yes (RBF) | Soft RBF | **Proposed Method** |
| **`occupancy_only`** | Zeros | Zeros | None | **Primary Static Baseline** |
| **`phase_koopman_only`** | Yes (RBF) | Zeros | Soft RBF | Intra-cycle phase dynamics alone |
| **`cycle_koopman_only`** | Zeros | Yes (RBF) | Soft RBF | Beat-to-beat cycle dynamics alone |
| **`linear_state_dmd`** | Yes (Linear) | Yes (Linear) | Identity/PCA | Lifting ablation (linear DMD on KME state) |
| **`soft_markov`** | Yes (Counts) | Yes (Counts) | Simplex | Inverse-covariance ablation (empirical Markov counts) |
| **`phase_order_shuffled`** | Destroyed | Preserved | Soft RBF | Phase-dynamics destroyer |
| **`beat_order_shuffled`** | Preserved | Destroyed | Soft RBF | Cycle-dynamics destroyer |
| **`global_shuffled`** | Destroyed | Destroyed | Soft RBF | Complete serial chronology null |
| **`operator_only`** | Yes (RBF) | Yes (RBF) | Soft RBF | Operator sufficiency without occupancy |
| **`phase_aware_linear`** | Linear | Linear | — | Classifier complexity probe |
| **`spectral_only`** | Invariants | Invariants | Soft RBF | Secondary spectral sensitivity |

### 2. Hypotheses & Success Criteria
- **$H_{\text{phase}}$**: $\text{AUROC}(\texttt{full\_dual\_koopman}) > \text{AUROC}(\texttt{phase\_order\_shuffled})$ (Phase progression encodes diagnostic information).
- **$H_{\text{cycle}}$**: $\text{AUROC}(\texttt{full\_dual\_koopman}) > \text{AUROC}(\texttt{beat\_order\_shuffled})$ (Cycle progression encodes diagnostic information).
- **$H_{\text{dyn}}$**: $\text{AUROC}(\texttt{full\_dual\_koopman}) > \text{AUROC}(\texttt{occupancy\_only})$ (Dynamics add value beyond static occupancy).
- **$H_{\text{lift}}$**: $\text{AUROC}(\texttt{full\_dual\_koopman}) > \text{AUROC}(\texttt{linear\_state\_dmd})$ (Nonlinear RBF lifting is superior to linear DMD).
- **$H_{\text{cov}}$**: $\text{AUROC}(\texttt{full\_dual\_koopman}) > \text{AUROC}(\texttt{soft\_markov})$ (Inverse-covariance Koopman solve is superior to empirical transition counts).

### 3. Hyperparameter Sweep Grid
- Learning rates: $\{1\times 10^{-4}, 3\times 10^{-4}, 1\times 10^{-3}\}$
- Weight decays: $\{1\times 10^{-5}, 1\times 10^{-4}, 1\times 10^{-3}\}$
- Batch size: 2048, Max epochs: 100, Patience: 10, Seed: 42
- Metric: Multilabel macro AUROC, macro AUPRC, macro F1 on PTB-XL validation set.\n