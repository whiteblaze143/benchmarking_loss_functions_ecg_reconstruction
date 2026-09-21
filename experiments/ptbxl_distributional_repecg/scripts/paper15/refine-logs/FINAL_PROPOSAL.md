# Final Proposal: Paper 15 — Modular Phase-Transition Dynamics in ECG Representation Space (P15-V2)

## Problem Anchor
Cardiac electrophysiology traverses 16 distinct physiological phases per cycle (atrial depolarization, ventricular rapid depolarization, early repolarization, plateau, terminal repolarization). Existing models either model the sequence as an entangled autoregressive black box or collapse the cycle into a static representation, failing to isolate phase-specific transition laws from patient-specific voltage states.

## Core Thesis
We investigate **phase-specific transition modularity in a fixed ECG representation space**. The periodic cardiac cycle is factorized into 16 cyclic transition operators:
$$\hat{Z}_{(g+1)\bmod 16} = Z_g + M_g(Z_g), \quad g \in \mathbb{Z}_{16}$$
where the final transition $15 \to 0$ closes the periodic cardiac cycle.

Autonomy is formalized not as causal independence from state (since $M_g$ explicitly takes $Z_g$ as input), but as **stability of the functional transition law $M_g$ under changes in the distribution of its input state $P(Z_g)$**, provided the underlying electrodynamic transition law itself remains unchanged.

Disentangling the fixed coordinate state $Z$ from the transition operators $M_g$ allows:
1. Distinct, heterogeneous phase dynamics (e.g. rapid ventricular depolarization vs slow repolarization) to be learned without mutual parameter interference.
2. Stability under input-state distribution shifts (e.g. baseline voltage shift, heart rate variation), enabling robust transfer.
3. Diagnostic probing across a 4-tier representational hierarchy:
   $$\text{State } Z \rightarrow \text{Observed Dynamics } \Delta \rightarrow \text{Predicted Transitions } \hat{\Delta} \rightarrow \text{Transition Innovations } R$$
   directly determining where pathological clinical information resides.

## Embedded Inductive Biases & Architectural Contracts
1. **Fixed Coordinate Grounding**: Latent states $Z_g \in \mathbb{R}^{64}$ are extracted via a train-only fitted and frozen representation mapping ($X_g^{\text{PhaseKME}} \to Z_g$), eliminating latent gauge/collapse issues ($Z_g \to c_g$).
2. **16-Transition Periodic Cyclic Ring**: All 16 transitions $g \in \mathbb{Z}_{16}$ are modeled, preserving the cyclic topology of the heartbeat without arbitrary cut points.
3. **Strong Shared Controls**: Benchmarked against:
   - `shared_phase_conditioned`: A shared MLP $M_{\text{cond}}(Z_g, e_g)$ provided with one-hot phase identity.
   - `shared_capacity_matched`: A shared MLP matched in parameter budget to the 16 modular networks ($\Delta < 0.1\%$).
   - `shared_same_width`: A shared MLP with identical per-transition hidden width.
4. **Post-Training Frozen Derangement Kill Test**: Freezing the trained modular model and evaluating under fixed derangement $\pi(g) = (g+8)\bmod 16$ without retraining demonstrates that operators have specialized to phase-specific dynamics ($L_{\text{perm}} \gg L_{\text{ordered}}$).
5. **Simple-Transition Baselines**: Non-linear modular operators must strictly outperform Identity ($\hat{Z}=Z_g$), Phase Mean ($\hat{Z}=\mu_{g+1}$), Mean Residual ($\hat{Z}=Z_g+\bar{\Delta}_g$), and Linear Phase-Specific baselines.
6. **Stagewise Training**: Primary mechanism learning is purely label-free ($\mathcal{L}_{\text{trans}}$ on folds 1–6). Disease classification is conducted via frozen linear diagnostic probes on the resulting representation hierarchy ($Z, \Delta, \hat{\Delta}, R$).

## Rejected Complexity
We reject overclaiming "Independent Causal Mechanisms" without structural DAG interventions; we reject joint BCE+transition optimization as a primary mechanism which permits classification labels to artificially deform transition dynamics; and we reject unconstrained learnable encoders that lack information-preservation constraints.
