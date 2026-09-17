# Pipeline Summary: Paper 05
## The Koopman Operator in Lifted RKHS

### Executive Summary
Paper 05 establishes an operator-theoretic formulation of cardiac electrophysiology:
1. Cardiac states are distribution-valued representations $P_{b,g}$ in a reproducing kernel Hilbert space $\mathcal{H}_k$, approximated in finite KME coordinates $\hat{\mu}_{b,g} \in \mathbb{R}^{128}$.
2. Observables $\psi(\hat{\mu}) \in \Delta^{31}$ define soft cluster memberships over global anchors.
3. Decoupled Koopman operators capture:
   - **Intra-cycle phase progression** ($K_{\text{phase}}$): how states evolve within a single cardiac cycle.
   - **Beat-to-beat cycle dynamics** ($K_{\text{cycle}}$): how matched cardiac phases change across successive heartbeats.
4. Estimators use normalized sample moments and scale-relative ridge penalties, ensuring mathematical invariance to record duration and beat counts.
5. The model directly falsifies the static hypothesis: proving whether transition laws add diagnostic value beyond static distribution occupancy.\n