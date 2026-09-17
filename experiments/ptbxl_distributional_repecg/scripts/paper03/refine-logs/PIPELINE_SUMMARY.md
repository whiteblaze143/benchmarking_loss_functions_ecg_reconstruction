# Pipeline Summary: Paper 03
# Phase-Path-Signature: Distributional Path Signatures on Cardiac Phase Trajectories

## Executive Summary
Paper 03 establishes a rigorous mathematical and clinical bridge between **Rough Path Theory** and **Cardiac Electrophysiology**. By mapping multilead continuous voltage loops within 16 cardiac phase cells into their free-Lie-algebra log-signatures, whitening via truncated PCA, and nonparametrically embedding the beat ensemble via a Nyström Kernel Mean Embedding, Paper 03 represents the multibeat ECG as a continuous $C_{16}$-periodic distribution field:
$$\theta \longmapsto \mu_{P(\operatorname{LogSig}(\gamma_{b, \theta}))} \in \mathcal{H}_{k_{\text{sig}}}$$

## Architectural & Invariance Guarantees
1. **Chen's Identity & Reparameterization Invariance**: The pure geometric descriptor $d^{\text{sig}} = [\gamma(0), \gamma(1), \operatorname{LogSig}_3(\gamma)]$ is strictly invariant to non-linear monotone intra-cell time warping $\phi(t)$.
2. **Trajectory Chirality & Orientation**: Reversing the trajectory time arrow inverts the sign of the level-2 antisymmetric Lévy area ($A^{i,j}(\gamma^{\leftarrow}) = -A^{i,j}(\gamma)$).
3. **Internal Path Ordering**: Scrambling internal samples while conserving point marginals completely destroys the log-signature ($>50\%$ distance), isolating path geometry from point cloud distributions.
4. **Cyclic Group Equivariance**: The sequence of 16 phase-cell embeddings is processed by `PhaseCNN` with circular residual blocks, guaranteeing discrete $C_{16}$ shift equivariance.
