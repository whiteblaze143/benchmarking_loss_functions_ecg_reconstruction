# Pipeline Summary: Paper 01 (Distributional Recurrence Operator)

**Problem**: Prior ECG recurrence methods compare individual trajectory states or scalar samples. How can we construct recurrence between local distributions of electrical states across beats to capture non-linear cardiac dynamics while remaining robust to morphological jitter and lead sparsity?  
**Final Method Thesis**: Distributional recurrence between phase-conditioned ECG state distributions using Maximum Mean Discrepancy (MMD) in RKHS: $R_{gh} = \exp\left(-\frac{\operatorname{MMD}^2(P_g, P_h)}{\tau}\right)$ captures beat-to-beat variability and spatial loop dynamics, demoting conventional Euclidean point recurrence to a baseline.  
**Final Verdict**: `MAJOR REVISION — NOT YET PRODUCTION READY`  
**Date**: 2026-09-17  

---

## Final Deliverables
- Proposal: `experiments/ptbxl_distributional_repecg/scripts/paper01/refine-logs/FINAL_PROPOSAL.md`
- Review summary: `experiments/ptbxl_distributional_repecg/scripts/paper01/refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `experiments/ptbxl_distributional_repecg/scripts/paper01/refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `experiments/ptbxl_distributional_repecg/scripts/paper01/refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `experiments/ptbxl_distributional_repecg/scripts/paper01/refine-logs/EXPERIMENT_TRACKER.md`

---

## Contribution Snapshot
- **Dominant Contribution**: Distributional MMD Recurrence Operator $R_{gh} = \exp(-\operatorname{MMD}^2(P_g, P_h)/\tau)$ over empirical beat distributions $P_g$, processed by a 2D circular convolutional network.
- **Novelty vs Prior Art**: Explicit separation from existing Euclidean recurrence-plot CNNs (Mathunjwa et al.) by operating on non-parametric distributions rather than point vectors.
- **Explicitly Rejected Complexity**: Rejected fixed-threshold binary recurrence plots, handcrafted RQA feature engineering, and intra-record variance normalization as an unverified default.

---

## Must-Prove Claims
1. **Distributional Superiority**: MMD recurrence separates multi-modal phase distributions where mean-distance and Euclidean recurrence fail (Synthetic World A).
2. **Clinical Non-Inferiority**: Achieves predictive noninferiority ($\Delta \operatorname{AUROC} > -0.01$) compared to SOTA 1D raw waveform models (~0.928–0.930) on PTB-XL official benchmark folds.
3. **Sparse-Lead Advantage (GraphECG Protocol)**: $\Delta_{\text{MMD-vs-Euclid}}(k)$ widens as lead count $k$ decreases ($12 \to 6 \to 3 \to 1$), proving distributional recurrence provides superior inductive bias under observation sparsity.
4. **Split-Half Reproducibility**: Odd-even beat stability (NFS) of MMD recurrence exceeds Euclidean recurrence by $>15\%$.
5. **Scale Disentanglement Synergy**: Factorized $[R_{\text{shape}}, a]$ outperforms naive amplitude removal.

---

## Main Risks & Pre-Production Action Items
1. **Implementation Upgrade**: The representation generator must be updated from single-beat Euclidean distances to MMD $U$-statistics across beats.
2. **Synthetic Proofs**: Execute Synthetic Worlds A–D to verify distribution sensitivity and cyclic equivariance.
3. **Prior-Art Replications**: Implement RQA/XGBoost and ResNet-18 controls.
4. **Official Splits**: Restructure training scripts to train on Folds 1–8, validate on Fold 9, and test on Fold 10.

---

## Next Action
Paper 01's research plan is now rigorously anchored in representation theory and literature prior art. We will proceed to update the Paper 01 implementation code according to this plan before triggering the production sweep.
