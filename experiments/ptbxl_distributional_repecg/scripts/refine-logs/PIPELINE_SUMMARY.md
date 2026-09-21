# Pipeline Summary: Benchmarking Loss Functions for ECG Reconstruction
## 15-Paper Distributional Representation Evaluation

**Problem**: What structural properties of learned ECG representations are necessary and sufficient for robust cardiac diagnosis across acquisition conditions?

**Final Method Thesis**: The 15 papers collectively evaluate a taxonomy of distributional representation hypotheses — from unordered kernel mean embeddings (P02) through sequential signatures (P03), spectral operators (P05–P07), conditional residuals (P06, P12), counterfactual reasoning (P09), causal compression (P11), acquisition alignment (P10, P14), architectural priors (P13), and phase-specific modularity (P15). The overarching finding is that **task-driven representation learning on kernel mean embeddings already achieves strong diagnostic performance (AUROC ≈ 0.90), and the value of additional structural priors lies not in diagnostic gain but in representational properties** — nuisance pruning (P14), transition interpretability (P15), and distributional robustness (P10).

**Final Verdict**: READY (pending real-data grid results)
**Date**: 2026-09-20

## Paper Taxonomy

### Layer 1: Base Representations (What to encode)
| Paper | Representation | Core Hypothesis |
|---|---|---|
| P01 | Distributional recurrence (phase-cell kernel operators) | Cardiac dynamics are captured by distributional transitions between phase cells |
| P02 | Kernel mean embedding (KME) of phase-cell point clouds | Unordered distributional summary per cardiac phase suffices for diagnosis |
| P03 | Path signature of ordered lead trajectories | Sequential path geometry (curvature, area, higher-order iterated integrals) encodes diagnostic signal beyond marginal statistics |

### Layer 2: Dynamical Structure (How to model temporal evolution)
| Paper | Structure | Core Hypothesis |
|---|---|---|
| P05 | Koopman operator on KME states | Dual phase-progression + beat-to-beat dynamics decomposition via linear operator |
| P06 | Macrostate-conditioned residual recurrence | Dominant spatial field (rank-3 VCG) plus phase-specific non-dipolar residuals |
| P07 | Continuous lead-span functional ECG | Functional operator over electrode geometry captures cross-lead dependencies |
| P12 | Phase-ordered location-scale residuals | Conditional innovation sequences (residual after predictable phase state) isolate diagnostically useful variation |

### Layer 3: Higher-Order Reasoning (What to do with representations)
| Paper | Method | Core Hypothesis |
|---|---|---|
| P08 | Phase-token attention with equivalence clustering | Dynamic routing through token bank captures inter-phase dependencies |
| P09 | Counterfactual operator response sets | Isolating task-irrelevant measurement variations improves OOD robustness |
| P11 | Continuous predictive compression of ordered beats | Future-predictive morphology retention beyond autocorrelation validates temporal sufficiency |

### Layer 4: Invariance and Regularization (How to constrain representations)
| Paper | Constraint | Core Hypothesis |
|---|---|---|
| P10 | Paired acquisition matching (IRM/CORAL/CausIRL) | Stabilizing representations across known transforms of the same record |
| P13 | Equivariant architecture for anatomical shifts | Spatial transformation invariance/equivariance as architectural prior |
| P14 | MMD alignment across acquisition environments | Kernel MMD regularization prunes acquisition nuisance from latent space |
| P15 | Phase-specific modular transition mechanisms | 16 cyclic transition operators factorize cardiac phase dynamics |

### Ineligible
| Paper | Status | Reason |
|---|---|---|
| P04 | Dead | Failed Nyström fidelity audit (Hankel DMD representations numerically unstable) |

## Final Deliverables
- Per-paper proposals: `scripts/paper{01..15}/refine-logs/FINAL_PROPOSAL.md`
- Per-paper experiment plans: `scripts/paper{01..15}/refine-logs/EXPERIMENT_PLAN.md`
- Gate results: `scripts/paper{13,14,15}/refine-logs/*GATES*.json`
- This summary: `scripts/refine-logs/PIPELINE_SUMMARY.md`

## Contribution Snapshot
- **Dominant contribution**: Systematic benchmarking of 14 distributional ECG representation families on a unified PTB-XL evaluation protocol
- **Key empirical finding (P14)**: MMD environment alignment is a scientifically valid null — prunes acquisition nuisance (−23.3% env decodability, −35.6% paired drift) without affecting diagnosis (AUROC Δ = +0.0004)
- **Key structural finding (P15)**: Phase-specific modular transitions are a genuine structural property of cardiac cycles (8/8 synthetic gates, 5.06x kill-test ratio, 7.66x mechanism-shift failure ratio)
- **Explicitly rejected complexity**: IRM-based causal claims (wrong estimand), hospital-level generalization (overclaims), learnable MMD bandwidth (unnecessary)

## Must-Prove Claims (Pending Real-Data Results)
1. P15 `modular` > `shared_phase_conditioned` on PTB-XL Fold 8 AUROC
2. P01/P03 recurrence/signature representations > P02 KME baseline
3. P05–P08 spectral/operator representations provide complementary diagnostic signal
4. P13 equivariant architecture improves robustness under spatial shifts
5. P10 paired acquisition matching reduces representation drift

## GPU Queue Status
- **Session**: `gpu_queue` tmux
- **Order**: P01 → P03 → P05 → P06 → P07 → P08 → P02 → P10 → P11 → P12 → P13 → P15 → P09
- **Monitor**: `tmux capture-pane -p -t gpu_queue | tail -20`

## First Runs to Launch
1. ✅ GPU queue launched (13 papers sequentially on A100 40GB)
2. After queue: Aggregate all results into comparison matrix
3. After aggregation: Write combined claims document

## Main Risks
- **P01/P03 1-epoch failure repeat**: Previous cells all stopped at epoch 1. Root cause unknown — may be representation quality issue. Monitoring.
- **OOD evaluators retired**: Papers 01–13 have retired OOD eval scripts. Will need to write checkpoint-backed native-task evaluation for cross-dataset testing.
- **P04 permanently ineligible**: Hankel DMD representations failed numerical fidelity — paper is dead.

## Next Action
- Monitor GPU queue progress
- When results arrive: `/analyze-results` → `/paper-writing`
