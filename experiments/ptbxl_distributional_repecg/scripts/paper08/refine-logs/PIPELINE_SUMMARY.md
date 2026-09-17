# Pipeline Summary: Paper 08 — Phase-Token Attention & Dynamic Routing vs Equivalence Clustering

## Executive Summary
Paper 08 investigates the mechanisms of cross-phase information routing in the distributional repECG framework:
1. **Dynamic vs Static Routing**: Quantifies whether patient-specific dynamic attention weights outperform static learned routing matrices.
2. **Local vs Long-Range Cross-Phase Interaction**: Measures the contribution of long-range cardiac phase attention compared to local cyclic conduction neighborhoods ($w=2$ on $C_{16}$).
3. **Phase-Order Sensitivity**: Evaluates positional embedding dependence through phase-scrambling and phase-agnostic set controls.
4. **Continuous vs Discrete Representation**: Compares continuous RKHS embeddings against 64-cluster KMeans codebooks and statistically verified equivalence classes (`equivalence.py`).

## Deliverables
- `src/repecg/paper08_tokens/model.py`: Modular `PhaseTokenTransformer` with full mask, static routing, and attention extraction support.
- `src/repecg/paper08_tokens/equivalence.py`: Statistical bootstrap bounds and complete-linkage clustering.
- `tests/test_paper08_synthetic_recovery.py`: 8 mathematically verified synthetic proof worlds.
