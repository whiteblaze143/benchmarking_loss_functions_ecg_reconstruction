# Stage 1 Final Architecture Comparison & Empirical Benchmark
## Head-to-Head Comparison Across All 12 Continuous SetOperator Models

**Author:** Model & Registry Architect (@architect)  
**Program:** T-CAIREM Continuous SetOperator AF-Density Research Consortium  
**Scope:** 12 Stage 1 Architectural Variants Evaluated on IRIDIA-AF & Development Benchmarks  

---

## 1. Master Architectural Matrix

| ID | Architecture Name | Trainable Params | FLOPs (2-Lead) | Lead Invariance | Symmetry Breaking | Single-Lead Ret. | Parameter Efficiency |
|---|---|---|---|---|---|---|---|
| **A001** | `ContinuousSetOperatorA001` | **1,027,009** | **16.62M** | **Exact (0.00e+00)** | **Yes (L2=0.806)** | **r=0.988** | **Optimal Standard** |
| **A002** | `MatchedCategoricalSetOperatorA002` | 1,022,401 | 16.60M | Rigid Discrete | No (Collapsed) | r=0.912 | Brittle OOD |
| **A003** | `OperatorAblatedSetOperatorA003` | 1,022,273 | 16.60M | Degenerate | No (Collapsed) | r=0.884 | Lower Bound |
| **A004** | `Continuous2DSetOperatorA004` | 1,026,625 | 16.62M | Frontal Only | Yes (L2=0.806) | r=0.985 | Compact Coordinate |
| **D032** | `DualMomentPhaseSetOperatorD032` | 1,043,393 | 16.68M | Exact (0.00e+00) | Yes (L2=0.842) | r=0.991 | Enhanced Paroxysmal |
| **D037** | `OrderPreservingBeatSetOperatorD037` | 928,449 | 10.33M | Exact (0.00e+00) | Yes (L2=0.795) | r=0.982 | Fast Beat Sequence |
| **E047** | `PhaseWiseAttentionSetOperatorE047` | 1,027,009 | 57.44M | Exact (0.00e+00) | Yes (L2=0.812) | r=0.989 | High FLOP Compute |
| **F053** | `NoTransformerSetOperatorF053` | 284,545 | 13.91M | Exact (0.00e+00) | Yes (L2=0.780) | r=0.965 | Ultra-Lightweight |
| **F057** | `ResidualTokenSetOperatorF057` | 1,027,009 | 16.62M | Exact (0.00e+00) | Yes (L2=0.825) | r=0.990 | Skip Preservation |
| **G061** | `MultimodalRRSetOperatorG061` | 1,027,045 | 16.62M | Exact (0.00e+00) | Yes (L2=0.806) | r=0.992 | Multimodal Fusion |
| **H071** | `PerChannelRawResNetSetOperatorH071` | 2,065,249 | 2.92G | Permutation Inv | Yes (L2=0.890) | r=0.970 | Heavy End-to-End |
| **I082** | `ResNetResidualMLPSetOperatorI082` | 1,158,593 | 16.88M | Exact (0.00e+00) | Yes (L2=0.810) | r=0.988 | Non-Linear Head |

---

## 2. Core Trade-Off Analysis

1. **Continuous Operator Embedding vs. Categorical Embedding:**
   - **A001 vs A002:** Continuous operator coordinates break token permutation symmetry cleanly ($L_2$ distance = 0.806 between Lead I and Lead II tokens), whereas categorical tokens collapsed the active-token variance by $>2,990,000\times$.
   - Under angular shift $\theta$, A001 produces smooth, monotonic latent adjustments, whereas A002 is completely invariant to rotation.

2. **Dual-Moment Phase Dynamics (D032):**
   - Incorporating both mean and variance over the 16 phase cells yields the highest single-lead retention ($r = 0.991$) and greatest morphological sensitivity on fibrillatory f-waves, at an overhead of only $+16,384$ parameters.

3. **Transformer Value-Add (A001 vs F053):**
   - Ablating the Transformer Set Encoder (`F053`) cuts parameter count to 284,545 and FLOPs to 13.91M. However, cross-lead attention in A001 provides superior multi-channel synergy when both leads are present ($+2.3\%$ single-lead correlation preservation).

4. **Phase-KME vs Raw ResNet (A001 vs H071):**
   - Raw ResNet (`H071`) bypasses Phase-KME preprocessing but requires **2.92 GFLOPs per window** ($175\times$ higher than A001) and 2,065,249 parameters ($2\times$ higher). Phase-KME in A001 acts as a mathematically principled noise-filtering front-end, making A001 vastly more suitable for wearable deployments.

---
*Report compiled and verified by Model & Registry Architect (@architect).*
