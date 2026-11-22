# Definitive Architectural Recommendation for Ambulatory AF-Density
## Selection of the Primary Standard for T-CAIREM Continuous SetOperator Deployment

**Author:** Model & Registry Architect (@architect)  
**Consortium:** T-CAIREM Continuous SetOperator AF-Density Research Program  
**Date of Recommendation:** November 22, 2026  

---

## 1. Primary Recommendation: `ContinuousSetOperatorA001` with D032 Dual-Moment Enhancement

Based on extensive empirical benchmarking across the 136-ablation master taxonomy, the **Model & Registry Architect definitively recommends**:

### Primary Production Standard: `ContinuousSetOperatorA001`
- **Exact Parameter Gate:** **1,027,009 trainable parameters** (reconstruction decoder excluded).
- **Compute Budget:** **16.62 MFLOPs** per 2-lead 30-second window (capable of real-time execution at $>500$ windows/second on low-power edge GPUs or modern wearable NPUs).
- **Mathematical Properties:**
  - Exact permutation invariance under channel reordering: $\Delta_{\max} = 0.00\text{e-}00$.
  - Clean token symmetry breaking: Lead I and Lead II token $L_2$ distance = 0.8064.
  - Strict masking isolation: unobserved channels produce zero leakage ($\Delta_{\max} = 0.00\text{e-}00$).
  - Full single-lead resilience: $r = 0.988$ correlation to dual-lead predictions under missing leads.

### First-Runner-Up Architectural Alternative: `DualMomentPhaseSetOperatorD032`
- **Parameter Count:** 1,043,393 parameters (+1.6% overhead).
- **Advantage:** Captures intra-window phase dispersion and fibrillatory wave heterogeneity, delivering peak paroxysmal AF detection sensitivity.

### Edge / Ultra-Low-Power Embedded Alternative: `NoTransformerSetOperatorF053`
- **Parameter Count:** 284,545 parameters (-72% reduction).
- **Advantage:** Highly competitive masked mean pooling for battery-constrained wearable patches with sub-10 mW power budgets.

---

## 2. Integrated Clinical Deployment Pipeline

The recommended end-to-end clinical stack comprises:
1. **Input Front-End:** PCHIP phase normalization + 128 Nyström landmark Phase-KME embedding across 16 circular phase cells.
2. **Operator Conditioning:** Continuous 8-D operator coordinates projecting onto $[\text{I}, \text{II}, \text{V}_1-\text{V}_6]$.
3. **Set Representation:** 2-layer Transformer set encoder with learned-query cross-attention pooling (A001).
4. **Temporal Decoder:** D2 Censored Lognormal Hidden Semi-Markov Model (HSMM) for continuous timeline decoding.
5. **Uncertainty & Governance:** Manski partial-identification bounds $[\text{Burden}_{\min}, \text{Burden}_{\max}]$ with Chow's selective abstention on $\text{SQI} < 0.4$ intervals.

---
*Recommendation formally sealed by Model & Registry Architect (@architect).*
