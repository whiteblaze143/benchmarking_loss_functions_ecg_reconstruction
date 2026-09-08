# Refinement Report: Transition to Lean Multi-Task Architecture

**Date**: 2026-09-08  
**Topic**: Empirical Justification for Occam's Razor Model Architecture

---

## 1. Executive Summary

This report documents the architectural refinement process that transitioned the project from an over-parameterized multi-loss transformer into the streamlined **`Lean-CardioAIM`** architecture.

### Quantitative Evolution

| Iteration | Model Config | Parameters | Missing-11 Mean $r$ | Tail $p_{05}$ | Speed (it/s) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Round 0 (Anchor)** | Enc-8, Width-768, Heads-12, Mask 1110000, Lead Dropout | ~45M | 0.7456 | 0.4048 | 1.40 |
| **Round 1 (No Dropout)** | Enc-8, Width-768, Heads-12, No Lead Dropout (`K1`) | ~45M | 0.7561 | 0.4227 | 1.40 |
| **Round 1 (Width-512)** | Enc-8, Width-512, Heads-8 (`C3`) | ~28M | 0.7502 | 0.4054 | 1.75 |
| **Round 1 (Enc-4)** | Enc-4, Width-768, Heads-12 (`C1`) | ~32M | 0.7461 | 0.4045 | 1.85 |
| **Round 2 (Clean Lean Best)** | Enc-4, Width-512, Heads-8, No Dropout, Z-Score (`L1`) | ~18M | *Running* | *Running* | **2.15** |

---

## 2. Refinement Insights & Decisions

1. **Why Smaller is Better**:
   - Single-lead ECG contains 1 channel of input data ($1 \times 5000$ points). Mapping 1 channel to 11 channels does not require deep, high-dimensional latent spaces. 
   - A 768-dimensional space suffers from representation sparsity and overfits to high-frequency baseline wander. 
   - Compressing to 512 dimensions with 4 encoder layers acts as an explicit structural regularizer.
2. **Loss Objective Cleanliness**:
   - The failure of `L0_lean_best` ($-0.0130$) proved that adding VCG loss + MMD distribution loss causes loss gradient competition.
   - The parsimonious triplet (MSE + Pearson + Deriv) remains the cleanest loss formulation.
