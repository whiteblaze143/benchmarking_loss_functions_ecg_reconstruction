# Experiment Plan: Paper 15 — Modular Phase-Transition Dynamics (P15-V2)

## Objective
Benchmark phase-specific transition modularity against strong shared controls, linear models, and simple baselines across the 16 cyclic transitions of the cardiac cycle, and quantify diagnostic information across the representational hierarchy ($Z, \Delta, \hat{\Delta}, R$).

## Variants to Test
- **full (modular)**: 16 separate transition mechanisms $M_g: Z_g \rightarrow Z_{(g+1)\bmod 16}$.
- **shared_phase_conditioned**: 1 shared MLP receiving $[Z_g, e_g]$ with one-hot phase indicator.
- **capacity_matched_shared**: 1 shared MLP with matched parameter capacity ($\Delta < 0.1\%$).
- **shared_same_width**: 1 shared MLP with identical per-transition width.
- **linear_modular**: 16 separate linear transition maps $A_g Z_g + b_g$.
- **linear_probe**: Linear baseline on phase representations.

## Audited 8-Gate Inductive Bias Stack (All PASSED)

1. **P15-G0: State Grounding & Cyclic Ring Contract**
   - Grounding across all 16 phases; isolation $\Delta_{\text{others}} = 0.0$; cyclic ring closure $15 \to 0$; active gradient flow across all 16 modules.
   - **Result**: PASSED.
2. **P15-G1: Transition Law Recovery**
   - Accurate recovery of known synthetic phase-specific transition laws ($R^2 = 0.8863$, MSE $= 0.1137$).
   - **Result**: PASSED.
3. **P15-G2: Input-Distribution Autonomy (State-Shift Transfer vs Mechanism-Shift Failure)**
   - Transfer under state distribution shift ($P_A(Z) \to P_B(Z)$): transfer ratio = **0.76** (loss remains low).
   - Failure under true mechanism shift ($M_A \to M_C$): degradation ratio = **7.66x**.
   - **Result**: PASSED.
4. **P15-G3: Post-Training Frozen Permutation Kill Test**
   - Freezing the trained modular model and evaluating under cyclic derangement $\pi(g) = (g+8)\bmod 16$ degrades transition MSE by **5.06x** (ordered: 0.1202 vs permuted: 0.6082).
   - **Result**: PASSED.
5. **P15-G4: Strong Shared Controls Benchmark**
   - Modular (loss: 0.1230) beats phase-agnostic shared same-width (0.3043) and capacity-matched shared (0.3016).
   - Capacity matching delta = **0.0877%** ($< 1.0\%$).
   - **Result**: PASSED.
6. **P15-G5: Cyclic Origin Invariance**
   - Consistency across phase origins $0, 4, 8, 12$: maximum relative deviation is **0.22%** ($< 10\%$).
   - **Result**: PASSED.
7. **P15-G6: Information Preservation & Non-Collapse Audit**
   - Effective rank $r_{\text{eff}} = 26.93 \ge 8.0$; minimum phase variance = 0.9827; zero phase collapse.
   - **Result**: PASSED.
8. **P15-G7: Simple-Transition Baselines Benchmark**
   - Modular nonlinear model (0.0920) strictly beats Identity (7.4752; 98.77% improvement), Phase Mean (24.6452), Mean Residual (6.2347), and Linear modular (0.1024; 10.21% improvement).
   - **Result**: PASSED.

## Execution Strategy
- Inductive bias verification: `scripts/paper15/run_paper15_gates.py` (8/8 PASS).
- Unit tests: `tests/test_paper15_causal_factorization.py` (8/8 PASS).
- Real-data grid execution: `scripts/paper15/run_paper15_grid.sh` across all 6 variants with transition evaluations and simple baselines logged in cell summaries.
