# Experiment Tracker: Paper 06
## Execution and Verification Checklist

- [x] **Adversarial Prior Art Audit**: Grounded against VCG dipole modeling (Frank, 1956; Kors, 1990) and conditional MMD (Fukumizu, 2008).
- [x] **Ontology Fortification**: Renamed to dominant rank-3 spatial macrostate and non-macro residual.
- [x] **Physical mV Coordinates**: Ensured spatial decomposition operates in physical mV.
- [x] **5D Orthonormal Complement**: Formulated $V_\perp \in \mathbb{R}^{8 \times 5}$.
- [x] **Anchor-Conditioned MMD**: Formally specified AC-MMD estimator $\omega_{ghk}$.
- [x] **Effective-Overlap Ineligibility Audit**: Analyzed 35.8% exclusion and planned selection bias audit.
- [x] **Matched Comparison Hierarchy**: Structured $[R_{\text{macro}}, R_{\text{cond-res}}]$ vs $R_{\text{macro}}$ and vs $[R_{\text{macro}}, R_{\text{marg-res}}]$.
- [x] **Synthetic Proof Worlds**: 8 proof worlds designed in `test_paper06_synthetic_recovery.py`.
- [ ] **Codebase Enhancements**:
  - [ ] `src/repecg/paper06_conditional/residual.py`
  - [ ] `src/repecg/paper06_conditional/__init__.py`
  - [ ] `src/repecg/common/models.py`
  - [ ] `src/repecg/common/variants.py`
  - [ ] `scripts/paper06/run_paper06_grid.sh`
- [ ] **Synthetic Test Suite Execution**: Run `pytest tests/test_paper06_synthetic_recovery.py -v`.
- [ ] **Cross-Paper Regression Tests**: Run all paper models and mechanisms.
- [ ] **1-Epoch GPU Smoke Test**: Execute on PTB-XL development data.\n