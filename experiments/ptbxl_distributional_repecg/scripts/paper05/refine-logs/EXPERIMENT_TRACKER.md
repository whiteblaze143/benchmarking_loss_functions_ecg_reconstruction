# Experiment Tracker: Paper 05
## Execution and Verification Checklist

- [x] **Adversarial Prior Art Audit**: Grounded against Ghosh & Monfared (2026) and HAVOK (2017).
- [x] **Mathematical Ontology**: Clarified Koopman on observables of distribution-valued KME states.
- [x] **Dual Operator Decoupling**: Separated $K_{\text{phase}}$ from $K_{\text{cycle}}$.
- [x] **Normalized Moment Estimator**: Implemented $G = \frac{1}{M} Z_- Z_-^\top, A = \frac{1}{M} Z_+ Z_-^\top$.
- [x] **Scale-Relative Regularization**: Locked $\lambda = \max(\alpha \frac{\operatorname{Tr}(G)}{d}, 10^{-6})$.
- [x] **Baselines Added**: `linear_state_dmd` and `soft_markov`.
- [x] **Multi-Seed Shuffles**: Structured 3 destruction controls with $S=5$ independent seeds.
- [x] **Analytical Independence Null**: Implemented $K_{\text{iid}}$ and $K_{\text{excess}}$.
- [x] **Synthetic Recovery Suite**: 9 worlds designed in `tests/test_paper05_synthetic_recovery.py`.
- [x] **External Oracle Verification**: Cross-validated with PyDMD EDMD.
- [ ] **Codebase Updates**:
  - [ ] `src/repecg/paper05_koopman/operator.py`
  - [ ] `src/repecg/common/models.py`
  - [ ] `src/repecg/common/variants.py`
  - [ ] `scripts/paper05/run_paper05_grid.sh`
- [ ] **Synthetic Test Suite Execution**: Run `pytest tests/test_paper05_synthetic_recovery.py -v`.
- [ ] **Cross-Paper Regression Tests**: Run all model and mechanism tests.
- [ ] **1-Epoch GPU Smoke Test**: Execute on PTB-XL development representations.\n