# Experiment Tracker: Paper 08

## Milestone Checklist

- [x] **Phase 1: Literature Audit & Prior Art Demarcation**
  - Demarcate against Beat-Synchronous Transformers (2025) and HeartLLM (2025/2026).
  - Formulate Phase-Token Routing vs Local/Discrete Equivalence.
- [x] **Phase 2: Mathematical Formulation & Architecture Design**
  - Define cyclic banded attention mask on $C_{16}$.
  - Define static routing and uniform attention controls.
  - Define continuous vs discrete equivalence clustering.
- [x] **Phase 3: Implementation & Module Refactoring**
  - Implement `PhaseTokenTransformer` in `src/repecg/paper08_tokens/model.py`.
  - Export in `src/repecg/paper08_tokens/__init__.py`.
  - Update `src/repecg/common/models.py` and `variants.py`.
- [x] **Phase 4: Synthetic Proof Suite**
  - Build `tests/test_paper08_synthetic_recovery.py` with 8 proof worlds.
  - Run and verify 8/8 tests pass.
- [x] **Phase 5: Cross-Paper Regression Verification**
  - Run full regression suite across Papers 01–08.
- [x] **Phase 6: GPU Smoke Training Verification**
  - Run 1-epoch smoke test in persistent `tmux` session.
