# Refinement Report: Paper 10

**Initial approach:** a generic information bottleneck / hospital invariance story.

**Refined approach:** paired acquisition matching with separate stability, robustness, shortcut-resistance, and impossibility tests.

**Verdict:** RETHINK

## Evidence Driving the Change

- Environment IDs are absent from the current representation artifacts and trainer.
- The existing objective is BCE only; named IRM, CORAL, and CausIRL variants do not execute distinct losses.
- Site membership cannot isolate hardware from population and prevalence differences.

## Required Before Any Production Launch

1. Versioned raw-waveform paired transform builder with fold/provenance contracts.
2. Intervention-admissibility and preprocessing-survival audit; semantic label copying is not evidence of preserved diagnostic information.
3. Four synthetic worlds: independent environment, train-only spurious shortcut, genuinely causal environment, and destructive transform.
4. Distinct tested ERM, IRMv1, CORAL, auxiliary-only, pair-only, and paired objectives, including gradient and parameter-update tests.
5. Augmentation-matched ERM, ten surgical mismatch permutations, norm-leakage probes, and patient-equal coverage-aware evaluation.
