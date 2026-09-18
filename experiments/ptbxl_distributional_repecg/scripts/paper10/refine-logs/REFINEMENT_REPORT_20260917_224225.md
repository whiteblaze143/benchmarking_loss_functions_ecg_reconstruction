# Refinement Report: Paper 10

**Initial approach:** a generic information bottleneck / hospital invariance story.

**Refined approach:** a paired, controlled acquisition-intervention stability test.

**Verdict:** RETHINK

## Evidence Driving the Change

- Environment IDs are absent from the current representation artifacts and trainer.
- The existing objective is BCE only; named IRM, CORAL, and CausIRL variants do not execute distinct losses.
- Site membership cannot isolate hardware from population and prevalence differences.

## Required Before Any Production Launch

1. Versioned raw-waveform paired transform builder with fold/provenance contracts.
2. Distinct tested ERM, IRMv1, CORAL, and paired objectives.
3. Synthetic identifiability, wrong-pair, and collapse tests.
4. Fold-7-only selection and patient-equal robust evaluation.
