# Refinement Report: Paper 07 — Continuous Measurement-Operator ECG

## Audit Findings & Revisions Table

| Component | Pre-Refinement Status | Production-Locked Revision | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Operator Space** | Ambiguous parameterization | Strict unit sphere $\mathbb{S}^7 \subset \mathbb{R}^8$ ($||q||_2 = 1$) | Enforces scale-invariant projection geometry. |
| **Response Atom** | Raw voltages | Phase-space derivative atom $[x_q, \dot{x}_q]$ | Preserves velocity and phase trajectory geometry. |
| **Orientation Law** | Implicit | Explicit paired negation training + symmetric KL loss | Enforces physical law $x_{-q} = -x_q$ without diagnostic disruption. |
| **Target Holdout** | Unverified | Strict exclusion guarantee: target $t \notin C_{\text{context}}$ | Prevents label/response leakage during reconstructive inversion. |
| **Categorical Control** | Separate script | Fully capacity-matched paired architecture | Ensures performance gap reflects operator inductive bias, not parameter count. |
| **OOD Evaluation** | Unsafe runtime script | Fail-closed retirement with frozen test banks | Guarantees test bank integrity and leakage prevention. |
