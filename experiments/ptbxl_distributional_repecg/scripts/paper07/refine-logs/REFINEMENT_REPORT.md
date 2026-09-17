# Refinement Report: Paper 07 — Continuous Lead-Span Functional ECG

## Systematic Modifications Table

| Component | Pre-Refinement Status | Production-Locked Revision | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Object Ontology** | Arbitrary physical lead on $S^7$ | Continuous lead-span measurement functional $\ell_q(x) = q^\top x$ | Acknowledges that physical leads outside the span require multi-position data. |
| **Polarity Symmetry** | Approximate KL loss on $f(q)$ vs $f(-q)$ | Exact structural $\mathbb{Z}_2$ projective atom $g_q = [q x_q, q \dot{x}_q]$ | Resolves non-odd Phase-KME bug; enforces invariance by construction on $\mathbb{RP}^7$. |
| **Benchmark Suite** | Binary `continuous` vs `categorical` (strawman UNK) | 10-variant hierarchy: `nearest_known_operator`, `q_ablated`, `analytic_pinv`, `LMMSE`, `GraphECG` | Prevents trivial victory over deliberately crippled baseline. |
| **Reconstructive Claim** | "Reconstructive inversion of 8-D field" | "Auxiliary operator-response prediction" governed by $\operatorname{rank}(Q L) = 3$ | Prevents mathematically false claim that $m < 8$ inverts generic 8-D space. |
| **Operator Separation** | Signed spherical distance | Projective distance $d_{\mathbb{RP}}(q, p) = \arccos |q^\top p| \ge \delta$ | Respects unoriented nature of measurement axes ($q \sim -q$). |
| **Units** | Standardized per-lead voltages | Strict physical mV prior to projection $x_q = q^\top x$ | Prevents lead variance scaling from distorting spatial projection angles. |
