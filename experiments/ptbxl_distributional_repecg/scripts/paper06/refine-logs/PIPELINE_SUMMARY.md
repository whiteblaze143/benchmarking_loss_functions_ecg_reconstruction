# Pipeline Summary: Paper 06
## Macrostate-Conditioned Residual Recurrence for ECG

### Executive Summary
Paper 06 provides a mathematically grounded, clinically defensible framework for evaluating whether ECG residual distributions contain diagnostic information beyond the dominant low-rank spatial cardiac field:
1. Decomposes physical mV voltages into a dominant rank-3 spatial macrostate $z(t)$ and an explicit 5D orthonormal residual complement $u(t)$.
2. Partitions the 3D macrostate space into soft anchor clusters.
3. Evaluates Anchor-Conditioned MMD (AC-MMD) between cardiac phase sectors conditioned on shared macrostates.
4. Identifies the effective-overlap ineligibility cliff as an estimator constraint across disparate cardiac phases, performing a selection bias audit.
5. Employs matched information controls ($[R_{\text{macro}}, R_{\text{cond-res}}]$ vs $[R_{\text{macro}}, R_{\text{marg-res}}]$ and vs $R_{\text{macro}}$) to definitively test the conditional residual hypothesis.\n