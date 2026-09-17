# Paper 12: Structural-Innovation repECG

## The Core Question
The normal sinus rhythm of a heart is highly predictable (a P wave is almost always followed by a QRS complex). True pathological information often lies in the *innovation*—the unpredictable residual that deviates from the cycle (like a sudden ectopic beat).

## The Architecture
A Conditional Normalizing Flow (CNF). The flow receives the autoregressive past phase states $Z_{<g}$ and explicitly isolates the structural innovation $U_g = f_\theta^{-1}(Z_g ; Z_{<g})$—the part of the signal that mathematically could not be predicted by the past.

## The Mechanism (Biological Context)
This disentangles the deterministic, healthy cyclic rhythm of the heart from unpredictable, localized structural anomalies. By isolating $U_g$, we can pinpoint exactly when and where the heart did something unexpected.

## The Falsification & Control
- **Matched Control**: Raw residual prediction.
- **Falsification**: We reverse the phase order during training ($Z_G, \dots, Z_1$). Because causation flows forward in time, reversing the causal directionality must destroy the validity of the structural innovations.
