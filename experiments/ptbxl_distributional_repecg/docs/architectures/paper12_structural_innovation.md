# Paper 12: Structural-Innovation repECG

## The Core Question
The normal sinus rhythm of a healthy heart is highly predictable (a P wave is almost always followed by a QRS complex, which is followed by a T wave). True pathological diagnostic information often lies in the *innovation*—the unpredictable residual that deviates from the cycle (like a sudden ectopic beat or a dropped block).

## The Architecture
We utilize a **Conditional Normalizing Flow (CNF)**. The flow receives the autoregressive past phase states $Z_{<g}$ and explicitly isolates the structural innovation $U_g = f_\theta^{-1}(Z_g ; Z_{<g})$—the exact mathematical part of the signal that could not be predicted by the past.

## The Biological Context
This disentangles the deterministic, healthy cyclic rhythm of the heart from unpredictable, localized structural anomalies. By isolating $U_g$, a doctor can pinpoint exactly when and where the heart did something physiologically unexpected.

## Rigorous Falsification & Controls
- **Matched Control**: Raw residual prediction using an L2 loss.
- **Falsification (The Kill Test)**: We artificially reverse the phase order during training ($Z_G, \dots, Z_1$). Because biological causation physically flows forward in time, reversing the causal directionality must mathematically destroy the validity of the structural innovations.
