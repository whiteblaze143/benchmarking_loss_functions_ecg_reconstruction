# Paper 10: Interventional repStat

## The Core Question
Real-world medical data is messy. Different hospitals use different sampling rates, noise filters, amplifiers, and lead subsets. How do we make the model completely immune to this?

## The Architecture
We factorize the network into two distinct parts: $Z_A$ and $Z_S$.
- $Z_A$ (Intervention Vector): explicitly predicts the messy dataset intervention (e.g., is this from a 250Hz machine or a 500Hz machine?).
- $Z_S$ (Diagnostic Vector): is constrained via an MMD penalty to be completely invariant to the intervention. Diagnosis is predicted *exclusively* from $Z_S$.

## The Mechanism (Biological Context)
This guarantees that the diagnostic features are completely invariant to the hospital's data collection practices, preventing the model from diagnosing a patient based on the brand of ECG machine used.

## The Falsification & Control
- **Matched Control**: Standard domain adversarial training (DANN).
- **Falsification**: A random, non-causal perturbation is applied rather than the controlled dataset intervention. The invariance must specifically protect against known data-collection interventions.
