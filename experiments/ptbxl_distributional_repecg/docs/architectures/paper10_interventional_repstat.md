# Paper 10: Interventional repStat

## The Core Question
Real-world medical data is messy. Different hospitals use different sampling rates, analog noise filters, operational amplifiers, and lead subsets. How do we make the model completely immune to the hospital's specific hardware?

## The Architecture
We explicitly factorize the network into two strictly distinct latent spaces: $Z_A$ and $Z_S$.
- **$Z_A$ (Intervention Vector)**: explicitly attempts to predict the messy dataset intervention (e.g., "Is this recording from a 250Hz EchoNext machine or a 500Hz PTB-XL machine?").
- **$Z_S$ (Diagnostic Vector)**: is mathematically constrained via an MMD penalty to be completely invariant to the intervention. Diagnosis of the disease is predicted *exclusively* from $Z_S$.

## The Biological Context
This guarantees that the diagnostic features are completely invariant to the hospital's data collection practices. It physically prevents the neural network from diagnosing a patient based on the brand of ECG machine the hospital bought, forcing it to rely only on the patient's actual biology.

## Rigorous Falsification & Controls
- **Matched Control**: Standard Domain Adversarial Neural Networks (DANN).
- **Falsification (The Kill Test)**: A random, non-causal perturbation is applied rather than the controlled, known dataset hardware intervention. The invariance mechanism must specifically protect against known data-collection interventions, but fail to protect against unstructured random noise.
