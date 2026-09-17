# Paper 13: Counterfactual Distribution Surgery

## The Core Question
How do we mathematically prove that a specific wave *causes* a diagnosis, rather than just being correlated with it? (e.g., Does the elevated ST-segment actually trigger the Myocardial Infarction diagnosis, or is the model looking at something else?)

## The Mathematical Framework: Pearl's Do-Calculus
We implement Judea Pearl's structural `do()` calculus inside the neural network.

## The Architecture
During the forward pass of the network, we surgically slice out specific phase-cell KMEs $P_g$ (e.g., the T-wave) and explicitly replace them with a matched healthy reference distribution: $do(P_g = P_g^{ref})$. 
We then measure:
1. **Necessity**: Does the MI diagnosis disappear when we surgically remove the pathological wave?
2. **Sufficiency**: Does the MI diagnosis appear when we surgically inject a diseased wave into a healthy patient's signal?

## The Biological Context
This moves deep learning interpretability far beyond correlational heatmaps (like Grad-CAM, which only highlights where the model looked) into true counterfactual causal reasoning. It provides mathematical proof that the model relies on the correct, doctor-approved biological structures to make life-or-death decisions.

## Rigorous Falsification & Controls
- **Matched Control**: Standard feature masking (e.g., dropout or zeroing out features).
- **Falsification (The Kill Test)**: Applying mismatched or completely random distribution surgeries instead of context-aligned causal surgery. The rigorous necessity and sufficiency causal scores must collapse to baseline if the surgery is random.
