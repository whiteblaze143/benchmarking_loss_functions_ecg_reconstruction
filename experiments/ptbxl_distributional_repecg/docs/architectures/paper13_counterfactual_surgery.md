# Paper 13: Counterfactual Distribution Surgery

## The Core Question
How do we prove that a specific wave *causes* a diagnosis, rather than just being correlated with it? (e.g., Does the ST-elevation actually trigger the Myocardial Infarction diagnosis?)

## The Math
Pearl's structural `do()` calculus.

## The Architecture
During the forward pass of the network, we surgically slice out specific phase-cell KMEs $P_g$ and replace them with a matched healthy reference: $do(P_g = P_g^{ref})$. 
We then measure:
- **Necessity**: Does the diagnosis disappear when we remove the pathological wave?
- **Sufficiency**: Does the diagnosis appear when we inject a diseased wave into a healthy heart signal?

## The Mechanism (Biological Context)
This moves model interpretability beyond correlational heatmaps (like Grad-CAM, which just highlights where the model looked) into true counterfactual causal reasoning, proving that the model relies on the correct biological structures.

## The Falsification & Control
- **Matched Control**: Standard feature masking (dropout).
- **Falsification**: Applying mismatched or completely random distribution surgeries instead of context-aligned causal surgery. The rigorous necessity/sufficiency scores should collapse.
