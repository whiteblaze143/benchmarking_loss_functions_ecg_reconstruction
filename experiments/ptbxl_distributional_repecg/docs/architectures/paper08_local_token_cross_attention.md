# Paper 08: Local Token Cross-Attention

## The Core Question
For Patient A (who has a conduction block), the most important diagnostic clue is early in the heartbeat (the P-R interval). For Patient B (who has ischemia), it's late (the ST segment). Because diseases manifest at different times, how can the network dynamically focus its attention differently for every patient?

## The Architecture
We treat the sequence of phase-cell KMEs as independent "tokens" (conceptually identical to how words in a sentence are treated by Large Language Models like GPT). We apply a **Transformer multi-head self-attention and cross-attention mechanism** over these phase tokens.

## The Biological Context
Unlike static CNN kernels (which are rigid and always look at the same fixed temporal window), attention weights allow the network to dynamically route information between distinct cardiac phases based on the patient's specific presentation. This provides incredibly high interpretability: a cardiologist can extract the attention weight matrices to see exactly which phase tokens the network deemed most important for diagnosing that specific patient.

## Rigorous Falsification & Controls
- **Matched Control**: Rigid CNN sequence modeling (Paper 02).
- **Falsification (The Kill Test)**: Scrambling the attention matrices or applying a uniform-attention mask during inference. If the model truly relies on dynamic, patient-specific routing, forcing uniform attention will collapse its performance.
