# Paper 8: Local Token Cross-Attention

## The Core Question
For Patient A (who has a heart block), the most important diagnostic clue is early in the heartbeat. For Patient B (who has ischemia), it's late. How can the network dynamically focus its attention?

## The Architecture
We treat the sequence of phase-cell KMEs as "tokens" (exactly like words in a sentence for a Large Language Model). We apply a Transformer multi-head self-attention and cross-attention mechanism over these phase tokens.

## The Mechanism (Biological Context)
Unlike static CNN kernels (which always look at the same fixed window), attention weights allow the network to dynamically route information between distinct cardiac phases. This provides high interpretability: we can extract the attention maps to see exactly which phase tokens the network deemed most important for diagnosing a specific patient.

## The Falsification & Control
- **Matched Control**: CNN sequence modeling (Paper 2).
- **Falsification**: Scrambling the attention matrices or applying a uniform-attention mask. If the model truly relies on dynamic routing, forcing uniform attention will collapse its performance.
