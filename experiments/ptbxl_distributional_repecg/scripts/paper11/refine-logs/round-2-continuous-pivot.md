# Round 2 Refinement — Continuous Predictive Compression

## Anchor Check

The unchanged problem is causal-prefix compression of ordered beats. The categorical formulation failed because a four-state bottleneck was not reliably sufficient. Replacing it with a continuous code changes the mechanism but not the problem; claiming causal-state recovery would be drift and is rejected.

## Simplicity Check

The revised method has one trainable encoder-decoder and one explicit bottleneck. It removes the categorical classifier, state embeddings, diagnosis head, and any state-label recovery claim. No foundation-model component is needed for this signal-level mechanism.

## Evidence and Decision

The original continuous 400-update protocol failed on seed 44 and remains failed. A diagnostic showed this was an insufficient optimization budget. Continuous G1 v2 fixed 1,000 updates, retained all numerical margins, used fresh seeds 45–47, and passed every gate. The evidence supports a synthetic continuous predictive-compression claim only. Real-ECG G0 is the next gate.
