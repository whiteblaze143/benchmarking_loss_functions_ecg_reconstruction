# Paper 01: Distributional Recurrence Operator

## The Core Question
How do different phases of the cardiac cycle interact with each other statistically? 

Traditional recurrence plots in physics measure whether the raw scalar voltage at time $t_i$ is close to the voltage at $t_j$. This model upgrades that concept to the distributional level: **How similar is the electrical *distribution* of the heart right now to its distribution 200 milliseconds ago?**

## The Mathematical Framework: Maximum Mean Discrepancy (MMD)
We utilize a concept from kernel statistics called **Maximum Mean Discrepancy (MMD)**. MMD is a rigorous statistical test that measures the distance between two probability distributions by calculating the distance between their Kernel Mean Embeddings in the RKHS.

If $P$ is the distribution of the phase cell early in the beat, and $Q$ is the distribution late in the beat, MMD calculates how mathematically distinct these two distributions are without needing to estimate their probability density functions directly.

## The Architecture
1. We compute an $N \times N$ dense recurrence matrix. Every cell $(i, j)$ in this matrix represents the MMD distance between phase cell $i$ and phase cell $j$.
2. This provides a holistic, pairwise map of how every phase of the heartbeat statistically interacts with every other phase.
3. This symmetric matrix is flattened into a 1D vector.
4. The vector is fed into a simple Linear Probe or a lightweight Multi-Layer Perceptron (MLP) to predict the 5 diagnostic cardiac superclasses (Normal, Myocardial Infarction, ST/T Change, Conduction Disturbance, Hypertrophy).

## The Biological Context
This operator captures long-range, non-linear temporal correlations. For example, it directly models how the geometry of the P-wave relates to the geometry of the T-wave. It achieves this without requiring deep, unstable recurrent networks like LSTMs, providing a stable, mathematically guaranteed measure of cyclic cardiac recurrence.

## Rigorous Falsification & Controls
In repECG, a model must prove that it is actually using the mathematical mechanism we designed, not just cheating.
- **Matched Control**: We compare the model against a naive "mean recurrence" control, which simply averages the physical voltages and computes standard Euclidean distance instead of comparing true RKHS distributions.
- **Falsification (The Kill Test)**: We randomly shuffle the temporal order of the phase blocks before computing recurrence. If the network is genuinely learning temporal recurrence dependencies, this shuffling must destroy its accuracy. If the accuracy remains high, the network was cheating, and the causal claim is falsified.
