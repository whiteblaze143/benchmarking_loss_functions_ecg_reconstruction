# Paper 1: Distributional Recurrence Operator

## The Core Question
Traditional recurrence plots measure whether the raw voltage at time $t_i$ is close to the voltage at $t_j$. This model upgrades that concept to the distributional level: How similar is the electrical *distribution* of the heart right now to its distribution 200 milliseconds ago?

## The Math
We compute an $N \times N$ Maximum Mean Discrepancy (MMD) recurrence matrix. MMD is a statistical test that measures the distance between two distributions in the RKHS. The matrix captures the MMD distance between every phase cell and every other phase cell across the entire cardiac cycle.

## The Architecture
The dense $N \times N$ recurrence matrix is flattened into a 1D vector and fed into a simple Linear Probe or a lightweight Multi-Layer Perceptron (MLP) to predict the 5 diagnostic superclasses.

## The Mechanism (Biological Context)
This operator captures long-range, non-linear temporal correlations (e.g., how the P-wave distribution relates to the T-wave distribution) without requiring deep recurrent networks (like LSTMs). It provides a holistic, pairwise map of how every phase of the heartbeat statistically interacts with every other phase.

## The Falsification & Control
- **Matched Control**: Must outperform a naive "mean recurrence" control (which just averages the voltages instead of comparing distributions).
- **Falsification**: Destroyed by shuffling the temporal order of the phase blocks. If the network is actually using temporal recurrence, this shuffling should destroy its accuracy.
