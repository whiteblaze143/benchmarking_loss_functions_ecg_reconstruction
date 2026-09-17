# Paper 4: Hankel Dynamics

## The Core Question
Can we model the heartbeat using physics? Specifically, can we treat the heart as a series of oscillating springs and physical dynamical systems?

## The Math
We construct Hankel matrices (matrices of time-shifted data) from the phase cells. We then apply **Dynamic Mode Decomposition (DMD)**. DMD is a linear algebra technique that extracts the eigenvalues—which represent the growth/decay rates and oscillation frequencies—of the local underlying dynamical system.

## The Architecture
A shallow neural network that takes the sequential dynamic modes (the eigenvalues and eigenvectors extracted from the Hankel matrices) as its input features to diagnose the heart.

## The Mechanism (Biological Context)
This provides a strictly physics-inspired view of the ECG. Abnormalities like ischemia (lack of oxygen to the heart muscle) physically alter the cellular action potentials, which manifests as changes in the local decay rates of the repolarization phases (the T-wave). DMD explicitly extracts this decay rate as a measurable mathematical feature.

## The Falsification & Control
- **Matched Control**: Standard temporal pooling.
- **Falsification**: Shuffling the local phase cells to destroy the smooth dynamical transition. If the model is relying on the physical decay rate, breaking the sequence will destroy the DMD eigenvalues and collapse the accuracy.
