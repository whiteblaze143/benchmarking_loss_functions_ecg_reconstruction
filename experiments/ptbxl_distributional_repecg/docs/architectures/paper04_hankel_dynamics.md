# Paper 04: Hankel Dynamics & Dynamic Mode Decomposition

## The Core Question
Can we model the beating heart using the physics of dynamical systems? Specifically, can we treat the heart as a series of oscillating springs and physical state transitions, rather than just a pattern recognition problem?

## The Mathematical Framework: DMD
We rely on **Dynamic Mode Decomposition (DMD)**, a linear algebra technique heavily used in fluid dynamics and physics.
1. We construct a **Hankel Matrix** from the temporal sequence of phase cells. A Hankel matrix is formed by shifting the time series data so that each column represents a delayed sliding window of the heart state.
2. We apply DMD to this matrix to extract the underlying linear operator governing the state transition.
3. DMD yields **eigenvalues**. The magnitude of an eigenvalue dictates the growth or decay rate of the biological signal, and its phase dictates the oscillation frequency.

## The Architecture
We extract the dynamic modes (the complex eigenvalues and their corresponding eigenvectors) from the Hankel matrices spanning the heartbeat. These modes are passed into a shallow neural network that uses them as its sole input features to diagnose the heart.

## The Biological Context
This provides a strictly physics-inspired view of the ECG. Abnormalities like myocardial ischemia (lack of oxygen to the heart muscle) physically alter the cellular action potentials of the cardiac myocytes. This biological delay directly manifests as measurable changes in the local decay rates of the repolarization phases (the T-wave). DMD explicitly and mathematically extracts this decay rate as a clear feature, bridging deep learning with physical biophysics.

## Rigorous Falsification & Controls
- **Matched Control**: Standard temporal pooling over the phase cells.
- **Falsification (The Kill Test)**: We shuffle the local phase cells chronologically. DMD relies entirely on the smooth dynamical transition of time. If the model is genuinely relying on the physical decay rate of the dynamical system, breaking the sequence will destroy the DMD eigenvalues and cause an immediate collapse in diagnostic accuracy.
