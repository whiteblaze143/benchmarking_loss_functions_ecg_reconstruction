# Paper 02: Local Phase Kernel Mean Embeddings (Phase-CNN)

## The Core Question
Do we really need to explicitly calculate and store a massive, dense $N \times N$ recurrence matrix (as in Paper 01), or can a neural network simply look at the chronological sequence of local KMEs and figure out the transitions itself?

## The Mathematical Object
Instead of a pairwise matrix, the mathematical object fed into this model is just the raw temporal sequence of the kernel means: $\mathbf{K} = (K_1, K_2, \dots, K_G)$. 
Here, $G$ is the number of phase cells in a heartbeat, and each $K_i$ is the $D$-dimensional Nyström approximation of that cell's distribution.

## The Architecture
We employ a lightweight 1D Convolutional Neural Network (CNN). 
The 1D-CNN slides its convolutional filters sequentially over the temporal sequence of KME vectors, treating the sequence of distributions exactly like a 1D sequence of pixels.

## The Biological Context
CNNs are highly computationally efficient and translationally invariant. By sliding a CNN over statistical KMEs rather than raw waveforms, the network is forced to focus on macro-state transitions (e.g., the smooth, distributional transition from the QRS complex into the ST segment) rather than overfitting to high-frequency micro-voltage noise. The receptive field of the CNN naturally aggregates neighboring phase distributions into larger, recognizable physiological macro-states.

## Rigorous Falsification & Controls
- **Matched Control**: A standard 1D-CNN with identical capacity applied directly to the raw 8-lead physical voltages.
- **Falsification (The Kill Test)**: We apply a heavy temporal smoothing filter (a moving average) to the sequence of KMEs. If the model is genuinely learning local, sharp transitions between distinct phase distributions, over-smoothing the phases will collapse these boundaries and significantly destroy the model's accuracy.
