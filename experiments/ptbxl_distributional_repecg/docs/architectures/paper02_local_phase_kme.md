# Paper 2: Local Phase Kernel Mean Embeddings

## The Core Question
Do we really need to explicitly calculate a dense, heavy $N \times N$ recurrence matrix, or can a neural network just look at the sequence of local KMEs and figure out the transitions itself?

## The Math & Object
Instead of a pairwise matrix, the mathematical object here is just the raw temporal sequence of kernel means $\mathbf{K} = (K_1, K_2, \dots, K_G)$.

## The Architecture
A lightweight 1D Convolutional Neural Network (CNN). The 1D-CNN slides its filters sequentially over the temporal sequence of KME vectors. 

## The Mechanism (Biological Context)
CNNs are highly efficient and translationally invariant. By sliding a CNN over KMEs rather than raw waveforms, the network focuses on macro-state transitions (e.g., the smooth transition from QRS to ST segment) rather than overfitting to high-frequency micro-voltage noise. The receptive field naturally aggregates neighboring phase distributions into larger physiological macro-states.

## The Falsification & Control
- **Matched Control**: Standard 1D-CNN applied to the raw 8-lead physical voltages.
- **Falsification**: We apply a temporal smoothing filter (a mechanism-use destroyer). If the model is genuinely learning transitions between local distributions, over-smoothing the phases will collapse the transitions and destroy the model's accuracy.
