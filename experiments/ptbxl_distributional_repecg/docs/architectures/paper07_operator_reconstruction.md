# Paper 7: Operator Reconstruction Auxiliary

## The Core Question
How do we prevent a neural network from "cheating"? Deep networks often discard all the actual biology of the heart and just memorize the single pixel that gives away the diagnosis (shortcut learning).

## The Architecture
A dual-head network acting as a severe information bottleneck. A shared encoder processes the KMEs.
1. **Head 1**: The primary diagnostic classification head (guesses the disease).
2. **Head 2**: A dense generative reconstruction decoder (e.g., a transposed CNN) that is forced to perfectly reconstruct the original, raw, physical-mV ECG waveform *using only the KME latent bottleneck*.

## The Mechanism (Biological Context)
This mathematically guarantees biological grounding. If the model can accurately reconstruct the physical raw voltage signal from its internal representations, we have proof that it hasn't discarded the biophysical reality of the heart just to optimize the diagnostic loss function.

## The Falsification & Control
- **Matched Control**: The exact same architecture trained without the reconstruction head.
- **Falsification**: We measure the structural similarity (SSIM) and MSE of the reconstructed waveform. The kill rule states that the diagnostic accuracy must remain stable even when the reconstruction penalty is highly weighted.
