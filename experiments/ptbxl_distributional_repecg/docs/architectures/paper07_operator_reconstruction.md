# Paper 07: Operator Reconstruction Auxiliary

## The Core Question
How do we prevent a neural network from "cheating"? Deep learning networks often discard all the actual biology of the heart and just memorize a single spurious pixel that gives away the diagnosis (shortcut learning). How do we prove the model actually understands the heart?

## The Architecture
We build a dual-head network acting as a severe **information bottleneck**. A shared encoder processes the KMEs into a tight latent space.
1. **Head 1 (Diagnostic)**: The primary classification head that guesses the disease.
2. **Head 2 (Generative)**: A dense generative reconstruction decoder (e.g., a transposed CNN) that is forced to perfectly reconstruct the original, raw, physical-mV 8-lead ECG waveform *using only the KME latent bottleneck*.

## The Biological Context
This mathematically guarantees biological grounding. If the model can accurately reconstruct the physical raw voltage signal from its internal representations, we have absolute mathematical proof that it hasn't discarded the biophysical reality of the heart just to optimize the diagnostic loss function. The network is forced to retain a holistic understanding of the patient's cardiac structure.

## Rigorous Falsification & Controls
- **Matched Control**: The exact same architecture trained without the reconstruction generative head.
- **Falsification (The Kill Test)**: We measure the Structural Similarity (SSIM) and Mean Squared Error (MSE) of the reconstructed physical waveform. The rule states that the diagnostic accuracy must remain highly stable even when the reconstruction penalty is highly weighted. If accuracy collapses when forced to reconstruct, the baseline model was cheating.
