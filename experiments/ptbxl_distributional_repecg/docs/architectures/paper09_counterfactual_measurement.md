# Paper 9: Counterfactual Measurement-Operator

## The Core Question
An ECG is just a 2D "shadow" of the heart's true 3D electrical activity. Can the model learn the true 3D physics of the heart, independent of where the doctor happened to place the physical electrodes?

## The Architecture
The model learns a causal representation $Z_S$ from context pairs (the waveform + the operator $q$ used to record it). We then attach a generative network $g_\theta(Z_S, q_*)$ and ask it a counterfactual question: "What would the waveform look like if we had placed the electrodes at angle $q_*$ instead of $q$?"

## The Mechanism (Biological Context)
This forces the representation to isolate the true biological state of the heart (the 3D dipole) from the artifactual measurement operator (the camera angle/lead vector). It separates the *heart* from the *camera*.

## The Falsification & Control
- **Matched Control**: Standard representation learning without operator prediction.
- **Falsification**: Destroyed by randomly mismatching the operator $q$ and the waveform during training. If the model cannot predict the counterfactual lead, the causal claim fails.
