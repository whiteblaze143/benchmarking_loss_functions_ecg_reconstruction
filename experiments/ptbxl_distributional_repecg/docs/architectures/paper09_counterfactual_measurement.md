# Paper 09: Counterfactual Measurement-Operator

## The Core Question
An ECG is just a 2D "shadow" (a projection) of the heart's true 3D electrical activity. Can the model learn the true 3D physics of the heart, independent of where the doctor happened to physically stick the electrodes on the patient's chest?

## The Architecture
The model learns a causal representation $Z_S$ from context pairs: (the recorded waveform + the spatial operator $q$ representing the physical lead vector used to record it). 
We then attach a generative network $g_\theta(Z_S, q_*)$ and ask it a strict counterfactual question: *"What would the waveform look like if we had placed the electrodes at angle $q_*$ instead of $q$?"*

## The Biological Context
This forces the representation to isolate the true biological state of the heart (the 3D electrical dipole vector) from the artifactual measurement operator (the camera angle/lead vector). It cleanly separates the *heart* from the *camera*.

## Rigorous Falsification & Controls
- **Matched Control**: Standard representation learning without spatial operator prediction.
- **Falsification (The Kill Test)**: We randomly mismatch the operator $q$ and the waveform during training. If the model cannot predict the counterfactual lead when presented with a new camera angle, the causal claim fails entirely.
