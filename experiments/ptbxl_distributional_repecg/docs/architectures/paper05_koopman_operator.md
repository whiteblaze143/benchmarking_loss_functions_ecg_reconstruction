# Paper 05: The Koopman Operator

## The Core Question
The heart is a wildly non-linear biological system, making it mathematically chaotic and difficult to predict over time. Can we force it to behave linearly?

## The Mathematical Framework: Koopman Theory
**Koopman Operator Theory** (from 1931) states a profound mathematical truth: *Any non-linear dynamical system can be represented as a perfectly linear system if you lift the state observations into an infinite-dimensional space.*

Because our Foundational Pipeline already lifts the ECG into an infinite-dimensional RKHS using Kernel Mean Embeddings, the system is perfectly primed for Koopman Theory. We use **Extended Dynamic Mode Decomposition (EDMD)** operating on the RKHS features to approximate this finite-dimensional linear Koopman operator, $K$.

## The Architecture
We design a specialized linear neural network layer operating strictly in the lifted KME space. Because the temporal evolution is linearized in this lifted space, the model predicts the forward temporal evolution of the ECG using simple matrix multiplication: $Z_{t+1} = K Z_t$.

## The Biological Context
By lifting the highly non-linear physical voltages of the ECG into the statistical RKHS, the chaotic transitions of the heart are mathematically linearized. This makes long-term prediction, stability analysis, and identifying invariant sub-spaces of the cardiac cycle analytically tractable. We can mathematically predict what the state of the heart will be in 50 milliseconds using a simple, closed-form linear equation.

## Rigorous Falsification & Controls
- **Matched Control**: A standard non-linear Recurrent Neural Network (RNN) operating on the same features.
- **Falsification (The Kill Test)**: We evaluate the model's ability to explicitly predict the *future* RKHS state. If Koopman theory holds and the operator was learned successfully, the linear operator must be able to project the state forward accurately. If the forward projection MSE is high, the model failed to learn the true Koopman operator.
