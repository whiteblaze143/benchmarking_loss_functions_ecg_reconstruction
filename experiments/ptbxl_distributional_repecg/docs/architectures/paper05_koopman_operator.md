# Paper 5: Koopman Operator

## The Core Question
The heart is a wildly non-linear system, making it mathematically difficult to predict. Can we force it to behave linearly?

## The Math
**Koopman Operator Theory** states that any non-linear dynamical system can be represented as a perfectly linear system if you lift it into an infinite-dimensional space. We use Extended Dynamic Mode Decomposition (EDMD) operating on the RKHS features to approximate this finite-dimensional linear Koopman operator.

## The Architecture
A specialized linear neural network layer operating in the lifted KME space. Because it is linear in this lifted space, it can model the forward temporal evolution of the ECG using simple matrix multiplication.

## The Mechanism (Biological Context)
By lifting the highly non-linear ECG into the RKHS, the transitions of the heart are linearized. This makes long-term prediction, stability analysis, and identifying invariant sub-spaces of the cardiac cycle analytically tractable. We can mathematically predict what the heart state will be in 50 milliseconds using a simple linear equation.

## The Falsification & Control
- **Matched Control**: A standard non-linear recurrent neural network (RNN).
- **Falsification**: We evaluate its ability to explicitly predict the *future* RKHS state. If Koopman theory holds, the linear operator should be able to project the state forward accurately.
