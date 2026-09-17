# Paper 11: repStat Causal-State ECG

## The Core Question
What truly defines a "state" in a biological system? If two ECGs look wildly different in the past but result in the exact same mathematical future, shouldn't they be considered the exact same state?

## The Mathematical Framework: Computational Mechanics
Based on Computational Mechanics and $\epsilon$-machines. A state is defined *exclusively* by its ability to predict the future. 
We map historical sequences of the ECG $H$ into continuous embeddings $h$. We then run a rigorous statistical test: If two different historical embeddings $h_i$ and $h_j$ produce the exact same future probability distribution $P(F \mid h)$, they are mapped into the exact same Causal State $\epsilon_k$.

## The Biological Context
This shifts the entire paradigm of clinical ECG analysis. We stop looking at *"what did the signal look like in the past"* and focus entirely on *"what are the statistical bounds on what the heart will do next"*. It naturally calculates the Statistical Complexity $C_\mu$ and Entropy Rate $h_\mu$ of the patient's rhythm.

## Rigorous Falsification & Controls
- **Matched Control**: Past-history predictive state (a standard RNN predicting the next step).
- **Falsification (The Kill Test)**: Chronologically shuffling the temporal links to break future predictability. This must immediately destroy the causal state machine's ability to cluster the states.
